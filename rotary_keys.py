"""CPU 旋转位置编码（RoPE）与分块键交付。

模块分为三层：

``View`` / ``Segment``
    纯函数式的分段旋转位置编码。支持 ``interleaved`` / ``split`` 两种通道
    排列，只旋转 ``rotary_dim`` 前缀，按全局位置所在段更换频率与原点。

``KeyStore``
    分块键的物理存储。同一源块可按不同位置视图、不同目标量化规格派生；
    派生表示按真实常驻字节计费，可被多个读者共享。交付出去的表示在读者
    归还前不可变，另一个会话扩窗或换表不会改写它。

``ExchangeRunner``
    读取项目的交换资料，驱动 ``borrow`` / ``return`` / ``retry`` 会话流程，
    并提供逐查询的可用键、相位来源、误差界与拒绝原因审计。

数值约定
--------
* 正角逆时针：``(a, b) -> (c*a - s*b, s*a + c*b)``，``c=cos θ``，``s=sin θ``。
* ``θ = (全局位置 - 段原点) × 段频率``，原点已包含会话重编号。
* 位置视图只改变旋转前缀；未旋转分量原样透传，但仍须满足目标数值契约。
* 键块 ``q`` 一律按 ``scale × (q - zero)`` 解码，无论表示原始键还是已旋转键。

误差约定
--------
交付误差由三项可证明上界相加：

``source``
    源表示相对其相位理想值的逐行 L2 上界（交换资料的 ``l2_source_error``）。
    源是已旋转键时，撤销/重做旋转是正交变换，L2 上界不变。
``arithmetic``
    float64 解码与旋转的算术误差上界。
``quantization``
    目标格式的舍入误差上界。int8 为 ``scale/2 × sqrt(宽度)``；越界直接
    拒绝而不得钳位。FP32 为相对舍入单位乘以范数。

打分误差上界为 ``query_norm_ceiling × L2 上界``。可交付性只用这些先验
上界判断，不使用“新旧解码值之差”这一事后量。存在原始 FP32 键块时按精确
契约办理（源误差记 0），否则按公开界判断能否用于新视图。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "Acquisition",
    "BlockPlan",
    "BorrowResult",
    "Contract",
    "DeliveredBlock",
    "ExchangeError",
    "ExchangeRunner",
    "KeyStore",
    "OutOfSegment",
    "Segment",
    "StoredBlock",
    "TargetFormat",
    "TargetOverflow",
    "View",
    "apply_rotation",
    "decode_int8",
    "quantize",
    "rotate_query",
    "STATUS_READY",
    "STATUS_WAITING",
    "STATUS_UNPROVABLE",
    "STATUS_IMPOSSIBLE",
]

# 请求级 / 块级状态。
STATUS_READY = "ready"
STATUS_WAITING = "waiting"
STATUS_UNPROVABLE = "unprovable"
STATUS_IMPOSSIBLE = "impossible"

# 相位来源，审计中必须能区分，绝不混用。
PHASE_EXACT = "exact_original"
PHASE_ROTATED = "rotated_source"
PHASE_NEUTRAL = "neutral_raw"

F32_UNIT = 2.0 ** -24          # float32 相对舍入单位（最近舍入，平局取偶）
F32_ABS = 2.0 ** -149          # 覆盖 float32 次正规数
F64_EPS = float(np.finfo(np.float64).eps)

_LAYOUTS = ("interleaved", "split")
_DTYPES = ("fp32", "int8")

# 块级拒绝 / 排除原因。
R_MISSING = "missing_block"
R_UNKNOWN_PHASE = "unknown_phase_view"
R_OUT_OF_SEGMENT = "position_out_of_segments"
R_NO_VISIBLE = "no_visible_positions"
R_OUT_OF_RANGE = "outside_session_range"
R_ERROR_BOUND = "error_bound_exceeded"
R_OVERFLOW = "target_overflow"
R_RESIDENT = "resident_budget"
R_TEMPORARY = "temporary_budget"
R_RELEASED = "released"


class ExchangeError(ValueError):
    """交换资料不满足约定。"""


class OutOfSegment(ExchangeError):
    """位置不在视图的任何半开区间内。"""


class TargetOverflow(ExchangeError):
    """目标 int8 定标无法表示该值；绝不钳位或回绕。"""


def _as_int_positions(positions: Iterable[int]) -> tuple[int, ...]:
    out: list[int] = []
    for p in positions:
        try:
            ip = int(p)
        except (TypeError, ValueError) as exc:
            raise ExchangeError(f"位置必须是整数：{p!r}") from exc
        if ip != p:
            raise ExchangeError(f"位置必须是整数：{p!r}")
        out.append(ip)
    return tuple(out)


# ---------------------------------------------------------------------------
# 旋转位置编码
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """全局位置半开区间 ``[start, end)`` 上的一对维度频率与原点。"""

    start: int
    end: int
    origin: float
    freqs: tuple[float, ...]

    def covers(self, position: int) -> bool:
        return self.start <= position < self.end

    def angle(self, position: int, pair: int) -> float:
        return (position - self.origin) * self.freqs[pair]

    def as_tuple(self) -> tuple:
        return (self.start, self.end, self.origin, self.freqs)


@dataclass(frozen=True)
class View:
    """一个位置视图：通道排列、旋转前缀、分段频率表与目标交付格式。"""

    name: str
    layout: str
    rotary_dim: int
    segments: tuple[Segment, ...]
    dtype: str = "fp32"
    scale: float | None = None
    zero: float = 0.0
    revision: int = 0

    def __post_init__(self) -> None:
        if self.layout not in _LAYOUTS:
            raise ExchangeError(f"未知通道排列：{self.layout!r}")
        if self.dtype not in _DTYPES:
            raise ExchangeError(f"未知目标格式：{self.dtype!r}")
        if self.rotary_dim < 0 or self.rotary_dim % 2:
            raise ExchangeError("rotary_dim 必须为非负偶数")
        if self.dtype == "int8":
            if self.scale is None or not math.isfinite(self.scale) or self.scale <= 0:
                raise ExchangeError("int8 视图需要正的有限 scale")
        pairs = self.rotary_dim // 2
        if not self.segments:
            raise ExchangeError("视图至少需要一个分段")
        previous_end: int | None = None
        for seg in self.segments:
            if seg.end <= seg.start:
                raise ExchangeError("分段区间必须非空")
            if len(seg.freqs) != pairs:
                raise ExchangeError(
                    f"分段 {seg.as_tuple()[:2]} 的频率数 {len(seg.freqs)} "
                    f"与 rotary_dim/2={pairs} 不符"
                )
            if previous_end is not None and seg.start < previous_end:
                raise ExchangeError("分段区间不得重叠")
            previous_end = seg.end

    def pairs(self) -> tuple[tuple[int, int], ...]:
        half = self.rotary_dim // 2
        if self.layout == "interleaved":
            return tuple((2 * i, 2 * i + 1) for i in range(half))
        return tuple((i, i + half) for i in range(half))

    def segment_for(self, position: int) -> Segment:
        for seg in self.segments:
            if seg.covers(position):
                return seg
        raise OutOfSegment(f"视图 {self.name!r} 的分段不覆盖位置 {position}")

    def angles(self, positions: Sequence[int]) -> np.ndarray:
        pos = _as_int_positions(positions)
        pairs = self.rotary_dim // 2
        out = np.empty((len(pos), pairs), dtype=np.float64)
        for row, p in enumerate(pos):
            seg = self.segment_for(p)
            for j in range(pairs):
                out[row, j] = seg.angle(p, j)
        return out

    def signature(self) -> tuple:
        """内容签名。频率表真正变化时签名变化，旧派生表示不会被误用。"""

        return (
            self.layout,
            self.rotary_dim,
            self.dtype,
            self.scale,
            self.zero,
            tuple(seg.as_tuple() for seg in self.segments),
        )

    def rotation_signature(self) -> tuple:
        """只描述旋转本身（布局、前缀、分段频率表），不含交付格式。

        相同旋转的派生表示才可共享键相位；仅“目标格式相同”绝不足以共享。
        """

        return (
            self.layout,
            self.rotary_dim,
            tuple(seg.as_tuple() for seg in self.segments),
        )

    def target_format(self) -> "TargetFormat":
        return TargetFormat(self.dtype, self.scale, self.zero)

    def label(self) -> str:
        return f"{self.name}@{self.revision}"

    @classmethod
    def from_spec(
        cls, name: str, spec: Mapping[str, Any], rotary_dim: int, revision: int = 0
    ) -> "View":
        segments = tuple(
            Segment(
                int(start),
                int(end),
                float(origin),
                tuple(float(f) for f in freqs),
            )
            for start, end, origin, freqs in spec["segments"]
        )
        return cls(
            name=name,
            layout=spec["layout"],
            rotary_dim=int(rotary_dim),
            segments=segments,
            dtype=spec.get("dtype", "fp32"),
            scale=spec.get("scale"),
            zero=float(spec.get("zero", 0.0)),
            revision=int(revision),
        )


def apply_rotation(
    vectors: Any,
    view: View,
    positions: Sequence[int],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """按视图旋转（或撤销旋转）向量行。

    只修改 ``rotary_dim`` 前缀；其余分量逐位保留。``vectors`` 可为
    ``(宽度,)`` 或 ``(行数, 宽度)``，行数必须与 ``positions`` 一致。
    返回 float64 副本，不修改输入。
    """

    arr = np.asarray(vectors, dtype=np.float64)
    single = arr.ndim == 1
    rows = arr.reshape(1, -1) if single else arr
    if rows.ndim != 2:
        raise ExchangeError("向量必须是 1 维或 2 维")
    if rows.shape[1] < view.rotary_dim:
        raise ExchangeError("向量宽度小于 rotary_dim")
    pos = _as_int_positions(positions)
    if len(pos) != rows.shape[0]:
        raise ExchangeError("位置数必须与向量行数一致")

    theta = view.angles(pos)
    out = rows.copy()
    sign = -1.0 if inverse else 1.0
    for j, (a, b) in enumerate(view.pairs()):
        angle = sign * theta[:, j]
        cos = np.cos(angle)
        sin = np.sin(angle)
        xa = out[:, a].copy()
        xb = out[:, b].copy()
        out[:, a] = cos * xa - sin * xb
        out[:, b] = sin * xa + cos * xb
    return out[0] if single else out


def rotate_query(query: Any, view: View, positions: Sequence[int]) -> np.ndarray:
    """把同一原始查询向量按多个位置旋转，返回 ``(位置数, 宽度)``。"""

    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1:
        raise ExchangeError("查询原始向量必须是一维")
    pos = _as_int_positions(positions)
    if not pos:
        return np.empty((0, q.shape[0]), dtype=np.float64)
    tiled = np.broadcast_to(q, (len(pos), q.shape[0])).copy()
    return apply_rotation(tiled, view, pos)


# ---------------------------------------------------------------------------
# 目标格式与量化
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetFormat:
    """交付目标格式：FP32，或对称 int8 定标 ``scale``（``zero`` 必须为 0）。

    对称定标的有效编码区间为 ``[-127, 127]``；越界即抛 ``TargetOverflow``，
    绝不钳位。``quant_l2_bound`` 给出逐行 L2 舍入上界，而不只是逐元素。
    """

    dtype: str
    scale: float | None = None
    zero: float = 0.0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ExchangeError(f"未知目标格式：{self.dtype!r}")
        if self.dtype == "int8":
            if self.scale is None or not math.isfinite(self.scale) or self.scale <= 0:
                raise ExchangeError("int8 目标需要正的有限 scale")
            if self.zero != 0:
                raise ExchangeError("对称 int8 目标要求 zero == 0")

    def signature(self) -> tuple:
        return (self.dtype, self.scale, self.zero)

    @property
    def itemsize(self) -> int:
        return 4 if self.dtype == "fp32" else 1

    def quantize(self, values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=np.float64)
        if self.dtype == "fp32":
            out = vals.astype(np.float32)
            finite = np.isfinite(out)
            if not bool(finite.all()):
                raise TargetOverflow("值超出 float32 范围")
            if np.any(np.abs(out) > np.finfo(np.float32).max):
                raise TargetOverflow("值超出 float32 范围")
            return out
        scaled = vals / float(self.scale) + self.zero
        encoded = np.rint(scaled)  # 最近舍入，平局取偶
        if np.any(encoded < -127) or np.any(encoded > 127):
            raise TargetOverflow(
                "int8 对称定标无法表示该值（有效区间 [-127, 127]）；"
                "不得钳位"
            )
        return encoded.astype(np.int8)

    def quant_l2_bound(self, row_count: int, width: int, norm_upper: float) -> float:
        """``row_count`` 行的总 L2 量化上界。"""

        if self.dtype == "fp32":
            return float(
                F32_UNIT * norm_upper * row_count
                + F32_ABS * math.sqrt(width * row_count)
            )
        return 0.5 * float(self.scale) * math.sqrt(width * row_count)


def decode_int8(q: Any, scale: float, zero: float = 0.0) -> np.ndarray:
    """按 ``scale × (q − zero)`` 解码 int8 键块，返回 float64。"""

    arr = np.asarray(q)
    if arr.dtype != np.int8:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ExchangeError("int8 解码要求整数输入")
        arr = arr.astype(np.int8)
    return float(scale) * (arr.astype(np.float64) - float(zero))


def quantize(values: Any, scale: float, zero: float = 0.0) -> np.ndarray:
    """按对称 int8 定标量化；``zero`` 仅供非对称源解码，目标要求 0。"""

    if zero != 0:
        raise ExchangeError("对称 int8 量化要求 zero == 0")
    return TargetFormat("int8", float(scale)).quantize(np.asarray(values, dtype=np.float64))


# ---------------------------------------------------------------------------
# 物理键块
# ---------------------------------------------------------------------------


@dataclass
class StoredBlock:
    """一份物理键块。

    ``q`` 一律按 ``scale × (q − zero)`` 解码。``phase`` 是存储时已带有的
    位置视图标签，``None`` 表示未旋转原始键。存在 ``original`` 时以 FP32
    数据定义原始键，源相位误差为零。
    """

    id: str
    positions: tuple[int, ...]
    q: np.ndarray
    scale: float
    zero: float
    l2_source_error: float
    phase: str | None = None
    source_view: View | None = None
    original: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = _as_int_positions(self.positions)
        raw = np.asarray(self.q)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ExchangeError(f"块 {self.id!r} 的 q 必须是整数分量")
        if raw.size and (int(raw.min()) < -128 or int(raw.max()) > 127):
            raise ExchangeError(f"块 {self.id!r} 的 q 超出 int8 分量范围")
        self.q = raw.astype(np.int8)
        if self.q.ndim != 2 or self.q.shape[0] != len(self.positions):
            raise ExchangeError(f"块 {self.id!r} 的 q 形状与位置数不符")
        self.q.flags.writeable = False
        if self.original is not None:
            original = np.asarray(self.original, dtype=np.float32)
            if original.shape != self.q.shape:
                raise ExchangeError(f"块 {self.id!r} 的 original 形状与 q 不符")
            original.flags.writeable = False
            self.original = original
        if not math.isfinite(self.l2_source_error) or self.l2_source_error < 0:
            raise ExchangeError("l2_source_error 必须是有限非负数")
        if not math.isfinite(self.scale):
            raise ExchangeError("源 scale 必须有限")

    @property
    def width(self) -> int:
        return int(self.q.shape[1])

    @property
    def decoded(self) -> np.ndarray:
        return decode_int8(self.q, self.scale, self.zero)

    @property
    def resident_bytes(self) -> int:
        total = int(self.q.nbytes)
        if self.original is not None:
            total += int(self.original.nbytes)
        return total

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], views: Mapping[str, View]
    ) -> "StoredBlock":
        phase = spec.get("phase")
        source_view = None
        if phase is not None:
            if phase not in views:
                raise ExchangeError(f"块 {spec['id']!r} 引用了未知相位视图 {phase!r}")
            source_view = views[phase]
        original = spec.get("original")
        return cls(
            id=str(spec["id"]),
            positions=tuple(spec["positions"]),
            q=np.asarray(spec["q"], dtype=np.int64),
            scale=float(spec["scale"]),
            zero=float(spec.get("zero", 0.0)),
            l2_source_error=float(spec.get("l2_source_error", 0.0)),
            phase=phase,
            source_view=source_view,
            original=None if original is None else np.asarray(original, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contract:
    """数值契约与预算。能力判断只用这里的先验量。"""

    width: int
    rotary_dim: int
    query_norm_ceiling: float
    score_error_limit: float
    resident_budget: int
    temporary_budget: int
    verify_declared_bounds: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ExchangeError("width 必须为正")
        if self.rotary_dim < 0 or self.rotary_dim > self.width:
            raise ExchangeError("rotary_dim 必须在 [0, width] 内")
        if self.rotary_dim % 2:
            raise ExchangeError("rotary_dim 必须为偶数")
        if not math.isfinite(self.query_norm_ceiling) or self.query_norm_ceiling < 0:
            raise ExchangeError("query_norm_ceiling 必须是有限非负数")
        if not math.isfinite(self.score_error_limit) or self.score_error_limit < 0:
            raise ExchangeError("score_error_limit 必须是有限非负数")
        if self.resident_budget < 0 or self.temporary_budget < 0:
            raise ExchangeError("预算不能为负")

    def check_view(self, view: View) -> None:
        if view.rotary_dim != self.rotary_dim:
            raise ExchangeError(
                f"视图 {view.name!r} 的 rotary_dim={view.rotary_dim} "
                f"与契约 {self.rotary_dim} 不符"
            )


# ---------------------------------------------------------------------------
# 交付计划与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockPlan:
    """一个块在一次请求中的处理结果与审计信息。"""

    block_id: str
    status: str
    requested_positions: tuple[int, ...] = ()
    delivered_positions: tuple[int, ...] = ()
    excluded_positions: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()
    phase_source: str | None = None
    l2_bound: float | None = None
    score_bound: float | None = None
    error_breakdown: Mapping[str, float] = field(default_factory=dict)
    resident_bytes: int = 0
    temporary_bytes: int = 0
    reused: bool = False
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "block": self.block_id,
            "status": self.status,
            "requested_positions": list(self.requested_positions),
            "delivered_positions": list(self.delivered_positions),
            "excluded_positions": list(self.excluded_positions),
            "reasons": list(self.reasons),
            "phase_source": self.phase_source,
            "l2_bound": self.l2_bound,
            "score_bound": self.score_bound,
            "error_breakdown": dict(self.error_breakdown),
            "resident_bytes": self.resident_bytes,
            "temporary_bytes": self.temporary_bytes,
            "reused": self.reused,
            "detail": self.detail,
        }


@dataclass
class DeliveredBlock:
    """交给点积消费者的键。数组只读，读者归还前不变。"""

    block_id: str
    positions: tuple[int, ...]
    keys: np.ndarray
    target: TargetFormat
    phase_source: str
    l2_bound: float
    score_bound: float
    reused: bool = False

    def decoded(self) -> np.ndarray:
        if self.target.dtype == "int8":
            return decode_int8(self.keys, self.target.scale, self.target.zero)
        return np.asarray(self.keys, dtype=np.float64)

    def dot(self, query_rows: Any) -> np.ndarray:
        """``query_rows`` 为 ``(nq, width)``，返回 ``(nq, nk)``。"""

        q = np.asarray(query_rows, dtype=np.float64)
        return q @ self.decoded().T


@dataclass
class BorrowResult:
    """一次查询级借用的结果。``WAITING`` 时不保留任何块。"""

    query_id: str
    status: str
    view: View
    plans: tuple[BlockPlan, ...]
    delivered: dict[str, DeliveredBlock]
    excluded: tuple[str, ...]
    waiting: tuple[str, ...] = ()
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY

    @property
    def delivered_ids(self) -> tuple[str, ...]:
        return tuple(self.delivered)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query_id,
            "view": self.view.label(),
            "status": self.status,
            "delivered": list(self.delivered),
            "waiting": list(self.waiting),
            "excluded": list(self.excluded),
            "blocks": [p.as_dict() for p in self.plans],
            "detail": self.detail,
        }


@dataclass
class Acquisition:
    """``KeyStore.acquire`` 的内部返回值。"""

    plan: BlockPlan
    block: DeliveredBlock | None
    rep: "_Rep | None"
    reused: bool = False


# ---------------------------------------------------------------------------
# 派生表示与物理存储
# ---------------------------------------------------------------------------


class _Rep:
    """一份物理派生表示。读者持有期间不可变。"""

    __slots__ = (
        "key",
        "block_id",
        "rotation_sig",
        "target_sig",
        "positions",
        "keys",
        "nbytes",
        "phase_source",
        "l2_bound",
        "score_bound",
        "source_error",
        "arithmetic_error",
        "quant_error",
        "holders",
        "aliases_source",
    )

    def __init__(self, *, key, block_id, rotation_sig, target_sig, positions,
                 keys, nbytes, phase_source, l2_bound, score_bound, source_error,
                 arithmetic_error, quant_error, aliases_source):
        self.key = key
        self.block_id = block_id
        self.rotation_sig = rotation_sig
        self.target_sig = target_sig
        self.positions = positions
        self.keys = keys
        self.keys.flags.writeable = False
        self.nbytes = int(nbytes)
        self.phase_source = phase_source
        self.l2_bound = float(l2_bound)
        self.score_bound = float(score_bound)
        self.source_error = float(source_error)
        self.arithmetic_error = float(arithmetic_error)
        self.quant_error = float(quant_error)
        self.holders: dict[str, int] = {}
        self.aliases_source = bool(aliases_source)

    def error_triple(self) -> dict[str, float]:
        """源相位、旋转算术、量化三项误差界，审计可逐项查看。"""

        return {
            "source": self.source_error,
            "arithmetic": self.arithmetic_error,
            "quantization": self.quant_error,
            "l2": self.l2_bound,
        }


class KeyStore:
    """分块键的物理存储、派生共享与预算记账。

    记账规则：

    * 源块（``q`` 与可选的 FP32 原值）常驻，计入驻留预算。
    * 派生表示按真实字节计费一次；多个读者共享同一表示不重复计费。
    * 与源物理等价的 int8 交付（同定标、同旋转、同布局）直接引用源字节，
      新增驻留为 0。
    * 归还后没有读者的派生表示立即回收，内存可被后续请求使用。
    * 构建过程中的临时缓冲单独计费，构建完成即释放。
    """

    def __init__(self, blocks: Mapping[str, StoredBlock], contract: Contract):
        self.contract = contract
        self.blocks: dict[str, StoredBlock] = dict(blocks)
        for block in self.blocks.values():
            if block.width != contract.width:
                raise ExchangeError(
                    f"块 {block.id!r} 宽度 {block.width} 与契约 {contract.width} 不符"
                )
        self._reps: dict[tuple, _Rep] = {}
        self._holder_reps: dict[str, dict[tuple, int]] = {}
        self._row_failures: dict[tuple, set[int]] = {}
        self.data_warnings: list[dict[str, Any]] = []
        self.peak_temporary = 0
        self._temporary_in_use = 0

    # -- 记账 -------------------------------------------------------------
    @property
    def source_bytes(self) -> int:
        return sum(block.resident_bytes for block in self.blocks.values())

    @property
    def derived_bytes(self) -> int:
        return sum(rep.nbytes for rep in self._reps.values())

    @property
    def resident_bytes(self) -> int:
        return self.source_bytes + self.derived_bytes

    @property
    def free_resident(self) -> int:
        return self.contract.resident_budget - self.resident_bytes

    @property
    def temporary_bytes(self) -> int:
        return self._temporary_in_use

    def residency(self) -> dict[str, int]:
        """按真实常驻字节区分：源、共享派生、新增派生、预算余量。"""

        return {
            "source": self.source_bytes,
            "derived": self.derived_bytes,
            "aliased": self.aliased_bytes(),
            "resident": self.resident_bytes,
            "free": self.free_resident,
            "peak_temporary": self.peak_temporary,
            "reps": len(self._reps),
        }

    def aliased_bytes(self) -> int:
        """与源字节物理共享、未单独计费的派生表示总字节。"""

        return sum(rep.nbytes for rep in self._reps.values()
                   if rep.aliases_source)

    def holders_of(self, block_id: str, rotation_sig: tuple, target_sig: tuple,
                   positions: tuple[int, ...]) -> tuple[tuple[str, int], ...]:
        rep = self._reps.get((block_id, rotation_sig, target_sig, positions))
        if rep is None:
            return ()
        return tuple(sorted(rep.holders.items()))

    # -- 物理等价判定 -----------------------------------------------------
    @staticmethod
    def _row_slice(block: StoredBlock,
                   positions: tuple[int, ...]) -> tuple[slice | list[int], int]:
        """在不复制的前提下选出块行。

        返回 ``(索引, 额外字节)``。位置按块内顺序连续时用切片视图，额外
        字节为 0；否则退回 fancy 索引，并如实计费复制出的字节。
        """

        idx = [block.positions.index(p) for p in positions]
        if idx and idx == list(range(idx[0], idx[0] + len(idx))):
            return slice(idx[0], idx[0] + len(idx)), 0
        if idx == list(range(len(block.positions))):
            return slice(0, len(block.positions)), 0
        return idx, int(len(idx) * block.width * block.q.itemsize)

    @staticmethod
    def _same_rotation(source: View | None, target: View,
                       positions: tuple[int, ...]) -> bool:
        if target.rotary_dim == 0:
            return True
        if source is None:
            return False
        if source.layout != target.layout:
            return False
        if source.rotary_dim != target.rotary_dim:
            return False
        if source.segments == target.segments:
            return True
        try:
            return bool(
                np.array_equal(source.angles(positions), target.angles(positions))
            )
        except OutOfSegment:
            return False

    def _can_alias_source(self, block: StoredBlock, target: TargetFormat,
                          view: View, positions: tuple[int, ...]) -> bool:
        if block.original is not None:
            # 原值定义精确契约，交付必须从原值重建。
            return False
        if target.dtype != "int8" or block.q.dtype != np.int8:
            return False
        if target.scale != block.scale or target.zero != block.zero:
            return False
        if block.phase is not None and block.source_view is None:
            return False
        return self._same_rotation(block.source_view, view, positions)

    # -- 误差界 -----------------------------------------------------------
    @staticmethod
    def _arithmetic_bound(pairs: int, norm_upper: float) -> float:
        if pairs == 0 or norm_upper == 0:
            return 0.0
        return 8.0 * F64_EPS * (pairs + 2) * norm_upper

    def _bounds(self, base_norm: float, target: TargetFormat, view: View,
                source_error: float, rotates: bool,
                undo: bool = False) -> tuple[float, float, float, float]:
        """返回 ``(arithmetic, quant, l2, score)``，均按单行上界。"""

        # 源相位误差下，真实行的范数至多为 base_norm + source_error。
        norm_upper = base_norm + source_error
        if rotates or undo:
            # 撤销源相位与重做目标相位各是一次旋转，各计一份算术界。
            passes = int(rotates) + int(undo)
            arithmetic = self._arithmetic_bound(view.rotary_dim // 2, norm_upper)
            arithmetic *= max(1, passes)
        else:
            # 仅解码与重组：int8 → float64 精确，另留一个极小上界。
            arithmetic = 4.0 * F64_EPS * norm_upper
        quant = target.quant_l2_bound(1, self.contract.width, norm_upper + arithmetic)
        l2 = source_error + arithmetic + quant
        score = self.contract.query_norm_ceiling * l2
        return arithmetic, quant, l2, score

    # -- 位置筛选 ---------------------------------------------------------
    @staticmethod
    def _select(block: StoredBlock, requested_positions: Sequence[int] | None,
                ranges: Sequence[Sequence[int]] | None
                ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回 ``(选中位置, 被范围排除的位置)``。"""

        if requested_positions is None:
            chosen = list(block.positions)
        else:
            wanted = set(_as_int_positions(requested_positions))
            chosen = [p for p in block.positions if p in wanted]
            missing = wanted - set(chosen)
            if missing:
                raise ExchangeError(
                    f"块 {block.id!r} 不包含位置 {sorted(missing)}"
                )
        if ranges is None:
            return tuple(chosen), ()
        spans = [(int(a), int(b)) for a, b in ranges]
        for a, b in spans:
            if b < a:
                raise ExchangeError(f"范围 [{a}, {b}) 非法")
        keep = [p for p in chosen if any(a <= p < b for a, b in spans)]
        skipped = [p for p in chosen if p not in set(keep)]
        return tuple(keep), tuple(skipped)

    def _split_visible(self, block: StoredBlock, view: View,
                       positions: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        visible, excluded = [], []
        for p in positions:
            try:
                view.segment_for(p)
                visible.append(p)
            except OutOfSegment:
                excluded.append(p)
        return tuple(visible), tuple(excluded)

    # -- 源数据重建 -------------------------------------------------------
    def _base_rows(self, block: StoredBlock, view: View,
                   positions: tuple[int, ...]
                   ) -> tuple[np.ndarray, str, bool, float, bool]:
        """返回 ``(原始键行, 相位来源, 是否重做旋转, 源误差, 是否撤销)``。"""

        if block.original is not None:
            index = [block.positions.index(p) for p in positions]
            base = np.asarray(block.original[index], dtype=np.float64)
            return base, PHASE_EXACT, view.rotary_dim > 0, 0.0, False
        index = [block.positions.index(p) for p in positions]
        decoded = block.decoded[index]
        source_error = float(block.l2_source_error)
        if block.source_view is None:
            return decoded, PHASE_NEUTRAL, view.rotary_dim > 0, source_error, False
        if self._same_rotation(block.source_view, view, positions):
            # 存储时已带有与目标视图相同的相位：原样交付，避免无谓的
            # 逆向/正向旋转引入算术误差。
            return decoded, PHASE_ROTATED, False, source_error, False
        # 已旋转键：先撤销源相位，再按目标视图重做；正交变换保持 L2 界。
        base = apply_rotation(decoded, block.source_view, positions, inverse=True)
        return base, PHASE_ROTATED, view.rotary_dim > 0, source_error, True

    @staticmethod
    def _temporary_need(rows: int, width: int, itemsize: int) -> int:
        # 源解码/base 缓冲 + 旋转输出 + 量化输出。
        return int(rows * width * (8 + 8 + itemsize))

    def _rotate_and_quantize(self, base: np.ndarray, view: View,
                             positions: tuple[int, ...],
                             target: TargetFormat,
                             rotate: bool) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(量化键行, 可交付掩码)``；越界行不钳位。"""

        rotated = (
            apply_rotation(base, view, positions)
            if rotate and view.rotary_dim > 0
            else base
        )
        if target.dtype == "fp32":
            encoded = rotated.astype(np.float32)
            good = np.isfinite(encoded).all(axis=1)
            good &= (np.abs(np.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0))
                     <= np.finfo(np.float32).max).all(axis=1)
            return encoded[good], good
        scaled = rotated / float(target.scale) + target.zero
        enc = np.rint(scaled)  # 最近舍入，平局取偶
        inside = (enc >= -127.0) & (enc <= 127.0)
        good = inside.all(axis=1)
        return enc[good].astype(np.int8), good

    # -- 单块获取 ---------------------------------------------------------
    def acquire(
        self,
        holder_id: str,
        block_id: str,
        view: View,
        positions: Sequence[int] | None = None,
        ranges: Sequence[Sequence[int]] | None = None,
    ) -> Acquisition:
        self.contract.check_view(view)
        target = view.target_format()
        block = self.blocks.get(block_id)
        if block is None:
            return Acquisition(
                BlockPlan(block_id, STATUS_IMPOSSIBLE, reasons=(R_MISSING,),
                          detail=f"存储中没有块 {block_id!r}"),
                None, None,
            )

        requested, out_of_range = self._select(block, positions, ranges)
        visible, excluded = self._split_visible(block, view, requested)
        reasons: list[str] = []
        if out_of_range:
            reasons.append(R_OUT_OF_RANGE)
        if excluded:
            reasons.append(R_OUT_OF_SEGMENT)
        excluded = out_of_range + excluded
        if not visible:
            return Acquisition(
                BlockPlan(
                    block_id, STATUS_IMPOSSIBLE,
                    requested_positions=requested,
                    excluded_positions=excluded,
                    reasons=tuple(reasons or [R_NO_VISIBLE]),
                    detail="视图不覆盖任何请求位置",
                ),
                None, None,
            )

        # 已知不可量化的行，直接排除而不重复构建。旋转不同则越界行可能不同。
        rotation_sig = view.rotation_signature()
        failure_key = (block_id, rotation_sig, target.signature())
        known_bad = self._row_failures.get(failure_key, set())
        if known_bad:
            dropped = tuple(p for p in visible if p in known_bad)
            if dropped:
                visible = tuple(p for p in visible if p not in known_bad)
                excluded = excluded + dropped
                reasons.append(R_OVERFLOW)
                if not visible:
                    return Acquisition(
                        BlockPlan(
                            block_id, STATUS_IMPOSSIBLE,
                            requested_positions=requested,
                            excluded_positions=excluded,
                            reasons=tuple(reasons),
                            detail="所有行都超出目标 int8 定标",
                        ),
                        None, None,
                    )

        # 未知相位：既非原始键，也没有对应频率表；不得混用。
        if block.phase is not None and block.source_view is None:
            return Acquisition(
                BlockPlan(
                    block_id, STATUS_IMPOSSIBLE,
                    requested_positions=requested,
                    excluded_positions=excluded,
                    reasons=(R_UNKNOWN_PHASE,),
                    detail=f"块声明相位 {block.phase!r}，但没有对应频率表",
                ),
                None, None,
            )

        # 物理等价：无需旋转且目标编码与源一致，可引用源字节。
        if self._can_alias_source(block, target, view, visible):
            index, extra = self._row_slice(block, visible)
            keys = block.q[index]
            if extra:
                keys = np.array(keys, dtype=np.int8, copy=True)
            source_error = float(block.l2_source_error)
            l2 = source_error
            score = self.contract.query_norm_ceiling * l2
            if score > self.contract.score_error_limit:
                return Acquisition(
                    self._reject(block_id, target, requested, visible, excluded,
                                 reasons, PHASE_ROTATED if block.source_view else PHASE_NEUTRAL,
                                 l2, score, R_ERROR_BOUND,
                                 "仅剩低精度源相位，公开误差界超出分数契约"),
                    None, None,
                )
            # 子集复制才占用预算。
            if extra > self.free_resident:
                return Acquisition(
                    BlockPlan(
                        block_id, STATUS_WAITING,
                        requested_positions=requested,
                        excluded_positions=excluded,
                        reasons=tuple(reasons + [R_RESIDENT]),
                        phase_source=PHASE_ROTATED if block.source_view else PHASE_NEUTRAL,
                        l2_bound=l2,
                        score_bound=score,
                        resident_bytes=extra,
                        detail=(
                            f"非连续子集需要 {extra} 驻留字节，"
                            f"当前可用 {max(self.free_resident, 0)}"
                        ),
                    ),
                    None, None,
                )
            phase = PHASE_ROTATED if block.source_view is not None else PHASE_NEUTRAL
            rep = self._rep_for(block_id, rotation_sig, target, visible, keys,
                                extra, 0.0, 0.0, phase, source_error, l2, score,
                                aliases=True)
            return self._deliver(holder_id, block_id, target, visible,
                                 requested, excluded, reasons, rep)

        key = (block_id, rotation_sig, target.signature(), visible)
        rep = self._reps.get(key)
        if rep is not None:
            return self._confirm(holder_id, block_id, target, visible, requested,
                                 excluded, reasons, rep)

        (base, phase_source, rotates, source_error, undo) = self._base_rows(
            block, view, visible
        )
        norms = np.linalg.norm(base, axis=1)
        base_norm = float(norms.max()) if norms.size else 0.0
        arithmetic, quant, l2, score = self._bounds(
            base_norm, target, view, source_error, rotates, undo
        )
        if self.contract.verify_declared_bounds:
            self._verify_declared(block, visible)
        if score > self.contract.score_error_limit:
            return Acquisition(
                self._reject(block_id, target, requested, visible, excluded,
                             reasons, phase_source, l2, score, R_ERROR_BOUND,
                             "源相位误差加上交付量化误差后，分数界超出契约",
                             error_breakdown={
                                 "source": source_error,
                                 "arithmetic": arithmetic,
                                 "quantization": quant,
                             }),
                None, None,
            )

        rows = len(visible)
        temporary = self._temporary_need(rows, self.contract.width, target.itemsize)
        if temporary > self.contract.temporary_budget:
            return Acquisition(
                BlockPlan(
                    block_id, STATUS_IMPOSSIBLE,
                    requested_positions=requested,
                    excluded_positions=excluded,
                    reasons=tuple(reasons + [R_TEMPORARY]),
                    phase_source=phase_source,
                    l2_bound=l2,
                    score_bound=score,
                    error_breakdown={
                        "source": source_error,
                        "arithmetic": arithmetic,
                        "quantization": quant,
                    },
                    temporary_bytes=temporary,
                    detail=(
                        f"临时区需要 {temporary} 字节，"
                        f"预算 {self.contract.temporary_budget}"
                    ),
                ),
                None, None,
            )

        nbytes = int(rows * self.contract.width * target.itemsize)
        free = self.free_resident
        if nbytes > free:
            return Acquisition(
                BlockPlan(
                    block_id, STATUS_WAITING,
                    requested_positions=requested,
                    excluded_positions=excluded,
                    reasons=tuple(reasons + [R_RESIDENT]),
                    phase_source=phase_source,
                    l2_bound=l2,
                    score_bound=score,
                    error_breakdown={
                        "source": source_error,
                        "arithmetic": arithmetic,
                        "quantization": quant,
                    },
                    resident_bytes=nbytes,
                    temporary_bytes=temporary,
                    detail=(
                        f"需要 {nbytes} 驻留字节，当前可用 {max(free, 0)}；"
                        "等待其他读者归还派生块"
                    ),
                ),
                None, None,
            )

        self._temporary_in_use += temporary
        self.peak_temporary = max(self.peak_temporary, self._temporary_in_use)
        try:
            keys, good = self._rotate_and_quantize(base, view, visible, target, rotates)
        finally:
            self._temporary_in_use = max(0, self._temporary_in_use - temporary)

        if not bool(good.all()):
            bad = tuple(p for p, keep in zip(visible, good) if not keep)
            self._row_failures.setdefault(failure_key, set()).update(bad)
            visible = tuple(p for p, keep in zip(visible, good) if keep)
            excluded = excluded + bad
            reasons.append(R_OVERFLOW)
            if not visible:
                return Acquisition(
                    BlockPlan(
                        block_id, STATUS_IMPOSSIBLE,
                        requested_positions=requested,
                        excluded_positions=excluded,
                        reasons=tuple(reasons),
                        detail="所有行都超出目标 int8 定标",
                    ),
                    None, None,
                )
            # 行数变化后重算量化的 L2 上界。
            nbytes = int(len(visible) * self.contract.width * target.itemsize)
            quant = target.quant_l2_bound(
                1, self.contract.width, base_norm + source_error + arithmetic
            )
            l2 = source_error + arithmetic + quant
            score = self.contract.query_norm_ceiling * l2
            if score > self.contract.score_error_limit:
                return Acquisition(
                    self._reject(block_id, target, requested, visible, excluded,
                                 reasons, phase_source, l2, score, R_ERROR_BOUND,
                                 "排除越界行后分数界仍超出契约",
                                 error_breakdown={
                                     "source": source_error,
                                     "arithmetic": arithmetic,
                                     "quantization": quant,
                                 }),
                    None, None,
                )

        rep = self._rep_for(block_id, rotation_sig, target, visible, keys,
                            nbytes, arithmetic, quant, phase_source,
                            source_error, l2, score, aliases=False)
        return self._deliver(holder_id, block_id, target, visible, requested,
                             excluded, reasons, rep)

    def _confirm(self, holder_id, block_id, target, visible, requested,
                 excluded, reasons, rep: _Rep) -> Acquisition:
        """共享已有派生表示前，复核它仍满足当前契约与预算。"""

        if rep.score_bound > self.contract.score_error_limit:
            return Acquisition(
                BlockPlan(
                    block_id, STATUS_UNPROVABLE,
                    requested_positions=requested,
                    excluded_positions=excluded + visible,
                    reasons=tuple(reasons + [R_ERROR_BOUND]),
                    phase_source=rep.phase_source,
                    l2_bound=rep.l2_bound,
                    score_bound=rep.score_bound,
                    error_breakdown=rep.error_triple(),
                    detail="已有派生表示的分数界超出契约，拒绝共享",
                ),
                None, None,
            )
        return self._deliver(holder_id, block_id, target, visible, requested,
                             excluded, reasons, rep)

    # -- 内部辅助 ---------------------------------------------------------
    def _verify_declared(self, block: StoredBlock,
                         positions: tuple[int, ...]) -> None:
        """有原始块可恢复时，核对声明的源误差界确实成立。

        “相对其相位理想值”指：相位为 ``None`` 时是原值本身，否则是原值按
        该相位旋转后的键。没有原值时只能信任公开界，不能从解码值反推。
        """

        declared = float(block.l2_source_error)
        if not math.isfinite(declared) or declared < 0:
            raise ExchangeError(
                f"块 {block.id!r} 的 l2_source_error 非法：{declared!r}"
            )
        if block.original is None:
            return
        index = [block.positions.index(p) for p in positions]
        original = np.asarray(block.original[index], dtype=np.float64)
        if block.source_view is None:
            ideal = original
        else:
            ideal = apply_rotation(original, block.source_view, positions)
        actual = float(np.linalg.norm(block.decoded[index] - ideal, axis=1).max())
        # original 以 FP32 存储，核对时容忍其表示误差。
        tolerance = max(1e-9, 1e-5 * declared)
        if actual > declared + tolerance:
            # 原值可恢复时以精确契约为准，不因声明不实而误拒，只记录告警。
            self.data_warnings.append(
                {
                    "block": block.id,
                    "kind": "declared_bound_too_small",
                    "declared": declared,
                    "actual": actual,
                }
            )
    def _rep_for(self, block_id, rotation_sig, target, positions, keys, nbytes,
                 arithmetic, quant, phase_source, source_error, l2, score, *,
                 aliases) -> _Rep:
        key = (block_id, rotation_sig, target.signature(), positions)
        rep = self._reps.get(key)
        if rep is not None:
            return rep
        rep = _Rep(
            key=key,
            block_id=block_id,
            rotation_sig=rotation_sig,
            target_sig=target.signature(),
            positions=positions,
            keys=keys,
            nbytes=nbytes,
            phase_source=phase_source,
            l2_bound=l2,
            score_bound=score,
            source_error=source_error,
            arithmetic_error=arithmetic,
            quant_error=quant,
            aliases_source=aliases,
        )
        self._reps[key] = rep
        return rep

    def _deliver(self, holder_id, block_id, target, visible, requested,
                 excluded, reasons, rep: _Rep) -> Acquisition:
        first_holder = not rep.holders
        added = rep.nbytes if first_holder else 0
        rep.holders[holder_id] = rep.holders.get(holder_id, 0) + 1
        self._holder_reps.setdefault(holder_id, {})
        self._holder_reps[holder_id][rep.key] = (
            self._holder_reps[holder_id].get(rep.key, 0) + 1
        )
        delivered = DeliveredBlock(
            block_id=block_id,
            positions=visible,
            keys=rep.keys,
            target=TargetFormat(*rep.target_sig),
            phase_source=rep.phase_source,
            l2_bound=rep.l2_bound,
            score_bound=rep.score_bound,
            reused=not first_holder,
        )
        plan = BlockPlan(
            block_id=block_id,
            status=STATUS_READY,
            requested_positions=requested,
            delivered_positions=visible,
            excluded_positions=excluded,
            reasons=tuple(reasons),
            phase_source=rep.phase_source,
            l2_bound=rep.l2_bound,
            score_bound=rep.score_bound,
            error_breakdown=rep.error_triple(),
            resident_bytes=added,
            temporary_bytes=0,
            reused=not first_holder,
            detail=(
                "物理等价表示，直接引用源字节"
                if rep.aliases_source
                else ("共享已有派生表示" if not first_holder else "")
            ),
        )
        return Acquisition(plan, delivered, rep, not first_holder)

    @staticmethod
    def _reject(block_id, target, requested, visible, excluded, reasons,
                phase_source, l2, score, reason, detail,
                *, error_breakdown=None) -> BlockPlan:
        return BlockPlan(
            block_id=block_id,
            status=STATUS_UNPROVABLE,
            requested_positions=requested,
            excluded_positions=excluded + visible,
            reasons=tuple(reasons + [reason]),
            phase_source=phase_source,
            l2_bound=l2,
            score_bound=score,
            error_breakdown=dict(error_breakdown or {}),
            detail=detail,
        )

    # -- 归还 -------------------------------------------------------------
    def release(self, holder_id: str, block_id: str | None = None) -> int:
        """归还该读者的全部或指定块的表示；没有其他读者时立即回收。

        返回实际回收的字节数。
        """

        held = self._holder_reps.get(holder_id)
        if not held:
            return 0
        if block_id is None:
            self._holder_reps.pop(holder_id, None)
            items = list(held.items())
        else:
            items = [(k, c) for k, c in held.items() if k[0] == block_id]
            for key, _ in items:
                held.pop(key, None)
            if not held:
                self._holder_reps.pop(holder_id, None)
        freed = 0
        for key, count in items:
            rep = self._reps.get(key)
            if rep is None:
                continue
            remaining = rep.holders.get(holder_id, 0) - count
            if remaining > 0:
                rep.holders[holder_id] = remaining
                continue
            rep.holders.pop(holder_id, None)
            if not rep.holders:
                freed += rep.nbytes
                del self._reps[key]
        return freed

    def release_all(self) -> int:
        freed = 0
        for holder in list(self._holder_reps):
            freed += self.release(holder)
        return freed


# ---------------------------------------------------------------------------
# 交换资料与会话
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySpec:
    """一个查询：原始向量、查询位置、位置视图与可见块（可选位置范围）。

    ``ranges`` 以全局位置的半开区间给出该会话可见的键范围；缺省表示块内
    全部位置。范围外的行既不交付也不参与打分，并在计划里列为排除位置。
    """

    id: str
    view: str
    positions: tuple[int, ...]
    blocks: tuple[str, ...]
    ranges: tuple[tuple[int, int], ...] | None = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "QuerySpec":
        raw_ranges = spec.get("ranges")
        ranges = None
        if raw_ranges is not None:
            ranges = tuple((int(a), int(b)) for a, b in raw_ranges)
            for a, b in ranges:
                if b < a:
                    raise ExchangeError(f"查询范围 [{a}, {b}) 非法")
        return cls(
            id=str(spec["id"]),
            view=str(spec["view"]),
            positions=_as_int_positions(spec["positions"]),
            blocks=tuple(str(b) for b in spec["blocks"]),
            ranges=ranges,
        )

    def visible(self, position: int) -> bool:
        if self.ranges is None:
            return True
        return any(a <= position < b for a, b in self.ranges)


@dataclass
class ExchangeDocument:
    """解析后的交换资料。"""

    rotary_dim: int
    width: int
    views: dict[str, View]
    blocks: dict[str, StoredBlock]
    queries: dict[str, QuerySpec]
    query: np.ndarray
    query_norm_ceiling: float
    score_error_limit: float
    resident_budget: int
    temporary_budget: int
    timeline: tuple[tuple[str, str], ...]

    def contract(self) -> Contract:
        return Contract(
            width=self.width,
            rotary_dim=self.rotary_dim,
            query_norm_ceiling=self.query_norm_ceiling,
            score_error_limit=self.score_error_limit,
            resident_budget=self.resident_budget,
            temporary_budget=self.temporary_budget,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExchangeDocument":
        rotary_dim = int(data["rotary_dim"])
        width = int(data["width"])
        views = {
            name: View.from_spec(name, spec, rotary_dim)
            for name, spec in data["views"].items()
        }
        blocks: dict[str, StoredBlock] = {}
        for spec in data["blocks"]:
            block = StoredBlock.from_spec(spec, views)
            blocks[block.id] = block
        if not blocks:
            raise ExchangeError("资料至少需要一个键块")
        queries = {
            str(spec["id"]): QuerySpec.from_spec(spec) for spec in data.get("queries", [])
        }
        query = np.asarray(data["query"], dtype=np.float64)
        if query.shape != (width,):
            raise ExchangeError("query 长度必须等于 width")
        ceiling = float(data["query_norm_ceiling"])
        if float(np.linalg.norm(query)) > ceiling + 1e-12:
            raise ExchangeError("query 范数超过 query_norm_ceiling")
        timeline = tuple(
            (str(event), str(target)) for event, target in data.get("timeline", [])
        )
        return cls(
            rotary_dim=rotary_dim,
            width=width,
            views=views,
            blocks=blocks,
            queries=queries,
            query=query,
            query_norm_ceiling=ceiling,
            score_error_limit=float(data["score_error_limit"]),
            resident_budget=int(data["budget_bytes"]["resident"]),
            temporary_budget=int(data["budget_bytes"]["temporary"]),
            timeline=timeline,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ExchangeDocument":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class _Session:
    """一个查询的借用状态。"""

    query: QuerySpec
    view: View
    holder: str
    result: BorrowResult | None = None
    attempts: int = 0


class ExchangeRunner:
    """把交换资料变成可借还的分块键交付会话。

    使用方式::

        runner = ExchangeRunner.from_path("samples/scene.json")
        result = runner.borrow("A")          # ready / waiting / unprovable
        keys = runner.available_keys("A")    # 只含已交付块
        scores = runner.score("A")           # (查询位置数, 键行数)
        runner.release("A")                  # 归还并回收派生块

    ``waiting`` 不会留下半成品占用：本次尝试获得的块会全部退回，
    调用方可在其他会话归还后 ``retry``。
    """

    def __init__(self, document: ExchangeDocument):
        self.document = document
        self.contract = document.contract()
        self.store = KeyStore(document.blocks, self.contract)
        self.views: dict[str, View] = dict(document.views)
        self._sessions: dict[str, _Session] = {}
        self._history: list[dict[str, Any]] = []
        self._view_revisions: dict[str, int] = {name: v.revision for name, v in self.views.items()}

    @classmethod
    def from_path(cls, path: str | Path) -> "ExchangeRunner":
        return cls(ExchangeDocument.from_path(path))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExchangeRunner":
        return cls(ExchangeDocument.from_dict(data))

    # -- 视图下发 ---------------------------------------------------------
    def install_view(self, name: str, spec: Mapping[str, Any]) -> View:
        """下发新的频率表。已存在的在途读者继续使用其旧视图对象。"""

        revision = self._view_revisions.get(name, -1) + 1
        view = View.from_spec(name, spec, self.document.rotary_dim, revision=revision)
        self._view_revisions[name] = revision
        self.views[name] = view
        return view

    # -- 借用 -------------------------------------------------------------
    def _session_for(self, query_id: str) -> _Session:
        query = self.document.queries.get(query_id)
        if query is None:
            raise ExchangeError(f"资料中没有查询 {query_id!r}")
        view = self.views.get(query.view)
        if view is None:
            raise ExchangeError(f"查询 {query_id!r} 引用了未知视图 {query.view!r}")
        self.contract.check_view(view)
        session = self._sessions.get(query_id)
        if session is None:
            session = _Session(query=query, view=view, holder=query_id)
            self._sessions[query_id] = session
        elif session.result is None and session.view is not view:
            # 上一轮已归还：下一次借用才采用新下发的频率表。
            session.view = view
            session.attempts += 1
            session.holder = f"{query_id}#{session.attempts}"
        # 在途结果继续使用其创建时的视图对象；下发新表不会改写它。
        return session

    def borrow(self, query_id: str) -> BorrowResult:
        """借用查询可见的全部块。

        已就绪或已有部分交付的会话重复借用时返回同一结果，不会自我阻塞。
        等待中的块不会进入交付集合，也不以零填充。
        """

        session = self._session_for(query_id)
        if session.result is not None:
            return session.result
        session.attempts += 1
        holder = f"{query_id}#{session.attempts}"
        session.holder = holder

        acquisitions: list[Acquisition] = []
        for block_id in session.query.blocks:
            acquisitions.append(
                self.store.acquire(
                    holder,
                    block_id,
                    session.view,
                    positions=None,
                    ranges=session.query.ranges,
                )
            )

        plans = [acq.plan for acq in acquisitions]
        delivered: dict[str, DeliveredBlock] = {}
        waiting: list[str] = []
        excluded: list[str] = []
        for acq in acquisitions:
            if acq.plan.ready and acq.block is not None:
                delivered[acq.plan.block_id] = acq.block
            elif acq.plan.status == STATUS_WAITING:
                waiting.append(acq.plan.block_id)
            else:
                excluded.append(acq.plan.block_id)

        if waiting:
            # 部分块可立即兑现、部分等待容量：等待块不进交付集合，
            # 也绝不以零值参与打分。
            status = STATUS_WAITING
            detail = (
                "部分块等待容量释放；已交付块可立即打分"
                if delivered
                else "容量不足，未交付任何块；等待其他会话归还派生块"
            )
        elif delivered:
            status = STATUS_READY
            detail = "" if not excluded else "部分块被排除，未参与打分"
        elif any(p.status == STATUS_UNPROVABLE for p in plans):
            status = STATUS_UNPROVABLE
            detail = "没有任何块能证明满足分数契约"
        else:
            status = STATUS_IMPOSSIBLE
            detail = "没有任何块可用"

        if not delivered:
            self.store.release(holder)

        result = BorrowResult(
            query_id=query_id,
            status=status,
            view=session.view,
            plans=tuple(plans),
            delivered=delivered,
            waiting=tuple(waiting),
            excluded=tuple(excluded),
            detail=detail,
        )
        session.result = result
        self._record("borrow", query_id, result)
        return result

    def retry(self, query_id: str) -> BorrowResult:
        """在容量释放后重试一次等待中的借用（重新读取当前视图）。"""

        session = self._sessions.get(query_id)
        if session is not None and session.result is not None:
            self.store.release(session.holder)
            session.result = None
        return self.borrow(query_id)

    def release(self, query_id: str) -> int:
        """归还查询持有的全部块；没有其他读者的派生块立即回收。"""

        session = self._sessions.get(query_id)
        if session is None:
            return 0
        freed = self.store.release(session.holder)
        session.result = None
        self._record("release", query_id, None, freed=freed)
        return freed

    # -- 查询审计 ---------------------------------------------------------
    def session_ids(self) -> tuple[str, ...]:
        """已创建借用的查询 id，按首次借用顺序。"""

        return tuple(self._sessions)

    def audit_all(self) -> dict[str, dict[str, Any]]:
        """所有尚未归还的会话的审计信息。"""

        return {
            qid: self.audit(qid)
            for qid, session in self._sessions.items()
            if session.result is not None
        }

    def available_keys(self, query_id: str) -> dict[str, DeliveredBlock]:
        """当前已交付给打分器、尚未归还的键块。"""

        session = self._sessions.get(query_id)
        if session is None or session.result is None:
            return {}
        return dict(session.result.delivered)

    def audit(self, query_id: str) -> dict[str, Any]:
        """单个查询的可用键、相位来源、精度界与拒绝原因。"""

        session = self._sessions.get(query_id)
        if session is None:
            raise ExchangeError(f"查询 {query_id!r} 尚未借用")
        if session.result is None:
            return {
                "query": query_id,
                "view": session.view.label(),
                "status": "released",
                "blocks": [],
                "delivered": [],
                "excluded": [],
                "reasons": [R_RELEASED],
            }
        result = session.result
        delivered = result.delivered
        blocks = []
        for plan in result.plans:
            entry = plan.as_dict()
            held = delivered.get(plan.block_id)
            if held is not None:
                entry["held"] = True
                entry["keys_shape"] = list(held.keys.shape)
                entry["target"] = held.target.dtype
                entry["scale"] = held.target.scale
                entry["phase_source"] = held.phase_source
                entry["l2_bound"] = held.l2_bound
                entry["score_bound"] = held.score_bound
            else:
                entry["held"] = False
            blocks.append(entry)
        return {
            "query": query_id,
            "view": result.view.label(),
            "status": result.status,
            "delivered": list(delivered),
            "waiting": list(result.waiting),
            "excluded": list(result.excluded),
            "reasons": sorted({r for p in result.plans for r in p.reasons}),
            "blocks": blocks,
            "residency": self.store.residency(),
        }

    # -- 打分 -------------------------------------------------------------
    def score(self, query_id: str) -> dict[str, np.ndarray]:
        """把已交付块送入点积消费者，返回每个块的 ``(查询位置数, 键行数)``。

        查询向量按本次查询视图在其每个位置旋转。缺失块不参与、不以零填充。
        """

        session = self._sessions.get(query_id)
        if session is None or session.result is None:
            raise ExchangeError(f"查询 {query_id!r} 没有可用的已交付键")
        result = session.result
        if not result.delivered:
            raise ExchangeError(
                f"查询 {query_id!r} 状态为 {result.status}，没有已交付块可打分"
            )
        # 等待中的块不在 result.delivered 中，因此不会以零值参与。
        query_rows = rotate_query(self.document.query, result.view, session.query.positions)
        out: dict[str, np.ndarray] = {}
        for block_id, delivered in result.delivered.items():
            out[block_id] = delivered.dot(query_rows)
        return out

    def score_error_bounds(self, query_id: str) -> dict[str, float]:
        session = self._sessions.get(query_id)
        if session is None or session.result is None:
            return {}
        return {
            block_id: held.score_bound
            for block_id, held in session.result.delivered.items()
        }

    # -- 时间线 -----------------------------------------------------------
    def _record(self, event: str, query_id: str, result: BorrowResult | None,
                *, freed: int = 0) -> None:
        self._history.append(
            {
                "event": event,
                "query": query_id,
                "status": None if result is None else result.status,
                "delivered": [] if result is None else sorted(result.delivered),
                "waiting": [] if result is None else sorted(result.waiting),
                "excluded": [] if result is None else sorted(result.excluded),
                "freed": int(freed),
                "resident": self.store.resident_bytes,
            }
        )

    def run_timeline(self) -> list[dict[str, Any]]:
        events = []
        for event, query_id in self.document.timeline:
            if event == "borrow":
                result = self.borrow(query_id)
                events.append({"event": event, "query": query_id,
                               "status": result.status,
                               "delivered": sorted(result.delivered),
                               "waiting": sorted(result.waiting),
                               "excluded": sorted(result.excluded)})
            elif event == "return":
                freed = self.release(query_id)
                events.append({"event": event, "query": query_id, "freed": freed})
            elif event == "retry":
                result = self.retry(query_id)
                events.append({"event": event, "query": query_id,
                               "status": result.status,
                               "delivered": sorted(result.delivered),
                               "waiting": sorted(result.waiting),
                               "excluded": sorted(result.excluded)})
            else:
                raise ExchangeError(f"未知时间线事件 {event!r}")
        return events

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._history)

    def residency(self) -> dict[str, int]:
        return self.store.residency()
