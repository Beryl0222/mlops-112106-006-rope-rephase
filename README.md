# 旋转键交接

`samples/scene.json` 是本地合成的位置视图和键块格式。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 audit_delivery.py samples/scene.json            # 全量流程、审计与分数
python3 audit_delivery.py samples/scene.json --audit A  # 单个查询的审计
python3 -m unittest discover -s tests -v
```

## `rotary_keys.py`

主要交付是 CPU 上的分段旋转位置编码与分块键读取接口。

```python
from rotary_keys import ExchangeRunner

runner = ExchangeRunner.from_path("samples/scene.json")
result = runner.borrow("A")        # ready / waiting / unprovable / impossible
runner.available_keys("A")         # 只含已交付块
scores = runner.score("A")         # 按查询位置旋转查询，与交付键点积
runner.audit("A")                  # 相位来源、三项误差界、拒绝原因
runner.release("A")                # 归还；无其他读者时立即回收派生字节
```

### 位置视图

* `interleaved` 取相邻通道对，`split` 取旋转前缀的前半与后半；只旋转
  `rotary_dim` 前缀，其余分量不做旋转。
* `segments` 以全局位置的半开区间给出每对维度的频率与原点，角度为
  `(全局位置 - 原点) × 频率`，正角逆时针。原点已包含会话重编号。
* 同一段物理存储可供多个读者使用，每个会话按自己的视图取位置视图。

### 相位与量化

* 键块 `q` 一律按 `scale × (q - zero)` 解码；`phase` 记录它已带有的视图。
  `original` 存在时以 FP32 原值定义原始键。
* 已旋转的历史键块先按源视图撤销相位，再按目标视图重做；两次旋转都是
  正交变换，L2 上界不变。存储相位与目标视图相同时跳过旋转。
* 目标格式为 FP32 或对称 int8 定标。int8 最近舍入、平局取偶；越界抛
  `TargetOverflow` 并按行排除，绝不钳位。未旋转维度也按目标格式量化。

### 误差界

交付 L2 上界 = 源相位界 + 旋转算术界 + 交付量化界；分数上界 =
`query_norm_ceiling × L2 上界`。是否有能力交付只由这些先验上界判断，
不使用“新旧解码值之差”。来源只剩低精度旧相位时按公开 `l2_source_error`
判断能否用于新视图；存在原始块时走精确契约并校验声明界确实成立。
不满足契约的块返回 `unprovable` 与原因，绝不进入打分。

### 预算与共享

* 源块常驻并计入预算；派生表示按真实字节计费一次，多读者共享不重复计费。
* 与源物理等价的 int8 交付（同定标、同旋转）直接引用源字节，新增 0 字节。
* 构建用的临时缓冲单独计费，构建完成即释放。
* 内存不足时返回 `waiting`：可立即兑现的块照常交付，其余块列入
  `waiting`，绝不以零值参与打分。其他会话归还后可 `retry`。
* 已交付但未读完的块在读者归还前不可变；新下发的频率表只影响之后的新借用。
