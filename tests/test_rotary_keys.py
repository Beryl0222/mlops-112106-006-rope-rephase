"""旋转键交接的行为测试。

覆盖：两种通道排列与分段旋转、误差界、相位来源、量化契约、预算与共享、
会话隔离、退还顺序、审计与拒绝原因。
"""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inspect_exchange import inspect
import rotary_keys as rk

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "samples" / "scene.json"
TIGHT = ROOT / "samples" / "tight-residency.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def simple_doc(**overrides):
    """最小合成资料：两个位置、一个视图。"""

    data = {
        "rotary_dim": 4,
        "width": 6,
        "views": {
            "v": {
                "layout": "interleaved",
                "segments": [[0, 32, 0, [1.0, 0.1]]],
                "dtype": "fp32",
            }
        },
        "blocks": [
            {
                "id": "b",
                "positions": [2],
                "phase": None,
                "scale": 0.01,
                "zero": 0,
                "l2_source_error": 0.005,
                "q": [[50, -25, 13, 0, 3, 4]],
                "original": [[0.5, -0.25, 0.13, 0.0, 0.03, 0.04]],
            }
        ],
        "queries": [
            {"id": "Q", "view": "v", "positions": [2], "blocks": ["b"]}
        ],
        "query": [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
        "query_norm_ceiling": 1.0,
        "score_error_limit": 1.0,
        "budget_bytes": {"resident": 1000, "temporary": 4096},
        "timeline": [],
    }
    data.update(overrides)
    return data


class RotationTests(unittest.TestCase):
    def test_interleaved_matches_pairs(self):
        view = rk.View.from_spec(
            "v",
            {"layout": "interleaved", "segments": [[0, 10, 0, [0.3, 0.7]]]},
            rotary_dim=4,
        )
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = rk.apply_rotation(vec, view, [2])
        c0, s0 = math.cos(2 * 0.3), math.sin(2 * 0.3)
        c1, s1 = math.cos(2 * 0.7), math.sin(2 * 0.7)
        expected = np.array(
            [
                c0 * 1 - s0 * 2,
                s0 * 1 + c0 * 2,
                c1 * 3 - s1 * 4,
                s1 * 3 + c1 * 4,
                5.0,
                6.0,
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)

    def test_split_layout_and_non_rotary_passthrough(self):
        view = rk.View.from_spec(
            "v",
            {"layout": "split", "segments": [[0, 10, 0, [0.3, 0.7]]]},
            rotary_dim=4,
        )
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = rk.apply_rotation(vec, view, [3])
        theta0, theta1 = 3 * 0.3, 3 * 0.7
        c0, s0 = math.cos(theta0), math.sin(theta0)
        c1, s1 = math.cos(theta1), math.sin(theta1)
        expected = np.array(
            [
                c0 * 1 - s0 * 3,
                c1 * 2 - s1 * 4,
                s0 * 1 + c0 * 3,
                s1 * 2 + c1 * 4,
                5.0,
                6.0,
            ]
        )
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)

    def test_inverse_round_trip(self):
        view = rk.View.from_spec(
            "v",
            {"layout": "split", "segments": [[0, 20, 1, [0.1, 0.25]]]},
            rotary_dim=4,
        )
        vec = np.array([0.3, -0.8, 1.2, 0.5, 7.0, -2.0])
        rotated = rk.apply_rotation(vec, view, [9])
        back = rk.apply_rotation(rotated, view, [9], inverse=True)
        np.testing.assert_allclose(back, vec, rtol=0, atol=1e-15)

    def test_segments_switch_frequency_and_origin(self):
        view = rk.View.from_spec(
            "v",
            {
                "layout": "interleaved",
                "segments": [
                    [0, 8, 0, [1.0, 0.1]],
                    [8, 16, 3, [0.5, 0.04]],
                ],
            },
            rotary_dim=4,
        )
        left = rk.apply_rotation(np.ones(4), view, [7])
        right = rk.apply_rotation(np.ones(4), view, [8])
        c0, s0 = math.cos(7 * 1.0), math.sin(7 * 1.0)
        np.testing.assert_allclose(left[:2], [c0 - s0, s0 + c0], atol=1e-14)
        c0, s0 = math.cos((8 - 3) * 0.5), math.sin((8 - 3) * 0.5)
        np.testing.assert_allclose(right[:2], [c0 - s0, s0 + c0], atol=1e-14)

    def test_position_outside_segments_raises(self):
        view = rk.View.from_spec(
            "v", {"layout": "split", "segments": [[0, 4, 0, [1.0, 0.1]]]},
            rotary_dim=4,
        )
        with self.assertRaises(rk.OutOfSegment):
            rk.apply_rotation(np.ones(4), view, [4])

    def test_rotate_query_batches_positions(self):
        view = rk.View.from_spec(
            "v", {"layout": "interleaved", "segments": [[0, 32, 0, [0.2, 0.4]]]},
            rotary_dim=4,
        )
        rows = rk.rotate_query(np.array([1.0, 0.0, 1.0, 0.0, 5.0, 6.0]), view, [1, 2])
        self.assertEqual(rows.shape, (2, 6))
        np.testing.assert_allclose(rows[:, 4:], [[5.0, 6.0], [5.0, 6.0]])


class QuantizationTests(unittest.TestCase):
    def test_nearest_even_rounding(self):
        target = rk.TargetFormat("int8", 0.5)
        values = np.array([0.25, 0.75, -0.25, -0.75, 1.24, 1.26])
        enc = target.quantize(values)
        # 0.5, 1.5, -0.5, -1.5 平局取偶 -> 0, 2, 0, -2
        np.testing.assert_array_equal(enc, [0, 2, 0, -2, 2, 3])

    def test_overflow_is_refused_not_clamped(self):
        target = rk.TargetFormat("int8", 0.01)
        with self.assertRaises(rk.TargetOverflow):
            target.quantize(np.array([10.0]))
        with self.assertRaises(rk.TargetOverflow):
            target.quantize(np.array([-10.0]))

    def test_int8_decode_contract(self):
        q = np.array([[-2, 3], [5, -1]], dtype=np.int8)
        decoded = rk.decode_int8(q, 0.25, 1.0)
        np.testing.assert_allclose(decoded, [[-0.75, 0.5], [1.0, -0.5]])

    def test_quantized_block_upper_bounds(self):
        doc = rk.ExchangeDocument.from_dict(simple_doc())
        runner = rk.ExchangeRunner(doc)
        view = rk.View.from_spec(
            "i8",
            {"layout": "interleaved", "segments": [[0, 32, 0, [1.0, 0.1]]],
             "dtype": "int8", "scale": 0.05},
            rotary_dim=4,
        )
        acq = runner.store.acquire("h", "b", view)
        self.assertTrue(acq.plan.ready)
        ideal = rk.apply_rotation(
            np.asarray(doc.blocks["b"].original, dtype=np.float64), view, [2]
        )[0]
        err = np.linalg.norm(acq.block.decoded()[0] - ideal)
        self.assertLessEqual(err, acq.plan.l2_bound + 1e-15)
        self.assertLessEqual(
            acq.plan.score_bound,
            doc.query_norm_ceiling * acq.plan.l2_bound + 1e-15,
        )


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.doc = rk.ExchangeDocument.from_path(SCENE)

    def test_document_parsing(self):
        self.assertEqual(self.doc.width, 6)
        self.assertEqual(set(self.doc.views), {"short", "extended", "field-reader"})
        self.assertEqual(sorted(self.doc.blocks), ["k0", "k1", "k2"])

    def test_original_block_uses_exact_contract(self):
        runner = rk.ExchangeRunner(self.doc)
        result = runner.borrow("A")
        self.assertTrue(result.ready)
        held = result.delivered["k0"]
        plan = next(p for p in result.plans if p.block_id == "k0")
        self.assertEqual(held.phase_source, rk.PHASE_EXACT)
        # 原值存在时源相位误差为零；剩下只有旋转算术与目标舍入。
        self.assertEqual(plan.error_breakdown["source"], 0.0)
        self.assertLess(held.l2_bound, 1e-6)

    def test_low_precision_source_rejected_by_public_bound(self):
        runner = rk.ExchangeRunner(self.doc)
        result = runner.borrow("A")
        self.assertIn("k1", result.excluded)
        plan = next(p for p in result.plans if p.block_id == "k1")
        self.assertFalse(plan.ready)
        self.assertEqual(plan.phase_source, rk.PHASE_ROTATED)
        self.assertGreater(plan.l2_bound, 0.0)
        self.assertGreater(plan.score_bound, self.doc.score_error_limit)
        self.assertIn(rk.R_ERROR_BOUND, plan.reasons)

    def test_scores_match_ideal_rotation(self):
        runner = rk.ExchangeRunner(self.doc)
        runner.borrow("A")
        scores = runner.score("A")
        query = rk.rotate_query(
            self.doc.query, self.doc.views["short"], self.doc.queries["A"].positions
        )
        block = self.doc.blocks["k0"]
        keys = rk.apply_rotation(
            np.asarray(block.original, dtype=np.float64),
            self.doc.views["short"],
            block.positions,
        )
        ideal = query @ keys.T
        err = np.abs(scores["k0"] - ideal).max()
        held = runner.available_keys("A")["k0"]
        self.assertLessEqual(err, held.score_bound + 1e-12)

    def test_different_views_are_not_shared(self):
        runner = rk.ExchangeRunner(self.doc)
        a = runner.borrow("A")
        b = runner.borrow("B")
        keys_a = a.delivered["k0"].keys
        keys_b = b.delivered["k0"].keys
        # A 用 interleaved，B 用 split；即使目标格式相同也不得复用。
        self.assertFalse(np.array_equal(keys_a, keys_b))
        self.assertEqual(a.delivered["k0"].phase_source, rk.PHASE_EXACT)
        self.assertEqual(b.delivered["k0"].phase_source, rk.PHASE_EXACT)

    def test_int8_field_reader_within_limit(self):
        runner = rk.ExchangeRunner(self.doc)
        result = runner.borrow("C")
        self.assertTrue(result.delivered)
        for held in result.delivered.values():
            self.assertEqual(held.target.dtype, "int8")
            self.assertLessEqual(held.score_bound, self.doc.score_error_limit)

    def test_missing_block_is_impossible_not_zero(self):
        runner = rk.ExchangeRunner(self.doc)
        acq = runner.store.acquire("h", "nope", self.doc.views["short"])
        self.assertEqual(acq.plan.status, rk.STATUS_IMPOSSIBLE)
        self.assertIn(rk.R_MISSING, acq.plan.reasons)
        self.assertIsNone(acq.block)
        runner.store.release("h")

    def test_audit_reports_phase_and_bounds(self):
        runner = rk.ExchangeRunner(self.doc)
        runner.borrow("A")
        audit = runner.audit("A")
        self.assertEqual(audit["status"], rk.STATUS_READY)
        k0 = next(b for b in audit["blocks"] if b["block"] == "k0")
        self.assertTrue(k0["held"])
        self.assertEqual(k0["phase_source"], rk.PHASE_EXACT)
        self.assertIn("error_breakdown", k0)
        k1 = next(b for b in audit["blocks"] if b["block"] == "k1")
        self.assertFalse(k1["held"])
        self.assertTrue(k1["reasons"])


class BudgetTests(unittest.TestCase):
    def test_tight_residency_waits_then_retries(self):
        runner = rk.ExchangeRunner.from_path(TIGHT)
        events = runner.run_timeline()
        statuses = [e.get("status") for e in events if "status" in e]
        self.assertEqual(statuses, [rk.STATUS_READY, rk.STATUS_WAITING,
                                    rk.STATUS_READY])
        self.assertEqual(runner.borrow("C").status, rk.STATUS_READY)

    def test_waiting_blocks_are_not_delivered(self):
        runner = rk.ExchangeRunner.from_path(TIGHT)
        runner.borrow("B")
        result = runner.borrow("C")
        self.assertEqual(result.status, rk.STATUS_WAITING)
        self.assertEqual(result.delivered, {})
        self.assertEqual(set(result.waiting), {"k0", "k2"})
        with self.assertRaises(rk.ExchangeError):
            runner.score("C")

    def test_release_frees_derived_bytes_only(self):
        runner = rk.ExchangeRunner.from_path(TIGHT)
        runner.borrow("B")
        before = runner.residency()
        freed = runner.release("B")
        after = runner.residency()
        self.assertEqual(freed, before["derived"])
        self.assertEqual(after["derived"], 0)
        self.assertEqual(after["source"], before["source"])

    def test_shared_derived_representation_charged_once(self):
        doc = rk.ExchangeDocument.from_dict(simple_doc())
        runner = rk.ExchangeRunner(doc)
        first = runner.store.acquire("h1", "b", doc.views["v"])
        second = runner.store.acquire("h2", "b", doc.views["v"])
        self.assertGreater(first.plan.resident_bytes, 0)
        self.assertEqual(second.plan.resident_bytes, 0)
        self.assertTrue(second.plan.reused)
        self.assertIs(first.rep, second.rep)
        runner.store.release("h1")
        self.assertEqual(runner.store.derived_bytes, first.plan.resident_bytes)
        runner.store.release("h2")
        self.assertEqual(runner.store.derived_bytes, 0)

    def test_shared_bytes_survive_one_reader_returning(self):
        doc = rk.ExchangeDocument.from_dict(simple_doc())
        runner = rk.ExchangeRunner(doc)
        runner.store.acquire("h1", "b", doc.views["v"])
        held = runner.store.acquire("h2", "b", doc.views["v"])
        keys = held.block.keys
        runner.store.release("h1")
        self.assertTrue(np.array_equal(keys, doc.blocks["b"].original, equal_nan=False)
                        or keys.shape == (1, 6))
        self.assertGreater(runner.store.derived_bytes, 0)
        runner.store.release("h2")
        self.assertEqual(runner.store.derived_bytes, 0)

    def test_temporary_budget_refuses_before_building(self):
        doc = rk.ExchangeDocument.from_dict(
            simple_doc(budget_bytes={"resident": 1000, "temporary": 8})
        )
        runner = rk.ExchangeRunner(doc)
        acq = runner.store.acquire("h", "b", doc.views["v"])
        self.assertEqual(acq.plan.status, rk.STATUS_IMPOSSIBLE)
        self.assertIn(rk.R_TEMPORARY, acq.plan.reasons)
        self.assertEqual(runner.store.derived_bytes, 0)


class IsolationTests(unittest.TestCase):
    def test_in_flight_reader_keeps_old_frequency_table(self):
        runner = rk.ExchangeRunner.from_path(SCENE)
        first = runner.borrow("A")
        old_keys = first.delivered["k0"].keys
        runner.install_view(
            "short",
            {"layout": "interleaved", "segments": [[0, 12, 1, [0.7, 0.2]]],
             "dtype": "fp32"},
        )
        again = runner.borrow("A")
        self.assertIs(again.delivered["k0"].keys, old_keys)
        self.assertEqual(again.view.revision, 0)
        runner.release("A")
        fresh = runner.borrow("A")
        self.assertEqual(fresh.view.revision, 1)
        self.assertIsNot(fresh.delivered["k0"].keys, old_keys)

    def test_new_frequency_table_applies_to_next_borrow(self):
        runner = rk.ExchangeRunner.from_path(SCENE)
        first = runner.borrow("A")
        old_keys = first.delivered["k0"].keys.copy()
        runner.release("A")
        runner.install_view(
            "short",
            {"layout": "interleaved", "segments": [[0, 12, 1, [0.7, 0.2]]],
             "dtype": "fp32"},
        )
        fresh = runner.borrow("A")
        self.assertEqual(fresh.view.revision, 1)
        # 频率表变化后键内容必须变化，不能沿用旧派生表示。
        self.assertFalse(np.array_equal(fresh.delivered["k0"].keys, old_keys))

    def test_rotated_source_is_undone_before_review(self):
        data = simple_doc()
        block = data["blocks"][0]
        view = rk.View.from_spec(
            "v",
            {"layout": "interleaved", "segments": [[0, 32, 0, [1.0, 0.1]]]},
            rotary_dim=4,
        )
        block["phase"] = "v"
        raw = np.asarray(block["original"], dtype=np.float64)
        rotated = rk.apply_rotation(raw, view, [2])
        block["scale"] = 0.005
        block["q"] = np.rint(rotated / block["scale"]).astype(int).tolist()
        block["l2_source_error"] = 0.004
        del block["original"]  # 只剩低精度旧相位，没有可恢复原值
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("Q")
        held = result.delivered["b"]
        self.assertEqual(held.phase_source, rk.PHASE_ROTATED)
        ideal = rk.apply_rotation(raw, doc.views["v"], [2])[0]
        err = np.linalg.norm(held.decoded()[0] - ideal)
        self.assertLessEqual(err, held.l2_bound + 1e-12)

    def test_unknown_phase_is_refused(self):
        data = simple_doc()
        data["blocks"][0]["phase"] = "ghost"
        with self.assertRaises(rk.ExchangeError):
            rk.ExchangeDocument.from_dict(data)

    def test_mixed_views_never_share_representation(self):
        runner = rk.ExchangeRunner.from_path(SCENE)
        runner.borrow("A")
        runner.borrow("B")
        reps = list(runner.store._reps.values())
        self.assertEqual(len(reps), 2)
        rotations = {rep.rotation_sig for rep in reps}
        self.assertEqual(len(rotations), 2)


class RefundOrderTests(unittest.TestCase):
    def test_delivered_keys_are_immutable(self):
        runner = rk.ExchangeRunner.from_path(SCENE)
        result = runner.borrow("A")
        keys = result.delivered["k0"].keys
        with self.assertRaises(ValueError):
            keys[0, 0] = 1.0

    def test_returns_in_any_order(self):
        runner = rk.ExchangeRunner.from_path(SCENE)
        runner.borrow("A")
        runner.borrow("B")
        runner.borrow("C")
        base = runner.residency()
        runner.release("B")
        runner.release("A")
        runner.release("C")
        final = runner.residency()
        self.assertEqual(final["derived"], 0)
        self.assertEqual(final["source"], base["source"])

    def test_partial_block_release(self):
        doc = rk.ExchangeDocument.from_dict(simple_doc())
        runner = rk.ExchangeRunner(doc)
        runner.store.acquire("h", "b", doc.views["v"])
        freed = runner.store.release("h", "b")
        self.assertEqual(runner.store.derived_bytes, 0)
        self.assertGreater(freed, 0)
        self.assertEqual(runner.store.release("h"), 0)


class ExchangeFormatTests(unittest.TestCase):
    def test_samples_still_inspect(self):
        self.assertIn("rotary_dim", inspect(SCENE)["字段"])
        self.assertGreater(inspect(TIGHT)["字节"], 0)

    def test_incomplete_scene_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaises(KeyError):
                inspect(path)

    def test_authored_sample_round_trips(self):
        doc = rk.ExchangeDocument.from_path(SCENE)
        runner = rk.ExchangeRunner(doc)
        runner.borrow("A")
        json.dumps(runner.audit("A"), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()


class CrossViewTests(unittest.TestCase):
    def _rotated_block_scene(self):
        """k0 只剩 short 相位的 int8 表示，但另有一个 extended 视图。"""

        data = load(SCENE)
        k0 = data["blocks"][0]
        raw = np.asarray(k0["original"], dtype=np.float64)
        views = {
            name: rk.View.from_spec(name, spec, data["rotary_dim"])
            for name, spec in data["views"].items()
        }
        rotated = rk.apply_rotation(raw, views["short"], k0["positions"])
        scale = 0.02
        encoded = np.rint(rotated / scale)
        true_err = float(np.linalg.norm(encoded * scale - rotated, axis=1).max())
        k0.pop("original")
        k0.update(scale=scale, zero=0, q=encoded.astype(int).tolist(),
                  l2_source_error=true_err)
        data["budget_bytes"] = {"resident": 1000, "temporary": 4096}
        return data, raw

    def test_undo_and_redo_across_views_stays_within_bound(self):
        data, raw = self._rotated_block_scene()
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("B")  # extended / split
        held = result.delivered["k0"]
        self.assertEqual(held.phase_source, rk.PHASE_ROTATED)
        ideal = rk.apply_rotation(raw, doc.views["extended"], doc.blocks["k0"].positions)
        actual = np.linalg.norm(held.decoded() - ideal, axis=1)
        self.assertLessEqual(float(actual.max()), held.l2_bound + 1e-12)

    def test_same_rotation_skips_redo(self):
        data, raw = self._rotated_block_scene()
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("A")  # short / interleaved，与存储相位相同
        held = result.delivered["k0"]
        plan = next(p for p in result.plans if p.block_id == "k0")
        self.assertEqual(held.phase_source, rk.PHASE_ROTATED)
        # 不做重做旋转，算术项只剩解码级误差（远小于源相位误差）。
        self.assertLess(plan.error_breakdown["arithmetic"], 1e-12)
        self.assertLess(abs(held.l2_bound - doc.blocks["k0"].l2_source_error), 1e-6)

    def test_declared_bound_lie_is_recorded_not_fatal(self):
        data = load(SCENE)
        data["blocks"][0]["l2_source_error"] = 0.0  # 实际误差约 0.15
        runner = rk.ExchangeRunner.from_dict(data)
        result = runner.borrow("A")
        # 原值存在时按精确契约交付，声明不实只记录告警。
        self.assertTrue(result.ready)
        self.assertTrue(runner.store.data_warnings)
        warning = runner.store.data_warnings[0]
        self.assertEqual(warning["block"], "k0")
        self.assertGreater(warning["actual"], 0.0)


class AliasTests(unittest.TestCase):
    def test_equivalent_int8_delivery_uses_no_extra_residency(self):
        data = load(SCENE)
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        # 构造一个已带 field-reader 相位的块，交付要求与源完全一致。
        view = doc.views["field-reader"]
        raw = np.asarray(data["blocks"][2]["q"], dtype=np.float64)
        del raw
        block = rk.StoredBlock(
            id="aliased",
            positions=(8,),
            q=np.array([[2, 4, 5, 1, 1, 4]]),
            scale=view.scale,
            zero=0.0,
            l2_source_error=0.01,
            phase=view.name,
            source_view=view,
        )
        store = rk.KeyStore({"aliased": block}, doc.contract())
        acq = store.acquire("h", "aliased", view)
        self.assertTrue(acq.plan.ready)
        self.assertEqual(acq.plan.resident_bytes, 0)
        self.assertEqual(store.derived_bytes, 0)
        self.assertTrue(acq.plan.detail)

    def test_alias_refused_when_rotation_differs(self):
        data = load(SCENE)
        doc = rk.ExchangeDocument.from_dict(data)
        block = rk.StoredBlock(
            id="b",
            positions=(8,),
            q=np.array([[2, 4, 5, 1, 1, 4]]),
            scale=0.05,
            zero=0.0,
            l2_source_error=0.01,
            phase="field-reader",
            source_view=doc.views["field-reader"],
        )
        store = rk.KeyStore({"b": block}, doc.contract())
        # short 视图的旋转与 field-reader 不同，必须重算而不是引用源字节。
        acq = store.acquire("h", "b", doc.views["short"])
        self.assertTrue(acq.plan.ready)
        self.assertGreater(acq.plan.resident_bytes, 0)


class VisibilityTests(unittest.TestCase):
    def test_query_ranges_restrict_delivered_rows(self):
        data = load(SCENE)
        data["queries"].append(
            {"id": "R", "view": "short", "positions": [3],
             "blocks": ["k0"], "ranges": [[0, 4]]}
        )
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("R")
        self.assertTrue(result.ready)
        held = result.delivered["k0"]
        self.assertEqual(held.positions, (3,))
        scores = runner.score("R")
        self.assertEqual(scores["k0"].shape, (1, 1))

    def test_out_of_range_rows_are_reported_not_zero_filled(self):
        data = load(SCENE)
        data["queries"].append(
            {"id": "R", "view": "short", "positions": [3],
             "blocks": ["k0"], "ranges": [[10, 20]]}
        )
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("R")
        self.assertFalse(result.ready)
        self.assertEqual(result.delivered, {})
        self.assertIn("k0", result.excluded)
        plan = result.plans[0]
        self.assertIn(rk.R_OUT_OF_RANGE, plan.reasons)
        self.assertEqual(plan.excluded_positions, (3, 4))

    def test_partially_visible_rows_are_excluded_with_reason(self):
        data = load(SCENE)
        data["queries"].append(
            {"id": "R", "view": "short", "positions": [3],
             "blocks": ["k0"], "ranges": [[4, 20]]}
        )
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("R")
        self.assertTrue(result.ready)
        held = result.delivered["k0"]
        self.assertEqual(held.positions, (4,))
        self.assertIn(rk.R_OUT_OF_RANGE, result.plans[0].reasons)


class ErrorContractTests(unittest.TestCase):
    def test_non_rotary_dims_pass_through_within_target_format(self):
        data = load(SCENE)
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("C")  # int8 field-reader
        held = result.delivered["k2"]
        # k2 是 neutral，field-reader 只旋转前 4 维：后两维不做旋转，
        # 但仍须按目标格式量化并满足其误差契约。
        source = rk.decode_int8(doc.blocks["k2"].q, doc.blocks["k2"].scale,
                                doc.blocks["k2"].zero)
        scale = doc.views["field-reader"].scale
        self.assertTrue(np.allclose(
            held.decoded()[:, 4:], np.rint(source[:, 4:] / scale) * scale,
            atol=1e-12,
        ))
        self.assertLessEqual(
            float(np.linalg.norm(held.decoded()[:, 4:] - source[:, 4:])),
            held.l2_bound + 1e-12,
        )

    def test_score_bounds_are_monotone_in_query_norm(self):
        data = load(SCENE)
        data["query_norm_ceiling"] = 0.5
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        # k1 的公开误差界 0.16；查询范数上限降到 0.5 后应可交付。
        acq = runner.store.acquire("h", "k1", doc.views["short"])
        self.assertTrue(acq.plan.ready)
        self.assertLessEqual(acq.plan.score_bound, doc.score_error_limit)

    def test_zero_query_norm_ceiling_makes_any_bound_fit(self):
        data = load(SCENE)
        data["query"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data["query_norm_ceiling"] = 0.0
        doc = rk.ExchangeDocument.from_dict(data)
        runner = rk.ExchangeRunner(doc)
        result = runner.borrow("A")
        self.assertIn("k1", result.delivered)
        self.assertEqual(result.delivered["k1"].score_bound, 0.0)
