"""lab/probes/act_map/port_cc.py: gate 3b (the 9B's "conclude" direction written into the 0.8B through the gate 1 maps),
the pure pieces: measures against the base, the dev scale rule, the verdict rule, free dev items, the edit check."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "act_map"))


def _items(gsm_ok, gsm_fin, math_ok, math_fin, ok2=None):
    """ok / finished flags per item -> bench_runner-like items (ok2 defaults to ok)."""
    out = []
    for bench, oks, fins in (("gsm8k", gsm_ok, gsm_fin), ("math500", math_ok, math_fin)):
        for i, (o, f) in enumerate(zip(oks, fins)):
            out.append({"bench": bench, "i": i, "ok": bool(o), "ok_independent": bool(o if ok2 is None else ok2),
                        "finish": "stop" if f else "length", "tokens": 500 if f else 2048, "loop": False})
    return out


def _flags(n, k):
    return [i < k for i in range(n)]


class TestMeasures(unittest.TestCase):
    def test_counts_and_paired_tests_against_the_base(self):
        import port_cc as pc
        base = _items(_flags(20, 2), _flags(20, 3), _flags(10, 0), _flags(10, 1))
        arm = _items(_flags(20, 10), _flags(20, 14), _flags(10, 2), _flags(10, 4))
        m = pc.measures(arm, base)
        g = m["gsm8k"]
        self.assertEqual((g["n"], g["correct"], g["finished"], g["cut"]), (20, 10, 14, 6))
        self.assertEqual((g["vs_base"]["correct"]["win"], g["vs_base"]["correct"]["loss"]), (8, 0))
        self.assertLess(g["vs_base"]["correct"]["p"], 0.05)
        self.assertEqual((g["vs_base"]["finished"]["win"], g["vs_base"]["finished"]["loss"]), (11, 0))
        self.assertEqual(m["math500"]["correct"], 2)
        self.assertEqual(m["math500"]["vs_base"]["correct"]["p"], 0.5)      # 2/0
        self.assertEqual(m["mean_tokens_gsm8k"], round((14 * 500 + 6 * 2048) / 20))


class TestChooseScale(unittest.TestCase):
    def test_most_correct_wins_ties_go_to_the_smaller_scale(self):
        import port_cc as pc
        dev = {0.5: {"correct": 6, "finished": 9}, 1.0: {"correct": 6, "finished": 20}, 2.0: {"correct": 4, "finished": 30}}
        self.assertEqual(pc.choose_scale(dev, {"correct": 2, "finished": 3}), 0.5)

    def test_floor_rule_uses_finished_when_no_scale_beats_the_base_in_correct(self):
        import port_cc as pc
        dev = {0.5: {"correct": 1, "finished": 4}, 1.0: {"correct": 2, "finished": 12}, 2.0: {"correct": 0, "finished": 12}}
        self.assertEqual(pc.choose_scale(dev, {"correct": 2, "finished": 3}), 1.0)

    def test_none_when_nothing_beats_the_base(self):
        import port_cc as pc
        dev = {0.5: {"correct": 1, "finished": 3}, 1.0: {"correct": 2, "finished": 2}}
        self.assertIsNone(pc.choose_scale(dev, {"correct": 2, "finished": 3}))


def _m(gsm_c, gsm_p, gsm_fin, fin_p, math_c, math_p, gsm_c2=None, math_c2=None, base=False):
    """A measures-like dict for decide(): counts and p-values only."""
    def arm(c, p, c2, fin, fp):
        # a grader-2 count given explicitly comes with its own (null) p-value: the graders disagree
        return {"correct": c, "correct2": c if c2 is None else c2, "finished": fin,
                "vs_base": {"correct": {"p": p}, "correct2": {"p": p if c2 is None else 1.0}, "finished": {"p": fp}}}
    return {"gsm8k": arm(gsm_c, gsm_p, gsm_c2, gsm_fin, fin_p), "math500": arm(math_c, math_p, math_c2, 0, 1.0)}


class TestDecide(unittest.TestCase):
    def base(self):
        return _m(2, 1.0, 3, 1.0, 1, 1.0)

    def test_pass_with_exp8_as_reference_when_own_does_not_beat_the_base(self):
        import port_cc as pc
        st = {"base": self.base(),
              "own": _m(4, 0.6, 8, 0.2, 1, 1.0),                 # not significant: no own reference
              "port": _m(25, 1e-4, 30, 1e-5, 6, 0.1),            # closes and is correct on GSM8K
              "embport": _m(3, 1.0, 5, 0.5, 1, 1.0),
              "placebo": _m(2, 1.0, 4, 1.0, 1, 1.0),
              "exp8": _m(40, 1e-9, 70, 1e-9, 15, 1e-3),
              "exp8_placebo": _m(2, 1.0, 3, 1.0, 1, 1.0)}
        d = pc.decide(st)
        self.assertEqual(d["reference"], "exp8")
        self.assertTrue(d["Q1 closes (GSM8K finished, p<0.05)"])
        self.assertTrue(d["Q2 correct above base (both graders, one benchmark)"])
        self.assertTrue(d["Q3 placebo flat"])
        self.assertTrue(d["Q4 port gain >= 0.5 x reference gain"])      # (23 + 5) = 28 vs 0.5 x (38 + 14) = 26
        self.assertEqual((d["port_gain"], d["reference_gain"]), (28, 52))
        self.assertEqual(d["verdict"], "PASS")
        self.assertTrue(d["Q5 port - embport >= 0.5 x port gain"])

    def test_own_is_the_reference_when_it_beats_the_base(self):
        import port_cc as pc
        st = {"base": self.base(), "own": _m(30, 1e-6, 40, 1e-6, 6, 0.06),
              "port": _m(10, 0.01, 20, 1e-3, 2, 1.0), "embport": _m(2, 1.0, 3, 1.0, 1, 1.0),
              "placebo": _m(2, 1.0, 3, 1.0, 1, 1.0), "exp8": _m(40, 1e-9, 70, 1e-9, 15, 1e-3),
              "exp8_placebo": _m(2, 1.0, 3, 1.0, 1, 1.0)}
        d = pc.decide(st)
        self.assertEqual(d["reference"], "own")
        self.assertFalse(d["Q4 port gain >= 0.5 x reference gain"])      # 9 vs 0.5 x 33
        self.assertEqual(d["verdict"], "FAIL")

    def test_not_decided_without_any_reference(self):
        import port_cc as pc
        st = {"base": self.base(), "own": _m(3, 1.0, 4, 1.0, 1, 1.0), "port": _m(10, 0.01, 20, 1e-3, 2, 1.0),
              "embport": _m(2, 1.0, 3, 1.0, 1, 1.0), "placebo": _m(2, 1.0, 3, 1.0, 1, 1.0),
              "exp8": _m(5, 0.3, 6, 0.4, 1, 1.0), "exp8_placebo": _m(2, 1.0, 3, 1.0, 1, 1.0)}
        d = pc.decide(st)
        self.assertIsNone(d["reference"])
        self.assertTrue(d["verdict"].startswith("NOT DECIDED"))

    def test_q2_needs_both_graders_and_q3_fails_on_a_helpful_placebo(self):
        import port_cc as pc
        st = {"base": self.base(), "own": _m(3, 1.0, 4, 1.0, 1, 1.0),
              "port": _m(20, 1e-4, 30, 1e-5, 2, 1.0, gsm_c2=3),       # grader 2 disagrees on GSM8K
              "embport": _m(2, 1.0, 3, 1.0, 1, 1.0),
              "placebo": _m(12, 0.01, 20, 0.01, 1, 1.0),              # placebo helps
              "exp8": _m(40, 1e-9, 70, 1e-9, 15, 1e-3), "exp8_placebo": _m(2, 1.0, 3, 1.0, 1, 1.0)}
        d = pc.decide(st)
        self.assertFalse(d["Q2 correct above base (both graders, one benchmark)"])
        self.assertFalse(d["Q3 placebo flat"])
        self.assertEqual(d["verdict"], "FAIL")

    def test_missing_arm_fails_its_criteria(self):
        import port_cc as pc
        st = {"base": self.base(), "own": _m(3, 1.0, 4, 1.0, 1, 1.0), "port": None,
              "embport": _m(2, 1.0, 3, 1.0, 1, 1.0), "placebo": None,
              "exp8": _m(40, 1e-9, 70, 1e-9, 15, 1e-3), "exp8_placebo": _m(2, 1.0, 3, 1.0, 1, 1.0)}
        d = pc.decide(st)
        self.assertEqual(d["verdict"], "FAIL")
        self.assertFalse(d["Q1 closes (GSM8K finished, p<0.05)"])


class TestFreeItems(unittest.TestCase):
    def test_first_indices_not_used_anywhere(self):
        import port_cc as pc
        used = {"gsm8k": set(range(0, 100)) | {101, 103}, "math500": {0, 1, 5}}
        f = pc.free_items(used, gsm_n=4, math_n=3, gsm_total=1319, math_total=500)
        self.assertEqual(f, {"gsm8k": [100, 102, 104, 105], "math500": [2, 3, 4]})


class TestEditCheck(unittest.TestCase):
    def test_lists_only_the_tensors_whose_bytes_differ(self):
        import gguf
        import port_cc as pc
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = np.arange(12, dtype=np.float32).reshape(3, 4)
            b = a.copy(); b[1, 2] += 1.0
            for name, t2 in (("base.gguf", a), ("edit.gguf", b)):
                w = gguf.GGUFWriter(str(td / name), "llama")
                w.add_tensor("blk.0.ffn_down.weight", a)
                w.add_tensor("blk.15.ffn_down.weight", t2)
                w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            self.assertEqual(pc.differing_tensors(td / "base.gguf", td / "edit.gguf"), ["blk.15.ffn_down.weight"])
            self.assertEqual(pc.differing_tensors(td / "base.gguf", td / "base.gguf"), [])


if __name__ == "__main__":
    unittest.main()
