"""lab/probes/cc_bias (roadmap gap 18): a constant logit bias on `</think>` as the control the conclude vector lacks.
Pick rule, comparison summary and the token-id lookup. Written before bias.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.probes.cc_bias import bias as cb  # noqa: E402


def item(bench, i, ok, finish="stop", tokens=1000, reasoning=500, ok2=None):
    return {"bench": bench, "i": i, "ok": ok, "ok_independent": ok if ok2 is None else ok2, "finish": finish,
            "tokens": tokens, "reasoning_chars": reasoning, "gold": "1", "pred": "1", "tail": ""}


class PickTest(unittest.TestCase):
    def test_closest_mean_tokens_to_vector_ties_to_smaller_bias(self):
        # vector mean tokens 900; +4 and +6 both 100 away -> the smaller bias
        arms = {2: 1500, 4: 1000, 6: 800, 8: 400}
        self.assertEqual(cb.pick(arms, 900), 4)
        self.assertEqual(cb.pick({2: 1500, 4: 1200, 6: 820, 8: 400}, 900), 6)

    def test_pick_uses_pooled_mean_over_benches(self):
        reports = {4: {"gsm8k": {"n": 100, "mean_tokens": 1000}, "math500": {"n": 50, "mean_tokens": 1600}},
                   6: {"gsm8k": {"n": 100, "mean_tokens": 700}, "math500": {"n": 50, "mean_tokens": 1300}}}
        pooled = cb.pooled_tokens(reports[4])
        self.assertAlmostEqual(pooled, (100 * 1000 + 50 * 1600) / 150)
        vec = {"gsm8k": {"n": 100, "mean_tokens": 735}, "math500": {"n": 50, "mean_tokens": 1359}}
        self.assertEqual(cb.pick({b: cb.pooled_tokens(r) for b, r in reports.items()}, cb.pooled_tokens(vec)), 6)


class CompareTest(unittest.TestCase):
    def test_summary_counts(self):
        base = [item("gsm8k", 0, True), item("gsm8k", 1, False, "length", 2048), item("gsm8k", 2, False),
                item("math500", 0, False, "length", 2048)]
        arm = [item("gsm8k", 0, False, tokens=300, reasoning=10), item("gsm8k", 1, True, tokens=500),
               item("gsm8k", 2, False, "stop", 400, reasoning=40), item("math500", 0, True, tokens=900)]
        vec = [item("gsm8k", 0, True), item("gsm8k", 1, True), item("gsm8k", 2, True), item("math500", 0, True)]
        s = cb.compare(base, arm, vec)
        g = s["gsm8k"]
        self.assertEqual(g["arm_ok"], 1)
        self.assertEqual(g["vec_ok"], 3)
        self.assertEqual(g["base_ok"], 1)
        self.assertEqual(g["arm_finished"], 3)
        self.assertEqual(g["arm_cut"], 0)
        self.assertEqual(g["arm_nothink"], 2)            # reasoning_chars < 50
        self.assertEqual(g["lost_base_ok_stop"], 1)      # item 0: base correct and finished, arm wrong
        self.assertEqual(g["gained"], 1)                 # item 1 (cut in base) now right
        self.assertEqual(g["vs_vec"]["win"], 0)
        self.assertEqual(g["vs_vec"]["loss"], 2)
        self.assertAlmostEqual(g["vs_vec"]["p"], 0.5, places=6)   # 0/2 discordant, exact binomial two-sided
        self.assertEqual(s["math500"]["arm_ok"], 1)

    def test_r1_rule(self):
        g = {"arm_ok": 80, "vec_ok": 83, "vs_vec": {"p": 0.5}}
        m = {"arm_ok": 33, "vec_ok": 35, "vs_vec": {"p": 0.6}}
        self.assertTrue(cb.r1({"gsm8k": g, "math500": m}))
        self.assertFalse(cb.r1({"gsm8k": dict(g, arm_ok=79), "math500": m}))
        self.assertFalse(cb.r1({"gsm8k": g, "math500": dict(m, arm_ok=32)}))
        self.assertFalse(cb.r1({"gsm8k": dict(g, vs_vec={"p": 0.01}), "math500": m}))


class TokenTest(unittest.TestCase):
    def test_close_think_id_from_tokenizer_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokenizer.json"
            p.write_text(json.dumps({"added_tokens": [{"id": 5, "content": "<think>"}, {"id": 7, "content": "</think>"}],
                                     "model": {"vocab": {"think": 3}}}), encoding="utf-8")
            self.assertEqual(cb.close_think_id(d), 7)

    def test_real_qwen35_9b_id(self):
        hf = ROOT / "hf_cache" / "qwen35-9b-hf"
        if not (hf / "tokenizer.json").exists():
            self.skipTest("no local Qwen3.5-9B tokenizer")
        self.assertEqual(cb.close_think_id(hf), 248069)


if __name__ == "__main__":
    unittest.main()
