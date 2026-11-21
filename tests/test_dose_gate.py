"""Dose gate for the Llama Part C (lab/probes/cc_llama/PLAN.md, Part C): per-scale summary against dev_base and the
pre-registered rule that decides whether sealed set 3 is used. Written before dose_gate.py."""
import unittest

from lab.probes.cc_llama.dose_gate import gate, summarize


def item(bench, i, ok, finish="stop", tokens=1000):
    return {"bench": bench, "i": i, "ok": ok, "ok_independent": ok, "finish": finish, "tokens": tokens,
            "reasoning_chars": 500, "gold": "1", "tail": "1"}


def arm(spec):
    """spec: list of (bench, i, ok, finish, tokens) -> items keyed by (bench, i)."""
    return {(b, i): item(b, i, ok, f, t) for b, i, ok, f, t in spec}


class DoseGateTest(unittest.TestCase):
    def test_summarize_counts_lost_only_among_base_correct_finished(self):
        base = arm([("gsm8k", 0, True, "stop", 1000), ("gsm8k", 1, True, "length", 2048), ("gsm8k", 2, False, "stop", 900),
                    ("math500", 0, False, "length", 2048), ("math500", 1, True, "stop", 1500)])
        vec = arm([("gsm8k", 0, False, "stop", 500), ("gsm8k", 1, False, "stop", 600), ("gsm8k", 2, True, "stop", 400),
                   ("math500", 0, True, "stop", 900), ("math500", 1, True, "stop", 700)])
        s = summarize(base, vec)
        self.assertEqual(s["gsm8k"]["ok"], 1)
        self.assertEqual(s["gsm8k"]["lost"], 1)          # item 0 only: item 1 was cut in the base
        self.assertEqual(s["gsm8k"]["gained"], 1)
        self.assertEqual(s["gsm8k"]["cut"], 0)
        self.assertAlmostEqual(s["gsm8k"]["token_change"], 1500 / 3948 - 1, places=3)
        self.assertEqual(s["math500"]["ok"], 2)
        self.assertEqual(s["math500"]["lost"], 0)

    def test_gate_rule_picks_best_scale_and_applies_thresholds(self):
        base_ok = {"gsm8k": 74, "math500": 22}
        summ = {"0.25": {"gsm8k": {"ok": 78, "lost": 5, "token_change": -0.25}, "math500": {"ok": 27, "lost": 1}},
                "0.35": {"gsm8k": {"ok": 77, "lost": 6, "token_change": -0.35}, "math500": {"ok": 29, "lost": 2}}}
        g = gate(summ, base_ok)
        self.assertEqual(g["chosen"], "0.35")              # max of gsm ok + math ok (106 vs 105)
        self.assertTrue(g["pass"])
        summ["0.35"]["gsm8k"]["lost"] = 8                    # lost > 6 at the chosen scale -> no sealed run
        self.assertFalse(gate(summ, base_ok)["pass"])
        summ["0.35"]["gsm8k"]["lost"] = 6
        summ["0.35"]["gsm8k"]["token_change"] = -0.15        # saving < 20 % -> no sealed run
        self.assertFalse(gate(summ, base_ok)["pass"])


if __name__ == "__main__":
    unittest.main()
