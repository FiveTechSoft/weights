"""lab/probes/skill2/skill2.py (lab/probes/skill2/PLAN.md): a second behaviour, "reason in the requested language"
(Spanish thinking on English problems), read from the base's own answers. Judge, row builder, splits, dev rule.
Written before skill2.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.probes.skill2 import skill2 as s2  # noqa: E402

ES = ("Primero calculamos cuántas manzanas quedan después de que ella vende la mitad, y luego sumamos las que "
      "compró por la mañana para obtener el total que tiene ahora.")
EN = ("First we compute how many apples are left after she sells half of them, and then we add the ones she "
      "bought in the morning to get the total she has now.")


class JudgeTest(unittest.TestCase):
    def test_spanish_and_english_detected(self):
        self.assertEqual(s2.lang(ES), "es")
        self.assertEqual(s2.lang(EN), "en")

    def test_short_or_numeric_text_is_unknown(self):
        self.assertEqual(s2.lang("16 - 3 - 4 = 9. 9 * 2 = 18."), "unk")
        self.assertEqual(s2.lang(""), "unk")

    def test_mixed_text_is_decided_by_the_majority_with_margin(self):
        self.assertEqual(s2.lang(ES + " " + EN[:40]), "es")
        self.assertEqual(s2.lang(ES[:60] + " " + EN), "en")

    def test_head_window_is_used(self):
        # the first 300 words decide; a Spanish head followed by an English tail is Spanish
        text = (ES + " ") * 20 + (EN + " ") * 60
        self.assertEqual(s2.lang(s2.head(text, 300)), "es")
        self.assertEqual(len(s2.head(text, 300).split()), 300)


class RowsTest(unittest.TestCase):
    def test_rows_pair_spanish_and_english_answers_and_drop_disagreements(self):
        gens = [{"i": 1, "prompt": "P1", "en": {"content": EN}, "es": {"content": ES}},
                {"i": 2, "prompt": "P2", "en": {"content": EN}, "es": {"content": EN}},   # did not comply: dropped
                {"i": 3, "prompt": "P3", "en": {"content": ES}, "es": {"content": ES}},   # English arm not English: dropped
                {"i": 4, "prompt": "P4", "en": {"content": "42"}, "es": {"content": ES}}]  # unknown: dropped
        pos, neg, stats = s2.build_rows(gens)
        self.assertEqual(len(pos), 1)
        self.assertEqual(len(neg), 1)
        self.assertEqual(pos[0]["prompt"], "P1")
        self.assertEqual(pos[0]["response"], ES)
        self.assertEqual(neg[0]["response"], EN)
        self.assertFalse(pos[0]["think"])
        self.assertIsNone(pos[0].get("focus"))
        self.assertEqual(stats, {"n": 4, "kept": 1, "es_noncompliant": 1, "en_noncompliant": 1, "unknown": 1})


class SplitTest(unittest.TestCase):
    def test_discovery_and_screen_are_train_items_disjoint_from_cc_discovery(self):
        sp = s2.split()
        self.assertEqual(sp["discovery_gsm_train"], list(range(64, 128)))
        self.assertEqual(sp["screen_gsm_train"], list(range(128, 160)))
        self.assertFalse(set(sp["discovery_gsm_train"]) & set(range(64)))   # cc used train 0-63

    def test_sealed_ids_exclude_dev_and_every_existing_sealed_set(self):
        existing = [{"gsm8k": [150, 151], "math500": [3, 4]}, {"gsm8k": [152], "math500": [5]}]
        s = s2.sealed_ids(seed=1, gsm_n=5, math_n=3, gsm_total=300, math_total=50, dev_gsm=range(100),
                          dev_math=[0, 1, 2], existing=existing)
        self.assertEqual(len(s["gsm8k"]), 5)
        self.assertEqual(len(s["math500"]), 3)
        self.assertFalse(set(s["gsm8k"]) & (set(range(100)) | {150, 151, 152}))
        self.assertFalse(set(s["math500"]) & {0, 1, 2, 3, 4, 5})
        self.assertEqual(s, s2.sealed_ids(seed=1, gsm_n=5, math_n=3, gsm_total=300, math_total=50, dev_gsm=range(100),
                                          dev_math=[0, 1, 2], existing=existing))   # deterministic


class RulesTest(unittest.TestCase):
    def test_screen_rule_picks_most_spanish_among_accuracy_safe_scales(self):
        base = {"es_think": 1, "ok": 10}
        arms = {"0.5": {"es_think": 8, "ok": 10}, "1.0": {"es_think": 20, "ok": 9}, "2.0": {"es_think": 26, "ok": 5},
                "4.0": {"es_think": 30, "ok": 0}}
        self.assertEqual(s2.choose_scale(base, arms, n=32), "1.0")       # 2.0 and 4.0 lose more than 3 correct
        arms["1.0"]["es_think"] = 12; arms["0.5"]["es_think"] = 12         # tie above base + 30 pp -> smaller scale
        self.assertEqual(s2.choose_scale(base, arms, n=32), "0.5")
        self.assertIsNone(s2.choose_scale(base, {"0.5": {"es_think": 5, "ok": 10}}, n=32))   # < base + 30 pp -> FAIL

    def test_dev_verdict_c1_c3(self):
        def items(n, es_flags, ok_flags):
            return [{"bench": "gsm8k", "i": k, "es_think": bool(e), "ok": bool(o), "finish": "stop", "tokens": 100}
                    for k, (e, o) in enumerate(zip(es_flags, ok_flags))]
        base = items(10, [0] * 10, [1] * 8 + [0] * 2)
        vec = items(10, [1] * 7 + [0] * 3, [1] * 7 + [0] * 3)
        pla = items(10, [0] * 10, [1] * 8 + [0] * 2)
        v = s2.dev_verdict(base, vec, pla)
        self.assertTrue(v["C1"])            # 7/0 discordant, p = 2/128 < 0.05, +70 pp
        self.assertTrue(v["C2"])            # placebo 0/0, p = 1, +0 pp
        self.assertTrue(v["C3"])            # 7 >= 8 - 3
        self.assertEqual(v["verdict"], "PASS")
        pla10 = items(10, [0] * 9 + [1], [1] * 8 + [0] * 2)                # +10 pp is not "below base + 10 pp"
        self.assertFalse(s2.dev_verdict(base, vec, pla10)["C2"])
        vec_bad = items(10, [1] * 7 + [0] * 3, [1] * 4 + [0] * 6)
        self.assertEqual(s2.dev_verdict(base, vec_bad, pla)["verdict"], "FAIL")   # C3: 4 < 5

    def test_binomial_two_sided(self):
        self.assertAlmostEqual(s2.binom_two_sided(7, 0), 2 / 128, places=9)
        self.assertEqual(s2.binom_two_sided(0, 0), 1.0)


class CompileTest(unittest.TestCase):
    def test_calib_rows_take_finished_english_traces_without_think_tag(self):
        gens = [{"i": 1, "prompt": "P1", "en": {"content": "x", "reasoning": "<think>\nstep one\nstep two", "finish": "stop"},
                 "es": {"content": "y", "reasoning": "r", "finish": "stop"}},
                {"i": 2, "prompt": "P2", "en": {"content": "", "reasoning": "cut...", "finish": "length"},
                 "es": {"content": "y", "reasoning": "r", "finish": "stop"}}]
        rows = s2.make_calib_rows(gens)
        self.assertEqual(rows, [{"prompt": "P1", "reasoning": "step one\nstep two"}])

    def test_fidelity_merged_equals_vector_item_by_item(self):
        def items(flags):
            return [{"bench": "gsm8k", "i": k, "es_think": bool(e), "ok": True, "finish": "stop", "tokens": 1}
                    for k, e in enumerate(flags)]
        vec = items([1] * 8 + [0] * 2)
        same = items([1] * 7 + [0] * 2 + [1])          # 1/1 discordant
        far = items([0] * 10)                           # 0/8 discordant, p = 2/256
        self.assertTrue(s2.fidelity(vec, same)["C4"])
        self.assertFalse(s2.fidelity(vec, far)["C4"])


class ReportTest(unittest.TestCase):
    def test_summarize_counts_spanish_thinking_only_on_the_head(self):
        rows = [{"bench": "gsm8k", "i": 0, "es_think": True, "es_answer": True, "ok": True, "ok_independent": True,
                 "finish": "stop", "tokens": 500, "loop": False},
                {"bench": "gsm8k", "i": 1, "es_think": False, "es_answer": True, "ok": False, "ok_independent": False,
                 "finish": "length", "tokens": 2048, "loop": True}]
        s = s2.summarize(rows)
        self.assertEqual(s["gsm8k"], {"n": 2, "es_think": 1, "es_answer": 2, "ok": 1, "ok_independent": 1, "cut": 1,
                                      "loops": 1, "mean_tokens": 1274})


if __name__ == "__main__":
    unittest.main()
