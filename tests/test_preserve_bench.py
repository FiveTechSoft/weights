"""engine/preserve_bench.py (lab/probes/preserve/PLAN.md): capability-preservation battery from local data. HumanEval
164 loader, garbage / loop measures, the 40-prompt home-made instruction-compliance set and its checks, the summary
schema. Written before the module."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import preserve_bench as pb  # noqa: E402


class HumanEvalLoaderTest(unittest.TestCase):
    def test_loads_164_tasks_with_fields(self):
        probs = pb.load_humaneval()
        self.assertEqual(len(probs), 164)
        self.assertEqual(probs[0]["task_id"], "HumanEval/0")
        for k in ("prompt", "test", "entry_point"):
            self.assertIn(k, probs[0])
        self.assertEqual(len(pb.load_humaneval(5)), 5)


class GarbageMeasuresTest(unittest.TestCase):
    def test_repeated_16gram_flags_loops_only(self):
        unit = "wait let me check the answer again because it should be sixty four not sixty two okay "
        looped = "some reasoning first. " + unit * 4
        g = pb.garbage_measures(looped)
        self.assertTrue(g["repeat_16gram"])
        clean = " ".join(f"word{i}" for i in range(400))
        self.assertFalse(pb.garbage_measures(clean)["repeat_16gram"])

    def test_nonascii_run_flags_fragment_salad_not_short_foreign_words(self):
        salad = "njąesson{id境风的专家的 vant美颜ära base过我emon automatértirlines经 reapsureمانiable lệ簧чай所以他们"
        self.assertTrue(pb.garbage_measures("x " * 50 + salad)["nonascii_run"])
        self.assertFalse(pb.garbage_measures("The answer is 42, también en español, c'est fini.")["nonascii_run"])
        self.assertEqual(pb.garbage_measures("")["nonascii_run"], False)


class ComplianceSetTest(unittest.TestCase):
    def test_forty_prompts_ten_per_kind_from_unused_gsm_indices(self):
        items = pb.compliance_set()
        self.assertEqual(len(items), 40)
        kinds = [it["kind"] for it in items]
        for k in pb.KINDS:
            self.assertEqual(kinds.count(k), 10)
        ids = [it["i"] for it in items]
        self.assertEqual(len(set(ids)), 40)
        self.assertTrue(all(i >= pb.GSM_START for i in ids))      # far from the dev / sealed ranges
        self.assertTrue(all(it["instruction"] and it["question"] for it in items))

    def test_checks(self):
        self.assertTrue(pb.check_compliance("end_done", "The answer is 18.\nDONE"))
        self.assertTrue(pb.check_compliance("end_done", "The answer is 18. DONE."))
        self.assertFalse(pb.check_compliance("end_done", "DONE\nThe answer is 18."))
        self.assertTrue(pb.check_compliance("words_not_digits", "She makes eighteen dollars a day."))
        self.assertFalse(pb.check_compliance("words_not_digits", "She makes 18 dollars a day."))
        self.assertTrue(pb.check_compliance("one_sentence", "Janet makes 18 dollars at the market every day."))
        self.assertTrue(pb.check_compliance("one_sentence", "She earns $2.50 per egg, so 18 dollars."))  # decimals ok
        self.assertFalse(pb.check_compliance("one_sentence", "First she has 9 eggs. Then she sells them for 18."))
        self.assertFalse(pb.check_compliance("one_sentence", ""))
        self.assertTrue(pb.check_compliance("all_caps", "JANET MAKES 18 DOLLARS A DAY."))
        self.assertFalse(pb.check_compliance("all_caps", "Janet makes 18 dollars a day."))
        self.assertFalse(pb.check_compliance("all_caps", "18"))                      # fewer than 10 letters
        with self.assertRaises(ValueError):
            pb.check_compliance("unknown", "x")


class SummaryTest(unittest.TestCase):
    def test_summary_schema_and_counts(self):
        rows = [
            {"bench": "humaneval", "id": "HumanEval/0", "ok": True, "finish": "stop", "tokens": 100,
             "repeat_16gram": False, "nonascii_run": False},
            {"bench": "humaneval", "id": "HumanEval/1", "ok": False, "finish": "length", "tokens": 2048,
             "repeat_16gram": True, "nonascii_run": False},
            {"bench": "compliance", "id": 1000, "kind": "end_done", "ok": True, "finish": "stop", "tokens": 300,
             "repeat_16gram": False, "nonascii_run": False},
            {"bench": "compliance", "id": 1001, "kind": "all_caps", "ok": False, "finish": "stop", "tokens": 200,
             "repeat_16gram": False, "nonascii_run": True},
        ]
        s = pb.summarize(rows)
        self.assertEqual(s["humaneval"], {"n": 2, "ok": 1, "closed": 1, "median_tokens": 1074})
        self.assertEqual(s["compliance"]["n"], 2)
        self.assertEqual(s["compliance"]["ok"], 1)
        self.assertEqual(s["compliance"]["by_kind"], {"end_done": [1, 1], "all_caps": [0, 1]})
        self.assertEqual(s["garbage"], {"n": 4, "repeat_16gram": 1, "nonascii_run": 1, "cut_at_max": 1})


if __name__ == "__main__":
    unittest.main()
