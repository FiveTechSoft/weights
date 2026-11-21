"""lab/probes/cc/name_test.py: the "name test" of the conclude direction outside maths (lab/probes/cc/PLAN.md, "Name
test"). Pure parts only: scoring a logic reply, summarising rows, the N1-N3 readings and the placebo bands."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "cc"))

import name_test as nt   # noqa: E402


class TestScoreLogic(unittest.TestCase):
    def test_answer_is_read_after_the_think_block(self):
        self.assertEqual(nt.logic_answer("<think>maybe YES, maybe NO</think>\n\nNO"), "NO")

    def test_whole_text_when_there_is_no_think_block(self):
        self.assertEqual(nt.logic_answer("Alda is a glim, so YES"), "YES")

    def test_cut_inside_think_scores_nothing(self):
        self.assertEqual(nt.logic_answer("<think>still reasoning about Borin"), "")

    def test_both_yes_and_no_is_rejected(self):
        self.assertEqual(nt.logic_answer("</think>YES. Actually NO"), "")


class TestSummarize(unittest.TestCase):
    def test_counts_correct_cut_and_median_tokens(self):
        rows = [{"correct": True, "finish": "stop", "tokens": 100}, {"correct": False, "finish": "length", "tokens": 2048},
                {"correct": True, "finish": "stop", "tokens": 300}]
        s = nt.summarize(rows)
        self.assertEqual((s["n"], s["correct"], s["cut"], s["median_tokens"]), (3, 2, 1, 300))


class TestReadings(unittest.TestCase):
    def setUp(self):
        self.base = {"he": {"n": 20, "correct": 20, "cut": 0, "median_tokens": 650},
                     "logic": {"n": 120, "correct": 100, "cut": 3, "median_tokens": 800}}

    def test_name_holds_when_shorter_and_cost_bounded_in_both(self):
        vec = {"he": {"n": 20, "correct": 19, "cut": 0, "median_tokens": 300},
               "logic": {"n": 120, "correct": 96, "cut": 0, "median_tokens": 400}}
        r = nt.readings(self.base, vec)
        self.assertTrue(r["N1"]["he"] and r["N1"]["logic"] and r["N2"]["he"] and r["N2"]["logic"])
        self.assertEqual(r["N3"], "not applicable")
        self.assertEqual(r["name"], "holds")

    def test_concludes_at_a_cost_when_accuracy_drops_in_one_domain(self):
        vec = {"he": {"n": 20, "correct": 15, "cut": 0, "median_tokens": 200},
               "logic": {"n": 120, "correct": 98, "cut": 0, "median_tokens": 400}}
        r = nt.readings(self.base, vec)
        self.assertFalse(r["N2"]["he"])
        self.assertEqual(r["name"], "concludes at a cost")

    def test_name_fails_when_not_shorter_in_a_domain(self):
        vec = {"he": {"n": 20, "correct": 20, "cut": 0, "median_tokens": 600},
               "logic": {"n": 120, "correct": 100, "cut": 0, "median_tokens": 400}}
        self.assertEqual(nt.readings(self.base, vec)["name"], "fails")

    def test_n3_applies_only_when_the_base_cuts_a_tenth(self):
        base = dict(self.base, logic={"n": 120, "correct": 60, "cut": 40, "median_tokens": 2048})
        vec = {"he": {"n": 20, "correct": 19, "cut": 0, "median_tokens": 300},
               "logic": {"n": 120, "correct": 90, "cut": 15, "median_tokens": 900}}
        r = nt.readings(base, vec)
        self.assertEqual(r["N3"], {"logic": True})          # 15 <= 40 / 2
        vec["logic"]["cut"] = 25
        self.assertEqual(nt.readings(base, vec)["N3"], {"logic": False})

    def test_placebo_bands(self):
        pl = {"he": {"n": 20, "correct": 19, "cut": 0, "median_tokens": 700},
              "logic": {"n": 120, "correct": 95, "cut": 2, "median_tokens": 730}}
        c = nt.control(self.base, pl)
        self.assertEqual(c, {"he": True, "logic": True})
        pl["logic"]["median_tokens"] = 600                   # -25 %: outside +-10 %
        self.assertEqual(nt.control(self.base, pl)["logic"], False)


if __name__ == "__main__":
    unittest.main()
