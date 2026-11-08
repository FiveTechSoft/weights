"""lab/probes/close_probe/distill_gen.py: training items disjoint from the eval set, and grading of a raw generation."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


class TestDistillGen(unittest.TestCase):
    def test_selection_is_deterministic_and_disjoint(self):
        import distill_gen as dg
        a = dg.select(10, 100, exclude={1, 2, 3})
        self.assertEqual(a, dg.select(10, 100, exclude={1, 2, 3}))
        self.assertEqual(len(set(a)), 10)
        self.assertFalse(set(a) & {1, 2, 3})

    def test_done_ids_reads_items_of_an_earlier_run(self):
        import tempfile
        import distill_gen as dg
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.jsonl"
            p.write_text('{"i": 4, "ok": true}\n{"i": 9, "ok": false}\n', encoding="utf-8")
            self.assertEqual(dg.done_ids([str(p)]), {4, 9})
        self.assertEqual(dg.done_ids([]), set())

    def test_grade_needs_a_closed_think_and_a_right_answer(self):
        import distill_gen as dg
        self.assertEqual(dg.grade("x \boxed{5} </think>\n\nSo \boxed{6}", "6"), (True, True))
        self.assertEqual(dg.grade("x \boxed{6} still thinking", "6"), (False, False))
        self.assertEqual(dg.grade("x</think>\n\n\boxed{7}", "6"), (True, False))


if __name__ == "__main__":
    unittest.main()
