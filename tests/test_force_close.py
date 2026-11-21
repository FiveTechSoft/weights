"""lab/probes/close_probe/force_close.py: first answer inside the reasoning."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


class TestFirstAnswer(unittest.TestCase):
    def test_ignores_empty_box_and_instruction(self):
        import force_close as fc
        self.assertIsNone(fc.first_answer("put the answer within \\boxed{}. So x = 3"))

    def test_first_nonempty_balanced(self):
        import force_close as fc
        t = "within \\boxed{} ... so \\boxed{\\frac{4}{3}} or maybe \\boxed{2}"
        self.assertEqual(fc.first_answer(t), "\\frac{4}{3}")

    def test_unclosed_box_is_not_an_answer(self):
        import force_close as fc
        self.assertIsNone(fc.first_answer("so \\boxed{\\frac{4}{3"))


if __name__ == "__main__":
    unittest.main()
