"""demos/math_transplant/demo.py: side-by-side rendering of base vs base + transplanted adapter."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demos" / "math_transplant"))


class TestRender(unittest.TestCase):
    def test_table_shows_both_sides_and_the_score(self):
        import demo
        rows = [{"i": 106, "answer": "12", "base": {"closed": False, "ok": False, "tokens": 2048, "pred": ""},
                 "lora": {"closed": True, "ok": True, "tokens": 330, "pred": "12"}},
                {"i": 185, "answer": "3", "base": {"closed": True, "ok": True, "tokens": 900, "pred": "3"},
                 "lora": {"closed": True, "ok": True, "tokens": 292, "pred": "3"}}]
        t = demo.render(rows)
        self.assertIn("106", t)
        self.assertIn("cut at 2048", t)
        self.assertIn("330 tok", t)
        self.assertIn("base 1/2", t)
        self.assertIn("transplant 2/2", t)


if __name__ == "__main__":
    unittest.main()
