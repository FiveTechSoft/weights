"""lab/probes/happy/happy.py: the pure parts of the happiness-steering measurement (lexicon score, sign test, split)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "happy"))


class TestHappy(unittest.TestCase):
    def test_lexicon_counts_affect_words_and_exclamations_per_100_words(self):
        import happy
        self.assertEqual(happy.lex1("The kettle boils water."), 0.0)
        self.assertAlmostEqual(happy.lex1("I am so happy and excited!"), 100 * 3 / 6)

    def test_judge_answer_skips_the_think_block(self):
        # MiMo ignores enable_thinking=False and opens with <think>: the verdict is what follows </think>
        import happy
        self.assertEqual(happy.judge_answer("<think>The reply is coherent.</think>\n\nYes"), "yes")
        self.assertEqual(happy.judge_answer("<think></think>3"), "3")
        self.assertEqual(happy.judge_answer("No"), "no")
        self.assertIsNone(happy.judge_answer("<think>The reply is coherent, fluent,"))   # cut inside the think block

    def test_sign_test(self):
        import happy
        self.assertEqual(happy.sign_p([0, 0]), 1.0)
        self.assertAlmostEqual(happy.sign_p([1] * 10), 2 / 1024)
        self.assertEqual(happy.sign_p([1, -1]), 1.0)

    def test_split_is_fixed_and_disjoint(self):
        import happy
        a = happy.prompts()
        b = happy.prompts()
        self.assertEqual(a, b)
        self.assertEqual((len(a["train"]), len(a["dev"]), len(a["test"])), (70, 10, 40))
        self.assertFalse(set(a["train"]) & set(a["test"]) or set(a["dev"]) & set(a["test"]))


if __name__ == "__main__":
    unittest.main()
