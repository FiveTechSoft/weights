"""lab/probes/close_probe/he_eval.py: HumanEval program assembly and sandboxed run."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
from he_eval import build_program, run_program  # noqa: E402

PROMPT = 'def add(a, b):\n    """sum"""\n'
TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


class TestHeEval(unittest.TestCase):
    def test_full_function_in_last_block_passes(self):
        content = ("x\n```python\ndef add(a, b):\n    return a - b\n```\nbetter:\n"
                   "```python\ndef add(a, b):\n    return a + b\n```")
        self.assertTrue(run_program(build_program(PROMPT, content, TEST, "add")))

    def test_body_only_gets_prompt_prepended_and_wrong_fails(self):
        self.assertTrue(run_program(build_program(PROMPT, "```python\n    return a + b\n```", TEST, "add")))
        self.assertFalse(run_program(build_program(PROMPT, "```python\n    return a * b\n```", TEST, "add")))

    def test_no_code_or_hang_fails(self):
        self.assertFalse(run_program(build_program(PROMPT, "no code here", TEST, "add")))
        self.assertFalse(run_program("while True: pass", timeout=2))


if __name__ == "__main__":
    unittest.main()
