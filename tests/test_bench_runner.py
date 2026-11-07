"""engine.bench_runner: loop detector and saved tail (lab/probes/delta_patterns/h4_plan.md)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.bench_runner as br  # noqa: E402

LOOP = ("so the answer is 64.\n" + "    Wait, I'll check if I should use \\boxed{64}.\n    I'll use \\boxed{\\$64}.\n"
        "    Okay.\n" * 4)
PROGRESS = "\n".join(f"Step {i}: add {i} to the total, which gives {i * (i + 1) // 2}." for i in range(1, 30))


class TestLoopTail(unittest.TestCase):
    def test_indecision_loop_is_detected(self):
        self.assertTrue(br.loop_tail(LOOP))

    def test_progressing_reasoning_is_not(self):
        self.assertFalse(br.loop_tail(PROGRESS))

    def test_short_lines_do_not_count(self):
        self.assertFalse(br.loop_tail("Okay.\nOkay.\nOkay.\nOkay.\nSo x = 3.\n"))


class TestRunTail(unittest.TestCase):
    def test_saves_long_tail_and_loop_flag(self):
        reasoning = "x" * 5000 + "\n" + LOOP
        fake = {"content": "", "reasoning": reasoning, "finish": "length", "tokens": 8192, "tok_s": 30.0}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "ask", return_value=fake):
            rep = br.run("t", 1, gsm_n=2, math_n=0, parallel=1, out_dir=d, log=lambda s: None)
        row = rep["items"][0]
        self.assertEqual(len(row["tail"]), br.TAIL_CHARS)
        self.assertEqual(br.TAIL_CHARS, 2000)
        self.assertTrue(row["loop"])
        self.assertEqual(rep["summary"]["gsm8k"]["loop_tails"], 2)


if __name__ == "__main__":
    unittest.main()
