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


class TestLoraField(unittest.TestCase):
    def test_ask_sends_per_request_lora_scales(self):
        """lora=[(id, scale)] -> the llama-server per-request field "lora": [{"id", "scale"}]; absent when None."""
        import io
        import json as _json
        from engine import bench_runner as br
        sent = []

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=0):
            sent.append(_json.loads(req.data))
            return R(_json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                  "usage": {"completion_tokens": 1}}).encode())
        old = br.urllib.request.urlopen
        br.urllib.request.urlopen = fake
        try:
            br.ask("http://x", "q", 10, lora=[(0, 0.5)])
            br.ask("http://x", "q", 10)
        finally:
            br.urllib.request.urlopen = old
        self.assertEqual(sent[0]["lora"], [{"id": 0, "scale": 0.5}])
        self.assertNotIn("lora", sent[1])


class TestExtraInstruction(unittest.TestCase):
    """extra: text appended after SUFFIX in the user message (prompt-control arm, cc/PLAN.md "Prompt control"); the
    default sends exactly question + SUFFIX, so every earlier run is unchanged."""

    def _sent(self, **kw):
        import io
        sent = []

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=0):
            sent.append(json.loads(req.data))
            return R(json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                 "usage": {"completion_tokens": 1}}).encode())
        old = br.urllib.request.urlopen
        br.urllib.request.urlopen = fake
        try:
            br.ask("http://x", "q", 10, **kw)
        finally:
            br.urllib.request.urlopen = old
        return sent[0]["messages"][0]["content"]

    def test_extra_is_appended_after_the_suffix_and_absent_by_default(self):
        self.assertEqual(self._sent(), "q" + br.SUFFIX)
        self.assertEqual(self._sent(extra=" Stop now."), "q" + br.SUFFIX + " Stop now.")

    def test_run_passes_extra_to_every_request_and_records_it(self):
        fake = {"content": "", "reasoning": "", "finish": "stop", "tokens": 3, "tok_s": 30.0}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "ask", return_value=fake) as m:
            rep = br.run("t", 1, gsm_n=2, math_n=0, parallel=1, out_dir=d, log=lambda s: None, extra=" Stop now.")
        self.assertEqual({c.kwargs.get("extra") for c in m.call_args_list}, {" Stop now."})
        self.assertEqual(rep["extra"], " Stop now.")
