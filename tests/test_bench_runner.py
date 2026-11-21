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


class TestThinkFallback(unittest.TestCase):
    """Templates llama-server does not split (Phi-4-mini-reasoning writes <think>...</think> inside content, no
    reasoning_content; lab/probes/cc_phi/PLAN.md P1): ask() splits them itself. Templates it does split are untouched."""

    def _ask(self, message):
        import io
        import json as _json
        from engine import bench_runner as br

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=0):
            return R(_json.dumps({"choices": [{"message": message, "finish_reason": "stop"}],
                                  "usage": {"completion_tokens": 1}}).encode())
        old = br.urllib.request.urlopen
        br.urllib.request.urlopen = fake
        try:
            return br.ask("http://x", "q", 10)
        finally:
            br.urllib.request.urlopen = old

    def test_think_tags_in_content_are_split_into_reasoning_and_content(self):
        r = self._ask({"content": "<think>\nLet me count.\nSo 3.\n</think>\nThe answer is \\boxed{3}."})
        self.assertEqual(r["reasoning"], "Let me count.\nSo 3.")
        self.assertEqual(r["content"], "The answer is \\boxed{3}.")

    def test_unclosed_think_is_all_reasoning(self):
        r = self._ask({"content": "<think>\nstill thinking"})
        self.assertEqual(r["reasoning"], "still thinking")
        self.assertEqual(r["content"], "")

    def test_server_split_is_kept_as_is(self):
        r = self._ask({"content": "ans", "reasoning_content": "why"})
        self.assertEqual((r["reasoning"], r["content"]), ("why", "ans"))

    def test_content_without_tags_is_unchanged(self):
        r = self._ask({"content": "just an answer"})
        self.assertEqual((r["reasoning"], r["content"]), ("", "just an answer"))


class TestServedInfo(unittest.TestCase):
    """Each report records what llama-server says it serves (/props model_path) and that file's sha256, so a run
    labelled "merged" that served the base (results §45.11) is visible in the JSON. Control vectors are not in /props."""

    def _props(self, payload, raise_=None):
        import io

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        seen = []

        def fake(req, timeout=0):
            seen.append(req.full_url if hasattr(req, "full_url") else req)
            if raise_:
                raise raise_
            return R(json.dumps(payload).encode())
        return fake, seen

    def test_served_info_hashes_the_model_path_from_props(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            p.write_bytes(b"GGUF" + b"\x01" * 100)
            fake, seen = self._props({"model_path": str(p), "chat_template": "x"})
            with mock.patch.object(br.urllib.request, "urlopen", fake):
                s = br.served_info("http://127.0.0.1:8097/v1/chat/completions")
            self.assertEqual(seen, ["http://127.0.0.1:8097/props"])
            self.assertEqual(s["model_path"], str(p))
            self.assertEqual(s["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())

    def test_served_info_is_none_when_props_fails_or_file_is_missing(self):
        fake, _ = self._props({}, raise_=OSError("no route"))
        with mock.patch.object(br.urllib.request, "urlopen", fake):
            self.assertIsNone(br.served_info("http://127.0.0.1:8097/v1/chat/completions"))
        fake, _ = self._props({"model_path": "Z:/nowhere/m.gguf"})
        with mock.patch.object(br.urllib.request, "urlopen", fake):
            s = br.served_info("http://127.0.0.1:8097/v1/chat/completions")
        self.assertEqual(s, {"model_path": "Z:/nowhere/m.gguf", "sha256": None})

    def test_run_records_served_in_the_report(self):
        fake = {"content": "", "reasoning": "", "finish": "stop", "tokens": 3, "tok_s": 30.0}
        served = {"model_path": "m.gguf", "sha256": "ab" * 32}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "ask", return_value=fake), \
                mock.patch.object(br, "served_info", return_value=served):
            rep = br.run("t", 1, gsm_n=2, math_n=0, parallel=1, out_dir=d, log=lambda s: None)
            self.assertEqual(rep["served"], served)
            self.assertEqual(json.loads((Path(d) / "t.json").read_text(encoding="utf-8"))["served"], served)


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
