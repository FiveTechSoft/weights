"""Tests for the ws_runner adapter logic without the external weightspace dep.

The fake `weightspace` module lets us pin down the adapter's contract:
  - cmd_info/cmd_edit/cmd_apply_patch print readable JSON and journal one line
  - _steps_from_transcript parses assistant+tool_calls lines and rejects empties
  - main() surfaces the documented ImportError path with exit code 2
"""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import ws_runner




class FakeR:
    def __repr__(self):
        return "<FakeR>"


class FakeW:
    WRITES = []

    def __getitem__(self, idx):
        return 0.0

    def __setitem__(self, idx, value):
        FakeW.WRITES.append((idx, value))

    def __repr__(self):
        return "<FakeW>"


class FakeTensor:
    def __init__(self, name, changed=True, l2=0.5, rel=0.25):
        self.name = name
        self.changed = changed
        self.l2_delta = l2
        self.rel_delta = rel


class FakeDiff:
    def __init__(self, tensors):
        self.tensors = tensors


class FakeWeightSpace:
    def __init__(self, path):
        self.path = path
        self.R = FakeR()
        self.W = FakeW()
        self._names = ["blk.0.attn_q.weight", "blk.0.attn_kv.weight"]

    def checkpoint(self):
        pass

    def restore(self):
        pass

    def revert(self):
        pass

    @classmethod
    def from_file(cls, path):
        return cls(path)

    def names(self, pattern):
        return self._names

    def shape(self, name):
        return (4, 4)

    def dtype(self, name):
        return "F32"

    def numel(self, name=None):
        return 16 if name else 32

    def __getitem__(self, name):
        return self._W

    def diff(self):
        return FakeDiff([FakeTensor("blk.0.attn_q.weight")])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("weights")

    def zero_(self):
        pass

    def commit(self):
        pass

    def apply_patch(self, path):
        pass


def _install_fake_weightspace():
    weightspace = mock.MagicMock()
    weightspace.WeightSpace = FakeWeightSpace
    weightspace.describe = lambda n, w, d: f"describe:{n}"
    return weightspace


class TestMainWithoutDependency(unittest.TestCase):

    def test_missing_weightspace_returns_2_with_documented_error(self):
        with mock.patch.dict(sys.modules, {"weightspace": None}):
            with mock.patch.object(sys, "argv", ["ws_runner", "info", "dummy.npz"]):
                with mock.patch.object(sys.stdout, "write"):
                    code = ws_runner.main()
        self.assertEqual(code, 2)


class TestJournal(unittest.TestCase):

    def test_journal_writes_one_compact_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws_runner.JRN = Path(tmp) / "ws_runner.jsonl"
            ws_runner.journal("info", {"file": "dummy.npz", "n": 2})
            lines = Path(tmp).joinpath("ws_runner.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["kind"], "info")
        self.assertEqual(rec["detail"]["n"], 2)


class TestCmdsWithFakeWeightspace(unittest.TestCase):

    def setUp(self):
        fake = _install_fake_weightspace()
        self.sys_modules = {"weightspace": fake}
        self._patched = mock.patch.dict(sys.modules, self.sys_modules)
        self._patched.start()
        self._journal_tmp = tempfile.TemporaryDirectory()
        ws_runner.JRN = Path(self._journal_tmp.name) / "ws_runner.jsonl"

    def tearDown(self):
        self._journal_tmp.cleanup()

    def test_cmd_info(self):
        a = Namespace(file="dummy.npz", pattern=None, describe=False)
        ws_runner.cmd_info(a)

    def test_cmd_edit_runs_program_and_saves(self):
        FakeW.WRITES.clear()
        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "prog.py"
            program.write_text("W[0, 0] = 1.0\n")
            out = Path(tmp) / "out.npz"
            a = Namespace(file="dummy.npz", program=str(program), out=str(out))
            ws_runner.cmd_edit(a)
            self.assertIn(((0, 0), 1.0), FakeW.WRITES)
            self.assertEqual(Path(out).read_text(), "weights")

    def test_cmd_apply_patch_saves_and_journals(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.npz"
            a = Namespace(file="dummy.npz", patch="patch.npz", out=str(out))
            ws_runner.cmd_apply_patch(a)
            self.assertEqual(Path(out).read_text(), "weights")


class TestStepsFromTranscript(unittest.TestCase):

    class FakeLLMResponse:
        def __init__(self, text, tool_calls):
            self.text = text
            self.tool_calls = tool_calls

    class FakeToolCall:
        def __init__(self, id, name, arguments=None):
            self.id = id
            self.name = name
            self.arguments = arguments or {}

    class FakeScriptedLLM:
        def __init__(self, steps):
            self.steps = steps

    def _transcript(self, tmp, lines):
        path = Path(tmp) / "transcript.jsonl"
        path.write_text("\n".join(lines))
        return path

    def _stub(self):
        llm_module = mock.MagicMock()
        llm_module.LLMResponse = self.FakeLLMResponse
        llm_module.base = mock.MagicMock()
        llm_module.base.ToolCall = self.FakeToolCall
        llm_module.mock = mock.MagicMock()
        llm_module.mock.ScriptedLLM = self.FakeScriptedLLM
        return llm_module

    def test_parses_assistant_tool_calls_and_ignores_other_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._transcript(tmp, [
                '{"role": "assistant", "content": "working", "tool_calls": [{"id": "1", "name": "cmd", "arguments": {"k": "v"}}]}',
                'not json',
                '{"role": "user", "content": "ignorar"}',
            ])
            llm_mod = self._stub()
            with mock.patch.dict(sys.modules, {"weightspace": mock.MagicMock(), "weightspace.llm": llm_mod,
                                              "weightspace.llm.base": llm_mod.base,
                                              "weightspace.llm.mock": llm_mod.mock}):
                llm = ws_runner._steps_from_transcript(str(path))
            self.assertEqual(len(llm.steps), 1)
            self.assertEqual(llm.steps[0].tool_calls[0].name, "cmd")

    def test_empty_transcript_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._transcript(tmp, ['{"role": "user", "content": "solo"}'])
            llm_mod = self._stub()
            with mock.patch.dict(sys.modules, {"weightspace": mock.MagicMock(), "weightspace.llm": llm_mod,
                                              "weightspace.llm.base": llm_mod.base,
                                              "weightspace.llm.mock": llm_mod.mock}):
                with self.assertRaises(SystemExit):
                    ws_runner._steps_from_transcript(str(path))


if __name__ == "__main__":
    unittest.main()
