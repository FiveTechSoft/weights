"""wsp.py read-only commands over a tiny synthetic GGUF — end-to-end via in-process cmd_* calls.

The fixture is exactly the pattern of tests/test_gguf_reader.py (GGUFWriter with known F32 tensors),
written into a temp dir. The journal file is redirected (wsp.PATH_JRNL) so tests never touch the
repo's journal/journal.jsonl. No real model, no network, no GPU.
"""
import argparse
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
import numpy as np  # noqa: E402

import wsp  # noqa: E402

TENSORS = {
    "blk.0.attn_q.weight": (4, 3),
    "blk.1.attn_v.weight": (3, 4),
    "token_embd.weight": (5, 4),
}


def _make_fixture(dir_path):
    rng = np.random.RandomState(17)
    path = dir_path / "tiny.gguf"
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_block_count(2)
    for name, (n_out, n_in) in TENSORS.items():
        data = (rng.randn(n_out, n_in) + 0.5).astype(np.float32)
        w.add_tensor(name, np.ascontiguousarray(data))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _buf(fn, a):
    out = io.StringIO()
    with redirect_stdout(out):
        fn(a)
    return out.getvalue()


class CapturedCmd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gguf_path = _make_fixture(self.dir)
        # redirect the journal (read counter, budget report) away from the repo's journal/
        self.journal = self.dir / "journal.jsonl"
        self.old_journal = wsp.PATH_JRNL
        wsp.PATH_JRNL = self.journal
        self.addCleanup(self._restore_journal)

    def _restore_journal(self):
        wsp.PATH_JRNL = self.old_journal
        self.tmp.cleanup()

    def st(self, **kwargs):
        ns = argparse.Namespace(**kwargs)
        return ns

    def test_info_prints_file_and_tensor_count(self):
        out = _buf(wsp.cmd_info, self.st(gguf=str(self.gguf_path)))
        self.assertIn(f"file: {self.gguf_path}", out)
        self.assertIn("tensors: 3", out)
        self.assertIn("quant:", out)
        self.assertIn("first/last:", out)

    def test_tensors_lists_all_and_filters(self):
        out = _buf(wsp.cmd_tensors, self.st(gguf=str(self.gguf_path), pat="*"))
        for name in TENSORS:
            self.assertIn(name, out)
        out = _buf(wsp.cmd_tensors, self.st(gguf=str(self.gguf_path), pat="blk*"))
        self.assertIn("blk.0.attn_q.weight", out)
        self.assertIn("blk.1.attn_v.weight", out)
        self.assertNotIn("token_embd.weight", out)
        # the filter is per-line on tensor names only
        out = _buf(wsp.cmd_tensors, self.st(gguf=str(self.gguf_path), pat="token*"))
        self.assertIn("token_embd.weight", out)
        self.assertNotIn("attn_q", out)

    def test_peek_anatomy_prints_header(self):
        out = _buf(wsp.cmd_peek, self.st(gguf=str(self.gguf_path), tensor="blk.0.attn_q.weight",
                                         row=None, col=None, k=8, json=False))
        self.assertIn("=== WSP PEEK: blk.0.attn_q.weight ===", out)
        self.assertIn("Elementos no nulos:", out)

    def test_peek_json_is_machine_readable(self):
        out = _buf(wsp.cmd_peek, self.st(gguf=str(self.gguf_path), tensor="blk.0.attn_q.weight",
                                         row=None, col=None, k=8, json=True))
        res = json.loads(out)
        self.assertEqual(res["tensor"], "blk.0.attn_q.weight")
        self.assertEqual(res["shape"], [4, 3])
        self.assertEqual(res["dtype"], "F32")
        self.assertIn("architecture", res)
        self.assertIn("nonzero", res)

    def test_peek_known_value_and_row_col_coordinates(self):
        out = _buf(wsp.cmd_peek, self.st(gguf=str(self.gguf_path), tensor="blk.0.attn_q.weight",
                                         row=0, col=0, k=8, json=True))
        res = json.loads(out)
        self.assertEqual(res["row"], 0)
        self.assertEqual(res["col"], 0)
        self.assertIsInstance(res["value"], float)
        self.assertEqual(res["abs"], abs(res["value"]))
        self.assertEqual(res["frac_row"], abs(res["value"]) / (res["row_norm"] + 1e-30))

    def test_peek_unknown_tensor_raises(self):
        with self.assertRaises(KeyError):
            _buf(wsp.cmd_peek, self.st(gguf=str(self.gguf_path), tensor="nope.nope",
                                       row=None, col=None, k=8, json=True))

    def test_sha_prints_and_journals_read(self):
        out = _buf(wsp.cmd_sha, self.st(gguf=str(self.gguf_path)))
        m = re.match(r"sha256 (tiny\.gguf): ([0-9a-f]{64})", out.strip())
        self.assertIsNotNone(m, out)
        self.assertTrue(self.journal.exists(), "sha must journal a read")
        rec = json.loads(self.journal.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(rec["kind"], "read")
        self.assertEqual(rec["detail"], "sha")

    def test_budget_prints_three_counters(self):
        out = _buf(wsp.cmd_budget, self.st())
        self.assertIn("read", out)
        self.assertIn("write", out)
        self.assertIn("fwd", out)

    def test_budget_counts_journal_reads(self):
        # journal two reads, then check the report reflects them
        wsp.journal("read", "a")
        wsp.journal("read", "b")
        out = _buf(wsp.cmd_budget, self.st())
        self.assertRegex(out, r"read\s+2")

    def test_cursorcache_does_not_leak_between_files(self):
        # re-opening the same fixture must return a reader for the same path
        r1 = wsp.reader_of(str(self.gguf_path))
        r2 = wsp.reader_of(str(self.gguf_path))
        self.assertIs(r1, r2)


if __name__ == "__main__":
    unittest.main()


class StatsRowsDeltaCase(unittest.TestCase):
    """cmd_stats/cmd_rows/cmd_delta over the same fixture — with the journal redirected."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = _make_fixture(self.dir)
        # a second fixture deterministically different (base + 0.1 shift in blk.0.attn_q.weight only)
        rng = np.random.RandomState(99)
        self.base_path = Path(self.dir) / "second.gguf"
        w = gguf.GGUFWriter(str(self.base_path), arch="qwen35")
        w.add_block_count(2)
        TENSORS = {
        "blk.0.attn_q.weight": (4, 3),
        "blk.1.attn_v.weight": (3, 4),
        "token_embd.weight": (5, 4),
    }  # noqa: E402
        for name, (n_out, n_in) in TENSORS.items():
            data = rng.randn(n_out, n_in).astype(np.float32) + 0.5
            if name == "blk.0.attn_q.weight":
                data = data + 0.1
            w.add_tensor(name, np.ascontiguousarray(data))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        self.journal = self.dir / "journal.jsonl"
        self.old_journal = wsp.PATH_JRNL
        wsp.PATH_JRNL = self.journal
        self.addCleanup(self._restore_journal)

    def _restore_journal(self):
        wsp.PATH_JRNL = self.old_journal
        self.tmp.cleanup()

    def st(self, **kwargs):
        return argparse.Namespace(**kwargs)

    def test_stats_rows_and_cols(self):
        out = _buf(wsp.cmd_stats, self.st(gguf=str(self.path), tensor="blk.0.attn_q.weight",
                                         ev="rows", k=2))
        self.assertIn("blk.0.attn_q.weight", out)
        self.assertIn("ev=rows", out)
        self.assertIn("top-norm filas:", out)
        out = _buf(wsp.cmd_stats, self.st(gguf=str(self.path), tensor="blk.0.attn_q.weight",
                                         ev="cols", k=2))
        self.assertIn("ev=cols", out)
        self.assertIn("top-norm cols:", out)

    def test_stats_bad_ev_exits(self):
        with self.assertRaises(SystemExit):
            _buf(wsp.cmd_stats, self.st(gguf=str(self.path), tensor="blk.0.attn_q.weight",
                                        ev="sideways", k=2))

    def test_stats_unknown_tensor_raises(self):
        with self.assertRaises(KeyError):
            _buf(wsp.cmd_stats, self.st(gguf=str(self.path), tensor="nope", ev="rows", k=2))

    def test_rows_prints_slice(self):
        out = _buf(wsp.cmd_rows, self.st(gguf=str(self.path), tensor="blk.0.attn_q.weight",
                                         rng="1:3", n=4, full=False))
        self.assertIn("row 1:", out)
        self.assertIn("row 2:", out)
        self.assertNotIn("row 0:", out)
        self.assertNotIn("row 3:", out)

    def test_rows_full_prints_vals(self):
        out = _buf(wsp.cmd_rows, self.st(gguf=str(self.path), tensor="blk.1.attn_v.weight",
                                         rng="0", n=3, full=True))
        self.assertIn("row 0:", out)
        self.assertIn("vals[:3]", out)

    def test_delta_reports_relative_change(self):
        out = _buf(wsp.cmd_delta, self.st(dist=str(self.path), base=str(self.base_path),
                                          tensor="blk.0.attn_q.weight", k=2, d="rows"))
        self.assertIn("delta rows(rel)", out)
        self.assertIn("top-rel-rows:", out)
        self.assertIn("blk.0.attn_q.weight", out)

    def test_delta_shape_mismatch_asserts(self):
        # second fixture has same shape, so build a fixture with a different shape instead
        w = gguf.GGUFWriter(str(self.base_path), arch="qwen35")
        w.add_block_count(2)
        w.add_tensor("blk.0.attn_q.weight", np.zeros((9, 3), dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        with self.assertRaises(AssertionError):
            _buf(wsp.cmd_delta, self.st(dist=str(self.path), base=str(self.base_path),
                                        tensor="blk.0.attn_q.weight", k=2, d="rows"))
