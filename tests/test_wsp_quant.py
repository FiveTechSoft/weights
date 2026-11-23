"""Tests for wsp.py's quantization kernels (pure numpy, no model needed)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gguf
import numpy as np
import wsp


class TestQ8QuantBlocks(unittest.TestCase):

    def test_shape_dtype_and_roundtrip(self):
        rng = np.random.default_rng(1)
        X = (rng.standard_normal((8, 32)) * 0.5).astype(np.float32)
        B = wsp._q8_0_quant_blocks(X)
        self.assertEqual(B.shape, (8, 34))
        self.assertEqual(B.dtype, np.uint8)
        D = gguf.dequantize(B, gguf.GGMLQuantizationType.Q8_0)
        self.assertEqual(D.shape, (8, 32))
        rel = float(np.mean(np.abs(D - X)) / np.mean(np.abs(X)))
        self.assertLess(rel, 0.02)


class TestQ6KQuantBlocks(unittest.TestCase):

    def test_shape_dtype_and_roundtrip(self):
        B, D, rel = wsp._verify_q6k_roundtrip()
        self.assertEqual(B.shape, (40, 210))
        self.assertEqual(B.dtype, np.uint8)
        self.assertLess(rel, 0.05)

    def test_zeros_input_produces_zeros(self):
        X = np.zeros((4, 256), dtype=np.float32)
        self.assertEqual(int(wsp._q6k_quant_blocks(X).max()), 0)


class TestQ6KIndexMap(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(wsp._q6k_index_map(0), (0, 0))
        self.assertEqual(wsp._q6k_index_map(31), (0, 31))
        self.assertEqual(wsp._q6k_index_map(32), (1, 0))
        self.assertEqual(wsp._q6k_index_map(255), (7, 31))


class TestMakeQxQuants(unittest.TestCase):

    def test_shapes_and_zero_input(self):
        X = np.zeros((2, 16), dtype=np.float64)
        scale, L = wsp._make_qx_quants(X)
        self.assertEqual(scale.shape, (2,))
        self.assertEqual(L.shape, (2, 16))
        self.assertTrue(np.all(scale == 0.0))
        self.assertTrue(np.all(L == 0))


class FakeTensor:
    def __init__(self, data, tensor_type, ne0, ne1):
        self.data = data
        self.tensor_type = tensor_type
        self.shape = (ne0, ne1)


class TestRequantBlocksBytes(unittest.TestCase):

    def test_only_mapped_blocks_are_rewritten(self):
        rng = np.random.default_rng(0)
        half0 = (rng.standard_normal(256) * 0.3).astype(np.float32)
        half1 = (rng.standard_normal(256) * 0.3).astype(np.float32)
        B0 = wsp._q6k_quant_blocks(half0.reshape(1, 256)).tobytes()
        B1 = wsp._q6k_quant_blocks(half1.reshape(1, 256)).tobytes()
        t = FakeTensor(np.frombuffer(B0 + B1, dtype=np.uint8).reshape(1, 420),
                       gguf.GGMLQuantizationType.Q6_K, 512, 1)
        original = t.data.copy()
        # row 0, block 0 gets new values; block 1 must stay byte-identical
        D = np.zeros((1, 512), dtype=np.float32)
        D[0, :256] = rng.standard_normal(256).astype(np.float32)
        out = wsp._requant_blocks_bytes(t, {0: [0]}, D)
        out = np.frombuffer(out, dtype=np.uint8).reshape(1, 420)
        self.assertFalse(np.array_equal(out[0, :210], original[0, :210]))
        self.assertTrue(np.array_equal(out[0, 210:], original[0, 210:]))


class TestBudgetCheck(unittest.TestCase):

    def test_blocks_over_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            jnl = Path(tmp) / "journal.jsonl"
            with mock.patch.object(wsp, "PATH_JRNL", jnl), \
                 mock.patch.object(wsp, "BUDGET", {"read": 1, "write": 1, "fwd": 1}):
                jnl.write_text(json.dumps({"kind": "read", "detail": "x"}) + "\n")
                with self.assertRaises(SystemExit):
                    wsp._budget_check("read", 1)

    def test_returns_count_under_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            jnl = Path(tmp) / "journal.jsonl"
            with mock.patch.object(wsp, "PATH_JRNL", jnl), \
                 mock.patch.object(wsp, "BUDGET", {"read": 5, "write": 5, "fwd": 5}):
                self.assertEqual(wsp._budget_check("read", 1), 1)
                jnl.write_text(json.dumps({"kind": "read"}) + "\n")
                self.assertEqual(wsp._budget_check("read", 2), 3)


class TestFingerprint(unittest.TestCase):

    def test_sha256_full_matches_hashlib(self):
        data = b"hello-world" * 10
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(data)
            self.assertEqual(wsp._sha256_full(str(path)), hashlib.sha256(data).hexdigest())

    def test_fingerprint_reports_size_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.bin"
            path.write_bytes(b"x" * (5 * 1024 * 1024))
            fp = wsp._fingerprint(str(path))
            self.assertEqual(fp["size"], 5 * 1024 * 1024)
            self.assertEqual(len(fp["sha"]), 32)

    def test_fingerprint_small_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.bin"
            path.write_bytes(b"abc")
            fp = wsp._fingerprint(str(path))
            self.assertEqual(fp["size"], 3)
            self.assertEqual(len(fp["sha"]), 32)


if __name__ == "__main__":
    unittest.main()
