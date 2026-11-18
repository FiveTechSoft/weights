"""engine/ternary_write.py and the WSP-Lang ops ternarize / imprint_ternary (lab/probes/ternary_nanowsp/PLAN.md).
Written before the code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import ternary_write as tw  # noqa: E402
from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig  # noqa: E402
import engine.wsplang as wsplang  # noqa: E402


class RowSolveTest(unittest.TestCase):
    def test_recovers_an_exactly_ternary_matrix(self):
        rng = np.random.default_rng(0)
        T = rng.integers(-1, 2, size=(6, 40)).astype(np.int8)
        s = rng.uniform(0.5, 2.0, size=6).astype(np.float32)
        W = (T * s[:, None]).astype(np.float32)
        codes, scales, resid = tw.solve_rows(W, iters=3)
        np.testing.assert_array_equal(codes, T)
        np.testing.assert_allclose(scales, s, rtol=1e-5)
        self.assertLess(float(np.max(resid)), 1e-6)

    def test_alternation_never_increases_the_residual(self):
        W = (np.random.default_rng(1).normal(size=(8, 64)) * 0.05).astype(np.float32)
        hist = tw.solve_rows(W, iters=4, history=True)[3]
        for r in range(len(hist) - 1):
            self.assertLessEqual(float(np.sum(hist[r + 1])), float(np.sum(hist[r])) + 1e-9)
        self.assertTrue(set(np.unique(tw.solve_rows(W)[0])) <= {-1, 0, 1})

    def test_dequant_is_scale_times_codes(self):
        W = (np.random.default_rng(2).normal(size=(4, 16)) * 0.1).astype(np.float32)
        codes, scales, _ = tw.solve_rows(W)
        np.testing.assert_allclose(tw.deq(codes, scales), codes.astype(np.float32) * scales[:, None])


class TernarizeCheckpointTest(unittest.TestCase):
    def test_writes_codes_scales_float_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m" / "model.safetensors"
            NanoWSP(seed=42).save_checkpoint(src)
            out = Path(d) / "tern"
            man = tw.ternarize_checkpoint(src, out, rule="absmean")
            self.assertTrue((out / "model.safetensors").exists())
            self.assertTrue((out / "ternary.safetensors").exists())
            self.assertTrue((out / "manifest.json").exists())
            names = set(man["tensors"])
            self.assertIn("layers.1.mlp_out.weight", names)
            self.assertNotIn("token_embd.weight", names)       # embeddings stay float
            self.assertNotIn("output.weight", names)
            from safetensors.torch import load_file
            t = load_file(str(out / "ternary.safetensors"))
            m = load_file(str(out / "model.safetensors"))
            c = t["layers.1.mlp_out.weight.codes"]; s = t["layers.1.mlp_out.weight.scales"]
            self.assertEqual(c.dtype, torch.int8)
            torch.testing.assert_close(m["layers.1.mlp_out.weight"], c.float() * s[:, None])
            torch.testing.assert_close(m["token_embd.weight"], load_file(str(src))["token_embd.weight"])
            for v in man["tensors"].values():
                self.assertTrue(0.0 < v["density"] <= 1.0)
                self.assertGreater(v["rel_err"], 0.0)
            with self.assertRaises(FileExistsError):
                tw.ternarize_checkpoint(src, out, rule="absmean")   # never over an existing out

    def test_rejects_out_inside_input(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m" / "model.safetensors"
            NanoWSP(seed=42).save_checkpoint(src)
            with self.assertRaises(ValueError):
                tw.ternarize_checkpoint(src, src.parent / "tern", rule="absmean")


class ImprintTernaryTest(unittest.TestCase):
    def test_recalls_written_pairs_and_reports(self):
        pairs = [("A->", "1"), ("B->", "2"), ("C->", "3"), ("D->", "4")]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m" / "model.safetensors"
            NanoWSP(seed=42).save_checkpoint(src)
            tern = Path(d) / "tern"
            tw.ternarize_checkpoint(src, tern, rule="absmean")
            out = Path(d) / "written"
            rep = tw.imprint_ternary(tern, pairs, layer=1, eta=4.0, rounds=3, out=out)
            self.assertEqual(rep["rounds"], 3)
            self.assertEqual(len(rep["key_residual"]), 4)                     # round 0 + 3 refinements
            self.assertLessEqual(rep["key_residual"][-1], rep["key_residual"][0] + 1e-9)
            self.assertGreaterEqual(rep["recall"], 0.75)
            self.assertTrue(0.0 < rep["density"] <= 1.0)
            model = NanoWSP(seed=0); model.load_checkpoint(out / "model.safetensors")
            self.assertGreaterEqual(model.evaluate_recall(pairs), 0.75)       # the float materialisation recalls
            from safetensors.torch import load_file
            t = load_file(str(out / "ternary.safetensors"))
            m = load_file(str(out / "model.safetensors"))
            c = t["layers.1.mlp_out.weight.codes"]; s = t["layers.1.mlp_out.weight.scales"]
            torch.testing.assert_close(m["layers.1.mlp_out.weight"], c.float() * s[:, None])  # exactly ternary

    def test_rounds_zero_is_one_projection(self):
        pairs = [("A->", "1"), ("B->", "2")]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m" / "model.safetensors"
            NanoWSP(seed=42).save_checkpoint(src)
            tern = Path(d) / "tern"
            tw.ternarize_checkpoint(src, tern, rule="absmean")
            rep = tw.imprint_ternary(tern, pairs, layer=1, eta=4.0, rounds=0, out=Path(d) / "w0")
            self.assertEqual(rep["rounds"], 0)
            self.assertEqual(len(rep["key_residual"]), 1)


class WsplangOpsTest(unittest.TestCase):
    def test_registered_as_write_ops(self):
        for op in ("ternarize", "imprint_ternary"):
            self.assertIn(op, wsplang.OPS_ALLOWED)
            self.assertIn(op, wsplang.IMPLEMENTED_OPS)
            self.assertNotIn(op, wsplang.READ_ONLY_OPS)

    def test_spec_table_lists_them_as_writes(self):
        spec = (ROOT / "docs" / "wsplang_specification.md").read_text(encoding="utf-8")
        table = spec.split("<!-- ops:begin -->")[1].split("<!-- ops:end -->")[0]
        rows = {r.split("|")[1].strip().strip("`"): r.split("|")[2].strip() for r in table.splitlines() if r.startswith("| `")}
        self.assertEqual(rows.get("ternarize"), "writes weights")
        self.assertEqual(rows.get("imprint_ternary"), "writes weights")


if __name__ == "__main__":
    unittest.main()
