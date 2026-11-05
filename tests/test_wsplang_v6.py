"""test_wsplang_v6.py — Unit Tests for WSP-Lang 0.6: Cross-Model Procrustes Alignment & Latent Vector Translation."""

import unittest
import tempfile
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import engine.wsplang as wsplang


class TestWSPLangV6(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create synthetic source (64-dim) and target (32-dim) embedding matrices for 100 anchor words
        torch.manual_seed(42)
        K = 100
        d_src = 64
        d_tgt = 32

        # Latent shared semantic factors Z in R^{K x 16}
        Z = torch.randn(K, 16)
        W_s = torch.randn(16, d_src)
        W_t = torch.randn(16, d_tgt)

        E_src = torch.matmul(Z, W_s) + 0.1 * torch.randn(K, d_src)
        E_tgt = torch.matmul(Z, W_t) + 0.1 * torch.randn(K, d_tgt)

        self.src_file = self.temp_path / "E_src.npz"
        self.tgt_file = self.temp_path / "E_tgt.npz"
        np.savez(self.src_file, embeddings=E_src.numpy())
        np.savez(self.tgt_file, embeddings=E_tgt.numpy())

        # Vector in source space
        u_src = torch.randn(d_src)
        u_src = u_src / torch.norm(u_src)
        self.vec_file = self.temp_path / "u_source.npz"
        np.savez(self.vec_file, u=u_src.numpy())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rosetta_align_procrustes(self):
        """Verify rosetta_align computes semi-orthogonal Procrustes matrix T."""
        out_rosetta = self.temp_path / "rosetta.npz"
        code = f"""experiment test_rosetta:
budgets:
    allow_write = true
steps:
    step rosetta_align source={self.src_file} target={self.tgt_file} out={out_rosetta}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

        plan = wsplang.plan_run(prog, steps)
        self.assertTrue(plan["wsplang_version"] in ("0.6.0", "0.7.0"))

        results = wsplang.execute_plan_v4(plan, prog, dry_run=False)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")

        result_data = res["result"]
        self.assertEqual(result_data["d_source"], 64)
        self.assertEqual(result_data["d_target"], 32)
        self.assertGreater(result_data["mean_anchor_cosine"], 0.70)
        self.assertLess(result_data["semi_orthogonality_error"], 1e-4)

        # Check saved file
        self.assertTrue(out_rosetta.exists())
        data = np.load(out_rosetta)
        T = data["T"]
        self.assertEqual(T.shape, (32, 64))

    def test_cross_project_vector(self):
        """Verify cross_project translates direction vector from source to target dimension."""
        out_rosetta = self.temp_path / "rosetta.npz"
        out_translated = self.temp_path / "u_translated.npz"

        code = f"""experiment test_cross_proj:
budgets:
    allow_write = true
steps:
    step rosetta_align source={self.src_file} target={self.tgt_file} out={out_rosetta}
    step cross_project vector={self.vec_file} rosetta={out_rosetta} out={out_translated}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v4(plan, prog, dry_run=False)
        self.assertEqual(len(results), 2)

        res_proj = results[1]
        self.assertEqual(res_proj["status"], "ok")
        self.assertEqual(res_proj["result"]["source_dim"], 64)
        self.assertEqual(res_proj["result"]["target_dim"], 32)

        # Check translated file on disk
        self.assertTrue(out_translated.exists())
        data = np.load(out_translated)
        u_t = data["u"]
        self.assertEqual(u_t.shape, (32,))
        norm = np.linalg.norm(u_t)
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
