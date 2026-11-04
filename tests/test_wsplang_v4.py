"""test_wsplang_v4.py — Unit Tests for WSP-Lang 0.4 Matrix Perturbation Theory & Leverage Scores."""

import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsplang


class TestWSPLangV4(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create key vector u
        np.random.seed(42)
        u_raw = np.random.randn(64).astype(np.float32)
        self.u_vec = u_raw / np.linalg.norm(u_raw)
        self.u_file = self.temp_path / "u_vector.npz"
        np.savez(self.u_file, u=self.u_vec)

        # 2. Create structured weight matrix with known spectral gap:
        # We construct W with a dominant top singular vector to have a clear eigengap delta
        torch.manual_seed(42)
        u1 = torch.randn(64, 1)
        u1 = u1 / torch.norm(u1)
        v1 = torch.randn(1, 128)
        v1 = v1 / torch.norm(v1)

        # Base noise matrix
        noise = 0.2 * torch.randn(64, 128)
        # Dominant singular value: 5.0 * u1 @ v1
        self.orig_W = 5.0 * (u1 @ v1) + noise

        self.shard_file = self.temp_path / "model.safetensors"
        save_file({
            "model.language_model.layers.14.mlp.down_proj.weight": self.orig_W.clone(),
            "model.language_model.layers.14.mlp.up_proj.weight": torch.randn(128, 64)
        }, str(self.shard_file))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_spectral_audit_and_leverage_identity(self):
        """Verify that spectral_audit extracts singular values, eigengap, and leverage scores.
        RandNLA identity: sum(leverage_scores) == rank q.
        """
        code = f"""experiment test_spectral_audit:
steps:
    step spectral_audit model.language_model.layers.14.mlp.down_proj.weight q=8 vector={self.u_file}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

        plan = wsplang.plan_run(prog, steps)
        self.assertEqual(plan["wsplang_version"], "0.4.0")

        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")

        audit = res["result"]
        self.assertIn("top_singular_values", audit)
        self.assertIn("spectral_gap_delta", audit)
        self.assertGreater(audit["spectral_gap_delta"], 1.0, "Expected clear spectral gap on constructed matrix")

        # Check RandNLA identity: sum of leverage scores equals q (8)
        lev_sum = audit["leverage_distribution"]["sum"]
        self.assertAlmostEqual(lev_sum, 8.0, delta=1e-4)

        # Check Davis-Kahan theoretical bound calculation
        self.assertIn("davis_kahan_bound_sin_theta", audit)
        self.assertIn("operator_norm_bound_delta_W", audit)
        self.assertLess(audit["davis_kahan_bound_sin_theta"], 1.0)

    def test_assert_weyl_bound_satisfied(self):
        """Verify Weyl's Inequality bound: |sigma_i(W - Delta W) - sigma_i(W)| <= ||Delta W||_2."""
        code = f"""experiment test_weyl:
steps:
    step assert_weyl model.language_model.layers.14.mlp.down_proj.weight vector={self.u_file} alpha=0.05 top_cols=0.2 q=5
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["result"]["weyl_satisfied"])
        self.assertGreater(res["result"]["margin"], 0.0)

    def test_assert_davis_kahan_subspace_stability(self):
        """Verify Davis-Kahan sin Theta bound: empirical sin Theta <= ||Delta W||_2 / delta."""
        code = f"""experiment test_davis_kahan:
steps:
    step assert_davis_kahan model.language_model.layers.14.mlp.down_proj.weight vector={self.u_file} alpha=0.05 top_cols=0.2 max_angle_deg=5.0
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["result"]["davis_kahan_satisfied"])
        self.assertLessEqual(res["result"]["empirical_rotation_degrees"], 5.0)

    def test_leverage_guided_column_selection(self):
        """Verify that selection=efficiency in project_out and delta_unlearn prioritizes low-leverage columns."""
        code = f"""experiment test_leverage_selection:
budgets:
    max_modified_weights = 100000
    allow_write = false
steps:
    step project_out model.language_model.layers.14.mlp.down_proj.weight vector={self.u_file} alpha=0.05 top_cols=0.1 selection=efficiency
    step delta_unlearn model.language_model.layers.14.mlp.down_proj.weight key={self.u_file} eta=1.0 target=0.0 top_cols=0.1 selection=efficiency
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 2)

        # Check project_out
        p_res = results[0]["result"]
        self.assertEqual(p_res["column_selection_strategy"], "efficiency_leverage")
        self.assertIsNotNone(p_res["mean_leverage_selected"])

        # Check delta_unlearn
        d_res = results[1]["result"]
        self.assertEqual(d_res["column_selection_strategy"], "efficiency_leverage")
        self.assertIsNotNone(d_res["mean_leverage_selected"])


if __name__ == "__main__":
    unittest.main()
