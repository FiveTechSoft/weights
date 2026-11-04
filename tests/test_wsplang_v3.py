"""test_wsplang_v3.py — Comprehensive Unit Tests for WSP-Lang 0.3 Fast Weight Programming Engine."""

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


class TestWSPLangV3(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create key vector k (attractor direction)
        np.random.seed(42)
        k_raw = np.random.randn(64).astype(np.float32)
        self.k_vec = k_raw / np.linalg.norm(k_raw)
        self.k_file = self.temp_path / "k_attractor.npz"
        np.savez(self.k_file, u=self.k_vec)

        # 2. Create orthogonal test vector k_perp
        rand_v = np.random.randn(64).astype(np.float32)
        k_perp = rand_v - np.dot(rand_v, self.k_vec) * self.k_vec
        self.k_perp = k_perp / np.linalg.norm(k_perp)
        # Verify strict orthogonality
        self.assertAlmostEqual(float(np.dot(self.k_vec, self.k_perp)), 0.0, places=6)

        # 3. Create dummy SafeTensors model shard
        torch.manual_seed(42)
        self.orig_W = torch.randn(64, 128, dtype=torch.float32)
        self.shard_file = self.temp_path / "model.safetensors"
        save_file({
            "model.language_model.layers.14.mlp.down_proj.weight": self.orig_W.clone(),
            "model.language_model.layers.14.mlp.up_proj.weight": torch.randn(128, 64)
        }, str(self.shard_file))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_delta_unlearn_mathematical_precision_and_zero_crosstalk(self):
        """Mathematical verification of Delta Rule (Widrow-Hoff) in WSP-Lang 0.3:
        1. Attractor projection energy along k is completely cancelled.
        2. Orthogonal concepts k_perp suffer EXACT ZERO cross-talk distortion!
        """
        out_model_dir = self.temp_path / "repaired_delta"
        code = f"""experiment test_delta_unlearn_fwp:
theoretical_foundation: Fast Weight Programmers (Schlag, Irie, Schmidhuber 2021)
budgets:
    max_modified_weights = 100000
    allow_write = true
steps:
    step delta_unlearn model.language_model.layers.14.mlp.down_proj.weight key={self.k_file} eta=1.0 target=0.0
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

        plan = wsplang.plan_run(prog, steps)
        self.assertEqual(plan["wsplang_version"], "0.3.0")

        results = wsplang.execute_plan_v3(
            plan, prog,
            model_path=str(self.temp_path),
            out_dir=str(out_model_dir),
            dry_run=False
        )

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["result"]["rule"], "Delta-Rule (Widrow-Hoff)")
        self.assertAlmostEqual(res["result"]["orthogonal_crosstalk_risk"], 0.0)

        # Load repaired weights from disk
        repaired_shard = load_file(str(out_model_dir / "model.safetensors"))
        W_repaired = repaired_shard["model.language_model.layers.14.mlp.down_proj.weight"]

        # Verification 1: Projection energy along k is practically zero (< 1e-5)
        k_t = torch.from_numpy(self.k_vec).float()
        energy_after = torch.norm(torch.matmul(k_t, W_repaired)).item()
        self.assertLess(energy_after, 1e-5, f"Expected near-zero energy along k, got {energy_after}")

        # Verification 2: Zero cross-talk on orthogonal vector k_perp
        k_perp_t = torch.from_numpy(self.k_perp).float()
        orig_perp_proj = torch.matmul(k_perp_t, self.orig_W)
        repaired_perp_proj = torch.matmul(k_perp_t, W_repaired)
        max_ortho_diff = torch.max(torch.abs(orig_perp_proj - repaired_perp_proj)).item()
        self.assertLess(max_ortho_diff, 1e-6, f"Expected 0 cross-talk on orthogonal vector, got max diff {max_ortho_diff}")

    def test_fwp_overlay_export(self):
        """Verify dynamic Fast Weight Overlay generation for runtime inference."""
        overlay_out = self.temp_path / "fwp_overlay.npz"
        code = f"""experiment test_overlay:
steps:
    step fwp_overlay model.language_model.layers.14.mlp.down_proj.weight key={self.k_file} alpha=0.05 out={overlay_out}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v3(plan, prog, dry_run=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        self.assertTrue(overlay_out.exists())

        # Check saved contents
        loaded = np.load(overlay_out)
        self.assertEqual(str(loaded["target_tensor"]), "model.language_model.layers.14.mlp.down_proj.weight")
        self.assertAlmostEqual(float(loaded["alpha"]), 0.05)

    def test_spectral_decay(self):
        """Verify spectral decay operator for runaway attractor dissipation."""
        code = """experiment test_decay:
steps:
    step spectral_decay model.language_model.layers.14.mlp.down_proj.weight decay=0.92
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v3(plan, prog, dry_run=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        self.assertAlmostEqual(results[0]["result"]["decay_factor_lambda"], 0.92)

    def test_assert_orthogonal_preservation(self):
        """Verify contract assertion for orthogonal concept preservation."""
        code = f"""experiment test_assert_ortho:
steps:
    step assert_orthogonal_preservation key={self.k_file}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v3(plan, prog, dry_run=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        self.assertTrue(results[0]["result"]["orthogonal_preservation_verified"])


if __name__ == "__main__":
    unittest.main()
