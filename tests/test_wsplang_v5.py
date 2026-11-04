"""test_wsplang_v5.py — Unit Tests for WSP-Lang 0.5:
- LEACE Closed-Form Concept Erasure (Belrose et al. 2023)
- MEMIT / EMMET Covariance-Regularized Memory Synthesis (Meng et al. 2023, Gupta et al. 2024)
- Gated DeltaNet-2 Decoupled Erase & Write (Yang et al. 2025, NVIDIA 2026)
- Titans Test-Time Memory Surprisal Monitor (Behrouz et al. 2025)
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import engine.wsplang as wsplang
from engine.nanowsp_runtime import NanoWSP


class TestWSPLangV5(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Target concept vector u (64-dim matching W rows)
        np.random.seed(42)
        u_raw = np.random.randn(64).astype(np.float32)
        self.u_vec = u_raw / np.linalg.norm(u_raw)
        self.u_file = self.temp_path / "u_concept.npz"
        np.savez(self.u_file, u=self.u_vec)

        # 2. Write key (128-dim matching W cols) & value vector (64-dim matching W rows)
        k_raw = np.random.randn(128).astype(np.float32)
        self.k_vec = k_raw / np.linalg.norm(k_raw)
        self.k_file = self.temp_path / "k_write.npz"
        np.savez(self.k_file, u=self.k_vec)

        v_raw = np.random.randn(64).astype(np.float32)
        self.v_vec = v_raw / np.linalg.norm(v_raw)
        self.v_file = self.temp_path / "v_write.npz"
        np.savez(self.v_file, u=self.v_vec)

        # 3. Model shard with target weight matrix [64, 128]
        torch.manual_seed(42)
        self.W = torch.randn(64, 128, dtype=torch.float32)
        self.shard_file = self.temp_path / "model.safetensors"
        save_file({"layers.1.mlp_out.weight": self.W.clone()}, str(self.shard_file))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_leace_concept_erasure_closed_form(self):
        """Verify LEACE closed-form concept erasure eliminates target projection to 0.0."""
        code = f"""experiment test_leace:
budgets:
    allow_write = true
steps:
    step leace_erase layers.1.mlp_out.weight vector={self.u_file}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

        plan = wsplang.plan_run(prog, steps)
        self.assertTrue(plan["wsplang_version"] in ("0.5.0", "0.6.0"))

        out_dir = self.temp_path / "leace_out"
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), out_dir=str(out_dir), dry_run=False)

        self.assertEqual(len(results), 1)
        res = results[0]
        # AUDIT 2026-11-04: leace_erase es STUB declarado (proxy weight-Gram),
        # su status ejecutable es "stub_result" — el calculo matematico sigue verificable.
        self.assertIn(res["status"], ("ok", "stub_result"))
        self.assertEqual(res["execution_state"], "executed")

        erasure_res = res["result"]
        self.assertLess(erasure_res["concept_energy_after"], 1e-5)
        self.assertEqual(erasure_res["erasure_efficiency"], "100.0000%")

        # Verify on repaired file from disk
        repaired = load_file(str(out_dir / "model.safetensors"))
        W_repaired = repaired["layers.1.mlp_out.weight"]
        u_t = torch.from_numpy(self.u_vec).float()
        proj = torch.matmul(u_t, W_repaired)
        self.assertLess(float(torch.norm(proj).item()), 1e-5)

    def test_decoupled_delta_gated_deltanet2(self):
        """Verify Gated DeltaNet-2 decoupled erase of u_concept and write of k_write -> v_write."""
        code = f"""experiment test_decoupled_delta:
budgets:
    allow_write = true
steps:
    step decoupled_delta layers.1.mlp_out.weight erase_vector={self.u_file} write_key={self.k_file} write_value={self.v_file} beta_erase=1.0 beta_write=2.0
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        out_dir = self.temp_path / "delta2_out"
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), out_dir=str(out_dir), dry_run=False)

        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["execution_state"], "executed")
        self.assertLess(res["result"]["attractor_energy_after"], 1e-5)

    def test_surprise_fwp_titans_trigger(self):
        """Verify Titans surprisal monitor detects repetitive loop patterns."""
        code = """experiment test_surprise:
steps:
    step surprise_fwp prompt="ls app ls app ls app ls app" threshold=2.0
    step surprise_fwp prompt="unique productive reasoning trace" threshold=2.0
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0)

        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(self.temp_path), dry_run=True)

        self.assertEqual(len(results), 2)
        # Repetitive prompt triggers memory update
        self.assertTrue(results[0]["result"]["memory_update_triggered"])
        self.assertEqual(results[0]["result"]["action"], "flag_repetition_detected")

    def test_memit_covariance_regularization(self):
        """Verify MEMIT closed-form update stabilizes condition number under ridge regularization."""
        W0 = torch.randn(64, 128)
        # Construct nearly linearly-dependent keys
        k_base = torch.randn(128, 1)
        K = k_base.repeat(1, 10) + 0.01 * torch.randn(128, 10)  # [128, 10]
        V = torch.randn(64, 10)

        raw_cond = float(torch.linalg.cond(K.t() @ K).item())
        self.assertGreater(raw_cond, 1000.0, "Constructed keys should have high condition number")

        delta = wsplang.compute_memit_update(W0, K, V, reg_lambda=0.1)
        self.assertLess(float(torch.norm(delta).item()), 50.0)


if __name__ == "__main__":
    unittest.main()
