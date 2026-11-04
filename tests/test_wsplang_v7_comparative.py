"""test_wsplang_v7_comparative.py — Unit tests for WSP-Lang 0.7 Comparative Drift & Transformation Analysis."""

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
import engine.comparative_drift as comparative_drift


class TestWSPLangV7Comparative(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_and_validate_comparative_ops(self):
        """Verifies DSL parsing and validation for all 4 comparative drift operations."""
        code = """experiment test_comp:
    model: MiMo-9B
    budgets:
        allow_write = false
    steps:
        step architectural_drift base=models/base.gguf top=5
        step ssm_dynamics_diff layer=1 base=models/base.gguf
        step spectral_delta blk.1.ffn_down.weight base=models/base.gguf k=3
        step semantic_delta blk.1.ssm_alpha.weight base=models/base.gguf top=4
"""
        prog = wsplang.parse_args_lines(code)
        parsed_items, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")
        steps = [s for s in parsed_items if "op" in s]
        self.assertEqual(len(steps), 4)
        self.assertEqual([s["op"] for s in steps], [
            "architectural_drift", "ssm_dynamics_diff", "spectral_delta", "semantic_delta"
        ])

    def test_feasibility_gate_read_only(self):
        """Ensures comparative ops pass feasibility gate with allow_write=false and GGUF backend."""
        base_dummy = self.temp_path / "dummy_base.gguf"
        base_dummy.write_text("dummy")

        code = f"""experiment test_gate:
    steps:
        step architectural_drift base={base_dummy}
        step spectral_delta blk.1.ffn_down.weight base={base_dummy}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        feas_errors, warnings = wsplang.check_feasibility(prog, steps, model_arg="C:/dummy/model.gguf")
        self.assertEqual(feas_errors, [], f"Comparative ops should pass feasibility on GGUF: {feas_errors}")

    def test_spectral_delta_low_rank_vs_diffuse(self):
        """Synthetic verification of low-rank vs diffuse SVD classification."""
        d_out, d_in = 64, 128
        torch.manual_seed(42)

        # 1. Base weights
        W_b = torch.randn(d_out, d_in)

        # 2. Low-rank update (Rank-1 steering manifold: u @ v^T)
        u = torch.randn(d_out, 1)
        v = torch.randn(d_in, 1)
        delta_lowrank = 2.0 * torch.matmul(u, v.t())
        W_target_lowrank = W_b + delta_lowrank

        # Compute low-rank SVD
        delta_t = delta_lowrank.float()
        U_t, S_t, V_t = torch.svd_lowrank(delta_t, q=5, niter=4)
        s_vals = S_t.cpu().numpy()
        tot_var = float(torch.norm(delta_t) ** 2)
        top1_var = float(s_vals[0] ** 2 / tot_var)
        self.assertGreater(top1_var, 0.85, "Rank-1 update must capture >85% variance in top-1 component")

        # 3. Diffuse full-rank update (Gaussian noise adaptation)
        delta_diffuse = 0.1 * torch.randn(d_out, d_in)
        U_d, S_d, V_d = torch.svd_lowrank(delta_diffuse, q=5, niter=4)
        s_vals_d = S_d.cpu().numpy()
        tot_var_d = float(torch.norm(delta_diffuse) ** 2)
        top1_var_d = float(s_vals_d[0] ** 2 / tot_var_d)
        self.assertLess(top1_var_d, 0.10, "Diffuse update must have <10% top-1 variance share")

    def test_architectural_family_categorization(self):
        """Verifies tensor categorization into architectural families."""
        self.assertEqual(comparative_drift.get_architectural_family("token_embd.weight"), "embeddings_head")
        self.assertEqual(comparative_drift.get_architectural_family("output.weight"), "embeddings_head")
        self.assertEqual(comparative_drift.get_architectural_family("blk.0.attn_norm.weight"), "norm")
        self.assertEqual(comparative_drift.get_architectural_family("blk.1.ssm_a"), "ssm_dynamics")
        self.assertEqual(comparative_drift.get_architectural_family("blk.1.ssm_beta.weight"), "ssm_dynamics")
        self.assertEqual(comparative_drift.get_architectural_family("blk.1.ffn_down.weight"), "mlp_feedforward")
        self.assertEqual(comparative_drift.get_architectural_family("blk.3.attn_q.weight"), "full_attention")
        self.assertEqual(comparative_drift.get_architectural_family("blk.7.attn_v.weight"), "full_attention")
        self.assertEqual(comparative_drift.get_architectural_family("blk.1.attn_qkv.weight"), "recurrent_attention_interface")


if __name__ == "__main__":
    unittest.main()
