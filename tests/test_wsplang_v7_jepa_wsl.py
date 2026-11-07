"""Unit test suite for WSP-Lang 0.7.0 (JEPA & Weight Space Learning integration).

Tests:
  1. Operator registration and version verification (0.7.0).
  2. compute_jepa_steer mathematical correctness and latent alignment.
  3. compute_vicreg_metrics variance floor and off-diagonal covariance decorrelation.
  4. compute_permute_align bipartite neuron matching on permuted weight spaces.
  5. Program validation, plan generation, and simulation execution of v0.7.0 operators.
"""
import unittest
from pathlib import Path
import numpy as np
import torch

from engine.wsplang import (
    __version__,
    OPS_ALLOWED,
    IMPLEMENTED_OPS,
    compute_jepa_steer,
    compute_vicreg_metrics,
    compute_permute_align,
    parse_args_lines,
    validate,
    plan_run,
    execute_plan_v4,
)


class TestWSPLangV7JepaWSL(unittest.TestCase):

    def test_version_and_ops_registration(self):
        self.assertEqual(__version__, "0.7.0")
        expected_ops = {
            "jepa_steer",
            "vicreg_preserve",
            "permute_align",
            "compile_synthetic",
            "assert_behavioral_drift"
        }
        for op in expected_ops:
            self.assertIn(op, OPS_ALLOWED, f"Operator {op} missing from OPS_ALLOWED")
            self.assertIn(op, IMPLEMENTED_OPS, f"Operator {op} missing from IMPLEMENTED_OPS")

    def test_jepa_steer_math(self):
        torch.manual_seed(42)
        d_out, d_in, M = 64, 128, 8
        W0 = torch.randn(d_out, d_in) * 0.1
        H_in = torch.randn(d_in, M)
        target_latent = torch.randn(d_out, 1)

        # Baseline alignment
        H_out_0 = torch.matmul(W0, H_in)
        cos_0 = float(torch.cosine_similarity(H_out_0, target_latent.expand(-1, M), dim=0).mean().item())

        # JEPA latent steer solve
        delta, cos_steered, delta_norm = compute_jepa_steer(
            W0, H_in, target_latent, reg_lambda=0.01, bg_weight=0.0
        )

        self.assertGreater(cos_steered, cos_0)
        self.assertGreater(cos_steered, 0.95, "Steered alignment should exceed 0.95")
        self.assertGreater(delta_norm, 0.0)

        # Verify with background covariance
        Sigma_bg = torch.eye(d_in) * 0.5
        delta_bg, cos_bg, norm_bg = compute_jepa_steer(
            W0, H_in, target_latent, Sigma_bg=Sigma_bg, reg_lambda=0.1, bg_weight=1.0
        )
        self.assertGreater(cos_bg, cos_0)

    def test_vicreg_metrics(self):
        torch.manual_seed(123)
        # 1. Well-conditioned full-rank matrix
        W_good = torch.randn(64, 64)
        m_good = compute_vicreg_metrics(W_good, var_floor=0.01, max_cov_offdiag=0.10)
        self.assertTrue(m_good["variance_pass"])
        self.assertTrue(m_good["covariance_decorrelation_pass"])
        self.assertTrue(m_good["vicreg_certified"])

        # 2. Rank-1 collapsed matrix (all rows identical)
        v = torch.randn(1, 64)
        W_collapsed = v.expand(64, -1).clone()
        m_coll = compute_vicreg_metrics(W_collapsed, var_floor=0.5, max_cov_offdiag=0.05)
        # Should fail offdiagonal covariance decorrelation (offdiagonal correlation = 1.0)
        self.assertFalse(m_coll["covariance_decorrelation_pass"])
        self.assertFalse(m_coll["vicreg_certified"])

    def test_permute_align(self):
        torch.manual_seed(999)
        d_out, d_in = 32, 16
        W_target = torch.randn(d_out, d_in)

        # Create known permutation of columns
        true_perm = torch.randperm(d_in)
        W_shuffled = W_target[:, true_perm]

        W_aligned, recovered_perm, sim_before, sim_after = compute_permute_align(
            W_shuffled, W_target, mode="columns"
        )

        self.assertGreater(sim_after, sim_before)
        self.assertGreaterEqual(sim_after, 0.95, "Permutation matching should recover high alignment")

    def test_program_validation_and_execution(self):
        code = """experiment test_v7_suite:
model: nanowsp3
budgets:
    unlock_tensors = output.weight, out.weight
steps:
    step vicreg_preserve layers.3.mlp_out.weight var_floor=0.05
    step assert_behavioral_drift max_ppl_increase=1.40
    step compile_synthetic dataset=wspx_recall lambda_base=30.0
"""
        prog = parse_args_lines(code)
        steps, warnings = validate(prog)
        self.assertEqual(len([s for s in steps if "op" in s]), 3)

        plan = plan_run(prog, steps)
        self.assertEqual(len(plan["pasos"]), 3)

        # Execute plan in simulation
        results = execute_plan_v4(plan, prog, dry_run=True)
        self.assertEqual(len(results), 3)
        for r in results:
            if r["op"] == "assert_behavioral_drift":
                # Its own verdict decides the status: a VIOLATION is a failed assertion, never "ok".
                expected = "ok" if r["result"]["behavioral_drift_pass"] else "assertion_failed"
                self.assertEqual(r["status"], expected, r)
            else:
                self.assertEqual(r["status"], "ok", f"Step failed: {r}")


if __name__ == "__main__":
    unittest.main()
