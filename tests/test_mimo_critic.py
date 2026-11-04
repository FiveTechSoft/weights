"""test_mimo_critic.py — Unit test for MiMo-9B u_critic extraction and causal contract.
"""
import unittest
from pathlib import Path
import numpy as np
import torch

from engine.wsp_causal_contract import (
    InterventionContract,
    EvaluationSuite,
    EvaluationSample,
    CausalVerificationRunner,
    InterventionState
)

ROOT = Path(__file__).resolve().parent.parent


class TestMiMoCritic(unittest.TestCase):

    def test_u_critic_vector_properties(self):
        vec_path = ROOT / "data" / "u_critic_L16.npz"
        self.assertTrue(vec_path.exists(), "u_critic_L16.npz must exist")
        data = np.load(vec_path)
        self.assertIn("u_critic", data)
        u = data["u_critic"]
        self.assertEqual(u.shape, (4096,))
        norm = np.linalg.norm(u)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_causal_contract_rollback_on_tie(self):
        # Contract requires minimum 15% holdout improvement over controls
        suite = EvaluationSuite(
            name="test_suite",
            dev_samples=[EvaluationSample("prompt_1", "type_a")],
            holdout_samples=[EvaluationSample("prompt_2", "type_a")],
            regression_samples=[EvaluationSample("prompt_3", "type_b")]
        )
        contract = InterventionContract(
            target_model_name="MockMiMo",
            target_tensor_name="layers.16.down_proj",
            hypothesis="Mock hypothesis",
            expected_behavior="Mock expected",
            evaluation_suite=suite,
            allowed_weight_fraction=0.05,
            min_holdout_improvement_pct=15.0,
            rollback_on_failure=True
        )

        w = torch.ones((64, 64))
        delta = torch.zeros((64, 64))

        runner = CausalVerificationRunner(contract, model=None, tokenizer=None)
        # eval_fn returns identical accuracy across candidate and controls
        manifest = runner.run_4way_paired_protocol(
            target_tensor=w,
            delta_w_candidate=delta,
            eval_fn=lambda p, s: {"success": True}
        )

        # Since candidate gain vs base/random is 0 (below min 15%), it must rollback
        self.assertEqual(manifest["final_state"], InterventionState.ROLLED_BACK.value)
        self.assertTrue(manifest["rollback_executed"])


if __name__ == "__main__":
    unittest.main()
