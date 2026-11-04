"""test_nanowsp.py — Unit Tests for NanoWSP Micro-Transformer and WSPLang Integration."""

import unittest
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig, ASCIITokenizer
import engine.wsplang as wsplang


class TestNanoWSP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.model = NanoWSP(seed=42)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nanowsp_initialization_and_determinism(self):
        """Verify model weights initialization, shapes, and deterministic seed behavior."""
        m1 = NanoWSP(seed=42)
        m2 = NanoWSP(seed=42)
        m3 = NanoWSP(seed=999)

        # Check total parameter count (~80.7k)
        total_p = m1.count_parameters()
        self.assertEqual(total_p, 80704)

        # Check key tensor dimensions
        self.assertEqual(m1.weights["token_embd.weight"].shape, torch.Size([100, 64]))
        self.assertEqual(m1.weights["layers.0.attn_q.weight"].shape, torch.Size([64, 64]))
        self.assertEqual(m1.weights["layers.1.mlp_in.weight"].shape, torch.Size([128, 64]))
        self.assertEqual(m1.weights["layers.1.mlp_out.weight"].shape, torch.Size([64, 128]))

        # Check determinism: seed 42 produces identical weights
        w1 = m1.weights["layers.0.attn_q.weight"]
        w2 = m2.weights["layers.0.attn_q.weight"]
        self.assertTrue(torch.allclose(w1, w2))

        # Check variation across different seeds
        w3 = m3.weights["layers.0.attn_q.weight"]
        self.assertFalse(torch.allclose(w1, w3))

    def test_nanowsp_forward_pass_and_logits_shape(self):
        """Verify causal autoregressive forward pass, trace extraction, and logits output shape."""
        prompt = "test"
        token_ids = self.model.tokenizer.encode(prompt, add_bos=True)
        self.assertEqual(token_ids[0], ASCIITokenizer.BOS_TOKEN)
        self.assertEqual(len(token_ids), 5)  # BOS + 4 chars

        logits, traces = self.model.forward(token_ids, return_hidden=True)
        self.assertIsNotNone(traces)
        # Logits shape: [1, T, vocab_size] = [1, 5, 100]
        self.assertEqual(logits.shape, torch.Size([1, 5, 100]))

        # Verify trace tensors exist for both layers
        self.assertIn("layer_0_attn_probs", traces)
        self.assertIn("layer_1_mlp_hidden", traces)
        self.assertEqual(traces["layer_1_mlp_hidden"].shape, torch.Size([1, 5, 128]))

    def test_single_associative_imprinting(self):
        """Verify factual memory imprinting in weight space via outer product without SGD."""
        prompt = "2+2="
        target = "4"

        # Baseline generation does not output target
        baseline_out = self.model.generate(prompt, max_new_tokens=1)
        self.assertNotEqual(baseline_out, f"{prompt}{target}")

        # Imprint association into Layer 1 MLP Out
        res = self.model.imprint_associative_rule(prompt, target, layer_idx=1, eta=4.0)
        self.assertEqual(res["imprint_status"], "ok")
        self.assertGreater(res["delta_frobenius"], 0.0)

        # Immediate recall verification
        post_out = self.model.generate(prompt, max_new_tokens=1)
        self.assertEqual(post_out, f"{prompt}{target}")

    def test_multi_associative_imprinting_with_moore_penrose(self):
        """Verify batch associative memory synthesis via Moore-Penrose pseudo-inverse decorelation."""
        pairs = [("A->", "1"), ("B->", "2"), ("C->", "3"), ("D->", "4")]

        # Imprint all 4 pairs simultaneously with cross-talk cancellation
        res = self.model.imprint_associative_batch(pairs, layer_idx=1, eta=5.0)
        self.assertEqual(res["batch_status"], "ok")
        self.assertEqual(res["pairs_count"], 4)
        self.assertLess(res["condition_number"], 50.0)

        # Verify 100% recall across all 4 associations
        acc = self.model.evaluate_recall(pairs)
        self.assertEqual(acc, 1.0)

    def test_unlearn_attractor_and_orthogonal_preservation(self):
        """Verify Delta-Rule unlearning suppresses projection energy while preserving orthogonal directions."""
        prompt = "loop_attractor"
        res = self.model.unlearn_attractor(prompt, layer_idx=1, eta=1.0)
        self.assertEqual(res["unlearn_status"], "ok")
        self.assertLess(res["energy_after"], 1e-4)
        self.assertGreater(res["energy_before"], res["energy_after"])

    def test_wsplang_plan_execution_on_nanowsp_safetensors(self):
        """End-to-end integration: export NanoWSP to SafeTensors, compile WSPLang program, and execute."""
        # 1. Save NanoWSP checkpoint to temporary directory
        model_dir = self.temp_path / "nanowsp_model"
        model_dir.mkdir()
        shard_path = model_dir / "model.safetensors"
        self.model.save_checkpoint(shard_path)

        # 2. Extract and save a key vector (dimension 64 for Layer 1 MLP Out rows)
        np.random.seed(42)
        key_vec = np.random.randn(64).astype(np.float32)
        key_vec = key_vec / np.linalg.norm(key_vec)
        key_file = self.temp_path / "nanowsp_key.npz"
        np.savez(key_file, u=key_vec)

        # 3. Create a customized WSPLang program pointing to our temp artifacts
        wspl_code = f"""experiment test_nanowsp_wsplang:
budgets:
    max_modified_weights = 10000
    allow_write = true
steps:
    step spectral_audit layers.1.mlp_out.weight q=8
    step assert_weyl layers.1.mlp_out.weight vector={key_file} alpha=0.05 top_cols=0.25 q=4
    step assert_davis_kahan layers.1.mlp_out.weight vector={key_file} alpha=0.05 top_cols=0.25 max_angle_deg=5.0
    step delta_unlearn layers.1.mlp_out.weight key={key_file} eta=1.0 target=0.0
    step assert_orthogonal_preservation layers.1.mlp_out.weight key={key_file}
"""
        prog = wsplang.parse_args_lines(wspl_code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

        plan = wsplang.plan_run(prog, steps)
        out_dir = self.temp_path / "nanowsp_repaired"
        results = wsplang.execute_plan_v4(plan, prog, model_path=str(model_dir), out_dir=str(out_dir), dry_run=False)

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r["status"], "ok", f"Step {r.get('op')} failed: {r}")

        # Check specific contract outputs
        audit_res = results[0]["result"]
        self.assertAlmostEqual(audit_res["leverage_distribution"]["sum"], 8.0, delta=1e-3)

        weyl_res = results[1]["result"]
        self.assertTrue(weyl_res["weyl_satisfied"])

        dk_res = results[2]["result"]
        self.assertTrue(dk_res["davis_kahan_satisfied"])

        delta_res = results[3]["result"]
        self.assertEqual(delta_res["execution_state"], "executed")
        self.assertEqual(delta_res["orthogonal_crosstalk_risk"], 0.0)

        ortho_res = results[4]["result"]
        self.assertTrue(ortho_res["orthogonal_preservation_verified"])


if __name__ == "__main__":
    unittest.main()
