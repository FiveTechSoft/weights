"""test_nanowsp2.py — Unit tests for NanoWSP-2 Micro-LLM & WSP-Lang 0.5 primitives."""

import unittest
from pathlib import Path
import torch

from engine.nanowsp2_runtime import NanoWSP2, NanoWSP2Config


class TestNanoWSP2(unittest.TestCase):

    def setUp(self):
        self.model = NanoWSP2(seed=1337)

    def test_nanowsp2_init_and_parameter_count(self):
        """Verify 4-layer architecture parameters match theoretical total."""
        c = self.model.config
        self.assertEqual(c.n_layers, 4)
        self.assertEqual(c.d_model, 128)
        self.assertEqual(c.n_heads, 4)
        self.assertEqual(c.d_head, 32)
        self.assertEqual(c.d_ff, 256)
        self.assertEqual(c.vocab_size, 256)

        total_counted = self.model.count_parameters()
        total_theoretical = c.total_parameters()
        self.assertEqual(total_counted, total_theoretical)
        self.assertEqual(total_counted, 599168)

    def test_nanowsp2_forward_and_trace(self):
        """Verify hierarchical causal forward pass across 4 layers."""
        tokens = self.model.tokenizer.encode("TEST:123", add_bos=True)
        logits, traces = self.model.forward(tokens, return_hidden=True)

        self.assertIsNotNone(traces)
        self.assertEqual(logits.shape, (1, len(tokens), self.model.config.vocab_size))

        for l in range(4):
            self.assertIn(f"layer_{l}_attn_probs", traces)
            self.assertIn(f"layer_{l}_mlp_hidden", traces)
            self.assertEqual(traces[f"layer_{l}_mlp_hidden"].shape[-1], self.model.config.d_ff)

    def test_nanowsp2_memit_multi_task_recall(self):
        """Verify 100% multi-task recall on arithmetic, relational knowledge, and syntax."""
        batch = [
            ("CALC:3+4=", "7"),
            ("CALC:9-5=", "4"),
            ("REL:ESP:CAP=", "M"), ("REL:ESP:CAP=M", "A"), ("REL:ESP:CAP=MA", "D"),
            ("CODE:[{", "}"), ("CODE:[{}", "]")
        ]
        res = self.model.memit_imprint_batch(batch, layer_idx=2, eta=5.0, reg_lambda=0.01)
        self.assertEqual(res["status"], "ok")
        self.assertLess(res["reg_condition"], res["unreg_condition"])

        # Test generation
        out_calc = self.model.generate("CALC:3+4=", max_new_tokens=1)
        self.assertTrue(out_calc.endswith("7"))

        out_rel = self.model.generate("REL:ESP:CAP=", max_new_tokens=3)
        self.assertTrue(out_rel.endswith("MAD"))

        out_code = self.model.generate("CODE:[{", max_new_tokens=2)
        self.assertTrue(out_code.endswith("}]"))

    def test_nanowsp2_leace_erasure(self):
        """Verify LEACE concept erasure achieves >99.99% erasure."""
        torch.manual_seed(42)
        u_concept = torch.randn(self.model.config.d_model)
        u_concept = u_concept / torch.norm(u_concept)

        res = self.model.leace_erase_concept(u_concept, "layers.1.mlp_out.weight", reg_lambda=0.05)
        self.assertEqual(res["status"], "ok")
        self.assertGreater(res["erasure_pct"], 99.99)
        self.assertLess(res["energy_after"], 1e-5)

    def test_nanowsp2_titans_loop_breaker(self):
        """Verify Titans test-time monitor interrupts repetitive runaway loops."""
        prompt = "LOOP:RUN LOOP:RUN LOOP:RUN "
        out_no = self.model.generate(prompt, max_new_tokens=12, enable_titans_monitor=False)
        out_with = self.model.generate(prompt, max_new_tokens=12, enable_titans_monitor=True)

        len_no = len(out_no[len(prompt):])
        len_with = len(out_with[len(prompt):])

        # Without monitor it runs all 12 tokens, with monitor it terminates early
        self.assertEqual(len_no, 12)
        self.assertLess(len_with, 12)


if __name__ == "__main__":
    unittest.main()
