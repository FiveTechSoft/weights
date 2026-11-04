"""test_nanowsp3.py — Unit Tests for NanoWSP-3 6-Layer Thematic Micro-LLM."""

import unittest
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from engine.nanowsp3_runtime import NanoWSP3, NanoWSP3Config


class TestNanoWSP3(unittest.TestCase):

    def setUp(self):
        self.config = NanoWSP3Config(
            vocab_size=256,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_head=32,
            d_ff=512,
            max_seq_len=64
        )
        self.model = NanoWSP3(self.config)

    def test_parameter_count_and_tensor_shapes(self):
        total_params = self.model.count_parameters()
        self.assertEqual(total_params, 3_280_128)
        self.assertEqual(self.model.weights["embedding.weight"].shape, (256, 256))
        self.assertEqual(self.model.weights["layers.0.mlp_in.weight"].shape, (512, 256))
        self.assertEqual(self.model.weights["layers.0.mlp_out.weight"].shape, (256, 512))

    def test_forward_pass_logits(self):
        tokens = self.model.tokenizer.encode("SCI: Z79=", add_bos=True)
        logits, traces = self.model.forward(tokens, return_hidden=True)
        self.assertEqual(logits.shape, (1, len(tokens), 256))
        self.assertIn("embed", traces)
        self.assertIn("layer_0_mlp_hidden", traces)
        self.assertIn("layer_5_after_mlp", traces)

    def test_thematic_imprinting_recall(self):
        domains = [
            {
                "name": "TEST_SCIENCE",
                "facts": [
                    ("SCI: Z79=", "A"),
                    ("SCI: H2O=", "W"),
                ]
            },
            {
                "name": "TEST_GEO",
                "facts": [
                    ("GEO: ESP=", "M"),
                    ("GEO: FRA=", "P"),
                ]
            }
        ]
        stats = self.model.imprint_thematic_corpus(domains, target_layer=3)
        self.assertEqual(stats["total_facts_imprinted"], 4)
        for d in domains:
            for p, exp in d["facts"]:
                hit, pred, prob = self.model.evaluate_recall(p, exp)
                self.assertTrue(hit, f"Failed recall on {p} -> {pred} (exp {exp})")
                self.assertGreater(prob, 0.90)

    def test_checkpoint_roundtrip(self):
        tmp_path = ROOT / "data" / "test_nanowsp3_tmp.safetensors"
        try:
            self.model.save_checkpoint(tmp_path)
            self.assertTrue(tmp_path.exists())
            model2 = NanoWSP3(self.config)
            model2.load_checkpoint(tmp_path)
            self.assertEqual(model2.count_parameters(), self.model.count_parameters())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
