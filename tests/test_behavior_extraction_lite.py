"""Synthetic checks for lab/probes/behavior_extraction/lite_text.py. No real model, no decoder, no training."""
import os, sys, tempfile, unittest
import torch
from safetensors.torch import save_file

sys.path.insert(0, "lab/probes/behavior_extraction")
from lite_text import LiteText, prefix_mean_bf16, PFX

HERE = "lab/probes/behavior_extraction"


class Lite(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.E = torch.randn(40, 16, generator=g).to(torch.bfloat16)
        self.d = tempfile.mkdtemp()
        save_file({PFX + "embed_tokens.weight": self.E}, self.d + "/model.safetensors")   # NO decoder tensors in the file
        self.q = LiteText(self.d, tokenizer=False)

    def test_embed_matches_table_without_decoder(self):
        ids = torch.tensor([[3, 3, 7], [0, 39, 7]])
        self.assertTrue(torch.equal(self.q.embed(ids), self.E[ids]))

    def test_logits_chunked_matches_dense(self):
        h = torch.randn(5, 16)
        ref = (h.to(torch.bfloat16) @ self.E.T).float()
        self.assertTrue(torch.equal(self.q.logits_last(h, chunk=7), ref))

    def test_prefix_mean_is_bf16_exact_and_ignores_padding(self):
        ids = torch.tensor([[0, 5, 6], [1, 2, 3]]); mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        m = prefix_mean_bf16(self.q.embed(ids), mask)
        self.assertTrue(torch.equal(m, m.to(torch.bfloat16).float()))          # same grid as the training cache
        ref = ((self.E[5].float() + self.E[6].float()) / 2).to(torch.bfloat16).float()
        self.assertTrue(torch.equal(m[0], ref))

    def test_scripts_do_not_load_full_model_in_train_or_xc(self):
        t = open(os.path.join(HERE, "train_aux.py")).read()
        self.assertNotIn("QwenText", t)
        e = open(os.path.join(HERE, "eval_aux.py")).read()
        self.assertIn('QwenText(model) if arm == "h" else LiteText(model)', e)


if __name__ == "__main__":
    unittest.main()
