"""Contract tests for long-context synthesis: windowed attention plus ridge-fitted MLP blocks, no backprop."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "lab"), str(ROOT / "lab" / "reference" / "karpathy_nanogpt")]

import llmness as lm  # noqa: E402
from engine import wsp_synth, wsp_train  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

ARCH = ROOT / "programs" / "arch" / "nanogpt.wsparch"
TEXT = (ROOT / "lab" / "data" / "shakespeare_char" / "input.txt").read_text(encoding="utf-8")
DATA = wsp_train.char_dataset(TEXT)
SHAPE = dict(block=64, layers=3, heads=6, d=384, ff=1536, dropout=0.0, bias=False)
SPEC = dict(order=3, free_blocks=1, windows=(8, 16, 32))
VAL_TEXT = TEXT[int(len(TEXT) * 0.9):][:20000]


def fit_model():
    return wsp_synth.compile_long_context(ARCH, DATA.train[:300000], DATA.chars, overrides=SHAPE, seed=1, **SPEC).eval()


class TestLongContextSynthesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_model()
        cls.adapter = lm.NanoGPTCharModel(cls.model, DATA.chars)

    def test_output_runs_unchanged_in_upstream_model(self):
        upstream = GPT(GPTConfig(vocab_size=len(DATA.chars), block_size=64, n_layer=3, n_head=6, n_embd=384,
                                 dropout=0.0, bias=False)).eval()
        upstream.load_state_dict(self.model.export_state_dict(), strict=True)
        x = torch.tensor(DATA.val[:64][None])
        with torch.no_grad():
            self.assertTrue(torch.equal(upstream(x, x)[0], self.model(x, x)[0]))

    def test_no_gradient_was_ever_computed(self):
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        info = self.model.synthesis_info
        self.assertEqual(info["backward_calls"], 0)
        self.assertEqual(info["provenance"], "closed_form_ridge")

    def test_old_tokens_influence_the_prediction(self):
        """Tokens outside the n-gram's reach must change the output; a table cannot do this."""
        x = torch.tensor(DATA.val[:64][None])
        corrupted = x.clone()
        corrupted[0, :50] = torch.tensor(DATA.val[1000:1050])
        with torch.no_grad():
            self.assertFalse(torch.allclose(self.model(x)[0], self.model(corrupted)[0], atol=1e-3))

    def test_step_size_never_worsens_its_own_holdout(self):
        """Validation can still get worse, and does: see results.md section 31."""
        for fit in self.model.synthesis_info["fits"]:
            self.assertLessEqual(fit["holdout_nll_after"], fit["holdout_nll_before"])

    def test_without_fit_the_free_block_is_silent(self):
        table_only = wsp_synth.compile_long_context(ARCH, DATA.train[:300000], DATA.chars, overrides=SHAPE, seed=1,
                                                    fit=False, **SPEC).eval()
        self.assertEqual(table_only.get_parameter("transformer.h.2.mlp.c_proj.weight").abs().sum().item(), 0.0)

    def test_step_size_never_read_validation(self):
        self.assertLessEqual(self.model.synthesis_info["fit_positions_end"], 300000)

    def test_same_seed_is_reproducible(self):
        other = fit_model()
        for (name, a), (_, b) in zip(self.model.named_parameters(), other.named_parameters()):
            self.assertTrue(torch.equal(a, b), name)

    def test_too_few_heads_for_order_and_windows_is_rejected(self):
        with self.assertRaises(wsp_synth.SynthesisError):
            wsp_synth.compile_long_context(ARCH, DATA.train[:5000], DATA.chars, overrides=SHAPE, seed=1,
                                           order=5, free_blocks=1, windows=(8, 16, 32))

    def test_no_free_block_is_rejected(self):
        with self.assertRaises(wsp_synth.SynthesisError):
            wsp_synth.compile_long_context(ARCH, DATA.train[:5000], DATA.chars, overrides=SHAPE, seed=1,
                                           order=3, free_blocks=0, windows=(8,))


if __name__ == "__main__":
    unittest.main()
