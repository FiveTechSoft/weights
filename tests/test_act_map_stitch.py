"""lab/probes/act_map/stitch.py (PLAN.md gate 2): put mapped states into a model's residual stream and score it."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "act_map"))

import stitch as st  # noqa: E402


def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


class TestPieces(unittest.TestCase):
    def test_gap_recovered(self):
        self.assertAlmostEqual(st.gap_recovered(3.0, 2.0, 1.0), 0.5)
        self.assertAlmostEqual(st.gap_recovered(3.0, 1.0, 1.0), 1.0)
        self.assertLess(st.gap_recovered(3.0, 4.0, 1.0), 0)

    def test_bootstrap_interval_of_a_paired_difference(self):
        rng = np.random.default_rng(0)
        n = rng.integers(50, 120, size=40).astype(float)
        a = n * 2.0 + rng.normal(0, 1, 40)
        b = n * 1.5 + rng.normal(0, 1, 40)          # b lower by ~0.5 per token
        lo, hi = st.boot_diff(a, b, n, reps=500, seed=1)
        self.assertGreater(lo, 0.4)
        self.assertLess(hi, 0.6)
        lo, hi = st.boot_diff(a, a, n, reps=100, seed=1)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_verdict_rule(self):
        good = {"G": 0.6, "diff_lo": 0.1, "nll_stitch": 1.0, "nll_shuffle": 2.0, "nll_mean": 2.5}
        bad = dict(good, G=0.4)
        self.assertTrue(st.verdict([good, good, bad])["pass"])
        self.assertFalse(st.verdict([good, bad, bad])["pass"])
        self.assertFalse(st.verdict([dict(good, diff_lo=-0.01)] * 2 + [bad])["pass"])
        self.assertFalse(st.verdict([dict(good, nll_mean=0.9)] * 3)["pass"])


class TestStitchOnTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = tiny()
        g = torch.Generator().manual_seed(3)
        cls.ids = [torch.randint(0, 64, (n,), generator=g).tolist() for n in (9, 6, 12)]
        with torch.no_grad():
            cls.hs = [cls.m(input_ids=torch.tensor([x]), output_hidden_states=True).hidden_states for x in cls.ids]

    def test_native_nll_is_the_models_own_loss_on_the_scored_positions(self):
        r = st.run_arm(self.m, self.ids, j=None, states=None, device="cpu", batch=2)
        with torch.no_grad():
            for t, x in enumerate(self.ids):
                lg = self.m(input_ids=torch.tensor([x])).logits[0]
                lp = torch.log_softmax(lg.float(), -1)
                want = -sum(float(lp[p, x[p + 1]]) for p in range(1, len(x) - 1))
                self.assertAlmostEqual(r["nll_sum"][t], want, places=4)
                self.assertEqual(r["n"][t], len(x) - 2)

    def test_injecting_the_models_own_state_j_reproduces_native_and_j_is_the_hidden_state_index(self):
        nat = st.run_arm(self.m, self.ids, j=None, states=None, device="cpu", batch=2)
        for j in (1, 2):
            own = [h[j][0, 1:] for h in self.hs]
            r = st.run_arm(self.m, self.ids, j=j, states=own, device="cpu", batch=2, ref_top1=nat["top1"])
            np.testing.assert_allclose(r["nll_sum"], nat["nll_sum"], atol=1e-4)
            self.assertEqual(r["top1_same"], nat["n"])
            wrong = [h[j - 1][0, 1:] for h in self.hs]           # off by one hidden state must change the loss
            r2 = st.run_arm(self.m, self.ids, j=j, states=wrong, device="cpu", batch=2)
            self.assertGreater(np.abs(np.array(r2["nll_sum"]) - nat["nll_sum"]).max(), 1e-3)

    def test_position_zero_is_left_native(self):
        z = [torch.zeros(len(x) - 1, 16) for x in self.ids]
        seen = {}

        def probe(mod, args, kwargs, out):
            seen["h"] = (out[0] if isinstance(out, tuple) else out).clone()
        h = self.m.model.layers[1].register_forward_hook(probe, with_kwargs=True)
        try:
            st.run_arm(self.m, self.ids[:1], j=1, states=z[:1], device="cpu", batch=1)
        finally:
            h.remove()
        # layer 1 sees layer 0's output (state 1) replaced at positions >= 1 but not at 0: its output at 0 is native
        torch.testing.assert_close(seen["h"][0, 0], self.hs[0][2][0, 0], atol=1e-5, rtol=0)

    def test_states_of_the_wrong_length_are_refused(self):
        bad = [torch.zeros(len(x), 16) for x in self.ids]
        with self.assertRaises(ValueError):
            st.run_arm(self.m, self.ids, j=1, states=bad, device="cpu", batch=2)


if __name__ == "__main__":
    unittest.main()
