"""lab/probes/close_probe/cf_distill.py: closed-form (no gradient) fit of the base's mlp.down_proj, layer by layer, so
its residual stream follows a teacher's on the teacher's own text (the WSP / MEMIT least-squares write, used for a
behaviour instead of a fact).

Checked on the real Qwen3.5-0.8B (CPU): a teacher that differs from the base only by a known change in some
down_proj matrices is matched to rounding error, and the change found is the known one on the directions the data
spans."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
M08 = ROOT / "hf_cache" / "qwen35-08b-hf"


class TestSolve(unittest.TestCase):
    def test_ridge_solve_recovers_a_linear_map(self):
        import cf_distill as cf
        g = torch.Generator().manual_seed(0)
        K = torch.randn(500, 20, generator=g, dtype=torch.float64)
        D = torch.randn(7, 20, generator=g, dtype=torch.float64)
        dW = cf.solve(K.T @ K, K.T @ (K @ D.T), mu=1e-9)
        self.assertEqual(tuple(dW.shape), (7, 20))
        self.assertTrue(torch.allclose(dW.double(), D, atol=1e-5))

    def test_heldout_error_in_closed_form_equals_direct(self):
        import cf_distill as cf
        g = torch.Generator().manual_seed(2)
        K = torch.randn(300, 12, generator=g, dtype=torch.float64)
        R = torch.randn(300, 5, generator=g, dtype=torch.float64)
        dW = torch.randn(5, 12, generator=g, dtype=torch.float64)
        direct = (R - K @ dW.T).pow(2).sum()
        closed = cf.residual_error(dW, K.T @ K, K.T @ R, R.pow(2).sum())
        self.assertAlmostEqual(float(closed), float(direct), places=6)

    def test_ridge_is_chosen_on_heldout(self):
        import cf_distill as cf
        g = torch.Generator().manual_seed(3)
        D = torch.randn(4, 30, generator=g, dtype=torch.float64)

        def data(n):
            K = torch.randn(n, 30, generator=g, dtype=torch.float64)
            return K, K @ D.T + 3.0 * torch.randn(n, 4, generator=g, dtype=torch.float64)
        Kt, Rt = data(40)                 # few noisy training rows: a tiny ridge overfits
        Kh, Rh = data(400)
        mus = (1e-6, 1e-2, 1e-1, 1.0)
        dW, mu, errs = cf.solve_select(Kt.T @ Kt, Kt.T @ Rt, Kh.T @ Kh, Kh.T @ Rh, Rh.pow(2).sum(), mus)
        self.assertEqual(errs[mu], min(errs.values()))
        self.assertGreater(mu, 1e-6)


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestTeacherLoader(unittest.TestCase):
    def test_reads_a_layer_without_mmap_and_matches_the_model(self):
        import cf_distill as cf
        from engine.weight_diff import LocalSafetensors
        from transformers import AutoModelForCausalLM
        state, get = cf.teacher_loader(M08)
        self.assertIsInstance(cf._READER, LocalSafetensors)     # safe_open per tensor crashed natively on Windows
        model = AutoModelForCausalLM.from_pretrained(str(M08), dtype=torch.bfloat16)
        sd = state(2)
        ref = cf.layers_of(model)[2].state_dict()
        self.assertEqual(set(sd), set(ref))
        for k in ref:
            self.assertTrue(torch.equal(sd[k].to(ref[k].dtype), ref[k]), k)


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestFit(unittest.TestCase):
    def test_teacher_with_known_down_proj_change_is_matched(self):
        import cf_distill as cf
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(M08), dtype=torch.float32)
        L = cf.layers_of(model)
        g = torch.Generator().manual_seed(1)
        known = {}
        for l in (1, 4):
            W = L[l].mlp.down_proj.weight
            known[l] = torch.randn(W.shape, generator=g) * W.std() * 0.3

        def teacher_state(l):
            sd = {k: v.clone() for k, v in L[l].state_dict().items()}
            if l in known:
                sd["mlp.down_proj.weight"] += known[l]
            return sd
        embed = model.get_input_embeddings().weight.detach()
        # more positions than down_proj inputs (3584), as in the real run (keys over-determine the change)
        seqs = [torch.randint(0, 5000, (n,), generator=g).tolist() for n in (1500, 1400, 1300, 400)]
        out = cf.fit(model, teacher_state, embed, seqs, layers=[0, 1, 2, 3, 4, 5], mu=1e-6, device="cpu",
                     heldout=1)
        for l in (1, 4):            # held-out sequence: the fitted change generalises, it is the known one
            self.assertGreater(out["stats"][l]["explained_heldout"], 0.99, out["stats"][l])
            rn = known[l].norm() / L[l].mlp.down_proj.weight.norm()
            self.assertAlmostEqual(out["stats"][l]["rel_norm"], rn.item(), delta=0.05 * rn.item())
        for l in (1, 4):
            self.assertGreater(out["stats"][l]["explained_train"], 0.999, out["stats"][l])
        for l in (0, 2, 3, 5):      # identical layers: nothing to fit, the change stays ~0
            self.assertLess(out["delta"][l].norm().item(), 1e-3 * L[l].mlp.down_proj.weight.norm().item())
        # on the directions the data spans the known change is recovered: compare its action on the keys
        K = out["keys_sample"][1]
        rel = ((K @ out["delta"][1].T) - (K @ known[1].T)).norm() / (K @ known[1].T).norm()
        self.assertLess(rel.item(), 1e-2)
        # the model's own weights are left as they were (the change is returned, not written in)
        self.assertEqual(len(L[1].mlp.down_proj._forward_hooks), 0)

    def test_teacher_with_known_attention_output_change_is_matched(self):
        """targets=("attn","down"): the attention output projection (linear_attn.out_proj in linear-attention layers,
        self_attn.o_proj in full-attention ones) is fitted first, to the teacher's stream after attention, then
        down_proj to the rest. A known change in the attention output of layers 1 (linear) and 3 (full) is found."""
        import cf_distill as cf
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(M08), dtype=torch.float32)
        L = cf.layers_of(model)
        self.assertEqual(cf.attn_out_name(L[1]), "linear_attn.out_proj")
        self.assertEqual(cf.attn_out_name(L[3]), "self_attn.o_proj")
        g = torch.Generator().manual_seed(4)
        known = {}
        for l in (1, 3):
            W = cf.attn_out(L[l]).weight
            known[l] = torch.randn(W.shape, generator=g) * W.std() * 0.3

        def teacher_state(l):
            sd = {k: v.clone() for k, v in L[l].state_dict().items()}
            if l in known:
                sd[cf.attn_out_name(L[l]) + ".weight"] += known[l]
            return sd
        embed = model.get_input_embeddings().weight.detach()
        seqs = [torch.randint(0, 5000, (n,), generator=g).tolist() for n in (1500, 1400, 1300, 400)]
        out = cf.fit(model, teacher_state, embed, seqs, layers=[0, 1, 2, 3, 4], mu=1e-6, device="cpu", heldout=1,
                     targets=("attn", "down"))
        for l in (1, 3):
            st = out["stats"][l]
            self.assertGreater(st["attn_explained_heldout"], 0.99, st)
            self.assertGreater(st["explained_heldout"], 0.99, st)
            rn = known[l].norm() / cf.attn_out(L[l]).weight.norm()
            self.assertAlmostEqual(st["attn_rel_norm"], rn.item(), delta=0.05 * rn.item())
            # the known change was all in attention: down_proj has nothing left to fix
            self.assertLess(out["delta"][l].float().norm().item(), 1e-2 * L[l].mlp.down_proj.weight.norm().item())
        for l in (0, 2, 4):
            self.assertLess(out["delta_attn"][l].float().norm().item(),
                            1e-3 * cf.attn_out(L[l]).weight.norm().item())
        self.assertEqual(len(cf.attn_out(L[1])._forward_hooks), 0)
        self.assertEqual(len(L[1].post_attention_layernorm._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
