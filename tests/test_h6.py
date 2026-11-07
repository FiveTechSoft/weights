"""H6 stage 0 (lab/probes/delta_patterns/h6_plan.md): data splits and the per-chunk fingerprint."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))

import fingerprint as fp  # noqa: E402
import make_data as md  # noqa: E402


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = md.make_splits()

    def test_sizes(self):
        self.assertEqual({k: len(v) for k, v in self.s.items()},
                         {"train": 20000, "dev": 200, "test_id": 1000, "ood_easy": 500, "ood_hard": 500})

    def test_deterministic(self):
        self.assertEqual(md.make_splits()["test_id"][:20], self.s["test_id"][:20])

    def test_no_pair_shared_across_splits(self):
        seen = {}
        for name, items in self.s.items():
            for a, b in items:
                self.assertNotIn((a, b), seen, f"{(a, b)} in {name} and {seen.get((a, b))}")
                seen[(a, b)] = name
        for a, b in md.SHOTS:
            self.assertNotIn((a, b), seen)

    def test_digit_counts(self):
        def ok(items, da, db):
            return all(len(str(a)) == da and len(str(b)) == db for a, b in items)
        self.assertTrue(ok(self.s["train"], 3, 3) and ok(self.s["test_id"], 3, 3) and ok(self.s["dev"], 3, 3))
        self.assertTrue(ok(self.s["ood_easy"], 3, 2) and ok(self.s["ood_hard"], 4, 3))
        self.assertTrue(all(len(str(a)) == 3 and len(str(b)) == 3 for a, b in md.SHOTS))

    def test_prompt_and_target(self):
        p = md.prompt(123, 45)
        self.assertTrue(p.endswith("123*45="))
        self.assertEqual(p.count("="), len(md.SHOTS) + 1)
        self.assertEqual(md.target(123, 45), "5535\n")


class TestFingerprintMath(unittest.TestCase):
    def test_eff_rank_of_rank_one_is_one_and_of_identity_is_n(self):
        self.assertAlmostEqual(fp.eff_rank(torch.tensor([3.0, 0, 0, 0])), 1.0, places=5)
        self.assertAlmostEqual(fp.eff_rank(torch.ones(8)), 8.0, places=4)

    def test_overlap_same_subspace_is_one_orthogonal_is_zero(self):
        Q = torch.linalg.qr(torch.randn(64, 8, generator=torch.Generator().manual_seed(0)))[0]
        self.assertAlmostEqual(fp.overlap(Q[:, :4], Q[:, :4] @ torch.linalg.qr(torch.randn(4, 4))[0]), 1.0, places=5)
        self.assertAlmostEqual(fp.overlap(Q[:, :4], Q[:, 4:]), 0.0, places=5)

    def test_top_subspace_finds_planted_directions_on_small_side(self):
        g = torch.Generator().manual_seed(1)
        U = torch.linalg.qr(torch.randn(32, 3, generator=g))[0]          # small side: 32 rows
        D = U @ torch.randn(3, 200, generator=g) * 10 + 0.01 * torch.randn(32, 200, generator=g)
        self.assertGreater(fp.overlap(fp.top_subspace(D, 3), U), 0.99)
        self.assertEqual(fp.top_subspace(D.T, 3).shape, (32, 3))        # transposed: still the 32-dim side

    def test_energy_share(self):
        D = torch.diag(torch.tensor([3.0, 4.0, 0.0]))
        self.assertAlmostEqual(fp.top_energy(D, 1), 16 / 25, places=5)


class TestChunkFingerprint(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(2)
        self.base = {"model.layers.0.mlp.up_proj.weight": torch.randn(48, 16, generator=g),
                     "model.layers.1.mlp.down_proj.weight": torch.randn(16, 48, generator=g)}
        self.dirs = torch.linalg.qr(torch.randn(16, 2, generator=g))[0]
        self.g = g

    def step(self, W, scale):
        out = {}
        for n, w in W.items():
            if w.shape[1] == 16:      # small side = columns
                out[n] = w + scale * torch.randn(w.shape[0], 2, generator=self.g) @ self.dirs.T
            else:
                out[n] = w + scale * self.dirs @ torch.randn(2, w.shape[1], generator=self.g)
        return out

    def test_increments_in_same_subspace_overlap_fully(self):
        w1 = self.step(self.base, 0.1)
        w2 = self.step(w1, 0.1)
        rep = fp.chunk_fingerprint(self.base, w1, w2, r=2)
        for row in rep["tensors"]:
            self.assertGreater(row["overlap_prev_inc"], 0.99)
            self.assertGreater(row["overlap_cum"], 0.99)
            self.assertLess(row["eff_rank"], 2.5)
        self.assertEqual({r["kind"] for r in rep["tensors"]}, {"mlp.up_proj", "mlp.down_proj"})
        self.assertAlmostEqual(sum(r["share"] for r in rep["tensors"]), 1.0, places=5)

    def test_first_chunk_has_no_previous(self):
        w1 = self.step(self.base, 0.1)
        rep = fp.chunk_fingerprint(self.base, self.base, w1, r=2)
        self.assertIsNone(rep["tensors"][0]["overlap_prev_inc"])
        self.assertIsNone(rep["tensors"][0]["overlap_cum"])

    def test_cross_seed_overlap(self):
        same = fp.cross_overlap(self.base, self.step(self.base, 0.1), self.step(self.base, 0.1), r=2)
        self.assertGreater(min(same.values()), 0.99)
        other = {n: w + 0.1 * torch.randn(w.shape, generator=self.g) for n, w in self.base.items()}
        diff = fp.cross_overlap(self.base, self.step(self.base, 0.1), other, r=2)
        self.assertLess(max(diff.values()), 0.6)


class TestAdamW(unittest.TestCase):
    def test_matches_torch_adamw_and_returns_the_applied_update(self):
        import train
        g = torch.Generator().manual_seed(3)
        p1 = torch.nn.Parameter(torch.randn(6, 4, generator=g))
        p2 = torch.nn.Parameter(p1.detach().clone())
        ref = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        opt = train.AdamW([p1], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        for _ in range(5):
            grad = torch.randn(6, 4, generator=g)
            p1.grad, p2.grad = grad.clone(), grad.clone()
            before = p1.detach().clone()
            stats = opt.step(lr=1e-3)
            ref.step()
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))
            self.assertAlmostEqual(stats[0]["u_norm"], float((p1.detach() - before).norm()), places=5)
            self.assertAlmostEqual(stats[0]["g_norm"], float(grad.norm()), places=4)


class TestSchedule(unittest.TestCase):
    def test_warmup_then_cosine(self):
        import train
        self.assertAlmostEqual(train.lr_at(0, 500), train.LR / train.WARMUP)
        self.assertAlmostEqual(train.lr_at(train.WARMUP - 1, 500), train.LR)
        self.assertAlmostEqual(train.lr_at(train.WARMUP, 500), train.LR)
        self.assertLess(train.lr_at(499, 500), train.LR * 0.01)
        self.assertEqual(train.LR, 3e-6)


class TestFingerprintDevice(unittest.TestCase):
    def test_same_result_in_float32(self):
        g = torch.Generator().manual_seed(5)
        D = torch.randn(40, 100, generator=g)
        self.assertGreater(fp.overlap(fp.top_subspace(D, 4), fp.top_subspace(D, 4, device="cpu", dtype=torch.float32)),
                           0.999)


class TestBatchAndSketch(unittest.TestCase):
    def test_labels_only_on_target(self):
        import train
        ids, labels = train.encode_pair([1, 2, 3], [7, 8])
        self.assertEqual(ids, [1, 2, 3, 7, 8])
        self.assertEqual(labels, [-100, -100, -100, 7, 8])

    def test_sketch_preserves_cosine(self):
        import train
        g = torch.Generator().manual_seed(4)
        a = torch.randn(64, 2000, generator=g)
        b = 0.6 * a + 0.8 * torch.randn(64, 2000, generator=g)
        R = train.sketch_matrix((64, 2000), 8, seed=0)
        sa, sb = train.sketch(a, R), train.sketch(b, R)
        exact = float((a * b).sum() / (a.norm() * b.norm()))
        approx = float((sa * sb).sum() / (sa.norm() * sb.norm()))
        self.assertEqual(tuple(sa.shape), (64, 8))
        self.assertAlmostEqual(approx, exact, delta=0.08)


class TestGraders(unittest.TestCase):
    def test_exact_and_independent(self):
        import eval_arith as ea
        self.assertEqual(ea.grade("5535\n123*", 5535), (True, True))
        self.assertEqual(ea.grade(" 5535", 5535), (False, True))     # exact needs the digits only
        self.assertEqual(ea.grade("553\n", 5535), (False, False))
        self.assertEqual(ea.grade("", 5535), (False, False))


class TestSynth(unittest.TestCase):
    """E_k = W0 + alpha * C_k and its norm-matched random control (h6_plan.md, synthetic continuation)."""

    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.base = {"a": torch.randn(8, 4, generator=g), "b": torch.randn(3, 5, generator=g)}
        self.ck = {n: w + 0.1 * torch.randn(w.shape, generator=g) for n, w in self.base.items()}

    def test_extrapolate(self):
        import synth
        e = synth.extrapolate(self.base, self.ck, 3.0)
        for n in self.base:
            self.assertTrue(torch.allclose(e[n] - self.base[n], 3.0 * (self.ck[n] - self.base[n]), atol=1e-6))
        one = synth.extrapolate(self.base, self.ck, 1.0)
        self.assertTrue(all(torch.allclose(one[n], self.ck[n], atol=1e-6) for n in self.base))

    def test_random_control_matches_norm_per_tensor_and_is_not_the_delta(self):
        import synth
        r = synth.random_like(self.base, self.ck, 3.0, seed=1)
        for n in self.base:
            d = self.ck[n] - self.base[n]
            self.assertAlmostEqual(float((r[n] - self.base[n]).norm()), float(3.0 * d.norm()), places=4)
            cos = float(((r[n] - self.base[n]) * d).sum() / ((r[n] - self.base[n]).norm() * d.norm()))
            self.assertLess(abs(cos), 0.8)
        r2 = synth.random_like(self.base, self.ck, 3.0, seed=1)
        self.assertTrue(all(torch.equal(r[n], r2[n]) for n in self.base))

    def test_mcnemar_exact(self):
        import synth
        self.assertEqual(synth.mcnemar([True] * 5, [True] * 5), 1.0)
        # 10 discordant, all favouring a: two-sided exact p = 2 * 0.5**10
        a = [True] * 10 + [False] * 3
        b = [False] * 10 + [False] * 3
        self.assertAlmostEqual(synth.mcnemar(a, b), 2 * 0.5 ** 10, places=10)
        # 6 vs 4 discordant: p = 2 * P(X <= 4 | n=10) = 0.7539
        a = [True] * 6 + [False] * 4
        self.assertAlmostEqual(synth.mcnemar(a, [not x for x in a]), 0.75390625, places=6)

    def test_recovery(self):
        import synth
        self.assertAlmostEqual(synth.recovery(0.40, 0.28, 0.52), 0.5)
        self.assertAlmostEqual(synth.recovery(0.28, 0.28, 0.52), 0.0)

    def test_pick_alpha_prefers_smaller_on_tie(self):
        import synth
        self.assertEqual(synth.pick_alpha({1: 0.3, 2: 0.5, 3: 0.5, 5: 0.1}), 2)


if __name__ == "__main__":
    unittest.main()
