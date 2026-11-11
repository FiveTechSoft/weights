"""H6 stage 0 (lab/probes/delta_patterns/h6_plan.md): data splits and the per-chunk fingerprint."""
import math
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


class TestTransfer(unittest.TestCase):
    """Stage 1 arms (h6_plan.md + addendum c): M2_r svd, M3_r random rescaled, X_r lane of the other seeds."""

    def setUp(self):
        self.g = torch.Generator().manual_seed(11)

    def test_project_with_top_svd_is_eckart_young(self):
        import transfer as tr
        for shape in ((30, 12), (12, 30)):
            D = torch.randn(shape, generator=self.g)
            U = tr.top_basis(D, 4)
            P = tr.project(D, U)
            Uf, S, Vh = torch.linalg.svd(D, full_matrices=False)
            ref = Uf[:, :4] @ torch.diag(S[:4]) @ Vh[:4]
            self.assertTrue(torch.allclose(P, ref, atol=1e-4), shape)
            self.assertEqual(U.shape, (12, 4))

    def test_full_basis_is_identity(self):
        import transfer as tr
        D = torch.randn(20, 8, generator=self.g)
        self.assertTrue(torch.allclose(tr.project(D, tr.top_basis(D, 8)), D, atol=1e-4))

    def test_random_basis_orthonormal_and_deterministic(self):
        import transfer as tr
        Q = tr.random_basis(16, 5, seed=3)
        self.assertTrue(torch.allclose(Q.T @ Q, torch.eye(5), atol=1e-5))
        self.assertTrue(torch.equal(Q, tr.random_basis(16, 5, seed=3)))

    def test_lane_basis_finds_shared_directions_of_other_runs(self):
        import transfer as tr
        shared = torch.linalg.qr(torch.randn(16, 3, generator=self.g))[0]
        mk = lambda: shared @ torch.randn(3, 50, generator=self.g) + 0.01 * torch.randn(16, 50, generator=self.g)
        Da, Db = mk(), mk()                       # (16 x 50): small side = rows
        L = tr.lane_basis(Da, Db, 3)
        self.assertGreater(fp.overlap(L, shared), 0.99)
        self.assertEqual(tr.lane_basis(Da.T, Db.T, 3).shape, (16, 3))
        own = mk()
        self.assertGreater(float(tr.project(own, L).norm() / own.norm()), 0.99)

    def test_bases_exist_for_narrow_matrices_below_rmax(self):
        import transfer as tr
        D = torch.randn(40, 16, generator=self.g)          # like in_proj_a/b: smaller side 16 < rmax
        U = tr.basis_upto(D, 256)
        self.assertEqual(U.shape, (16, 16))
        self.assertTrue(torch.allclose(tr.project(D, U[:, :10]), tr.project(D, tr.top_basis(D, 10)), atol=1e-4))

    def test_rescale_to(self):
        import transfer as tr
        A = torch.randn(5, 5, generator=self.g)
        self.assertAlmostEqual(float(tr.rescale_to(A, 2.5).norm()), 2.5, places=5)


class TestGradOnce(unittest.TestCase):
    def test_global_rescale(self):
        import grad_once as go
        d = {"a": torch.ones(2, 2), "b": 2 * torch.ones(1, 3)}
        out = go.global_rescale(d, 10.0)
        self.assertAlmostEqual(sum(float(v.pow(2).sum()) for v in out.values()) ** 0.5, 10.0, places=4)
        self.assertAlmostEqual(float(out["b"][0, 0] / out["a"][0, 0]), 2.0, places=5)


class TestH6f(unittest.TestCase):
    """H6f (h6_plan.md addendum d): shaped one-shot update and lane-coefficient training."""

    def setUp(self):
        self.g = torch.Generator().manual_seed(21)

    def test_shaped_update_has_per_matrix_norm_and_lies_in_lane(self):
        import h6f
        import transfer as tr
        G = {"a": torch.randn(30, 12, generator=self.g), "b": torch.randn(12, 40, generator=self.g)}
        lane = {n: tr.random_basis(12, 4, seed=i) for i, n in enumerate(G)}
        nu = {"a": 0.3, "b": 2.0}
        for use_sign in (False, True):
            d = h6f.shaped_update(G, lane, nu, alpha=0.5, use_sign=use_sign)
            for n in G:
                self.assertAlmostEqual(float(d[n].norm()), 0.5 * nu[n], places=5)
                self.assertTrue(torch.allclose(tr.project(d[n], lane[n]), d[n], atol=1e-5))
        d = h6f.shaped_update(G, lane, nu, alpha=1.0, use_sign=False)
        cos = float((d["a"] * -G["a"]).sum() / (d["a"].norm() * G["a"].norm()))
        self.assertGreater(cos, 0)                          # a descent direction

    def test_lane_norm(self):
        import h6f
        import transfer as tr
        U = tr.random_basis(12, 3, seed=0)
        D1 = U @ torch.randn(3, 20, generator=self.g)      # fully in the lane
        D2 = torch.randn(12, 20, generator=self.g)
        nu = h6f.lane_norm([D1, D2], U)
        ref = (float(D1.norm()) + float(tr.project(D2, U).norm())) / 2
        self.assertAlmostEqual(nu, ref, places=5)

    def test_coefficient_gradient_is_the_chain_rule(self):
        import h6f
        import transfer as tr
        for shape in ((30, 12), (12, 30)):
            d = min(shape)
            U = tr.random_basis(d, 4, seed=1)
            W0 = torch.randn(shape, generator=self.g)
            X = torch.randn(7, shape[1], generator=self.g)
            C = torch.zeros(h6f.coef_shape(shape, 4), requires_grad=True)
            loss = ((W0 + h6f.coef_apply(U, C, shape)) @ X.T).sin().sum()
            loss.backward()
            W = (W0 + h6f.coef_apply(U, C, shape)).detach().requires_grad_(True)
            (W @ X.T).sin().sum().backward()
            self.assertTrue(torch.allclose(h6f.coef_grad(W.grad, U, shape), C.grad, atol=1e-5), shape)

    def test_coef_apply_lies_in_lane(self):
        import h6f
        import transfer as tr
        for shape in ((30, 12), (12, 30)):
            U = tr.random_basis(12, 4, seed=2)
            C = torch.randn(h6f.coef_shape(shape, 4), generator=self.g)
            D = h6f.coef_apply(U, C, shape)
            self.assertEqual(tuple(D.shape), shape)
            self.assertTrue(torch.allclose(tr.project(D, U), D, atol=1e-5))

    def test_schedule_warmup_ten_percent_then_cosine(self):
        import h6f
        self.assertAlmostEqual(h6f.lr_at(0, 200, 1e-4), 1e-4 / 20)
        self.assertAlmostEqual(h6f.lr_at(19, 200, 1e-4), 1e-4)
        self.assertLess(h6f.lr_at(199, 200, 1e-4), 1e-6)


class TestH6g(unittest.TestCase):
    """H6g (h6_plan.md addendum e): forward-only (SPSA) search of the coefficients, its gate and its accounting."""

    def setUp(self):
        self.g = torch.Generator().manual_seed(33)

    def _linear(self, shapes):
        G = {n: torch.randn(s, generator=self.g, dtype=torch.float64) for n, s in shapes.items()}
        return G, (lambda C: sum(float((G[n] * C[n]).sum()) for n in G))

    def test_spsa_cosine_matches_its_expected_value(self):
        import h6g
        G, loss = self._linear({"a": (4, 50), "b": (30, 10)})          # d = 500
        C = {n: torch.zeros_like(v) for n, v in G.items()}
        d = h6g.n_coef(C)
        self.assertEqual(d, 500)
        for pairs in (16, 256):
            cos = [h6g.flat_cos(h6g.spsa(loss, C, 1e-3, pairs, self.g)[0], G) for _ in range(30)]
            self.assertAlmostEqual(sum(cos) / len(cos), h6g.expected_cos(pairs, d), delta=0.03)

    def test_spsa_is_unbiased(self):
        import h6g
        G, loss = self._linear({"a": (5, 8)})                          # d = 40
        C = {"a": torch.zeros(5, 8, dtype=torch.float64)}
        est, st = h6g.spsa(loss, C, 1e-3, 20000, self.g)
        self.assertGreater(h6g.flat_cos(est, G), 0.99)
        self.assertAlmostEqual(h6g.flat_norm(est) / h6g.flat_norm(G), 1.0, delta=0.05)
        self.assertAlmostEqual(st["loss"], 0.0, places=9)               # symmetric pairs around C = 0
        self.assertGreater(st["abs_diff"], 0)

    def test_spsa_does_not_move_the_coefficients(self):
        import h6g
        G, loss = self._linear({"a": (3, 7)})
        C = {"a": torch.randn(3, 7, generator=self.g, dtype=torch.float64)}
        before = C["a"].clone()
        h6g.spsa(loss, C, 1e-2, 4, self.g)
        self.assertTrue(torch.equal(C["a"], before))

    def test_directional_difference_is_the_gradient_norm_when_linear(self):
        import h6g
        G, loss = self._linear({"a": (6, 9), "b": (2, 5)})
        C = {n: torch.zeros_like(v) for n, v in G.items()}
        self.assertAlmostEqual(h6g.directional(loss, C, G, 0.3), h6g.flat_norm(G), places=8)

    def test_directional_difference_drops_when_the_step_leaves_the_linear_range(self):
        import h6g
        G = {"a": torch.randn(4, 4, generator=self.g, dtype=torch.float64)}
        loss = lambda C: math.tanh(float((G["a"] * C["a"]).sum()))      # noqa: E731
        C = {"a": torch.zeros(4, 4, dtype=torch.float64)}
        gn = h6g.flat_norm(G)
        self.assertAlmostEqual(h6g.directional(loss, C, G, 1e-4 / gn) / gn, 1.0, places=6)
        self.assertLess(h6g.directional(loss, C, G, 3.0 / gn) / gn, 0.5)

    def test_normalized_sgd_moves_each_coefficient_by_lr_in_rms(self):
        import h6g
        C = {"a": torch.zeros(4, 10), "b": torch.zeros(20, 3)}
        g = {"a": 5 * torch.randn(4, 10, generator=self.g), "b": 0.01 * torch.randn(20, 3, generator=self.g)}
        h6g.nsgd_step(C, g, 1e-3)
        self.assertAlmostEqual(h6g.flat_norm(C) / math.sqrt(h6g.n_coef(C)), 1e-3, places=8)
        self.assertLess(h6g.flat_cos(C, g), -0.999)                     # against the estimate
        h6g.nsgd_step(C, {n: torch.zeros_like(v) for n, v in g.items()}, 1e-3)   # zero estimate: no move, no nan
        self.assertTrue(all(torch.isfinite(v).all() for v in C.values()))

    def test_placebo_has_the_per_matrix_norm_and_another_direction(self):
        import h6g
        C = {"a": torch.randn(4, 100, generator=self.g), "b": 0.1 * torch.randn(100, 4, generator=self.g)}
        P = h6g.placebo_like(C, seed=5)
        for n in C:
            self.assertAlmostEqual(float(P[n].norm()), float(C[n].norm()), places=4)
        self.assertLess(abs(h6g.flat_cos(P, C)), 0.2)
        self.assertTrue(torch.equal(h6g.placebo_like(C, seed=5)["a"], P["a"]))

    def test_cost_and_evaluation_points(self):
        import h6g
        self.assertEqual(h6g.step_cost("spsa", pairs=8, batch=40), 640)
        self.assertEqual(h6g.step_cost("bp", batch=40), 120)
        self.assertEqual(h6g.eval_steps(1, h6g.step_cost("bp")), {200: 1})          # the H6f-b reference
        at = h6g.eval_steps(100, h6g.step_cost("spsa"))
        self.assertEqual(at, {38: 1, 375: 10, 3750: 100})
        self.assertEqual(h6g.eval_steps(10, 640), {38: 1, 375: 10})

    def test_select_and_tag(self):
        import h6g
        names = ["model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.up_proj.weight"]
        self.assertEqual(h6g.select(names, None), names)
        self.assertEqual(h6g.select(names, "mlp.down_proj"), names[:1])
        self.assertEqual(h6g.run_tag("spsa", "nsgd", "random", 1, "mlp.down_proj", 2),
                         "g_spsa_nsgd_random_r1_mlp.down_proj_seed2")
        self.assertEqual(h6g.run_tag("bp", "nsgd", "random", 20, None, 1), "g_bp_adam_random_r20_all_seed1")

    def test_gate_verdict_takes_the_largest_eps_that_passes(self):
        import h6g
        N = h6g.GATE_N[-1]

        def row(eps, ratio, share):
            return {"eps": eps, "ratio": ratio, "cos": {N: share * 0.01}, "expected": {N: 0.01}}
        self.assertEqual(h6g.gate_verdict([row(3e-5, 1.0, 1.0), row(1e-4, 0.9, 0.8), row(3e-4, 0.5, 1.0)]), 1e-4)
        self.assertEqual(h6g.gate_verdict([row(3e-5, 1.0, 0.2), row(1e-4, 1.4, 1.0)]), None)

    def test_forward_only_search_learns_a_toy_matrix_in_a_random_basis(self):
        """End to end without a model: W = W0 + U C, loss = |W X - Y|^2, target reachable inside the basis."""
        import h6f
        import h6g
        import train
        import transfer as tr
        shape, r = (12, 30), 3
        U = tr.random_basis(12, r, seed=4).double()
        W0 = torch.randn(shape, generator=self.g, dtype=torch.float64)
        X = torch.randn(30, 64, generator=self.g, dtype=torch.float64)
        Cstar = 0.05 * torch.randn(h6f.coef_shape(shape, r), generator=self.g, dtype=torch.float64)
        Y = (W0 + h6f.coef_apply(U, Cstar, shape)) @ X

        def loss(C):
            return float(((W0 + h6f.coef_apply(U, C["w"], shape)) @ X - Y).pow(2).mean())

        for opt in ("nsgd", "adam"):
            C = {"w": torch.nn.Parameter(torch.zeros_like(Cstar), requires_grad=False)}
            adam = train.AdamW([C["w"]]) if opt == "adam" else None
            start = loss(C)
            for step in range(400):
                est, _ = h6g.spsa(loss, C, 1e-4, 8, self.g)
                lr = h6f.lr_at(step, 400, 2e-3)
                if adam is None:
                    h6g.nsgd_step(C, est, lr)
                else:
                    C["w"].grad = est["w"]
                    adam.step(lr)
            self.assertLess(loss(C), 0.2 * start, opt)


if __name__ == "__main__":
    unittest.main()
