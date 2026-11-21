"""lab/probes/act_map (PLAN.md): a linear map between the residual streams of two models that read the same tokens."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "act_map"))

import act_map as am  # noqa: E402


class TestPieces(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(7)

    def test_split_keeps_texts_whole_and_disjoint(self):
        text_of = np.repeat(np.arange(20), 5)
        fit, val, test = am.split_texts(text_of, n_test=4)
        self.assertTrue(((fit.astype(int) + val + test) == 1).all())
        self.assertEqual(sorted(set(text_of[test])), [16, 17, 18, 19])
        self.assertEqual(sorted(set(text_of[val])), [14, 15])
        for m in (fit, val, test):
            self.assertEqual(int(m.sum()) % 5, 0)
        with self.assertRaises(ValueError):
            am.split_texts(text_of, n_test=19)

    def test_cka_is_one_under_rotation_and_scale_and_low_for_unrelated(self):
        X = torch.randn(300, 10, generator=self.g)
        Q = torch.linalg.qr(torch.randn(10, 10, generator=self.g))[0]
        self.assertAlmostEqual(am.linear_cka(X, 3.0 * X @ Q + 5.0), 1.0, places=6)
        self.assertLess(am.linear_cka(X, torch.randn(300, 10, generator=self.g)), 0.2)
        self.assertAlmostEqual(am.cka_from_grams(am.gram(X), am.gram(X @ Q)), am.linear_cka(X, X @ Q), places=9)

    def test_ridge_recovers_an_affine_map_between_different_widths(self):
        X = torch.randn(600, 20, generator=self.g)
        M = torch.randn(20, 8, generator=self.g)
        Y = X @ M + 2.0 + 0.01 * torch.randn(600, 8, generator=self.g)
        S = am.Source(X[:400])
        W, my, lam, v = am.fit_ridge(S, Y[:400], X[400:500], Y[400:500])
        self.assertEqual(lam, min(am.LAMS))                       # almost noiseless: the weakest ridge wins
        self.assertTrue(torch.allclose(W.float(), M, atol=0.05))
        self.assertGreater(am.r2(Y[500:], S.predict(X[500:], W, my)), 0.99)

    def test_ridge_strength_grows_when_there_is_nothing_to_learn(self):
        X = torch.randn(120, 60, generator=self.g)
        Y = torch.randn(120, 5, generator=self.g)
        S = am.Source(X[:80])
        _, _, lam, v = am.fit_ridge(S, Y[:80], X[80:], Y[80:])
        self.assertEqual(lam, max(am.LAMS))
        self.assertLess(v, 0.05)

    def test_r2_is_zero_for_the_mean_and_one_for_the_truth(self):
        Y = torch.randn(50, 4, generator=self.g)
        self.assertAlmostEqual(am.r2(Y, Y), 1.0)
        self.assertAlmostEqual(am.r2(Y, Y.mean(0).expand_as(Y)), 0.0, places=9)

    def test_depth_match_maps_the_ends_to_the_ends(self):
        self.assertEqual(am.depth_match(0, 25, 33), 0)
        self.assertEqual(am.depth_match(24, 25, 33), 32)
        self.assertEqual(am.depth_match(12, 25, 33), 16)

    def test_a_mean_difference_goes_through_the_map_without_the_intercept(self):
        X = torch.randn(400, 12, generator=self.g)
        mask = np.arange(400) % 4 == 0
        X[mask] += 1.5 * torch.randn(12, generator=self.g)
        M = torch.randn(12, 6, generator=self.g)
        Y = X @ M + 7.0
        S = am.Source(X)
        W, my = S.solve(Y, 1e-6)
        self.assertGreater(am.cos(am.mean_diff(X, mask) @ W, am.mean_diff(Y, mask)), 0.999)

    def test_contrast_masks(self):
        text_of = [0] * 8 + [1] * 4
        pos = list(range(1, 9)) + list(range(1, 5))
        m = am.contrast_masks(text_of, pos, [True] + [False] * 11)
        self.assertEqual(m["late"].tolist(), [False] * 5 + [True] * 3 + [False] * 2 + [True] * 2)
        self.assertEqual(int(m["digit"].sum()), 1)

    def test_verdict_rule(self):
        good = [{"r2": 0.8, "r2_emb": 0.3, "cos_late": 0.9}] * 6
        self.assertTrue(am.verdict(good, 0.0)["pass"])
        self.assertFalse(am.verdict(good, 0.2)["checks"]["shuffle"])
        self.assertFalse(am.verdict([{"r2": 0.8, "r2_emb": 0.75, "cos_late": 0.9}] * 6, 0.0)["checks"]["over_embeddings"])
        self.assertFalse(am.verdict([{"r2": 0.3, "r2_emb": 0.0, "cos_late": 0.9}] * 6, 0.0)["checks"]["median_r2"])
        self.assertFalse(am.verdict([{"r2": 0.8, "r2_emb": 0.3, "cos_late": 0.2}] * 6, 0.0)["checks"]["late_contrast"])


class TestFitOnSyntheticCaptures(unittest.TestCase):
    """Two fake 'models' built from one latent code: state 0 knows only the token, later states also the context."""

    def _make(self, store, name, d, n_states, Z_tok, Z_ctx, seed, related=True):
        g = torch.Generator().manual_seed(seed)
        out = store / name
        out.mkdir(parents=True)
        n = Z_tok.shape[0]
        for k in range(n_states):
            Z = Z_tok if k == 0 else torch.cat([Z_tok, Z_ctx], 1)
            if not related and k > 0:
                Z = torch.randn(Z.shape, generator=g)
            H = Z @ torch.randn(Z.shape[1], d, generator=g) + 0.05 * torch.randn(n, d, generator=g) + 3.0
            np.save(out / f"h{k:02d}.npy", H.numpy().astype(np.float16))
        return n_states, d

    def _run(self, related):
        g = torch.Generator().manual_seed(1)
        n_texts, per = 200, 20
        text_of = np.repeat(np.arange(n_texts), per).tolist()
        pos = (list(range(1, per + 1)) * n_texts)
        tok_ids = torch.randint(0, 30, (n_texts * per,), generator=g)
        E = torch.randn(30, 4, generator=g)
        Z_tok = E[tok_ids]
        Z_ctx = torch.randn(n_texts * per, 5, generator=g)
        Z_ctx[:, 0] += torch.tensor([2.0 if p >= 15 else 0.0 for p in pos])      # a context-only "late" signal
        with tempfile.TemporaryDirectory() as t:
            store, runs = Path(t) / "store", Path(t) / "runs"
            for name, d, ns, seed in (("A", 24, 5, 11), ("B", 12, 4, 12)):
                self._make(store, name, d, ns, Z_tok, Z_ctx, seed, related=related or name == "A")
                meta = {"n_states": ns, "d": d, "tokens": len(pos), "ids_sha256": am.ids_hash(tok_ids.tolist()),
                        "token_ids": tok_ids.tolist(), "text_of": text_of, "pos": pos,
                        "is_digit": (tok_ids < 5).tolist()}
                (store / name / "meta.json").write_text(json.dumps(meta))
            old = am.STORE, am.RUNS
            am.STORE, am.RUNS = store, runs
            try:
                am.fit(types.SimpleNamespace(a="A", b="B", n_test=30, n_cka=400, save_maps=True))
                rec = json.loads((runs / "A__B.json").read_text())
                from safetensors.torch import load_file
                maps = load_file(str(store / "maps_A__B.safetensors"))
            finally:
                am.STORE, am.RUNS = old
        return rec, maps

    def test_related_models_pass_in_both_directions(self):
        rec, maps = self._run(related=True)
        for k in ("A->B", "B->A"):
            v = rec[k]["verdict"]
            self.assertTrue(v["pass"], (k, v))
            self.assertGreater(v["median_r2"], 0.9)
            self.assertLess(abs(v["shuffle_r2"]), 0.05)
            for row in rec[k]["rows"]:
                self.assertGreater(row["r2"] - row["r2_emb"], 0.2)        # context is not in the embeddings
                self.assertGreater(row["cos_late"], 0.9)
                self.assertGreater(row["source_best"], 0)
        self.assertEqual(len(rec["A->B"]["rows"]), 3)                     # target states 1..3 of B
        self.assertEqual(tuple(maps["W.1"].shape), (24, 12))
        self.assertEqual(rec["tokens"], {"fit": 3060, "val": 340, "test": 600})

    def test_unrelated_models_fail(self):
        rec, _ = self._run(related=False)
        v = rec["A->B"]["verdict"]
        self.assertFalse(v["pass"], v)
        self.assertLess(v["median_r2"], 0.2)

    def test_different_tokens_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            store = Path(t)
            for name, h in (("A", "x"), ("B", "y")):
                (store / name).mkdir()
                (store / name / "meta.json").write_text(json.dumps({"ids_sha256": h, "text_of": [0], "n_states": 1}))
            old = am.STORE
            am.STORE = store
            try:
                with self.assertRaises(ValueError):
                    am.fit(types.SimpleNamespace(a="A", b="B", n_test=1, n_cka=10, save_maps=False))
            finally:
                am.STORE = old


if __name__ == "__main__":
    unittest.main()
