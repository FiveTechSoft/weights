"""lab/probes/cc/placebo_seeds.py: several same-norm random placebos of one control vector (one per seed), the check that
the generator reproduces the placebo already used (seed 2026), and the readings over the screen reports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "cc"))


def _vecs(d=64, layers=5, seed=0):
    g = np.random.default_rng(seed)
    return {N: (g.normal(size=d) * (N + 1)).astype(np.float32) for N in range(1, layers + 1)}


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestMake(unittest.TestCase):
    def test_placebos_keep_each_layer_norm_and_differ_by_seed(self):
        import placebo_seeds as ps
        from engine import contrast_vector as cv
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            vecs = _vecs()
            cv.write_gguf(vecs, td / "v.gguf", "qwen35")
            written = ps.make([1, 2], vec_path=td / "v.gguf", out_dir=td)
            self.assertEqual([p.name for p in written], ["v.placebo1.gguf", "v.placebo2.gguf"])
            p1, p2 = ps.read_vecs(td / "v.placebo1.gguf"), ps.read_vecs(td / "v.placebo2.gguf")
            self.assertEqual(sorted(p1), sorted(vecs))
            for N, v in vecs.items():
                self.assertAlmostEqual(float(np.linalg.norm(p1[N])), float(np.linalg.norm(v)), places=3)
                self.assertAlmostEqual(float(np.linalg.norm(p2[N])), float(np.linalg.norm(v)), places=3)
                self.assertLess(abs(_cos(p1[N], v)), 0.6)          # random direction in 64-d
                self.assertLess(abs(_cos(p1[N], p2[N])), 0.6)      # seeds differ
            # deterministic: the same seed again gives the same bytes of data
            ps.make([1], vec_path=td / "v.gguf", out_dir=td / "again")
            q1 = ps.read_vecs(td / "again" / "v.placebo1.gguf")
            for N in vecs:
                self.assertTrue(np.array_equal(p1[N], q1[N]))

    def test_model_hint_is_copied_from_the_vector(self):
        import placebo_seeds as ps
        from engine import contrast_vector as cv
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cv.write_gguf(_vecs(), td / "v.gguf", "somehint")
            ps.make([3], vec_path=td / "v.gguf", out_dir=td)
            self.assertEqual(ps.model_hint(td / "v.placebo3.gguf"), "somehint")


class TestCheck(unittest.TestCase):
    def test_regenerated_seed_matches_the_existing_placebo_only_with_the_same_seed(self):
        import placebo_seeds as ps
        from engine import contrast_vector as cv
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            vecs = _vecs()
            cv.write_gguf(vecs, td / "v.gguf", "qwen35")
            cv.write_gguf(cv.placebo(vecs, 2026), td / "v.placebo2026.gguf", "qwen35")   # as the op wrote it
            ok = ps.check(vec_path=td / "v.gguf", placebo_path=td / "v.placebo2026.gguf", seed=2026)
            self.assertTrue(ok["identical"])
            self.assertEqual(ok["layers"], len(vecs))
            bad = ps.check(vec_path=td / "v.gguf", placebo_path=td / "v.placebo2026.gguf", seed=7)
            self.assertFalse(bad["identical"])


def _report(path, oks, cut=None, tokens=None, ok2=None):
    cut = cut or [False] * len(oks)
    tokens = tokens or [1000] * len(oks)
    ok2 = ok2 or oks
    items = [{"bench": "math500", "i": i, "ok": bool(o), "ok_independent": bool(o2),
              "finish": "length" if c else "stop", "tokens": t, "loop": False}
             for i, (o, o2, c, t) in enumerate(zip(oks, ok2, cut, tokens))]
    path.write_text(json.dumps({"items": items, "summary": {}}), encoding="utf-8")


class TestReadings(unittest.TestCase):
    def test_distribution_counts_and_vector_rank(self):
        import placebo_seeds as ps
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n = 64
            base = [i < 8 for i in range(n)]                       # 8/64
            vec = [i < 38 for i in range(n)]                       # 38/64
            _report(td / "screen_base.json", base)
            _report(td / "screen_s1.0.json", vec)
            _report(td / "screen_pl1.json", [i < 6 for i in range(n)])                 # below base, not significant
            _report(td / "screen_pl2.json", [i < 8 for i in range(n)])                 # equal
            _report(td / "screen_pl3.json", [8 <= i < 32 for i in range(n)])           # 24 correct, disjoint: helps, p < 0.05
            r = ps.readings(td, [1, 2, 3])
            self.assertEqual(r["base"]["correct"], 8)
            self.assertEqual(r["vector"]["correct"], 38)
            self.assertEqual([s["correct"] for s in r["seeds"]], [6, 8, 24])
            self.assertEqual(r["median_correct"], 8)
            self.assertEqual(r["max_correct"], 24)
            self.assertEqual(r["half_gain_threshold"], 23)          # base + ceil((38 - 8) / 2)
            self.assertEqual(r["n_seeds_at_or_above_half_gain"], 1)
            self.assertEqual(r["n_seeds_better_than_base_p05"], 1)
            self.assertEqual(r["n_seeds_worse_than_base_p05"], 0)
            self.assertEqual(r["vector_rank"], 1)                     # 1 + number of seeds with correct >= vector
            self.assertEqual(r["seeds"][2]["vs_base"]["win"], 24)
            self.assertEqual(r["seeds"][2]["vs_base"]["loss"], 8)
            self.assertLess(r["seeds"][2]["vs_base"]["p"], 0.05)

    def test_vector_rank_counts_ties_against_the_vector(self):
        import placebo_seeds as ps
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n = 20
            _report(td / "screen_base.json", [i < 2 for i in range(n)])
            _report(td / "screen_s1.0.json", [i < 10 for i in range(n)])
            _report(td / "screen_pl1.json", [i < 10 for i in range(n)])     # ties the vector
            _report(td / "screen_pl2.json", [i < 12 for i in range(n)])     # beats it
            r = ps.readings(td, [1, 2])
            self.assertEqual(r["vector_rank"], 3)


if __name__ == "__main__":
    unittest.main()
