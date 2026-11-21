"""H9 (lab/probes/delta_patterns/h9_plan.md): warm-start initialisations."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h9"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h7"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))


def _norm(D):
    return float(torch.cat([D[n].double().flatten() for n in D]).norm())


class TestWarm(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.Cs = {t: {"a": torch.randn(2, 3, generator=g), "b": torch.randn(4, generator=g)}
                   for t in ("t1", "t2", "t3")}

    def test_warm_init_excludes_held_and_sums(self):
        import run_warm as rw
        W = rw.warm_init(self.Cs, "t2", 0.5)
        for n in ("a", "b"):
            self.assertTrue(torch.allclose(W[n], 0.5 * (self.Cs["t1"][n] + self.Cs["t3"][n])))

    def test_warm_init_is_deterministic(self):
        import run_warm as rw
        A, B = rw.warm_init(self.Cs, "t1", 0.5), rw.warm_init(self.Cs, "t1", 0.5)
        for n in A:
            self.assertTrue(torch.equal(A[n], B[n]))

    def test_rand_like_norm(self):
        import run_warm as rw
        W = rw.warm_init(self.Cs, "t2", 0.5)
        R1, R2 = rw.rand_like_norm(W, seed=9), rw.rand_like_norm(W, seed=9)
        self.assertAlmostEqual(_norm(R1), _norm(W), places=4)
        for n in W:
            self.assertEqual(R1[n].shape, W[n].shape)
            self.assertTrue(torch.equal(R1[n], R2[n]))
        cos = float(torch.cat([R1[n].flatten() for n in W]) @ torch.cat([W[n].flatten() for n in W])) / _norm(W) ** 2
        self.assertLess(abs(cos), 0.9)

    def test_snapshots(self):
        import run_warm as rw
        self.assertEqual(rw.snap_steps(200), {25, 50, 100, 200})
        self.assertIn(10, rw.snap_steps(10))


if __name__ == "__main__":
    unittest.main()
