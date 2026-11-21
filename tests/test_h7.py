"""H7 (lab/probes/delta_patterns/h7_plan.md): task registry, splits and the step-dataset trainer pieces."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h7"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))

import tasks as tk  # noqa: E402

NAMES = ["mul3x3", "mul4x3", "sq4", "mul2x2x1", "muladd1", "div6_2", "mod3_1", "add3x3", "rev4"]


class TestTasks(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(sorted(tk.TASKS), sorted(NAMES))

    def test_answers_are_right(self):
        import random
        r = random.Random(0)
        for name in NAMES:
            for _ in range(50):
                q, a = tk.TASKS[name](r)
                expr = q[:-1]
                if name == "rev4":
                    self.assertEqual(a, expr.split()[1][::-1])
                else:
                    self.assertEqual(int(a), eval(expr.replace("^", "**")), (name, q, a))

    def test_digit_shapes(self):
        import random
        r = random.Random(1)
        q, _ = tk.TASKS["mul4x3"](r)
        a, b = q[:-1].split("*")
        self.assertEqual((len(a), len(b)), (4, 3))
        for _ in range(200):
            q, _ = tk.TASKS["mod3_1"](r)
            a, b = q[:-1].split("%")
            self.assertEqual(len(a), 3)
            self.assertTrue(3 <= int(b) <= 19)

    def test_splits_sizes_disjoint_deterministic(self):
        for name in NAMES:
            s = tk.make_splits(name)
            self.assertEqual({k: len(v) for k, v in s.items() if k != "shots"},
                             {"train": 8000, "dev": 200, "test": 500})
            self.assertEqual(len(s["shots"]), 3)
            seen = set()
            for k in ("shots", "train", "dev", "test"):
                for q, _ in s[k]:
                    self.assertNotIn(q, seen, (name, k, q))
                    seen.add(q)
            self.assertEqual(tk.make_splits(name)["test"][:10], s["test"][:10])

    def test_prompt_and_target(self):
        s = tk.make_splits("mul3x3")
        p = tk.prompt("mul3x3", "123*456=")
        self.assertTrue(p.endswith("123*456="))
        self.assertEqual(p.count("\n"), 3)
        self.assertTrue(p.startswith(s["shots"][0][0] + s["shots"][0][1] + "\n"))
        self.assertEqual(tk.target("56088"), "56088\n")

    def test_grade(self):
        self.assertTrue(tk.grade("0321\nreverse", "0321"))
        self.assertFalse(tk.grade("321\n", "0321"))


class TestRunPieces(unittest.TestCase):
    def test_snapshot_schedule(self):
        import run_task as rt
        s = rt.snapshot_steps(200)
        self.assertEqual(s[:10], list(range(1, 11)))
        self.assertEqual(s[10:13], [15, 20, 25])
        self.assertEqual(s[-1], 200)
        self.assertEqual(len(s), 48)

    def test_shared_basis_is_deterministic_and_orthonormal(self):
        import run_task as rt
        shapes = {"model.layers.0.a.weight": (30, 12), "model.layers.0.b.weight": (12, 40)}
        U1, U2 = rt.shared_basis(shapes), rt.shared_basis(shapes)
        for n in shapes:
            self.assertTrue(torch.equal(U1[n], U2[n]))
            self.assertTrue(torch.allclose(U1[n].T @ U1[n], torch.eye(U1[n].shape[1]), atol=1e-5))
            self.assertEqual(U1[n].shape[0], min(shapes[n]))


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(5)

    def test_features_shape_and_adam_ratio(self):
        import predict as pr
        Ck, Ch, m, v = (torch.randn(4, 3, generator=self.g) for _ in range(4))
        v = v.abs()
        X = pr.features(Ck, Ch, m, v)
        self.assertEqual(tuple(X.shape), (12, 5))
        self.assertTrue(torch.allclose(X[:, 3], (m / (v.sqrt() + 1e-8)).flatten()))
        self.assertTrue(torch.all(X[:, 4] == 1))

    def test_fit_lin_recovers_planted_rule(self):
        import predict as pr
        w = torch.tensor([3.0, -1.0, 0.5, 0.0, 0.2], dtype=torch.float64)
        acc = pr.LinAcc()
        for _ in range(3):
            X = torch.randn(1000, 5, generator=self.g, dtype=torch.float64)
            X[:, 4] = 1
            acc.add(X, X @ w)
        self.assertTrue(torch.allclose(acc.solve(), w, atol=1e-6))

    def test_alpha_scale(self):
        import predict as pr
        Ck = {"a": torch.ones(2, 2)}
        self.assertAlmostEqual(pr.alpha_scale(Ck, {"a": 3 * torch.ones(2, 2)}), 3.0, places=6)

    def test_k_half(self):
        import predict as pr
        import run_task as rt
        snaps = set(rt.snapshot_steps(200))
        for k in pr.KS:
            self.assertIn(pr.K_HALF[k], snaps)
            self.assertIn(k, snaps)

    def test_r_traj(self):
        import predict as pr
        self.assertAlmostEqual(pr.r_traj(0.4, 0.2, 0.6), 0.5)


class TestGeometry(unittest.TestCase):
    def test_temporal_shares_of_a_straight_path_is_one_mode(self):
        import geometry as ge
        d = torch.randn(50, generator=torch.Generator().manual_seed(0))
        traj = [t * d for t in (0.1, 0.3, 0.7, 1.0)]
        sh = ge.temporal_shares(traj)
        self.assertAlmostEqual(sh[0], 1.0, places=5)

    def test_temporal_shares_of_a_turn_needs_two_modes(self):
        import geometry as ge
        a, b = torch.eye(4)[0], torch.eye(4)[1]
        sh = ge.temporal_shares([a, a + b, b * 2])
        self.assertLess(sh[0], 0.95)
        self.assertAlmostEqual(sh[1], 1.0, places=5)

    def test_cos_matrix(self):
        import geometry as ge
        A = {"x": torch.tensor([1.0, 0.0])}
        B = {"x": torch.tensor([0.0, 2.0])}
        M = ge.cos_matrix({"a": A, "b": B, "c": A})
        self.assertAlmostEqual(M["a"]["b"], 0.0)
        self.assertAlmostEqual(M["a"]["c"], 1.0)


if __name__ == "__main__":
    unittest.main()
