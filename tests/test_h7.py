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


if __name__ == "__main__":
    unittest.main()
