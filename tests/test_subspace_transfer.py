"""subspace_transfer: move a base model by a source model's delta, restricted to a subspace, tensor by tensor.

W' = W + alpha * P(D), D = W_source - W_base, P = projection on a k-dim subspace of the delta's smaller side
(the residual-stream side for the matrices of these models). The subspace comes from a delta signature
(engine.weight_diff.delta_signature), from the shared part of two signatures, or is random (the control).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.subspace_ops as so  # noqa: E402
import engine.weight_diff as wd  # noqa: E402


class TestProjection(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.D = torch.randn(40, 12, generator=g)          # tall: smaller side = 12 columns
        self.Q = torch.linalg.qr(torch.randn(12, 12, generator=g))[0]

    def test_full_basis_keeps_everything(self):
        self.assertTrue(torch.allclose(so.project_delta(self.D, self.Q), self.D, atol=1e-5))

    def test_orthogonal_basis_keeps_nothing(self):
        V, Vperp = self.Q[:, :4], self.Q[:, 4:]
        part = so.project_delta(self.D, V)
        self.assertTrue(torch.allclose(so.project_delta(part, Vperp), torch.zeros_like(part), atol=1e-5))

    def test_partial_is_orthogonal_projection(self):
        V = self.Q[:, :4]
        P = so.project_delta(self.D, V)
        self.assertTrue(torch.allclose(P, self.D @ V @ V.T, atol=1e-5))
        self.assertLessEqual(float(P.norm()), float(self.D.norm()) + 1e-5)

    def test_wide_matrix_projects_rows(self):
        Dw = self.D.T                                       # 12 x 40: smaller side = 12 rows
        V = self.Q[:, :4]
        self.assertTrue(torch.allclose(so.project_delta(Dw, V), V @ V.T @ Dw, atol=1e-5))

    def test_basis_must_match_smaller_side(self):
        with self.assertRaises(ValueError):
            so.project_delta(self.D, torch.randn(40, 3))


class TestBases(unittest.TestCase):
    def test_random_basis(self):
        V = so.random_basis(64, 8, seed=3)
        self.assertTrue(torch.allclose(V.T @ V, torch.eye(8), atol=1e-5))
        self.assertTrue(torch.equal(V, so.random_basis(64, 8, seed=3)))
        self.assertFalse(torch.equal(V, so.random_basis(64, 8, seed=4)))

    def test_shared_basis_finds_common_directions(self):
        g = torch.Generator().manual_seed(1)
        Q = torch.linalg.qr(torch.randn(64, 64, generator=g))[0]
        A = Q[:, [0, 1, 2, 3, 10, 11]]   # shares columns 0-3 with B
        B = Q[:, [0, 1, 2, 3, 20, 21]]
        S, cos = so.shared_basis(A, B, 4)
        common = Q[:, :4]
        self.assertGreater(float((common.T @ S).pow(2).sum()) / 4, 0.999)
        self.assertTrue(all(c > 0.999 for c in cos[:4]))


def save_model(directory, tensors, split):
    directory.mkdir(parents=True, exist_ok=True)
    s1 = {k: v for k, v in tensors.items() if k not in split}
    s2 = {k: v for k, v in tensors.items() if k in split}
    save_file(s1, str(directory / "model-00001-of-00002.safetensors"))
    save_file(s2, str(directory / "model-00002-of-00002.safetensors"))
    (directory / "config.json").write_text(json.dumps({"model_type": "toy"}))


class TestWSPLangOp(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        g = torch.Generator().manual_seed(2)
        self.base = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(16, 48, generator=g) for i in range(2)}
        self.base.update({f"model.layers.{i}.mlp.up_proj.weight": torch.randn(48, 16, generator=g) for i in range(2)})
        self.base["model.embed_tokens.weight"] = torch.randn(30, 16, generator=g)
        self.src = {k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in self.base.items()}
        split = {"model.layers.1.mlp.down_proj.weight", "model.layers.1.mlp.up_proj.weight"}
        self.base_dir, self.src_dir, self.out = self.dir / "base", self.dir / "src", self.dir / "out"
        save_model(self.base_dir, self.base, split)
        save_model(self.src_dir, self.src, split)
        sigs = {k: wd.delta_signature(k, self.base[k], self.src[k], k=4, sketch_dim=64) for k in self.base}
        self.sig = self.dir / "sig_src.pt"
        torch.save(sigs, self.sig)

    def run_op(self, args, budgets="allow_write = true\n    max_modified_weights = 100000"):
        import engine.wsplang as wsplang
        code = (f"experiment t:\nbudgets:\n    {budgets}\nsteps:\n    step subspace_transfer {args}\n")
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertFalse(errors, errors)
        return wsplang.execute_plan_v4(wsplang.plan_run(prog, steps), prog, model_path=str(self.base_dir))[0]

    def out_tensors(self):
        t = {}
        for f in self.out.glob("*.safetensors"):
            t.update(load_file(str(f)))
        return t

    def test_transfers_projected_delta_only_where_asked(self):
        r = self.run_op(f"tensors=*down_proj* source={self.src_dir.as_posix()} basis=sig:{self.sig.as_posix()} "
                        f"k=4 alpha=1.0 out={self.out.as_posix()}")
        self.assertEqual(r["status"], "ok", r)
        got = self.out_tensors()
        self.assertEqual(set(got), set(self.base))
        for k in self.base:
            if "down_proj" in k:
                V = torch.load(self.sig)[k]["basis"]
                want = self.base[k] + so.project_delta(self.src[k] - self.base[k], V)
                self.assertTrue(torch.allclose(got[k], want, atol=1e-5), k)
            else:
                self.assertTrue(torch.equal(got[k], self.base[k]), k)
        self.assertTrue((self.out / "config.json").exists())
        with safe_open(str(next(self.out.glob("*00002*.safetensors"))), "pt") as h:
            meta = h.metadata()
        self.assertEqual(meta["provenance"], "synthetic_transfer")
        self.assertIn(self.src_dir.name, meta["sources"])

    def test_random_control_has_same_norm_budget(self):
        r = self.run_op(f"tensors=*down_proj* source={self.src_dir.as_posix()} basis=random k=4 seed=0 "
                        f"alpha=1.0 out={self.out.as_posix()}")
        self.assertEqual(r["status"], "ok", r)
        self.assertIn("delta_norm_applied", r["result"])

    def test_protected_tensor_needs_unlock(self):
        r = self.run_op(f"tensors=*embed* source={self.src_dir.as_posix()} basis=sig:{self.sig.as_posix()} "
                        f"k=4 alpha=1.0 out={self.out.as_posix()}")
        self.assertNotEqual(r["status"], "ok")

    def test_needs_destination_and_leaves_input_alone(self):
        before = {f.name: f.read_bytes() for f in self.base_dir.glob("*.safetensors")}
        r = self.run_op(f"tensors=*down_proj* source={self.src_dir.as_posix()} basis=sig:{self.sig.as_posix()} k=4")
        self.assertEqual(r["status"], "error")
        self.assertEqual(before, {f.name: f.read_bytes() for f in self.base_dir.glob("*.safetensors")})

    def test_budget_is_enforced(self):
        r = self.run_op(f"tensors=*down_proj* source={self.src_dir.as_posix()} basis=sig:{self.sig.as_posix()} "
                        f"k=4 alpha=1.0 out={self.out.as_posix()}",
                        budgets="allow_write = true\n    max_modified_weights = 10")
        self.assertNotEqual(r["status"], "ok")


if __name__ == "__main__":
    unittest.main()
