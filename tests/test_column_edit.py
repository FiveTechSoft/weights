"""engine/column_edit.py (lab/probes/column_edit/PLAN.md): the WSP-Lang op edit_column / edit_neurons. One matrix,
one or a few columns (neurons), modes add / subtract / scale / replace (+ subtract_direction, the Exp 8 closed form),
a structural same-norm placebo, a report, a budget guard, and the ediciones program + npz that wsp.py's
add-cols-vectors applies to the GGUF. Written before the code. Orientation as HF: W is (out, in); columns = neurons."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import column_edit as ce  # noqa: E402
from engine import ternary as tn  # noqa: E402


def W(seed=0, shape=(6, 10)):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


class PlanEditTest(unittest.TestCase):
    def test_add_and_subtract_one_direction_normalised_by_default(self):
        W0 = W()
        u = np.array([3.0, 0, 0, 0, 0, 4.0], dtype=np.float32)          # norm 5
        dW = ce.plan_edit(W0, [2, 7], u, alpha=0.5, mode="add")
        self.assertEqual(dW.shape, W0.shape)
        np.testing.assert_allclose(dW[:, 2], 0.5 * u / 5.0)
        np.testing.assert_allclose(dW[:, 7], 0.5 * u / 5.0)
        self.assertEqual(int((dW != 0).sum()), 4)
        dS = ce.plan_edit(W0, [2, 7], u, alpha=0.5, mode="subtract")
        np.testing.assert_allclose(dS, -dW)
        raw = ce.plan_edit(W0, [2], u, alpha=0.5, mode="add", normalize_direction=False)
        np.testing.assert_allclose(raw[:, 2], 0.5 * u)

    def test_add_per_column_vectors_are_never_normalised(self):
        W0 = W()
        V = np.arange(12, dtype=np.float32).reshape(6, 2)
        dW = ce.plan_edit(W0, [1, 4], V, alpha=1.0, mode="add")
        np.testing.assert_allclose(dW[:, [1, 4]], V)
        self.assertTrue(np.all(dW[:, [0, 2, 3, 5, 6, 7, 8, 9]] == 0))

    def test_scale_and_replace(self):
        W0 = W()
        dW = ce.plan_edit(W0, [3], None, alpha=0.25, mode="scale")
        np.testing.assert_allclose(W0[:, 3] + dW[:, 3], 0.25 * W0[:, 3], atol=1e-6)
        self.assertTrue(np.all(np.delete(dW, 3, axis=1) == 0))
        u = np.array([0, 0, 2.0, 0, 0, 0], dtype=np.float32)
        dR = ce.plan_edit(W0, [8], u, alpha=1.5, mode="replace")
        np.testing.assert_allclose(W0[:, 8] + dR[:, 8], 1.5 * u / 2.0, atol=1e-6)   # replace by alpha * unit direction

    def test_subtract_direction_is_exp8_closed_form(self):
        W0 = W()
        u = np.array([3.0, 4.0, 0, 0, 0, 0], dtype=np.float32)
        dW = ce.plan_edit(W0, [0, 5], u, alpha=6.0, mode="subtract_direction")
        un = u / 5.0
        for c in (0, 5):
            np.testing.assert_allclose(dW[:, c], -6.0 * un * float(un @ W0[:, c]), atol=1e-6)
        self.assertTrue(np.all(dW[:, 1:5] == 0))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ce.plan_edit(W(), [0], np.ones(6, dtype=np.float32), 1.0, "multiply")
        with self.assertRaises(ValueError):
            ce.plan_edit(W(), [0, 1], np.ones((6, 3), dtype=np.float32), 1.0, "add")
        with self.assertRaises(ValueError):
            ce.plan_edit(W(), [10], np.ones(6, dtype=np.float32), 1.0, "add")       # column out of range
        with self.assertRaises(ValueError):
            ce.plan_edit(W(), [0], None, 1.0, "add")                                   # add needs a direction


class PlaceboTest(unittest.TestCase):
    def test_same_columns_same_norms_same_params_touched_orthogonal(self):
        W0 = W(1, (40, 12))
        u = np.random.default_rng(5).normal(size=40).astype(np.float32)
        dW = ce.plan_edit(W0, [2, 9, 11], u, alpha=0.7, mode="add")
        P = ce.structural_placebo(dW, [2, 9, 11], u, seed=2026)
        self.assertEqual(P.shape, dW.shape)
        self.assertTrue(np.all(P[:, [0, 1, 3, 4, 5, 6, 7, 8, 10]] == 0))
        self.assertEqual(int((P != 0).sum()), int((dW != 0).sum()))
        for c in (2, 9, 11):
            self.assertAlmostEqual(float(np.linalg.norm(P[:, c])), float(np.linalg.norm(dW[:, c])), places=5)
            self.assertLess(abs(float(P[:, c] @ u)) / (np.linalg.norm(u) * np.linalg.norm(P[:, c])), 1e-5)
        np.testing.assert_allclose(P, ce.structural_placebo(dW, [2, 9, 11], u, seed=2026))
        self.assertFalse(np.allclose(P, ce.structural_placebo(dW, [2, 9, 11], u, seed=1)))

    def test_placebo_from_realised_delta_for_scale_and_per_column_directions(self):
        W0 = W(2, (30, 8))
        dS = ce.plan_edit(W0, [4], None, alpha=0.5, mode="scale")      # realised delta = -0.5 * W0[:, 4]
        P = ce.structural_placebo(dS, [4], None, seed=3)
        self.assertAlmostEqual(float(np.linalg.norm(P[:, 4])), float(np.linalg.norm(dS[:, 4])), places=5)
        self.assertLess(abs(float(P[:, 4] @ dS[:, 4])) / (np.linalg.norm(dS[:, 4]) ** 2), 1e-5)
        V = np.random.default_rng(3).normal(size=(30, 2)).astype(np.float32)
        dW = ce.plan_edit(W0, [0, 7], V, 1.0, "add")
        Q = ce.structural_placebo(dW, [0, 7], V, seed=7)
        for j, c in enumerate((0, 7)):
            self.assertAlmostEqual(float(np.linalg.norm(Q[:, c])), float(np.linalg.norm(V[:, j])), places=5)
            self.assertLess(abs(float(Q[:, c] @ V[:, j])) / (np.linalg.norm(V[:, j]) ** 2), 1e-5)


class ReportAndGuardTest(unittest.TestCase):
    def test_report_fields(self):
        W0 = np.ones((4, 5), dtype=np.float32)
        u = np.array([1.0, 0, 0, 0], dtype=np.float32)
        dW = ce.plan_edit(W0, [1, 3], u, alpha=2.0, mode="add")
        P = ce.structural_placebo(dW, [1, 3], u, seed=42)
        r = ce.report(W0, dW, direction=u, cols=[1, 3], alpha=2.0, mode="add", tensor="t", placebo_dW=P,
                      placebo_seed=42, budget_params=2000)
        self.assertEqual(r["op"], "edit_column")
        self.assertEqual(r["tensor"], "t")
        self.assertEqual(r["column"], [1, 3])
        self.assertEqual(r["params_touched"], 2)
        self.assertAlmostEqual(r["delta_norm"], np.sqrt(8), places=6)
        self.assertAlmostEqual(r["rel_delta_norm"], np.sqrt(8) / np.sqrt(20), places=6)
        self.assertAlmostEqual(r["cosine_with_direction"], 1.0, places=6)
        self.assertEqual(r["alpha"], 2.0)
        self.assertEqual(r["mode"], "add")
        self.assertEqual(r["placebo_seed"], 42)
        self.assertAlmostEqual(r["placebo_delta_norm"], np.sqrt(8), places=5)
        self.assertEqual(r["placebo_params_touched"], 2)
        self.assertTrue(r["budget_params_ok"])
        self.assertEqual(r["col_norms"], {"1": 2.0, "3": 2.0})

    def test_budget_guard(self):
        W0 = W(0, (100, 20))
        dW = ce.plan_edit(W0, [0, 1, 2], np.ones(100, dtype=np.float32), 1.0, "add")   # 300 params
        ce.check_budget(dW, 300)
        with self.assertRaises(ce.BudgetError):
            ce.check_budget(dW, 299)

    def test_request_validation(self):
        ce.validate_request({"placebo": "true", "evaluate": "true"})
        ce.validate_request({"placebo": "false"})
        with self.assertRaises(ValueError):
            ce.validate_request({"placebo": "false", "evaluate": "true"})     # behaviour claim needs a placebo arm


class ProgramTest(unittest.TestCase):
    def test_program_and_npz_round_trip(self):
        W0 = W(4, (8, 16))
        u = np.random.default_rng(9).normal(size=8).astype(np.float32)
        dW = ce.plan_edit(W0, [3, 12], u, 1.5, "add")
        P = ce.structural_placebo(dW, [3, 12], u, seed=11)
        with tempfile.TemporaryDirectory() as d:
            out = ce.write_program(Path(d), "demo", [{"gguf_tensor": "blk.15.ffn_down.weight", "cols": [3, 12], "dW": dW}],
                                   placebos=[{"gguf_tensor": "blk.15.ffn_down.weight", "cols": [3, 12], "dW": P}],
                                   gguf_out=Path(d) / "demo.gguf", gguf_out_placebo=Path(d) / "demo_placebo.gguf")
            prog = json.loads(Path(out["program"]).read_text(encoding="utf-8"))
            self.assertEqual(prog["ops"][0]["type"], "add-cols-vectors")
            self.assertEqual(prog["ops"][0]["tensor"], "blk.15.ffn_down.weight")
            self.assertEqual((prog["ops"][0]["key"], prog["ops"][0]["cols_key"]), ("vec_0", "cols_0"))
            self.assertEqual(prog["out"], str(Path(d) / "demo.gguf"))
            with np.load(prog["ops"][0]["vec_file"]) as z:
                np.testing.assert_allclose(tn.delta_from_npz(z, 0, W0.shape), dW)
            progp = json.loads(Path(out["program_placebo"]).read_text(encoding="utf-8"))
            with np.load(progp["ops"][0]["vec_file"]) as z:
                np.testing.assert_allclose(tn.delta_from_npz(z, 0, W0.shape), P)
            self.assertTrue(Path(out["npz"]).exists() and Path(out["npz_placebo"]).exists())

    def test_gguf_name_mapping(self):
        self.assertEqual(ce.gguf_name("model.language_model.layers.15.mlp.down_proj.weight"), "blk.15.ffn_down.weight")
        self.assertEqual(ce.gguf_name("model.layers.3.mlp.up_proj.weight"), "blk.3.ffn_up.weight")
        self.assertEqual(ce.gguf_name("model.layers.3.self_attn.o_proj.weight"), "blk.3.attn_output.weight")
        with self.assertRaises(ValueError):
            ce.gguf_name("model.layers.3.input_layernorm.weight")


class HfApplyTest(unittest.TestCase):
    def test_edit_column_on_tiny_safetensors_is_bit_identical_to_numpy_reference(self):
        import torch
        from safetensors.torch import save_file
        from engine import wsplpatch as wp
        name = "model.language_model.layers.0.mlp.down_proj.weight"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); (d / "m").mkdir()
            W0 = torch.randn(8, 16, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
            save_file({name: W0}, str(d / "m" / "model.safetensors"), metadata={"format": "pt"})
            np.save(d / "u.npy", np.random.default_rng(1).normal(size=8).astype(np.float32))
            res = ce.run_edit_column(model=d / "m", tensor=name, cols=[5], direction=d / "u.npy", alpha=0.3, mode="add",
                                     placebo=True, placebo_seed=42, out=d / "out", budget_params=2000)
            # new dirs, input untouched
            self.assertTrue((d / "out" / "model.safetensors").exists())
            self.assertTrue((d / "out_placebo" / "model.safetensors").exists())
            self.assertTrue(torch.equal(wp.read_tensor(d / "m", name), W0))
            # bit-identical to the NumPy reference: (W0 + dW) rounded to bf16
            u = np.load(d / "u.npy"); u = u / np.linalg.norm(u)
            ref = W0.float().numpy().copy(); ref[:, 5] += 0.3 * u
            ref_bf16 = torch.from_numpy(ref).to(torch.bfloat16)
            self.assertTrue(torch.equal(wp.read_tensor(d / "out", name), ref_bf16))
            # report + manifest + patches for both arms, same params_touched
            rep = json.loads((d / "out" / "edit_column.report.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["params_touched"], 8)
            self.assertEqual(rep["placebo_params_touched"], 8)
            self.assertTrue(rep["budget_params_ok"])
            man = json.loads((d / "out" / "edit_column.manifest.json").read_text(encoding="utf-8"))
            self.assertIn("edit", man["tensor_hashes"]); self.assertIn("placebo", man["tensor_hashes"])
            self.assertNotEqual(man["tensor_hashes"]["edit"], man["tensor_hashes"]["placebo"])
            self.assertTrue(Path(res["patch"]).exists() and Path(res["patch_placebo"]).exists())
            self.assertEqual(json.loads(Path(res["patch"]).read_text(encoding="utf-8"))["wsplang_patch_version"], "1.0")
            # budget guard aborts before writing
            with self.assertRaises(ce.BudgetError):
                ce.run_edit_column(model=d / "m", tensor=name, cols=[5], direction=d / "u.npy", alpha=0.3, mode="add",
                                   placebo=False, out=d / "out2", budget_params=7)
            self.assertFalse((d / "out2").exists())


if __name__ == "__main__":
    unittest.main()
