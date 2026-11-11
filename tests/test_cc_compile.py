"""engine/cc_compile.py and the WSP-Lang op compile_control_vector (lab/probes/cc_weights/OP_PLAN.md): a llama.cpp
control vector written into mlp.down_proj as a rank-1 closed-form edit per layer, with a same-norm placebo, the G0 gate
on held-out tokens, a PEFT LoRA and merged Q6_K exports, and a report. Tiny random models on the CPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import cc_compile as cc  # noqa: E402
from engine import contrast_vector as cvx  # noqa: E402
import engine.wsplang as wsplang  # noqa: E402

M08 = ROOT / "hf_cache" / "qwen35-08b-hf"


def tiny_model(seed=0, layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def tiny_ids(n, seed=1, lo=6, hi=14):
    g = np.random.default_rng(seed)
    return [g.integers(0, 64, size=int(g.integers(lo, hi))).tolist() for _ in range(n)]


def tiny_vecs(layers=(1, 2), d=16, scale=0.3, seed=2):
    g = torch.Generator().manual_seed(seed)
    return {l: scale * torch.randn(d, generator=g) for l in layers}


def run(steps):
    body = "\n".join(f"    step {s}" for s in steps)
    prog = wsplang.parse_args_lines(f"experiment t:\nsteps:\n{body}\n")
    v, errors = wsplang.validate(prog)
    assert not errors, errors
    return wsplang.execute_plan_v4(wsplang.plan_run(prog, v), prog)


class TestRidge(unittest.TestCase):
    def test_exact_solution_gives_ones_and_ridge_shrinks_u(self):
        g = torch.Generator().manual_seed(0)
        K = torch.randn(200, 12, generator=g)
        u0 = torch.randn(12, generator=g)
        K = K - (K @ u0 - 1)[:, None] * u0 / (u0 @ u0)        # every row now has u0^T k = 1 exactly
        G, P = K.T @ K, K.sum(0)[:, None]
        u = cc.ridge_u(G, P, 1e-9)
        self.assertLess(float((K @ u - 1).abs().max()), 1e-4)
        self.assertLess(float(cc.ridge_u(G, P, 10.0).norm()), float(u.norm()))


class TestFit(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.fit_ids, self.ho_ids = tiny_ids(30), tiny_ids(8, seed=3)
        self.vecs = tiny_vecs()

    def test_fit_reaches_g0_and_reports_every_edited_layer(self):
        U, summ = cc.fit_layers(self.model, self.fit_ids, self.ho_ids, self.vecs, lam=1e-3, device="cpu")
        self.assertEqual(sorted(U), [1, 2])
        self.assertEqual(sorted(summ), [1, 2])
        for l in (1, 2):
            self.assertEqual(tuple(U[l].shape), (32,))
            self.assertLess(summ[l]["train_median_abs_err"], 0.5)
            for k in ("heldout_median_abs_err", "rel_norm_dW", "v_norm", "u_norm"):
                self.assertIn(k, summ[l])
        self.assertTrue(cc.g0(summ)["pass"])

    def test_fit_does_not_change_the_model(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        cc.fit_layers(self.model, self.fit_ids, self.ho_ids, self.vecs, lam=1e-3, device="cpu")
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_hook_edit_equals_weight_edit(self):
        """Two methods: adding v (u^T x) at the down_proj output == down_proj.weight += v u^T."""
        U, _ = cc.fit_layers(self.model, self.fit_ids, self.ho_ids, self.vecs, lam=1e-3, device="cpu")
        ids = torch.tensor([self.ho_ids[0]])
        hooks = []
        for l, u in U.items():
            v = self.vecs[l]
            hooks.append(cc.layers_of(self.model)[l].mlp.down_proj.register_forward_hook(
                lambda m, i, o, u=u, v=v: o + (i[0].float() @ u)[..., None] * v))
        with torch.no_grad():
            a = self.model(ids).logits
        for h in hooks:
            h.remove()
        m2 = tiny_model()
        with torch.no_grad():
            for l, u in U.items():
                cc.layers_of(m2)[l].mlp.down_proj.weight += torch.outer(self.vecs[l], u)
            b = m2(ids).logits
        self.assertTrue(torch.allclose(a, b, atol=1e-5), float((a - b).abs().max()))

    def test_later_layer_is_fitted_on_the_edited_stream(self):
        U_both, _ = cc.fit_layers(self.model, self.fit_ids, self.ho_ids, self.vecs, lam=1e-3, device="cpu")
        U_last, _ = cc.fit_layers(self.model, self.fit_ids, self.ho_ids, {2: self.vecs[2]}, lam=1e-3, device="cpu")
        self.assertFalse(torch.allclose(U_both[2], U_last[2], atol=1e-6))

    def test_vector_must_match_the_model(self):
        with self.assertRaises(ValueError):
            cc.fit_layers(self.model, self.fit_ids, self.ho_ids, {7: torch.ones(16)}, device="cpu")
        with self.assertRaises(ValueError):
            cc.fit_layers(self.model, self.fit_ids, self.ho_ids, {1: torch.ones(9)}, device="cpu")


class TestG0(unittest.TestCase):
    def test_median_of_medians_rule(self):
        s = lambda *e: {i: {"heldout_median_abs_err": x} for i, x in enumerate(e)}
        self.assertEqual(cc.g0(s(0.1, 0.9, 0.3))["median"], 0.3)
        self.assertEqual(cc.g0(s(0.1, 0.9, 0.3, 0.5))["median"], 0.4)
        self.assertEqual(cc.g0(s(0.1, 0.9, 0.3))["worst"], 0.9)
        self.assertTrue(cc.g0(s(0.5))["pass"])
        self.assertFalse(cc.g0(s(0.51))["pass"])


class TestSplitAndLora(unittest.TestCase):
    def test_split_is_seeded_and_disjoint(self):
        rows = list(range(20))
        a = cc.split_rows(rows, heldout=5, seed=11)
        self.assertEqual(a, cc.split_rows(rows, heldout=5, seed=11))
        fit, ho = a
        self.assertEqual((len(fit), len(ho)), (15, 5))
        self.assertEqual(sorted(fit + ho), rows)
        with self.assertRaises(ValueError):
            cc.split_rows(rows, heldout=20, seed=11)

    def test_lora_tensors_are_peft_names_and_rebuild_the_edit(self):
        model = tiny_model()
        U = {1: torch.randn(32), 2: torch.randn(32)}
        V = tiny_vecs()
        T, cfg = cc.lora_tensors(model, U, V)
        name = cc.down_proj_name(model, 1)
        self.assertEqual(name, "model.layers.1.mlp.down_proj")
        A, B = T[f"base_model.model.{name}.lora_A.weight"], T[f"base_model.model.{name}.lora_B.weight"]
        self.assertEqual((tuple(A.shape), tuple(B.shape)), ((1, 32), (16, 1)))
        dW = torch.outer(V[1], U[1])
        self.assertLess(float(((B.float() @ A.float()) - dW).norm() / dW.norm()), 0.01)   # bf16 storage
        self.assertEqual(len(T), 4)
        self.assertEqual((cfg["r"], cfg["lora_alpha"], cfg["target_modules"]), (1, 1, ["down_proj"]))


def write_q6k_gguf(path, n_out=8, n_in=512, seed=0):
    """A tiny GGUF with one Q6_K ffn_down (blk.1) and an f32 tensor beside it, quantized by wsp.py's port."""
    import gguf
    import wsp
    g = np.random.default_rng(seed)
    W = (0.05 * g.standard_normal((n_out, n_in))).astype(np.float32)
    q = wsp._q6k_quant_blocks(W.reshape(-1, 256)).reshape(n_out, -1)
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_tensor("blk.1.ffn_down.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
    w.add_tensor("blk.1.ffn_norm.weight", np.ones(n_in, np.float32))
    w.add_tensor("blk.2.ffn_down.weight", np.ones((n_out, n_in), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tensors(path):
    import gc
    import gguf
    r = gguf.GGUFReader(str(path))
    out = {t.name: (t.tensor_type, np.array(t.data).copy()) for t in r.tensors}
    del r
    gc.collect()
    return out


class TestMerge(unittest.TestCase):
    def test_merge_writes_the_direction_and_leaves_the_rest(self):
        import gguf
        g = torch.Generator().manual_seed(4)
        with tempfile.TemporaryDirectory() as t:
            base, dst = Path(t) / "base.gguf", Path(t) / "m.gguf"
            write_q6k_gguf(base)
            U, V = {1: 0.05 * torch.randn(512, generator=g)}, {1: 0.05 * torch.randn(8, generator=g)}
            stats = cc.merge_q6k(base, U, V, dst)
            self.assertTrue(dst.exists())
            self.assertFalse(Path(str(dst) + ".partial").exists())
            self.assertGreater(stats[1]["cos_written_dW"], 0.8)
            a, b = tensors(base), tensors(dst)
            self.assertTrue(np.array_equal(a["blk.1.ffn_norm.weight"][1], b["blk.1.ffn_norm.weight"][1]))
            self.assertFalse(np.array_equal(a["blk.1.ffn_down.weight"][1], b["blk.1.ffn_down.weight"][1]))
            W0 = gguf.dequantize(a["blk.1.ffn_down.weight"][1], gguf.GGMLQuantizationType.Q6_K)
            W1 = gguf.dequantize(b["blk.1.ffn_down.weight"][1], gguf.GGMLQuantizationType.Q6_K)
            dW = np.outer(V[1].numpy(), U[1].numpy())
            got = W1 - W0
            self.assertGreater(float((got * dW).sum() / (np.linalg.norm(got) * np.linalg.norm(dW))), 0.8)

    def test_merge_refuses_other_quant_types_and_existing_output(self):
        with tempfile.TemporaryDirectory() as t:
            base, dst = Path(t) / "base.gguf", Path(t) / "m.gguf"
            write_q6k_gguf(base)
            with self.assertRaises(ValueError):
                cc.merge_q6k(base, {2: torch.ones(512)}, {2: torch.ones(8)}, dst)
            self.assertFalse(dst.exists())
            dst.write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                cc.merge_q6k(base, {1: torch.ones(512)}, {1: torch.ones(8)}, dst)


class TestPipeline(unittest.TestCase):
    """compile_from_ids: fit vector -> G0 -> (stop | placebo fit, LoRA, merges) -> report."""

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.model = tiny_model()
        self.fit_ids, self.ho_ids = tiny_ids(30), tiny_ids(8, seed=3)

    def test_passing_vector_writes_both_arms_and_a_report(self):
        vecs = tiny_vecs()
        out = self.dir / "out"
        res = cc.compile_from_ids(self.model, self.fit_ids, self.ho_ids, vecs, out, placebo=None, placebo_seed=5,
                                  device="cpu")
        self.assertTrue(res["G0"]["pass"])
        for arm in ("vector", "placebo"):
            self.assertTrue((out / arm / "fit.pt").exists(), arm)
            self.assertTrue((out / arm / "peft" / "adapter_model.safetensors").exists(), arm)
        pl = torch.load(out / "placebo" / "fit.pt")["v"]
        for l in vecs:                                                   # same norm, other direction
            self.assertAlmostEqual(float(pl[l].norm()), float(vecs[l].norm()), places=4)
        self.assertIn("G0", res["placebo"])
        rep = json.loads((out / "report.json").read_text())
        self.assertEqual(rep["G0"]["pass"], True)
        self.assertIn("not run by this op", rep["note"])
        self.assertTrue((out / "report.md").exists())

    def test_failing_g0_writes_only_the_negative_report(self):
        vecs = tiny_vecs()
        out = self.dir / "out"
        res = cc.compile_from_ids(self.model, self.fit_ids, self.ho_ids, vecs, out, lam=1e6, device="cpu")
        self.assertFalse(res["G0"]["pass"])
        self.assertFalse((out / "vector" / "peft").exists())
        self.assertFalse((out / "placebo").exists())
        self.assertFalse(json.loads((out / "report.json").read_text())["G0"]["pass"])

    def test_existing_out_is_refused(self):
        out = self.dir / "out"
        out.mkdir()
        with self.assertRaises(wsplang.WriteGuardError):
            cc.compile_from_ids(self.model, self.fit_ids, self.ho_ids, tiny_vecs(), out, device="cpu")


class TestOp(unittest.TestCase):
    def test_op_is_registered_as_writing_new_files_only(self):
        self.assertIn("compile_control_vector", wsplang.OPS_ALLOWED)
        self.assertIn("compile_control_vector", wsplang.READ_ONLY_OPS)

    def test_missing_args_and_guards(self):
        with tempfile.TemporaryDirectory() as t:
            r = run([f"compile_control_vector model={t} out={t}/o"])[0]
            self.assertEqual(r["status"], "error", r)
            self.assertIn("compile_control_vector needs", json.dumps(r))
            vec = Path(t) / "v.gguf"
            cvx.write_gguf({1: np.ones(4, np.float32)}, vec, "llama")
            calib = Path(t) / "c.jsonl"
            calib.write_text('{"prompt": "a", "reasoning": "b"}\n')
            r = run([f"compile_control_vector model={t} vector={vec.as_posix()} calib={calib.as_posix()} "
                     f"out={t}/inside"])[0]
            self.assertEqual(r["status"], "error", r)
            self.assertIn("inside the input", json.dumps(r))
            self.assertFalse((Path(t) / "inside").exists())


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestOpOnQwen35(unittest.TestCase):
    """End to end through a .wspl step on the real 0.8B (CPU, two short traces per split, two edited layers)."""

    def test_wspl_step_runs_and_reports(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            from transformers import AutoConfig
            d = AutoConfig.from_pretrained(str(M08)).get_text_config().hidden_size
            g = np.random.default_rng(0)
            vec = t / "v.gguf"
            cvx.write_gguf({3: 0.2 * g.standard_normal(d).astype(np.float32),
                            5: 0.2 * g.standard_normal(d).astype(np.float32)}, vec, "qwen35")
            rows = [{"prompt": f"What is {i} + {i}?", "reasoning": f"I add {i} and {i}. That gives {2 * i}. Done."}
                    for i in range(6)]
            calib = t / "c.jsonl"
            calib.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            out = t / "out"
            r = run([f"compile_control_vector model={M08.as_posix()} vector={vec.as_posix()} "
                     f"calib={calib.as_posix()} out={out.as_posix()} heldout=2 device=cpu lora=0"])[0]
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(r["execution_state"], "executed")
            res = r["result"]
            self.assertEqual(sorted(int(k) for k in res["vector"]["layers"]), [3, 5])
            self.assertEqual((res["calib"]["fit"], res["calib"]["heldout"]), (4, 2))
            self.assertEqual(len(res["inputs"]["vector_sha256"]), 64)
            self.assertTrue((out / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
