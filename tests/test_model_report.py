"""engine.model_report: a fast fingerprint of how a model differs from its base, from blocks of rows.

Synthetic models with an exact, known recipe: a full fine-tune, a LoRA on attention + MLP, a directional
ablation (one global direction, residual-writing matrices, layers 2-5) and an unchanged copy. The classifier must
name each recipe and its details.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.model_report as mr  # noqa: E402
import engine.weight_diff as wd  # noqa: E402

D, I, L = 48, 96, 8   # hidden, MLP width, layers


def base_model(g):
    t = {"model.embed_tokens.weight": torch.randn(200, D, generator=g),
         "lm_head.weight": torch.randn(200, D, generator=g), "model.norm.weight": torch.ones(D)}
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(D)
        t[p + "mlp.gate_proj.weight"] = torch.randn(I, D, generator=g)
        t[p + "mlp.up_proj.weight"] = torch.randn(I, D, generator=g)
        t[p + "mlp.down_proj.weight"] = torch.randn(D, I, generator=g)
        if i % 4 == 3:
            for m in ("q", "k", "v"):
                t[p + f"self_attn.{m}_proj.weight"] = torch.randn(D, D, generator=g)
            t[p + "self_attn.o_proj.weight"] = torch.randn(D, D, generator=g)
        else:
            t[p + "linear_attn.in_proj_qkv.weight"] = torch.randn(3 * D, D, generator=g)
            t[p + "linear_attn.in_proj_b.weight"] = torch.randn(8, D, generator=g)
            t[p + "linear_attn.out_proj.weight"] = torch.randn(D, D, generator=g)
            t[p + "linear_attn.A_log"] = torch.randn(8, generator=g).float()
    return t


def recipe(name, base, g):
    t = {k: v.clone() for k, v in base.items()}
    if name == "full":
        for k, v in t.items():
            if v.dim() == 2 or "layernorm" in k:
                t[k] = v + 0.05 * v.abs().mean() * torch.randn(v.shape, generator=g)
    elif name == "lora":
        for k, v in t.items():
            if v.dim() == 2 and ("self_attn" in k or ".mlp." in k):
                B, A = torch.randn(v.shape[0], 4, generator=g), torch.randn(4, v.shape[1], generator=g)
                t[k] = v + 0.02 * B @ A
    elif name in ("ablation", "ablation_per_layer"):
        # ablation: one global direction (classic abliteration); per layer: a direction for each layer
        # (Heretic ARA, found in validation on xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic).
        r_global = torch.randn(D, generator=g)
        for k, v in t.items():
            layer = wd.layer_of(k)
            if layer is not None and 2 <= layer <= 5 and k.endswith(("o_proj.weight", "out_proj.weight",
                                                                      "down_proj.weight")):
                r = r_global if name == "ablation" else torch.randn(D, generator=torch.Generator().manual_seed(layer))
                r = r / r.norm()
                t[k] = v - torch.outer(r, r @ v)
    return t


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        g = torch.Generator().manual_seed(0)
        base = base_model(g)
        cls.paths = {}
        for name in ("base", "full", "lora", "ablation", "ablation_per_layer", "same"):
            t = base if name in ("base", "same") else recipe(name, base, g)
            p = d / name
            p.mkdir()
            out = {k: (v.float() if k.endswith("A_log") else v.to(torch.bfloat16)) for k, v in t.items()}
            if name == "full":   # a different dtype for A_log must not count as a change
                out = {k: (v.to(torch.bfloat16) if k.endswith("A_log") else v) for k, v in out.items()}
            save_file(out, str(p / "model.safetensors"))
            cls.paths[name] = p

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fp(self, name, rows=16):
        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        return mr.fingerprint(reader(self.paths["base"]), reader(self.paths[name]), rows=rows)


class TestClassify(Synthetic):
    def test_full_fine_tune(self):
        c = mr.classify(self.fp("full"))
        self.assertEqual(c["method"], "full_fine_tune", c)
        self.assertIn("linear_attn.A_log", c["untouched"])        # dtype cast only, not a change
        self.assertIn("mlp.gate_proj.weight", c["touched"])

    def test_lora(self):
        c = mr.classify(self.fp("lora"))
        self.assertEqual(c["method"], "lora_like", c)
        self.assertEqual(set(c["touched"]), {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                                             "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                             "self_attn.v_proj.weight", "self_attn.o_proj.weight"})
        self.assertLessEqual(c["rank_estimate"], 4)
        self.assertIn("linear_attn.in_proj_qkv.weight", c["untouched"])

    def test_directional_ablation(self):
        c = mr.classify(self.fp("ablation"))
        self.assertEqual(c["method"], "directional_ablation", c)
        self.assertEqual(c["layers"], [2, 5])
        self.assertEqual(c["rank_estimate"], 1)
        self.assertGreater(c["direction_consistency"], 0.99)
        self.assertTrue(set(c["touched"]) <= {"self_attn.o_proj.weight", "linear_attn.out_proj.weight",
                                              "mlp.down_proj.weight"})
        self.assertEqual(c["ablation_directions"], "global")

    def test_directional_ablation_with_a_direction_per_layer(self):
        c = mr.classify(self.fp("ablation_per_layer"))
        self.assertEqual(c["method"], "directional_ablation", c)
        self.assertEqual(c["layers"], [2, 5])
        self.assertEqual(c["rank_estimate"], 1)
        self.assertEqual(c["ablation_directions"], "per_layer")

    def test_identical(self):
        self.assertEqual(mr.classify(self.fp("same"))["method"], "identical")


class TestParallelReads(Synthetic):
    """Remote reads are latency-bound (validation: 773 s for ~500 MB in series); blocks are fetched concurrently."""

    def test_reads_overlap_and_results_do_not_change(self):
        import threading
        import time

        class Counting:
            def __init__(self, inner):
                self.inner, self.now, self.peak, self.lock = inner, 0, 0, threading.Lock()

            def keys(self):
                return self.inner.keys()

            def shape(self, n):
                return self.inner.shape(n)

            def _wrap(self, f, *a):
                with self.lock:
                    self.now += 1
                    self.peak = max(self.peak, self.now)
                time.sleep(0.01)
                try:
                    return f(*a)
                finally:
                    with self.lock:
                        self.now -= 1

            def get(self, n):
                return self._wrap(self.inner.get, n)

            def get_rows(self, n, s, c):
                return self._wrap(self.inner.get_rows, n, s, c)

        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        base, model = Counting(reader(self.paths["base"])), Counting(reader(self.paths["lora"]))
        fp = mr.fingerprint(base, model, rows=16, workers=8)
        self.assertGreater(max(base.peak, model.peak), 1)
        serial = mr.fingerprint(reader(self.paths["base"]), reader(self.paths["lora"]), rows=16, workers=1)
        strip = lambda f: [{k: v for k, v in r.items() if not k.startswith("_")} for r in f["tensors"]]
        self.assertEqual(strip(fp), strip(serial))


class TestCardAtlasUsable(unittest.TestCase):
    def test_card_flags(self):
        api = {"cardData": {"base_model": "Qwen/Qwen3.5-9B", "license": "other", "license_name": "webai-nc-1.0"},
               "tags": ["lora", "text-generation"]}
        readme = "Made with Heretic using Arbitrary-Rank Ablation. Non-commercial use only. Distilled from X."
        c = mr.card_info("someone/model", fetch_json=lambda url: api, fetch_text=lambda url: readme)
        self.assertEqual(c["base_model"], "Qwen/Qwen3.5-9B")
        self.assertIn("heretic", c["flags"])
        self.assertIn("distillation", c["flags"])
        self.assertIn("non_commercial", c["flags"])

    def test_card_words_need_word_boundaries(self):
        """'properties' contains 'ties', 'prompt' contains 'rl'-free text... flags match whole words only."""
        c = mr.card_info("x/y", fetch_json=lambda u: {"cardData": {}, "tags": []},
                         fetch_text=lambda u: "Model properties and capabilities; world knowledge; unparalleled.")
        self.assertEqual(c["flags"], [])

    def test_atlas_nearest(self):
        prof = {"a.w": {"rel_median": 0.3}, "b.w": {"rel_median": 0.2}, "c.w": {"rel_median": 0.1},
                "d.w": {"rel_median": 0.0}}
        atlas = {"m1": {"family": "agent", "profile": {"a.w": 3.0, "b.w": 2.0, "c.w": 1.0, "d.w": 0.5}},
                 "m2": {"family": "control", "profile": {"a.w": 0.1, "b.w": 0.2, "c.w": 0.3, "d.w": 0.4}}}
        near = mr.atlas_similarity(prof, atlas)
        self.assertEqual(near[0]["model"], "m1")
        self.assertAlmostEqual(near[0]["spearman"], 1.0)
        self.assertEqual(near[-1]["model"], "m2")

    def test_usable_mentions_method_specific_advice(self):
        ab = mr.usable({"method": "directional_ablation", "layers": [9, 23], "ablation_directions": "per_layer",
                        "rank_estimate": 1, "touched": ["mlp.down_proj.weight"]}, {"flags": ["non_commercial"]}, [])
        text = " ".join(ab)
        self.assertIn("project_out", text)
        self.assertIn("non-commercial", text)
        ft = mr.usable({"method": "full_fine_tune", "touched": ["mlp.gate_proj.weight"]}, {}, [])
        self.assertIn("subspace_transfer", " ".join(ft))


class TestWSPLangOp(Synthetic):
    def test_model_report_step(self):
        import engine.wsplang as wsplang
        out = Path(self.tmp.name) / "report_lora.md"
        code = (f"experiment r:\nsteps:\n    step model_report model={self.paths['lora'].as_posix()} "
                f"base={self.paths['base'].as_posix()} rows=16 out={out.as_posix()}\n")
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertFalse(errors, errors)
        feas = wsplang.check_feasibility(prog, steps)
        self.assertFalse([e for e in feas[0] if "NO implementada" in e], feas)
        r = wsplang.execute_plan_v4(wsplang.plan_run(prog, steps), prog)[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["result"]["method"], "lora_like")
        self.assertIn("lora_like", out.read_text(encoding="utf-8"))


class TestReport(Synthetic):
    def test_markdown_report(self):
        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        rep = mr.build_report(reader(self.paths["base"]), reader(self.paths["ablation"]), rows=16,
                              model_name="ablation", base_name="base")
        md = mr.render_markdown(rep)
        self.assertIn("directional_ablation", md)
        self.assertIn("layers 2-5", md)
        self.assertEqual(rep["headers"]["shape_mismatch"], [])


if __name__ == "__main__":
    unittest.main()
