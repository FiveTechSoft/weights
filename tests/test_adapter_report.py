"""engine.adapter_report: what a PEFT LoRA adapter changes, read from the adapter alone (plus base row blocks).

Synthetic fixtures with an exact recipe: a tiny base (2 layers), an adapter with r = 2 on q_proj and down_proj whose
A and B are known, so every number (delta norm, rank, block ratio, parameter count) has a closed form.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.adapter_report as ar  # noqa: E402

D, I, L, R = 16, 32, 2, 2
PREFIX = "base_model.model."


def base_tensors(g):
    t = {"model.embed_tokens.weight": torch.randn(50, D, generator=g),
         "lm_head.weight": torch.randn(50, D, generator=g), "model.norm.weight": torch.ones(D)}
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(D)
        t[p + "self_attn.q_proj.weight"] = torch.randn(D, D, generator=g)
        t[p + "self_attn.o_proj.weight"] = torch.randn(D, D, generator=g)
        t[p + "mlp.up_proj.weight"] = torch.randn(I, D, generator=g)
        t[p + "mlp.down_proj.weight"] = torch.randn(D, I, generator=g)
    return t


def adapter_tensors(g, same_q_direction=True):
    """Layer 0 with the PEFT prefix, layer 1 without it (both styles seen in the wild).
    q_proj: B shared across layers when same_q_direction (one output-side direction for all layers), A with two
    orthonormal rows weighted 2 and 1 (singular values 2:1 -> leading direction is B's first column, rank90 = 2
    because the first direction holds 80 % of the energy).
    down_proj of layer 1: A rank one (second row zero) -> rank90 = 1."""
    t = {}
    Bq = torch.linalg.qr(torch.randn(D, R, generator=g))[0]
    for i in range(L):
        stem = (PREFIX if i == 0 else "") + f"model.layers.{i}."
        Aq = (torch.diag(torch.tensor([2.0, 1.0])) @ torch.linalg.qr(torch.randn(D, R, generator=g))[0].T).contiguous()  # (R, D)
        t[stem + "self_attn.q_proj.lora_A.weight"] = Aq
        t[stem + "self_attn.q_proj.lora_B.weight"] = (Bq if same_q_direction else torch.randn(D, R, generator=g)).contiguous()
        Ad = torch.randn(R, I, generator=g)
        if i == 1:
            Ad[1] = 0
        t[stem + "mlp.down_proj.lora_A.weight"] = Ad
        t[stem + "mlp.down_proj.lora_B.weight"] = torch.randn(D, R, generator=g)
    return t


def write_adapter(d: Path, tensors, alpha=4, rslora=False, peft_type="LORA"):
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    cfg = {"peft_type": peft_type, "r": R, "lora_alpha": alpha, "use_rslora": rslora,
           "target_modules": ["q_proj", "down_proj"], "base_model_name_or_path": "org/tiny-base",
           "task_type": "CAUSAL_LM"}
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return d


class TestWSPLangOp(unittest.TestCase):
    """`step adapter_report adapter=<dir> base=<dir> out=<md>` through the runtime: registered, read-only, writes md + json."""

    def test_adapter_report_step(self):
        import engine.wsplang as wsplang
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            g = torch.Generator().manual_seed(7)
            base_dir = root / "base"
            base_dir.mkdir()
            save_file(base_tensors(g), str(base_dir / "model.safetensors"))
            ad_dir = write_adapter(root / "adapter", adapter_tensors(g), alpha=4)
            out = root / "rep" / "adapter.md"
            code = (f"experiment r:\nsteps:\n    step adapter_report adapter={ad_dir.as_posix()} "
                    f"base={base_dir.as_posix()} rows=8 out={out.as_posix()}\n")
            prog = wsplang.parse_args_lines(code)
            steps, errors = wsplang.validate(prog)
            self.assertFalse(errors, errors)
            feas = wsplang.check_feasibility(prog, steps)
            self.assertFalse([e for e in feas[0] if "NO implementada" in e], feas)
            r = wsplang.execute_plan_v4(wsplang.plan_run(prog, steps), prog)[0]
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(r["result"]["modules"], 4)
            self.assertEqual(r["result"]["r"], R)
            self.assertTrue(out.exists() and out.with_suffix(".json").exists())
            self.assertIn("adapter_report", wsplang.READ_ONLY_OPS)


class AdapterReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        g = torch.Generator().manual_seed(7)
        cls.base = base_tensors(g)
        cls.base_dir = root / "base"
        cls.base_dir.mkdir()
        save_file(cls.base, str(cls.base_dir / "model.safetensors"))
        cls.ad_t = adapter_tensors(g)
        cls.ad_dir = write_adapter(root / "adapter", cls.ad_t, alpha=4)
        cls.rs_dir = write_adapter(root / "adapter_rslora", cls.ad_t, alpha=4, rslora=True)
        cls.bad_dir = write_adapter(root / "adapter_ia3", cls.ad_t, peft_type="IA3")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def delta(self, layer, module):
        stem = (PREFIX if layer == 0 else "") + f"model.layers.{layer}.{module}"
        return 2.0 * self.ad_t[stem + ".lora_B.weight"] @ self.ad_t[stem + ".lora_A.weight"]   # scale alpha/r = 2

    def test_load_adapter_parses_config_and_maps_both_name_styles(self):
        a = ar.load_adapter(str(self.ad_dir))
        self.assertEqual(a["r"], R)
        self.assertEqual(a["alpha"], 4)
        self.assertAlmostEqual(a["scale"], 2.0)
        self.assertFalse(a["rslora"])
        self.assertEqual(a["base_model"], "org/tiny-base")
        self.assertEqual(sorted(a["modules"]),
                         ["model.layers.0.mlp.down_proj.weight", "model.layers.0.self_attn.q_proj.weight",
                          "model.layers.1.mlp.down_proj.weight", "model.layers.1.self_attn.q_proj.weight"])
        m = a["modules"]["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(m["A"], PREFIX + "model.layers.0.self_attn.q_proj.lora_A.weight")
        self.assertEqual(m["B"], PREFIX + "model.layers.0.self_attn.q_proj.lora_B.weight")

    def test_rslora_scale_is_alpha_over_sqrt_r(self):
        a = ar.load_adapter(str(self.rs_dir))
        self.assertTrue(a["rslora"])
        self.assertAlmostEqual(a["scale"], 4 / math.sqrt(R))

    def test_rejects_non_lora_adapter(self):
        with self.assertRaises(ValueError):
            ar.load_adapter(str(self.bad_dir))

    def test_delta_norm_exact_by_two_methods(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)))
        rows = {r["tensor"]: r for r in rep["modules"]}
        for layer in range(L):
            for module in ("self_attn.q_proj", "mlp.down_proj"):
                row = rows[f"model.layers.{layer}.{module}.weight"]
                want = float(self.delta(layer, module).norm())
                self.assertAlmostEqual(row["delta_norm"], want, places=4)        # method 1: trace identity
                self.assertAlmostEqual(row["delta_norm_svd"], want, places=4)    # method 2: singular values
                self.assertEqual(row["r"], R)
                self.assertEqual(row["params_adapter"], R * (self.delta(layer, module).shape[0] + self.delta(layer, module).shape[1]))

    def test_effective_rank_90(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)))
        rows = {r["tensor"]: r for r in rep["modules"]}
        self.assertEqual(rows["model.layers.0.self_attn.q_proj.weight"]["effective_rank_90"], 2)   # energies 80 % / 20 %
        self.assertEqual(rows["model.layers.1.mlp.down_proj.weight"]["effective_rank_90"], 1)      # A rank one
        self.assertTrue(all(r["effective_rank_90"] <= R for r in rep["modules"]))

    def test_params_two_methods_and_totals(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)))
        want = sum(v.numel() for v in self.ad_t.values())
        self.assertEqual(rep["totals"]["params_from_shapes"], want)
        self.assertEqual(rep["totals"]["params_from_header"], want)
        self.assertEqual(rep["totals"]["modules"], 4)
        self.assertEqual(rep["totals"]["layers"], [0, 1])
        self.assertEqual(rep["totals"]["bytes"], (self.ad_dir / "adapter_model.safetensors").stat().st_size)

    def test_block_ratio_against_base(self):
        rows_n = 8
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)), base=str(self.base_dir), rows=rows_n)
        rows = {r["tensor"]: r for r in rep["modules"]}
        name = "model.layers.1.mlp.down_proj.weight"
        start = D // 2 - rows_n // 2
        dW = self.delta(1, "mlp.down_proj")[start:start + rows_n]
        W = self.base[name][start:start + rows_n]
        self.assertAlmostEqual(rows[name]["rel_delta_block"], float(dW.norm() / W.norm()), places=4)
        self.assertEqual(rows[name]["block_start"], start)
        self.assertEqual(rows[name]["block_rows"], rows_n)
        self.assertTrue(all(r["shape_matches_base"] for r in rep["modules"]))

    def test_families_with_base(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)), base=str(self.base_dir), rows=8)
        self.assertEqual(rep["families"]["touched"], ["mlp.down_proj.weight", "self_attn.q_proj.weight"])
        for k in ("model.embed_tokens.weight", "lm_head.weight", "mlp.up_proj.weight", "self_attn.o_proj.weight",
                  "model.norm.weight"):   # tensor_kind keeps the model. prefix of non-layer tensors, as model_report does
            self.assertIn(k, rep["families"]["untouched"])
        self.assertFalse(rep["families"]["embeddings_touched"])
        self.assertFalse(rep["families"]["head_touched"])
        self.assertEqual(rep["base"]["in_card"], "org/tiny-base")
        self.assertEqual(rep["base"]["given"], str(self.base_dir))

    def test_families_without_base_still_lists_touched(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)))
        self.assertEqual(rep["families"]["touched"], ["mlp.down_proj.weight", "self_attn.q_proj.weight"])
        self.assertIsNone(rep["families"]["untouched"])

    def test_direction_consistency_per_kind(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)))
        dc = rep["direction_consistency"]
        self.assertGreater(dc["self_attn.q_proj.weight"], 0.99)   # same B in both layers -> same leading output direction
        self.assertIn("mlp.down_proj.weight", dc)
        self.assertLessEqual(dc["mlp.down_proj.weight"], 1.0)

    def test_markdown_has_sections_and_numbers(self):
        rep = ar.adapter_report(ar.load_adapter(str(self.ad_dir)), base=str(self.base_dir), rows=8)
        md = ar.to_markdown(rep)
        for s in ("# Adapter report", "r = 2", "alpha = 4", "## Families", "## Per kind", "## Modules",
                  "self_attn.q_proj.weight", "mlp.down_proj.weight", "rel_delta"):
            self.assertIn(s, md)


if __name__ == "__main__":
    unittest.main()
