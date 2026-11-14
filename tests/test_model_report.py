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


NEOX = [("model.embed_tokens.", "gpt_neox.embed_in."), ("lm_head.", "embed_out."),
        ("model.norm.", "gpt_neox.final_layer_norm."), ("self_attn.o_proj.", "attention.dense."),
        ("linear_attn.out_proj.", "attention.dense."), ("mlp.down_proj.", "mlp.dense_4h_to_h."),
        ("mlp.up_proj.", "mlp.dense_h_to_4h."), ("model.layers.", "gpt_neox.layers.")]


def to_neox(name):
    for a, b in NEOX:
        name = name.replace(a, b)
    return name


class TestOtherArchitecture(unittest.TestCase):
    """The same recipes saved with GPT-NeoX names (no Qwen-style names on disk) must classify the same way."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        g = torch.Generator().manual_seed(0)
        base = {k: v for k, v in base_model(g).items() if "linear_attn" not in k and "gate_proj" not in k}
        cls.paths = {}
        for name in ("base", "ablation", "lora"):
            t = base if name == "base" else recipe(name, base, g)
            p = d / name
            p.mkdir()
            save_file({to_neox(k): v.to(torch.bfloat16) for k, v in t.items()}, str(p / "model.safetensors"))
            cls.paths[name] = p

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fp(self, name):
        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        return mr.fingerprint(reader(self.paths["base"]), reader(self.paths[name]), rows=16)

    def test_names_on_disk_are_neox(self):
        self.assertIn("gpt_neox.layers.3.attention.dense.weight", wd.LocalSafetensors(self.paths["base"]).keys())

    def test_neox_ablation(self):
        c = mr.classify(self.fp("ablation"))
        self.assertEqual(c["method"], "directional_ablation", c)
        self.assertEqual(c["layers"], [2, 5])
        self.assertEqual(c["ablation_directions"], "global")

    def test_neox_lora(self):
        self.assertEqual(mr.classify(self.fp("lora"))["method"], "lora_like")


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


class TestTransferCandidates(unittest.TestCase):
    """Directions a model shares with studied improvements, estimated from blocks of rows (column side only)."""

    def test_shared_subspace_found_and_unrelated_not(self):
        g = torch.Generator().manual_seed(9)
        Dh, rows, layers = 64, 256, 6
        names = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(layers)]
        base = {n: torch.randn(rows, Dh, generator=g) for n in names}
        S = {n: torch.linalg.qr(torch.randn(Dh, 4, generator=g))[0] for n in names}   # per-layer shared subspace
        other = {n: torch.linalg.qr(torch.randn(Dh, 4, generator=g))[0] for n in names}

        def tuned(sub, scale=1.0):
            return {n: base[n] + scale * torch.randn(rows, 4, generator=g) @ sub[n].T
                    + 0.01 * torch.randn(rows, Dh, generator=g) for n in names}

        atlas_model, target, control = tuned(S), tuned(S, 0.5), tuned(other)
        sigs = {"improved": {n: wd.delta_signature(n, base[n], atlas_model[n], k=4, sketch_dim=64) for n in names}}

        class R:
            def __init__(self, t): self.t = t
            def keys(self): return list(self.t)
            def shape(self, n): return tuple(self.t[n].shape)
            def get(self, n): return self.t[n]
            def get_rows(self, n, s, c): return self.t[n][s:s + c]

        for model, expect in ((target, True), (control, False)):
            fp = mr.fingerprint(R(base), R(model), rows=32, workers=1, k=4)
            tc = mr.transfer_candidates(fp, sigs, k=4)
            row = tc["by_kind"]["mlp.gate_proj.weight"]["improved"]
            self.assertEqual(row["shared"], expect, row)
            self.assertEqual("mlp.gate_proj.weight" in tc["candidates"], expect)


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


class TestHiddenParts(Synthetic):
    """Tensors the text report hides (MTP head, vision tower) still count when one side lacks them (Arete-27B
    dropped Qwen3.8-27B's MTP head; the first report did not say so)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from safetensors.torch import load_file
        d = Path(cls.tmp.name)
        for name, extra in (("base_mtp", {"mtp.fc.weight": torch.randn(D, 2 * D), "mtp.norm.weight": torch.ones(D),
                                          "model.visual.blocks.0.attn.qkv.weight": torch.randn(3 * D, D)}),
                            ("full_no_mtp", {"model.visual.blocks.0.attn.qkv.weight": torch.randn(3 * D, D)})):
            src = cls.paths["base"] if name == "base_mtp" else cls.paths["full"]
            t = load_file(str(src / "model.safetensors"))
            t.update({k: v.to(torch.bfloat16) for k, v in extra.items()})
            (d / name).mkdir()
            save_file(t, str(d / name / "model.safetensors"))
            cls.paths[name] = d / name

    def report(self):
        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        return mr.build_report(reader(self.paths["base_mtp"]), reader(self.paths["full_no_mtp"]), rows=16,
                               model_name="m", base_name="b")

    def test_hidden_groups_only_in_one_side(self):
        h = self.report()["headers"]
        self.assertEqual(h["hidden_only_in_base"], {"mtp": 2})
        self.assertEqual(h["hidden_only_in_model"], {})

    def test_markdown_and_usable_name_the_missing_mtp_head(self):
        rep = self.report()
        md = mr.render_markdown(rep)
        self.assertIn("mtp (2 tensors)", md)
        self.assertIn("MTP", " ".join(rep["usable"]))


class TestAtlasFromAnotherBase(Synthetic):
    """The atlas is Qwen3.5-9B. For a model on another base (other shapes) the direction comparison does not apply,
    and the report must say so instead of printing an empty table and 'no shared directions' (Arete-27B)."""

    def sigs(self, dim):
        g = torch.Generator().manual_seed(3)
        n, norm = "model.layers.0.mlp.gate_proj.weight", "model.layers.0.input_layernorm.weight"
        a, w = torch.randn(I, dim, generator=g), torch.ones(D)
        # 1-D norms keep their shape across scales (head_dim 256, DeltaNet norm 128 in Qwen3.5-9B and 3.8-27B):
        # they must not make another base look like the atlas base.
        return {"improved": {n: wd.delta_signature(n, a, a + 0.1 * torch.randn(I, dim, generator=g), k=4,
                                                   sketch_dim=32),
                             norm: wd.delta_signature(norm, w, w + 0.01, k=4, sketch_dim=32)}}

    def report(self, sigs):
        reader = lambda p: wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
        atlas = {"x": {"family": "f", "profile": {"mlp.gate_proj.weight": 0.1, "mlp.up_proj.weight": 0.2,
                                                  "mlp.down_proj.weight": 0.3}}}
        return mr.build_report(reader(self.paths["base"]), reader(self.paths["full"]), rows=16, model_name="m",
                               base_name="b", atlas=atlas, atlas_sigs=sigs, transfer_rows=32)

    def test_other_base_is_labelled_not_applicable(self):
        rep = self.report(self.sigs(D + 16))
        self.assertFalse(rep["transfer"]["applicable"])
        md = mr.render_markdown(rep)
        self.assertIn("not applicable", md)
        self.assertNotIn("| type |  |", md)
        self.assertIn("another base", md)
        self.assertNotIn("No column-side tensor type shares", " ".join(rep["usable"]))

    def test_same_base_still_compared(self):
        rep = self.report(self.sigs(D))
        self.assertTrue(rep["transfer"]["applicable"])
        self.assertNotIn("another base", mr.render_markdown(rep))


BASE_SPLIT = r"[\p{L}\p{M}]+|\p{N}|\s+|[^\s\p{L}\p{M}\p{N}]+"   # Qwen3.5: letters keep their combining marks
OLD_SPLIT = r"\p{L}+|\p{N}|\s+|[^\s\p{L}\p{N}]+"                # Qwen2-era: marks split off (F25)


def write_side_files(d: Path, split: str, extra_tokens=(), template="{{ messages }}", config=None):
    """config/tokenizer/template of a tiny model; WordLevel vocab, so a different split gives different ids."""
    import json
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    words = ["hello", "world", " ", "नमस्ते", "न", "म", "स", "त", "ते", "्", "े", "1", "2", "!", ","]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(["<unk>"] + words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(split), behavior="isolated")
    if extra_tokens:
        tok.add_special_tokens(list(extra_tokens))
    d.mkdir(parents=True, exist_ok=True)
    tok.save(str(d / "tokenizer.json"))
    (d / "chat_template.jinja").write_text(template, encoding="utf-8")
    cfg = config or {"model_type": "qwen3_5", "text_config": {"hidden_size": 48, "rope_parameters":
                                                                  {"rope_theta": 1e7, "partial_rotary_factor": 0.25}},
                     "transformers_version": "4.57.0"}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestNonWeights(unittest.TestCase):
    """Config, tokenizer and chat template: what a hash cannot tell, an encode test across scripts does (F25)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        write_side_files(d / "base", BASE_SPLIT)
        write_side_files(d / "same", BASE_SPLIT)
        # MiMo-like: old split, two added tokens, same config content saved by a newer library in another layout
        write_side_files(d / "mimo", OLD_SPLIT, extra_tokens=["<|audio_start|>", "<|audio_end|>"],
                         config={"model_type": "qwen3_5",
                                 "text_config": {"hidden_size": 48, "partial_rotary_factor": 0.25,
                                                 "rope_parameters": {"rope_theta": 1e7}, "pad_token_id": None},
                                 "transformers_version": "5.12.1"})
        write_side_files(d / "arch", BASE_SPLIT, template="{{ messages }}{% if tools %}tools{% endif %}",
                         config={"model_type": "qwen3_5", "text_config": {"hidden_size": 64},
                                 "transformers_version": "4.57.0"})
        cls.d = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def nw(self, name):
        return mr.nonweight_diff(mr.side_files(str(self.d / "base")), mr.side_files(str(self.d / name)))

    def test_identical(self):
        r = self.nw("same")
        self.assertEqual(r["config"]["changed"], [])
        self.assertTrue(r["tokenizer"]["pre_tokenizer_same"])
        self.assertEqual(r["encode"]["differs"], [])
        self.assertEqual(r["template"]["status"], "same")
        self.assertEqual(r["verdict"], [])

    def test_serialization_only_config_changes_are_not_changes(self):
        r = self.nw("mimo")
        self.assertEqual(r["config"]["changed"], [], r["config"])
        cosmetic = " ".join(r["config"]["cosmetic"])
        self.assertIn("transformers_version", cosmetic)
        self.assertIn("partial_rotary_factor", cosmetic)        # moved, same value

    def test_real_config_change(self):
        r = self.nw("arch")
        self.assertIn("text_config.hidden_size", [c["key"] for c in r["config"]["changed"]])
        self.assertEqual(r["template"]["status"], "different")

    def test_pre_tokenizer_and_added_tokens(self):
        r = self.nw("mimo")
        t = r["tokenizer"]
        self.assertTrue(t["vocab_same"])
        self.assertFalse(t["pre_tokenizer_same"])
        self.assertEqual(sorted(t["added_only_in_model"]), ["<|audio_end|>", "<|audio_start|>"])

    def test_encode_test_names_the_scripts_that_change(self):
        r = self.nw("mimo")
        self.assertIn("hindi", r["encode"]["differs"])
        self.assertNotIn("english", r["encode"]["differs"])
        self.assertTrue(any("hindi" in v for v in r["verdict"]), r["verdict"])

    def test_report_section(self):
        nw = self.nw("mimo")
        md = mr.render_markdown({"model": "m", "base": "b", "rows_per_block": 16, "headers": {
            "only_in_base": [], "only_in_model": [], "shape_mismatch": [], "dtype_differs": []},
            "classification": {"method": "identical", "touched": [], "untouched": []}, "profile": {},
            "nonweights": nw, "usable": []})
        self.assertIn("## Outside the weights", md)
        self.assertIn("pre-tokenizer", md)
        self.assertIn("hindi", md)

    def test_missing_files_are_reported_not_fatal(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir(exist_ok=True)
        r = mr.nonweight_diff(mr.side_files(str(self.d / "base")), mr.side_files(str(empty)))
        self.assertIn("tokenizer.json", r["files"]["only_in_base"])
        self.assertIsNone(r["encode"])


if __name__ == "__main__":
    unittest.main()
