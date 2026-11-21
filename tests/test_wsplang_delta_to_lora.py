"""WSPLang op delta_to_lora: a fine-tune's change (model - base) as a reusable LoRA adapter (PEFT dir, optional GGUF).

Writes an adapter beside the models, never inside an input model and never over one."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as wsplang  # noqa: E402


def run(steps):
    body = "\n".join(f"    step {s}" for s in steps)
    prog = wsplang.parse_args_lines(f"experiment t:\nsteps:\n{body}\n")
    v, errors = wsplang.validate(prog)
    assert not errors, errors
    return wsplang.execute_plan_v4(wsplang.plan_run(prog, v), prog)


class TestDeltaToLora(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        g = torch.Generator().manual_seed(0)
        pre = "model.language_model.layers"     # multimodal Qwen3.5 naming, as in the real checkpoints
        self.names = {"mlp": f"{pre}.0.mlp.down_proj.weight", "att": f"{pre}.0.self_attn.o_proj.weight",
                      "lin": f"{pre}.1.linear_attn.out_proj.weight"}
        base = {n: torch.randn(24, 40, generator=g) for n in self.names.values()}
        base[f"{pre}.0.input_layernorm.weight"] = torch.ones(24)
        self.delta = {n: torch.randn(24, 3, generator=g) @ torch.randn(3, 40, generator=g) for n in self.names.values()}
        tuned = {n: (t + self.delta[n] if n in self.delta else t + 0.5) for n, t in base.items()}
        for name, ts in (("base", base), ("tuned", tuned)):
            (self.dir / name).mkdir()
            save_file(ts, str(self.dir / name / "model.safetensors"))
        self.base, self.tuned = (self.dir / "base").as_posix(), (self.dir / "tuned").as_posix()

    def test_exact_low_rank_delta_round_trips_in_peft_format(self):
        out = self.dir / "lora"
        r = run([f"delta_to_lora base={self.base} model={self.tuned} rank=4 out={out.as_posix()}"])[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["result"]["matrices"], 3)
        self.assertGreater(r["result"]["mean_kept"], 0.999)
        T = load_file(str(out / "adapter_model.safetensors"))
        cfg = json.loads((out / "adapter_config.json").read_text())
        self.assertEqual((cfg["r"], cfg["lora_alpha"], cfg["peft_type"]), (4, 4, "LORA"))
        for n, D in self.delta.items():
            stem = "base_model.model." + n[: -len(".weight")]
            B, A = T[stem + ".lora_B.weight"].float(), T[stem + ".lora_A.weight"].float()
            self.assertEqual(tuple(A.shape), (4, 40))
            self.assertLess(float((B @ A - D).norm() / D.norm()), 0.02)   # bf16 storage
        self.assertEqual(len(T), 6)                                       # no norm vector in a LoRA
        e = json.loads((out / "energy.json").read_text())
        self.assertIn(self.names["mlp"], e)
        # exact SVD: the share kept at each smaller rank comes free (true rank 3 here)
        self.assertLess(e[self.names["mlp"]]["kept_at"]["1"], 0.99)
        self.assertGreater(e[self.names["mlp"]]["kept_at"]["4"], 0.999)
        self.assertGreater(r["result"]["kept_weighted_at"]["4"], 0.999)

    def test_kinds_filter_and_rank_below_true_rank(self):
        out = self.dir / "lora_mlp"
        r = run([f"delta_to_lora base={self.base} model={self.tuned} rank=1 kinds=mlp out={out.as_posix()}"])[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["result"]["matrices"], 1)
        self.assertLess(r["result"]["mean_kept"], 0.999)
        cfg = json.loads((out / "adapter_config.json").read_text())
        self.assertEqual(cfg["target_modules"], ["down_proj"])

    def test_refuses_out_inside_input_or_existing(self):
        r = run([f"delta_to_lora base={self.base} model={self.tuned} rank=2 out={self.tuned}/lora"])[0]
        self.assertEqual(r["status"], "error")
        self.assertFalse((self.dir / "tuned" / "lora").exists())
        (self.dir / "taken").mkdir()
        r = run([f"delta_to_lora base={self.base} model={self.tuned} rank=2 out={(self.dir / 'taken').as_posix()}"])[0]
        self.assertEqual(r["status"], "error")

    def test_lowrank_is_the_exact_truncation_on_a_flat_spectrum(self):
        # randomized SVD with little oversampling missed by 11-16 % on real Qwen3.5 deltas (flat spectra)
        from engine.lora_export import lowrank
        g = torch.Generator().manual_seed(1)
        D = torch.randn(300, 200, generator=g)
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        X = (U[:, :16] * S[:16]) @ Vh[:16]
        A, B, kept = lowrank(D, 16)
        self.assertLess(float((B @ A - X).norm() / X.norm()), 1e-4)
        self.assertAlmostEqual(kept, float(S[:16].pow(2).sum() / S.pow(2).sum()), places=5)

    def test_model_on_the_hub_is_read_by_ranges(self):
        """model=hf:<owner>/<repo> reads the fine-tune from the Hugging Face hub by byte ranges (nothing written to
        disk); here the hub is a stub that serves the local file's bytes."""
        from engine import weight_diff as wd
        src = self.dir / "tuned" / "model.safetensors"
        seen = []

        def files(repo, revision="main", subfolder=""):
            seen.append((repo, revision))
            return ["model.safetensors"]

        def fetch(repo, revision="main", retries=5):
            def f(name, start, end):
                with open(src, "rb") as fh:
                    fh.seek(start)
                    return fh.read(end - start + 1)
            return f
        old = wd.hf_files, wd.hf_fetch
        wd.hf_files, wd.hf_fetch = files, fetch
        try:
            out = self.dir / "lora_hub"
            r = run([f"delta_to_lora base={self.base} model=hf:someone/tuned-9b rank=4 out={out.as_posix()}"])[0]
        finally:
            wd.hf_files, wd.hf_fetch = old
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(seen, [("someone/tuned-9b", "main")])
        self.assertGreater(r["result"]["mean_kept"], 0.999)
        self.assertEqual(r["result"]["verdict"], "likely")

    def test_verdict_from_the_energy_kept_at_rank_16(self):
        """Measured on 9B: Jackrong keeps 58.6 % at rank 16 and transfers; MiMo keeps 3.8 % and does not.
        verdict: likely >= 0.5, unlikely < 0.2, uncertain between (a prediction made before any eval)."""
        self.assertEqual(wsplang.transfer_verdict({"16": 0.586}), "likely")
        self.assertEqual(wsplang.transfer_verdict({"16": 0.038}), "unlikely")
        self.assertEqual(wsplang.transfer_verdict({"16": 0.35}), "uncertain")
        self.assertEqual(wsplang.transfer_verdict({}), "uncertain")

    def test_identical_model_is_no_change_not_likely(self):
        """A 'fine-tune' whose weights equal the base (OmniCoder-9B, OxCoder-9B on the hub, 2026-11-10) has no energy to
        keep: the verdict must say no_change, not likely (the old code divided 0 by 0 and reported 100 % kept)."""
        out = self.dir / "lora_same"
        r = run([f"delta_to_lora base={self.base} model={self.base} rank=4 out={out.as_posix()}"])[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["result"]["delta_norm"], 0.0)
        self.assertEqual(r["result"]["verdict"], "no_change")
        self.assertEqual(wsplang.transfer_verdict({"16": 1.0}, delta_norm=0.0), "no_change")
        self.assertEqual(wsplang.transfer_verdict({"16": 0.586}, delta_norm=12.3), "likely")

    def test_passes_cli_gate(self):
        prog = wsplang.parse_args_lines("experiment g:\nsteps:\n    step delta_to_lora base=a model=b rank=8 out=o\n")
        steps, errors = wsplang.validate(prog)
        self.assertFalse(errors, errors)
        res = wsplang.check_feasibility(prog, steps, model_arg=str(ROOT / "hf_cache" / "qwen35-08b-hf"))
        errs = res[0] if isinstance(res, tuple) else res.get("errors", [])
        self.assertFalse([e for e in errs if "NO implementada" in e], errs)


if __name__ == "__main__":
    unittest.main()
