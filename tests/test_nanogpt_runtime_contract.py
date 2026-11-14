"""Contract tests for the nanoGPT runtime, its WSPLang entry point and its parity reports."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "lab"), str(ROOT / "lab" / "reference" / "karpathy_nanogpt")]

from model import GPT as OfficialGPT, GPTConfig as OfficialConfig  # noqa: E402
from engine.nanogpt_runtime import GPTConfig, NanoGPT  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402

SMALL = dict(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)


def run_program(code):
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    assert not errors, errors
    return execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)


def init_program(out, extra=""):
    return f"""experiment t:
model: nanoGPT
budgets:
    max_modified_weights = 100000000
    allow_write = true
steps:
    step init_architecture model=nanoGPT layers=2 d_model=32 heads=2 vocab=65 max_seq=32 seed=1337 out={out.as_posix()} {extra}
"""


class TestSeededW0(unittest.TestCase):
    def test_same_seed_gives_official_w0(self):
        torch.manual_seed(1337)
        off = OfficialGPT(OfficialConfig(**SMALL))
        ours = NanoGPT(GPTConfig(**SMALL), seed=1337)
        off_p = dict(off.named_parameters())
        for name, p in ours.model.named_parameters():
            self.assertTrue(torch.equal(off_p[name], p), name)

    def test_weight_tying_survives_init(self):
        ours = NanoGPT(GPTConfig(**SMALL), seed=1)
        self.assertIs(ours.model.transformer.wte.weight, ours.model.lm_head.weight)


class TestCheckpointLoading(unittest.TestCase):
    def _save(self, sd):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "m.safetensors"
        save_file(sd, str(p))
        return p

    def _clean_sd(self, m):
        return {k: v.clone() for k, v in m.export_state_dict().items()}

    def test_missing_tensor_is_rejected(self):
        m = NanoGPT(GPTConfig(**SMALL), seed=1)
        sd = self._clean_sd(m)
        del sd["transformer.h.0.mlp.c_fc.weight"]
        with self.assertRaises(RuntimeError):
            m.load_checkpoint(self._save(sd))

    def test_unknown_tensor_is_rejected(self):
        m = NanoGPT(GPTConfig(**SMALL), seed=1)
        sd = self._clean_sd(m)
        sd["transformer.h.0.mystery"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            m.load_checkpoint(self._save(sd))

    def test_legacy_causal_mask_buffers_are_accepted(self):
        m = NanoGPT(GPTConfig(**SMALL), seed=1)
        sd = self._clean_sd(m)
        sd["transformer.h.0.attn.bias"] = torch.tril(torch.ones(1, 1, 32, 32))
        m.load_checkpoint(self._save(sd))


class TestInitArchitectureOp(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.out = Path(d.name) / "m.safetensors"

    def test_omitted_d_ff_defaults_to_four_times_d_model(self):
        res = run_program(init_program(self.out))[0]["result"]
        self.assertEqual(res["d_ff"], 128)
        off = sum(p.numel() for p in OfficialGPT(OfficialConfig(**SMALL)).parameters())
        self.assertEqual(res["total_parameters"], off)
        self.assertEqual(load_file(str(self.out))["transformer.h.0.mlp.c_fc.weight"].shape[0], 128)

    def test_explicit_d_ff_is_respected(self):
        res = run_program(init_program(self.out, "d_ff=64"))[0]["result"]
        self.assertEqual(res["d_ff"], 64)

    def test_bias_and_dropout_reach_the_config(self):
        res = run_program(init_program(self.out, "bias=false dropout=0.1"))[0]["result"]
        self.assertIs(res["bias"], False)
        self.assertEqual(res["dropout"], 0.1)
        self.assertNotIn("transformer.h.0.mlp.c_fc.bias", load_file(str(self.out)))

    def test_small_nanowsp_defaults_unchanged(self):
        """d_ff default for non-nanoGPT architectures keeps its historical value."""
        out = self.out.as_posix()
        code = init_program(self.out).replace("model=nanoGPT layers=2 d_model=32 heads=2 vocab=65 max_seq=32 seed=1337",
                                              "model=NanoWSP layers=2 d_model=64 heads=2 vocab=100")
        self.assertIn(out, code)
        self.assertEqual(run_program(code)[0]["result"]["d_ff"], 128)


class TestCompileSyntheticHonesty(unittest.TestCase):
    CODE = """experiment t:
model: nanowsp3
steps:
    step compile_synthetic dataset={ds} lambda_base=30.0 model={ckpt}
"""

    def _run(self, ds, ckpt):
        prog = parse_args_lines(self.CODE.format(ds=ds, ckpt=ckpt))
        steps, _ = validate(prog)
        return execute_plan_v4(plan_run(prog, steps), prog, dry_run=True)[0]

    def test_never_reports_hardcoded_speedup_or_success(self):
        ckpt = ROOT / "lab" / "runs" / "nanowsp3_pretrained.safetensors"
        if not ckpt.exists():
            self.skipTest("nanowsp3 checkpoint not available")
        r = self._run("wspx_recall", ckpt.as_posix())
        self.assertEqual(r["status"], "ok")
        self.assertNotIn("analytical_speedup", r["result"])
        self.assertIs(r["result"]["weights_written"], False)
        self.assertNotEqual(r["result"]["status"], "compiled_successfully")

    def test_writes_only_the_readout_and_reports_measured_accuracy(self):
        ckpt = ROOT / "lab" / "runs" / "nanowsp3_pretrained.safetensors"
        if not ckpt.exists():
            self.skipTest("nanowsp3 checkpoint not available")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.safetensors"
            code = f"""experiment t:
model: nanowsp3
budgets:
    allow_write = true
steps:
    step compile_synthetic dataset=wspx_recall lambda_base=30.0 model={ckpt.as_posix()} out={out.as_posix()}
"""
            prog = parse_args_lines(code)
            steps, errors = validate(prog)
            self.assertEqual(errors, [])
            r = execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)[0]
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(Path(r["result"]["base_checkpoint"]), ckpt)
            base, edited = load_file(str(ckpt)), load_file(str(out))
        self.assertEqual(sorted(base), sorted(edited))
        changed = [k for k in base if not torch.equal(base[k], edited[k])]
        self.assertEqual(changed, ["out.weight" if "out.weight" in base else "output.weight"])
        res = r["result"]
        self.assertIs(res["weights_written"], True)
        self.assertEqual(res["provenance"], "closed_form_surgery_on_gradient_base")
        self.assertGreater(res["train_accuracy_after"], res["train_accuracy_before"])
        self.assertIn("holdout_accuracy_after", res)

    def test_unknown_dataset_is_an_error(self):
        r = self._run("shakespeare_char", "lab/runs/nanowsp3_pretrained.safetensors")
        self.assertEqual(r["status"], "error")
        self.assertIn("shakespeare_char", r["error"])


class TestParityVerdict(unittest.TestCase):
    def test_official_conclusion_depends_on_results(self):
        import verify_official_karpathy_parity as v
        ok = v.build_verdict({"a": {"status": "PASS"}}, max_diff=0.0, commit="c")
        bad = v.build_verdict({"a": {"status": "PASS"}, "b": {"status": "FAIL"}}, max_diff=1.0, commit="c")
        self.assertTrue(ok["independent_parity_certified"])
        self.assertFalse(bad["independent_parity_certified"])
        self.assertNotEqual(ok["conclusion"], bad["conclusion"])
        self.assertIn("b", bad["conclusion"])
        self.assertNotIn("bit a bit", bad["conclusion"])

    def test_pinned_hash_ignores_line_endings(self):
        """Git converts line endings per platform; the pin must identify the same upstream file."""
        import verify_karpathy_vendorized as v
        with tempfile.TemporaryDirectory() as d:
            lf, crlf = Path(d) / "lf.py", Path(d) / "crlf.py"
            lf.write_bytes(b"a = 1\nb = 2\n")
            crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
            self.assertEqual(v.sha256(lf), v.sha256(crlf))

    def test_pin_matches_the_reference_file(self):
        import verify_karpathy_vendorized as v
        self.assertEqual(v.sha256(v.REF / "model.py"), v.UPSTREAM_MODEL_SHA256)

    def test_vendorized_interpretation_is_computed(self):
        import verify_karpathy_vendorized as v
        self.assertEqual(v.parity_verdict(0.0), "PASSES_fp32")
        self.assertEqual(v.parity_verdict(0.01), "CLOSE_BUT_NOT_EXACT")
        self.assertEqual(v.parity_verdict(2.36), "DIVERGES")
        for d in (0.0, 0.01, 2.36):
            self.assertNotEqual(v.interpretation(d), "verdict")
        self.assertNotIn("sospechoso", v.interpretation(0.0))


class TestNoHardcodedVerdicts(unittest.TestCase):
    """A success claim written as a literal survives any result; it must be computed."""
    PATTERNS = [
        r'"[^"\n]*(?:certified|verified|real_llm)[^"\n]*"\s*:\s*True\b',
        r'"verdict"\s*:\s*"[A-Z_]*PASS[A-Z_]*"\s*,',
        r'"why_[a-z0-9_]*is_real_llm"',
    ]

    def test_lab_scripts_compute_their_verdicts(self):
        import re
        offenders = []
        for path in sorted((ROOT / "lab").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            for pattern in self.PATTERNS:
                offenders += [f"{path.name}: {m.group(0)}" for m in re.finditer(pattern, source)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
