"""Contract tests for decompiling nanoGPT weights into WSPLang n-gram programs and compiling them back."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from engine import wsp_arch, wsp_decompile, wsp_synth, wsp_train  # noqa: E402

ARCH = ROOT / "programs" / "arch" / "nanogpt.wsparch"
TEXT = (ROOT / "lab" / "data" / "shakespeare_char" / "input.txt").read_text(encoding="utf-8")
DATA = wsp_train.char_dataset(TEXT)
V = len(DATA.chars)
SHAPE = dict(block=64, layers=2, heads=6, d=384, ff=1536, dropout=0.0, bias=False)


def logits(model, n=4):
    x = torch.tensor(np.stack([DATA.val[i * 64:(i + 1) * 64] for i in range(n)]))
    with torch.no_grad():
        return torch.log_softmax(model(x, x)[0].double(), -1)


class TestTextFormat(unittest.TestCase):
    def test_program_text_round_trips_exactly(self):
        table = wsp_synth.WittenBellTable(DATA.train[:20000], V, 3)
        decoded = wsp_decompile.UnitTable.from_table(table, DATA.chars, limit=200)
        text = wsp_decompile.format_program(decoded)
        again = wsp_decompile.parse_program(text)
        self.assertEqual(again.chars, DATA.chars)
        self.assertEqual(again.order, 3)
        self.assertEqual([c for c, _ in again.units()], [c for c, _ in decoded.units()])
        for (_, a), (_, b) in zip(again.units(), decoded.units()):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_program_is_readable(self):
        table = wsp_synth.WittenBellTable(DATA.train[:20000], V, 3)
        text = wsp_decompile.format_program(wsp_decompile.UnitTable.from_table(table, DATA.chars, limit=50))
        self.assertIn("ngram vocab=65 order=3", text)
        self.assertIn('unit "th"', text)

    def test_contexts_with_quotes_and_newlines_survive(self):
        units = [((), np.zeros(V)), ((DATA.chars.index('\n'),), np.ones(V)), ((DATA.chars.index("'"),), -np.ones(V))]
        table = wsp_decompile.UnitTable(DATA.chars, 2, units)
        again = wsp_decompile.parse_program(wsp_decompile.format_program(table))
        self.assertEqual([c for c, _ in again.units()], [c for c, _ in units])


class TestStructuralDecompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = wsp_synth.WittenBellTable(DATA.train, V, 4)
        cls.model = wsp_synth.compile_ngram(ARCH, cls.table, overrides=SHAPE, seed=1).eval()

    def test_recovers_used_contexts_in_order(self):
        decoded = wsp_decompile.decompile_ngram(self.model, DATA.chars)
        used = self.model.synthesis_info["units_used"]
        expected = [c for c, _ in self.table.units(limit=used)]
        self.assertEqual(decoded.order, 4)
        self.assertEqual([c for c, _ in decoded.units()], expected)

    def test_recovers_corrections_up_to_a_constant(self):
        decoded = wsp_decompile.decompile_ngram(self.model, DATA.chars)
        for (_, got), (_, want) in zip(decoded.units(), self.table.units(limit=300)):
            np.testing.assert_allclose(got - got.mean(), want - want.mean(), atol=2e-3)

    def test_round_trip_reproduces_the_weights(self):
        program = wsp_decompile.format_program(wsp_decompile.decompile_ngram(self.model, DATA.chars))
        again = wsp_synth.compile_ngram(ARCH, wsp_decompile.parse_program(program), overrides=SHAPE, seed=1).eval()
        for (name, a), b in zip(again.named_parameters(), self.model.parameters()):
            self.assertLess((a - b).abs().max().item(), 1e-6, name)

    def test_round_trip_with_another_seed_matches_the_table_compiled_with_it(self):
        """Seeds change the basis; float32 differences between seeds (about 5e-3) are the compiler's, not ours."""
        program = wsp_decompile.format_program(wsp_decompile.decompile_ngram(self.model, DATA.chars))
        again = wsp_synth.compile_ngram(ARCH, wsp_decompile.parse_program(program), overrides=SHAPE, seed=7).eval()
        direct = wsp_synth.compile_ngram(ARCH, self.table, overrides=SHAPE, seed=7).eval()
        self.assertLess((logits(again) - logits(direct)).abs().max().item(), 1e-3)

    def test_random_weights_are_not_decompilable(self):
        w0 = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=V), seed=1337).eval()
        with self.assertRaises(wsp_decompile.DecompileError):
            wsp_decompile.decompile_ngram(w0, DATA.chars)


class TestBehavioralDecompile(unittest.TestCase):
    def test_table_model_yields_its_own_table(self):
        """A model that is an order-3 table predicts the same at every occurrence of a 2-char context."""
        table = wsp_synth.WittenBellTable(DATA.train, V, 3)
        model = wsp_synth.compile_ngram(ARCH, table, overrides=SHAPE, seed=1).eval()
        decoded = wsp_decompile.behavioral_table(model, DATA.train[:30000], DATA.chars, order=3)
        for context in [(DATA.chars.index("t"), DATA.chars.index("h")), (DATA.chars.index(" "), DATA.chars.index("a"))]:
            np.testing.assert_allclose(decoded.log_prob(context), table.log_prob(context), atol=2e-3)

    def test_units_are_ranked_by_count_and_suffix_closed(self):
        table = wsp_synth.WittenBellTable(DATA.train, V, 3)
        model = wsp_synth.compile_ngram(ARCH, table, overrides=SHAPE, seed=1).eval()
        decoded = wsp_decompile.behavioral_table(model, DATA.train[:30000], DATA.chars, order=3)
        seen = set()
        for context, _ in decoded.units(limit=500):
            if context:
                self.assertIn(context[1:], seen)
            seen.add(context)


def run_steps(lines, allow_write=True):
    from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate
    body = "\n".join(f"    step {line}" for line in lines)
    code = f"""experiment t:
model: nanoGPT
budgets:
    max_modified_weights = 100000000
    allow_write = {str(allow_write).lower()}
steps:
{body}
"""
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    assert not errors, errors
    return execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)


class TestWSPLangOps(unittest.TestCase):
    SHAPE_ARGS = "block=64 layers=2 heads=6 d=384 ff=1536 dropout=0.0 bias=false"

    def setUp(self):
        import tempfile
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)

    def compiled(self):
        out = (self.dir / "m.safetensors").as_posix()
        res = run_steps([f"synthesize method=ngram order=3 arch=programs/arch/nanogpt.wsparch "
                         f"data=lab/data/shakespeare_char/input.txt out={out} {self.SHAPE_ARGS} seed=1"])[0]
        self.assertEqual(res["status"], "ok", res)
        return out

    def load(self, path):
        model = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=V), seed=1)
        model.load(path)
        return model.eval()

    def test_decompile_then_synthesize_from_program_reproduces_weights(self):
        model_path = self.compiled()
        program = (self.dir / "m.wspngram").as_posix()
        again = (self.dir / "again.safetensors").as_posix()
        res = run_steps([
            f"decompile method=structural arch=programs/arch/nanogpt.wsparch model={model_path} out={program} "
            f"{self.SHAPE_ARGS}",
            f"synthesize method=ngram table={program} arch=programs/arch/nanogpt.wsparch out={again} "
            f"{self.SHAPE_ARGS} seed=1"])
        self.assertTrue(all(r["status"] == "ok" for r in res), res)
        self.assertTrue(Path(program).read_text(encoding="utf-8").startswith(wsp_decompile.FORMAT_HEADER))
        self.assertEqual(res[0]["result"]["order"], 3)
        a, b = self.load(model_path), self.load(again)
        for (name, x), y in zip(a.named_parameters(), b.parameters()):
            self.assertLess((x - y).abs().max().item(), 1e-6, name)

    def test_behavioral_decompile_reports_what_it_is(self):
        model_path = self.compiled()
        program = (self.dir / "b.wspngram").as_posix()
        res = run_steps([f"decompile method=behavior order=3 data=lab/data/shakespeare_char/input.txt "
                         f"limit_tokens=20000 arch=programs/arch/nanogpt.wsparch model={model_path} out={program} "
                         f"{self.SHAPE_ARGS}"])[0]
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["result"]["method"], "behavior")
        self.assertIn("long-range context is dropped", res["result"]["note"])
        self.assertGreater(res["result"]["units"], 100)

    def test_structural_decompile_of_trained_weights_is_an_error(self):
        w0 = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=V), seed=1337)
        path = (self.dir / "w0.safetensors").as_posix()
        w0.save(path, metadata={"vocab_chars": __import__("json").dumps(DATA.chars)})
        res = run_steps([f"decompile method=structural arch=programs/arch/nanogpt.wsparch model={path} "
                         f"out={(self.dir / 'x.wspngram').as_posix()} {self.SHAPE_ARGS}"])[0]
        self.assertEqual(res["status"], "error")
        self.assertIn("orthonormal", res["error"])

    def test_program_from_a_gradient_model_keeps_that_provenance(self):
        """Compiling a program distilled from a trained model is not gradient-free synthesis, and must say so."""
        import json
        from safetensors import safe_open
        teacher = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=V), seed=1337)
        path = (self.dir / "teacher.safetensors").as_posix()
        teacher.save(path, metadata={"vocab_chars": json.dumps(DATA.chars), "provenance": "gradient_descent"})
        program = (self.dir / "t.wspngram").as_posix()
        out = (self.dir / "student.safetensors").as_posix()
        res = run_steps([
            f"decompile method=behavior order=2 data=lab/data/shakespeare_char/input.txt limit_tokens=5000 "
            f"arch=programs/arch/nanogpt.wsparch model={path} out={program} {self.SHAPE_ARGS}",
            f"synthesize method=ngram table={program} arch=programs/arch/nanogpt.wsparch out={out} "
            f"{self.SHAPE_ARGS} seed=1"])
        self.assertTrue(all(r["status"] == "ok" for r in res), res)
        self.assertEqual(wsp_decompile.parse_program(Path(program).read_text(encoding="utf-8")).provenance,
                         "distilled_from:gradient_descent")
        with safe_open(out, framework="pt") as f:
            meta = f.metadata()
        self.assertEqual(meta["provenance"], "closed_form_compile_of:distilled_from:gradient_descent")
        self.assertEqual(meta["method"], "wspngram_program")
        self.assertEqual(res[1]["result"]["provenance"], meta["provenance"])

    def test_structural_program_of_synthesized_weights_stays_closed_form(self):
        model_path = self.compiled()
        program = (self.dir / "c.wspngram").as_posix()
        run_steps([f"decompile method=structural arch=programs/arch/nanogpt.wsparch model={model_path} "
                   f"out={program} {self.SHAPE_ARGS}"])
        self.assertEqual(wsp_decompile.parse_program(Path(program).read_text(encoding="utf-8")).provenance,
                         "closed_form")

    def test_decompile_without_write_budget_is_simulated(self):
        model_path = self.compiled()
        program = self.dir / "s.wspngram"
        res = run_steps([f"decompile method=structural arch=programs/arch/nanogpt.wsparch model={model_path} "
                         f"out={program.as_posix()} {self.SHAPE_ARGS}"], allow_write=False)[0]
        self.assertEqual(res["execution_state"], "simulated")
        self.assertFalse(program.exists())


if __name__ == "__main__":
    unittest.main()
