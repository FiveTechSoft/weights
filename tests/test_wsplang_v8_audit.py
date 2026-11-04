"""AUDIT 2026-11-04: feasibility gate — ops no-implementadas, backend GGUF, artefactos y stubs."""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsplang


def parse(text):
    prog = wsplang.parse_args_lines(text)
    steps, errors = wsplang.validate(prog)
    return prog, steps, errors


class TestFeasibilityGate(unittest.TestCase):

    def test_unimplemented_op_rejected(self):
        prog, steps, _ = parse("experiment t:\nsteps:\n    step poke blk.5.ffn_up.weight 10 20 0.01\n")
        errors, _ = wsplang.check_feasibility(prog, steps, model_arg=None)
        self.assertTrue(any("poke" in e and "NO implementada" in e for e in errors), str(errors))

    def test_write_op_on_gguf_rejected(self):
        prog, steps, _ = parse(
            "experiment t:\nsteps:\n    step delta_unlearn blk.14.ffn_down.weight vector=data/u_loop_L01.npz eta=0.5\n")
        errors, _ = wsplang.check_feasibility(prog, steps, model_arg="C:/x/mimo.gguf")
        self.assertTrue(any("delta_unlearn" in e and "safetensors" in e for e in errors), str(errors))

    def test_write_op_op_acepta_safetensors(self):
        prog, steps, _ = parse(
            "experiment t:\nsteps:\n    step delta_unlearn blk.14.ffn_down.weight vector=data/u_loop_L01.npz eta=0.5\n")
        errors, _ = wsplang.check_feasibility(prog, steps, model_arg="C:/x/model.safetensors")
        self.assertFalse(any("delta_unlearn" in e for e in errors), str(errors))

    def test_missing_artifact_rejected(self):
        prog, steps, _ = parse(
            "experiment t:\nsteps:\n    step compare retos/inexistente_totalmente.json\n")
        errors, _ = wsplang.check_feasibility(prog, steps, model_arg="C:/x/mimo.gguf")
        self.assertTrue(any("inexistente" in e for e in errors), str(errors))

    def test_read_only_gguf_ok(self):
        prog, steps, _ = parse(
            "experiment t:\nsteps:\n    step peek blk.5.ffn_up.weight 10 20\n    step compare exp8_control/results_eval_n5.json\n")
        errors, _ = wsplang.check_feasibility(prog, steps, model_arg="C:/x/mimo.gguf")
        self.assertEqual(errors, [], str(errors))

    def test_stub_flagged(self):
        prog, steps, _ = parse(
            "experiment t:\nsteps:\n    step leace_erase blk.5.ffn_up.weight vector=data/u_loop_L01.npz\n")
        _, warnings = wsplang.check_feasibility(prog, steps, model_arg="C:/x/model.safetensors")
        self.assertTrue(any("STUB" in w for w in warnings), str(warnings))

    def test_real_program_feasible(self):
        src = Path(ROOT / "programs" / "verify_mit_loop_suppression.wspl")
        prog = wsplang.parse_args_lines(src.read_text(encoding="utf-8"))
        steps, errors = wsplang.validate(prog)
        feas, _ = wsplang.check_feasibility(prog, steps, model_arg=None)
        self.assertEqual(errors + feas, [], str(errors + feas))


if __name__ == "__main__":
    unittest.main()
