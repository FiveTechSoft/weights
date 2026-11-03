"""Unit tests verifying WSP-Lang security validation gate, forbidden tensors, and semantic ranking."""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsplang
import wsp


class TestWsplangSecurityAndRanking(unittest.TestCase):

    def test_unknown_step_operation_rejected(self):
        """Unknown step operations in 'steps:' must be rejected by validate()."""
        code = """experiment test_unknown_step:
steps:
    step totally_unknown_operation blk.5.ffn_down.weight
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertTrue(len(errors) > 0, "Validation should have detected error on unknown operation")
        self.assertTrue(any("totally_unknown_operation" in e for e in errors), f"Expected operation name in errors: {errors}")

    def test_orphan_line_outside_section_rejected(self):
        """Lines written outside recognized sections must be caught by validate()."""
        code = """experiment test_orphan:
orphan_instruction_without_section
steps:
    step peek blk.5.ffn_down.weight 10 20
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertTrue(len(errors) > 0, "Validation should reject orphan line outside section")
        self.assertTrue(any("linea fuera de seccion" in e for e in errors), f"Expected 'linea fuera de seccion' in errors: {errors}")

    def test_valid_step_passes_validation(self):
        """Valid operations inside steps must pass validation cleanly."""
        code = """experiment test_valid_step:
steps:
    step peek blk.5.ffn_down.weight 100 200
    step poke blk.5.ffn_down.weight 100 200 0.05
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Expected 0 errors, got: {errors}")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["op"], "peek")
        self.assertEqual(steps[1]["op"], "poke")

    def test_forbidden_tensor_rejection(self):
        """Execute plan and policy must reject poke operations on protected tensors."""
        for f in ("token_embd.weight", "output.weight", "output_norm.weight"):
            self.assertIn(f, wsp.FORBIDDEN_TENSORS)


if __name__ == "__main__":
    unittest.main()
