"""test_wsplang_v2.py — Comprehensive Unit Tests for WSP-Lang 0.2 Engine."""

import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsplang


class TestWSPLangV2(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy vector
        self.vec = np.random.randn(64).astype(np.float32)
        self.vec = self.vec / np.linalg.norm(self.vec)
        self.vec_file = self.temp_path / "u_dummy.npz"
        np.savez(self.vec_file, u=self.vec)

        # Create dummy SafeTensors model
        self.dummy_weight = torch.randn(64, 128, dtype=torch.float32)
        self.dummy_shard = self.temp_path / "model.safetensors"
        save_file({
            "model.language_model.layers.14.mlp.down_proj.weight": self.dummy_weight,
            "model.language_model.layers.14.mlp.up_proj.weight": torch.randn(128, 64)
        }, str(self.dummy_shard))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_backward_compatibility_v1_programs(self):
        """Verify that existing WSP-Lang 0.1 programs parse cleanly with zero errors."""
        prog_path = ROOT / "programs" / "fix_mimo_tool_loops.wspl"
        text = prog_path.read_text(encoding="utf-8")
        prog = wsplang.parse_args_lines(text)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Legacy program validation failed: {errors}")
        self.assertTrue(len(steps) >= 3)

    def test_v2_new_step_operations_validated(self):
        """Verify that WSP-Lang 0.2 step operations are recognized."""
        code = f"""experiment test_v2_features:
budgets:
    max_modified_weights = 1000
    max_delta_norm = 0.05
    allow_write = true
steps:
    step project_out model.language_model.layers.14.mlp.down_proj.weight vector={self.vec_file} alpha=0.05
    step audit model.language_model.layers.14.mlp.down_proj.weight vector={self.vec_file}
    step export_control_vector {self.vec_file} out={self.temp_path / 'cv.gguf'}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertEqual(len(errors), 0, f"Expected 0 errors, got: {errors}")
        plan = wsplang.plan_run(prog, steps)
        self.assertTrue(plan["wsplang_version"] in ("0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0"))
        self.assertEqual(plan["budgets"]["max_modified_weights"], 1000)
        self.assertTrue(plan["budgets"]["allow_write"])

    def test_security_rejection_forbidden_tensors(self):
        """Ensure forbidden tensors (embeddings, heads) are rejected from modification."""
        code = f"""experiment test_security:
steps:
    step project_out lm_head.weight vector={self.vec_file} alpha=0.05
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v2(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "security_rejection")
        self.assertIn("SECURITY POLICY REJECTION", results[0]["error"])

    def test_budget_exceeded_rejection(self):
        """Ensure budget constraints abort execution when limits are exceeded."""
        code = f"""experiment test_budget:
budgets:
    max_modified_weights = 10
steps:
    step project_out model.language_model.layers.14.mlp.down_proj.weight vector={self.vec_file} alpha=0.05 top_cols=0.5
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v2(plan, prog, model_path=str(self.temp_path), dry_run=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("BUDGET EXCEEDED", results[0]["error"])

    def test_safetensors_project_out_execution(self):
        """Verify successful rank-1 projection subtraction on SafeTensors."""
        out_model_dir = self.temp_path / "repaired"
        code = f"""experiment test_surgery:
budgets:
    max_modified_weights = 50000
    max_delta_norm = 1.0
    allow_write = true
steps:
    step project_out model.language_model.layers.14.mlp.down_proj.weight vector={self.vec_file} alpha=0.05 top_cols=0.25 top_rows=30
    step audit model.language_model.layers.14.mlp.down_proj.weight vector={self.vec_file}
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v2(
            plan, prog,
            model_path=str(self.temp_path),
            out_dir=str(out_model_dir),
            dry_run=False
        )

        self.assertEqual(len(results), 2)
        step1 = results[0]
        self.assertEqual(step1["status"], "ok")
        self.assertTrue(step1["result"]["modified_parameters"] > 0)
        self.assertTrue(step1["result"]["written_to_disk"])
        self.assertTrue(step1["result"]["directional_energy_after"] < step1["result"]["directional_energy_before"])

        # Verify repaired file exists and untouched tensor is preserved
        from safetensors.torch import load_file
        repaired_tensors = load_file(str(out_model_dir / "model.safetensors"))
        self.assertIn("model.language_model.layers.14.mlp.down_proj.weight", repaired_tensors)
        self.assertIn("model.language_model.layers.14.mlp.up_proj.weight", repaired_tensors)

    def test_export_control_vector(self):
        """Verify control vector export compiles valid GGUF file."""
        cv_out = self.temp_path / "cv_test.gguf"
        code = f"""experiment test_cv:
steps:
    step export_control_vector {self.vec_file} out={cv_out} scale=-0.5
"""
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        plan = wsplang.plan_run(prog, steps)
        results = wsplang.execute_plan_v2(plan, prog, dry_run=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        self.assertTrue(cv_out.exists())
        self.assertTrue(cv_out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
