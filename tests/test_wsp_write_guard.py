"""Write safety of every WSPLang op that writes weights, and exit codes of failed steps.

Rules under test (engine.wsplang.guard_model_write):
  - an edit never overwrites the model it read; it needs out= or --out-dir;
  - the tensor set, shapes and dtypes of the edited file match the source;
  - changed elements and max |delta| respect max_modified_weights / max_delta_norm;
  - protected tensors (embeddings, readout, final norms) change only when listed in unlock_tensors.
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as wsplang  # noqa: E402
from engine.nanowsp_runtime import NanoWSP  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(steps, model_path, out_dir=None, budgets="allow_write = true", dry_run=False):
    lines = "\n".join(f"    {b.strip()}" for b in budgets.strip().splitlines())
    body = "\n".join(f"    step {s}" for s in steps)
    code = f"experiment guard_test:\nbudgets:\n{lines}\nsteps:\n{body}\n"
    prog = wsplang.parse_args_lines(code)
    steps_v, errors = wsplang.validate(prog)
    assert not errors, errors
    plan = wsplang.plan_run(prog, steps_v)
    return wsplang.execute_plan_v4(plan, prog, model_path=str(model_path),
                                   out_dir=str(out_dir) if out_dir else None, dry_run=dry_run)


class Fixture(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        torch.manual_seed(0)
        self.model_dir = self.dir / "model"
        self.model_dir.mkdir()
        self.model = self.model_dir / "model.safetensors"
        save_file({
            "layers.1.mlp_out.weight": torch.randn(8, 16),
            "transformer.wte.weight": torch.randn(10, 8),
            "lm_head.weight": torch.randn(10, 8),
        }, str(self.model))
        u = np.random.RandomState(0).randn(8).astype(np.float32)
        self.u_file = self.dir / "u.npz"
        np.savez(self.u_file, u=u / np.linalg.norm(u))
        self.before = digest(self.model)

    def assertUntouched(self):
        self.assertEqual(digest(self.model), self.before, "input model was modified")


class TestNoOverwriteOfInput(Fixture):
    EDITS = [
        "leace_erase layers.1.mlp_out.weight vector={u}",
        "decoupled_delta layers.1.mlp_out.weight erase_vector={u}",
        "spectral_decay layers.1.mlp_out.weight decay=0.5",
        "delta_unlearn layers.1.mlp_out.weight key={u} eta=1.0 target=0.0",
        "project_out layers.1.mlp_out.weight vector={u} alpha=0.05",
    ]

    def test_edit_without_destination_is_an_error_and_leaves_input_intact(self):
        for edit in self.EDITS:
            with self.subTest(edit=edit.split()[0]):
                r = run([edit.format(u=self.u_file.as_posix())], self.model_dir)[0]
                self.assertEqual(r["status"], "error", r)
                self.assertIn("out=", r["error"])
                self.assertUntouched()

    def test_out_dir_equal_to_input_dir_is_refused(self):
        r = run([self.EDITS[0].format(u=self.u_file.as_posix())], self.model_dir, out_dir=self.model_dir)[0]
        self.assertEqual(r["status"], "error", r)
        self.assertIn("overwrite", r["error"])
        self.assertUntouched()

    def test_edit_with_out_dir_writes_there_and_reports_measured_change(self):
        out = self.dir / "edited"
        r = run([self.EDITS[2]], self.model_dir, out_dir=out)[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertUntouched()
        edited = load_file(str(out / "model.safetensors"))
        self.assertTrue(torch.allclose(edited["layers.1.mlp_out.weight"],
                                       0.5 * load_file(str(self.model))["layers.1.mlp_out.weight"]))
        self.assertEqual(r["write_guard"]["modified_parameters"], 8 * 16)
        self.assertEqual(r["write_guard"]["tensors_changed"], ["layers.1.mlp_out.weight"])

    def test_allow_write_false_writes_nothing(self):
        out = self.dir / "edited"
        r = run([self.EDITS[2]], self.model_dir, out_dir=out, budgets="allow_write = false")[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertFalse(out.exists())


class TestBudgetsOnEveryWrite(Fixture):
    def test_budget_applies_to_ops_that_never_checked_it(self):
        out = self.dir / "edited"
        for edit in ("spectral_decay layers.1.mlp_out.weight decay=0.5",
                     f"leace_erase layers.1.mlp_out.weight vector={self.u_file.as_posix()}"):
            with self.subTest(edit=edit.split()[0]):
                r = run([edit], self.model_dir, out_dir=out,
                        budgets="allow_write = true\nmax_modified_weights = 10")[0]
                self.assertEqual(r["status"], "error", r)
                self.assertIn("max_modified_weights", r["error"])
                self.assertFalse((out / "model.safetensors").exists())

    def test_max_delta_norm_applies(self):
        r = run(["spectral_decay layers.1.mlp_out.weight decay=0.5"], self.model_dir, out_dir=self.dir / "e",
                budgets="allow_write = true\nmax_delta_norm = 0.0001")[0]
        self.assertEqual(r["status"], "error", r)
        self.assertIn("max_delta_norm", r["error"])


class TestProtectedTensors(Fixture):
    def test_wte_is_protected(self):
        r = run(["spectral_decay transformer.wte.weight decay=0.5"], self.model_dir, out_dir=self.dir / "e")[0]
        self.assertIn(r["status"], ("error", "security_rejection"), r)
        self.assertIn("transformer.wte.weight", r["error"])

    def test_unlock_tensors_allows_a_named_protected_tensor(self):
        r = run(["spectral_decay transformer.wte.weight decay=0.5"], self.model_dir, out_dir=self.dir / "e",
                budgets="allow_write = true\nunlock_tensors = transformer.wte.weight")[0]
        self.assertEqual(r["status"], "ok", r)


class TestTieWeights(Fixture):
    def test_without_out_is_an_error(self):
        r = run(["tie_weights source=transformer.wte.weight target=lm_head.weight"], self.model,
                budgets="allow_write = true\nunlock_tensors = lm_head.weight")[0]
        self.assertEqual(r["status"], "error", r)
        self.assertUntouched()

    def test_target_is_protected_unless_unlocked(self):
        out = self.dir / "tied.safetensors"
        r = run([f"tie_weights source=transformer.wte.weight target=lm_head.weight out={out.as_posix()}"], self.model)[0]
        self.assertEqual(r["status"], "error", r)
        self.assertIn("lm_head.weight", r["error"])
        self.assertFalse(out.exists())

    def test_ties_into_out(self):
        out = self.dir / "tied.safetensors"
        r = run([f"tie_weights source=transformer.wte.weight target=lm_head.weight out={out.as_posix()}"], self.model,
                budgets="allow_write = true\nunlock_tensors = lm_head.weight")[0]
        self.assertEqual(r["status"], "ok", r)
        sd = load_file(str(out))
        self.assertTrue(torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"]))
        self.assertUntouched()

    def test_shape_mismatch_is_an_error(self):
        out = self.dir / "tied.safetensors"
        r = run([f"tie_weights source=layers.1.mlp_out.weight target=lm_head.weight out={out.as_posix()}"], self.model,
                budgets="allow_write = true\nunlock_tensors = lm_head.weight")[0]
        self.assertEqual(r["status"], "error", r)
        self.assertIn("shape", r["error"])
        self.assertFalse(out.exists())


class TestNanoWSPWriters(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.model = self.dir / "model.safetensors"
        NanoWSP().save_checkpoint(self.model)
        self.before = digest(self.model)

    def test_imprint_without_out_is_an_error(self):
        r = run(["imprint_association prompt=ab target=c"], self.model)[0]
        self.assertEqual(r["status"], "error", r)
        self.assertEqual(digest(self.model), self.before)

    def test_imprint_with_out_writes_a_copy(self):
        out = self.dir / "imprinted.safetensors"
        r = run([f"imprint_association prompt=ab target=c out={out.as_posix()}"], self.model)[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(digest(self.model), self.before)
        self.assertGreater(r["write_guard"]["modified_parameters"], 0)
        self.assertTrue(out.exists())

    def test_memit_without_out_is_an_error(self):
        r = run(["memit_imprint layers.1.mlp_out.weight prompt=ab target=c"], self.model)[0]
        self.assertEqual(r["status"], "error", r)
        self.assertEqual(digest(self.model), self.before)


class TestGuardFunction(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.src = Path(d.name) / "src.safetensors"
        self.out = Path(d.name) / "out.safetensors"
        self.base = {"a.weight": torch.zeros(4, 4), "b.weight": torch.ones(3)}
        save_file(self.base, str(self.src))
        self.budgets = {"max_modified_weights": None, "max_delta_norm": None, "unlock_tensors": []}

    def guard(self, candidate, out=None, written=()):
        return wsplang.guard_model_write(candidate, out or self.out, self.src, self.budgets, set(written))

    def test_counts_changed_elements_and_max_delta(self):
        cand = {k: v.clone() for k, v in self.base.items()}
        cand["a.weight"][0, :2] = torch.tensor([0.5, -2.0])
        report = self.guard(cand)
        self.assertEqual(report["modified_parameters"], 2)
        self.assertAlmostEqual(report["max_abs_delta"], 2.0)

    def test_added_tensor_is_refused(self):
        cand = dict(self.base, **{"foreign.weight": torch.zeros(2)})
        with self.assertRaisesRegex(wsplang.WriteGuardError, "foreign.weight"):
            self.guard(cand)

    def test_removed_tensor_is_refused(self):
        with self.assertRaisesRegex(wsplang.WriteGuardError, "b.weight"):
            self.guard({"a.weight": self.base["a.weight"]})

    def test_overwriting_source_is_refused_unless_written_this_run(self):
        with self.assertRaisesRegex(wsplang.WriteGuardError, "overwrite"):
            self.guard(dict(self.base), out=self.src)
        self.guard(dict(self.base), out=self.src, written={self.src.resolve()})


class TestUnlockBudgetParsing(unittest.TestCase):
    def test_unlock_tensors_list(self):
        prog = wsplang.parse_args_lines("experiment t:\nbudgets:\n    unlock_tensors = lm_head.weight, output.weight\n")
        self.assertEqual(wsplang.extract_budgets(prog)["unlock_tensors"], ["lm_head.weight", "output.weight"])


class TestExportControlVector(unittest.TestCase):
    def test_no_file_without_allow_write(self):
        with tempfile.TemporaryDirectory() as d:
            vec = Path(d) / "v.npz"
            np.savez(vec, u=np.ones(8, dtype=np.float32))
            out = Path(d) / "cv.gguf"
            r = run([f"export_control_vector {vec.as_posix()} out={out.as_posix()}"], Path(d),
                    budgets="allow_write = false")[0]
            self.assertFalse(out.exists(), r)


class TestFailedStepsExitNonZero(unittest.TestCase):
    def test_exit_code_of_results(self):
        self.assertEqual(wsplang.run_exit_code([{"status": "ok"}, {"status": "stub_result"}]), 0)
        for bad in ("error", "security_rejection", "assertion_failed"):
            self.assertEqual(wsplang.run_exit_code([{"status": "ok"}, {"status": bad}]), 1, bad)

    def test_failed_assertion_is_assertion_failed_not_ok(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model.safetensors"
            torch.manual_seed(0)
            save_file({"w": torch.randn(16, 16)}, str(model))
            u = np.zeros(16, dtype=np.float32)
            u[0] = 1
            vec = Path(d) / "u.npz"
            np.savez(vec, u=u)
            # max_angle_deg=0 cannot hold for any non-zero edit
            r = run([f"assert_davis_kahan w vector={vec.as_posix()} alpha=0.5 top_cols=1.0 max_angle_deg=0"],
                    model, budgets="allow_write = false")[0]
            self.assertEqual(r["status"], "assertion_failed", r)

    def test_cli_exits_one_when_a_step_fails(self):
        with tempfile.TemporaryDirectory() as d:
            prog = Path(d) / "p.wspl"
            prog.write_text("experiment exit_code_test:\nsteps:\n    step audit no.such.tensor\n", encoding="utf-8")
            model = Path(d) / "model.safetensors"
            save_file({"w": torch.zeros(2, 2)}, str(model))
            argv = ["wsplang", str(prog), "--run", "--model", str(model)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(wsplang, "DIR_PL", Path(d)), \
                    mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as cm:
                    wsplang.main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
