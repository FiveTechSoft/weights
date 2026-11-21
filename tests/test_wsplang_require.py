"""`step require` (WSP-Lang 0.8.4): a guard on a JSON artefact that stops a program before it spends compute.

A require whose file already exists (and that no earlier step writes) is evaluated when the program is checked, so a
doomed program is refused before any step runs; one on a file an earlier step writes is evaluated when reached. A
failed require stops the run: its status is guard_failed, every later step is skipped_by_guard, exit code 1."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as w  # noqa: E402
from engine import wsplang_require as rq  # noqa: E402

REPORT = {"G0": {"median": 0.03, "pass": True},
          "vector": {"layers": {"1": {"merge_cos": 0.88}, "2": {"merge_cos": 0.95}, "3": {"merge_cos": 0.83}}},
          "energy": [{"rank": 16, "retained": 0.586}, {"rank": 32, "retained": 0.71}],
          "verdict": "transfer"}


def write(d, name, obj):
    p = Path(d) / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def program(*step_lines):
    text = "experiment t:\nsteps:\n" + "".join(f"    step {s}\n" for s in step_lines)
    prog = w.parse_args_lines(text)
    steps, errors = w.validate(prog)
    return prog, steps, errors


class TestParse(unittest.TestCase):
    def test_positional_form(self):
        g = rq.parse_require('lab/r.json:G0.median <= 0.5 why="G0 gate"')
        self.assertEqual((g["file"], g["path"], g["cmp"], g["value"], g["agg"], g["why"]),
                         ("lab/r.json", "G0.median", "<=", 0.5, "all", "G0 gate"))

    def test_windows_path_keeps_drive_colon(self):
        g = rq.parse_require(r"C:\weights\lab\r.json:energy.0.retained >= 0.4")
        self.assertEqual(g["file"], r"C:\weights\lab\r.json")
        self.assertEqual(g["path"], "energy.0.retained")

    def test_named_form(self):
        g = rq.parse_require("file=lab/r.json key=vector.layers.*.merge_cos cmp=>= value=0.8 agg=min")
        self.assertEqual((g["file"], g["path"], g["cmp"], g["value"], g["agg"]),
                         ("lab/r.json", "vector.layers.*.merge_cos", ">=", 0.8, "min"))

    def test_values_bool_and_string(self):
        self.assertIs(rq.parse_require("r.json:G0.pass == true")["value"], True)
        self.assertEqual(rq.parse_require('r.json:verdict == "transfer"')["value"], "transfer")

    def test_bad_forms_raise(self):
        for bad in ("r.json:G0.median", "r.txt:a >= 1", "r.json:a => 1", "r.json:a >= 1 agg=median",
                    "r.json: >= 1", 'r.json:verdict >= "x"'):
            with self.assertRaises(ValueError, msg=bad):
                rq.parse_require(bad)


class TestEvaluate(unittest.TestCase):
    def ev(self, spec):
        return rq.evaluate(rq.parse_require(spec), REPORT)

    def test_scalar_pass_and_fail(self):
        self.assertTrue(self.ev("r.json:G0.median <= 0.5")["ok"])
        r = self.ev("r.json:G0.median < 0.01")
        self.assertFalse(r["ok"])
        self.assertEqual(r["observed"], 0.03)

    def test_list_index_and_bool(self):
        self.assertTrue(self.ev("r.json:energy.0.retained >= 0.4")["ok"])
        self.assertTrue(self.ev("r.json:G0.pass == true")["ok"])
        self.assertTrue(self.ev('r.json:verdict != "harms"')["ok"])

    def test_wildcard_aggregates(self):
        self.assertFalse(self.ev("r.json:vector.layers.*.merge_cos >= 0.85")["ok"])            # all (default)
        self.assertTrue(self.ev("r.json:vector.layers.*.merge_cos >= 0.85 agg=any")["ok"])
        r = self.ev("r.json:vector.layers.*.merge_cos >= 0.85 agg=mean")
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["observed"], (0.88 + 0.95 + 0.83) / 3)
        self.assertEqual(self.ev("r.json:vector.layers.*.merge_cos >= 0.8 agg=min")["observed"], 0.83)
        self.assertEqual(self.ev("r.json:energy.*.retained >= 0.7 agg=max")["n_values"], 2)

    def test_missing_key_fails_with_reason(self):
        r = self.ev("r.json:G0.nope >= 0")
        self.assertFalse(r["ok"])
        self.assertIn("missing", r["reason"])

    def test_non_numeric_value_under_numeric_comparison_fails(self):
        r = self.ev("r.json:verdict >= 1")
        self.assertFalse(r["ok"])


class TestStaticCheck(unittest.TestCase):
    def test_existing_file_passing_gives_no_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "r.json", REPORT)
            prog, steps, errors = program(f"require {p}:G0.median <= 0.5")
            self.assertEqual(errors, [])
            errs, _ = w.check_feasibility(prog, steps)
            self.assertEqual(errs, [])

    def test_existing_file_failing_is_refused_before_any_step(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "r.json", REPORT)
            prog, steps, _ = program(f"require {p}:G0.median < 0.01")
            errs, _ = w.check_feasibility(prog, steps)
            self.assertEqual(len(errs), 1)
            self.assertIn("require", errs[0])
            self.assertIn("0.03", errs[0])

    def test_missing_file_not_written_by_an_earlier_step_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            prog, steps, _ = program(f"require {Path(d) / 'nope.json'}:a >= 1")
            errs, _ = w.check_feasibility(prog, steps)
            self.assertEqual(len(errs), 1)
            self.assertIn("nope.json", errs[0])

    def test_file_an_earlier_step_writes_is_deferred(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "later.json"
            write(d, "later.json", {"a": 0})       # stale copy: must not be judged now
            prog, steps, _ = program(f"bench_table add={out} out={out}", f"require {out}:a >= 1")
            errs, _ = w.check_feasibility(prog, steps)
            self.assertFalse([e for e in errs if "require" in e], errs)

    def test_file_inside_an_earlier_out_dir_is_deferred(self):
        with tempfile.TemporaryDirectory() as d:
            prog, steps, _ = program(f"bench_table add=x.json out={Path(d) / 'cc'}",
                                     f"require {Path(d) / 'cc' / 'report.json'}:G0.pass == true")
            errs, _ = w.check_feasibility(prog, steps)
            self.assertFalse([e for e in errs if "require" in e], errs)

    def test_malformed_require_is_a_check_error(self):
        prog, steps, _ = program("require r.json:G0.median")
        errs, _ = w.check_feasibility(prog, steps)
        self.assertTrue(any("require" in e for e in errs), errs)


class TestRun(unittest.TestCase):
    def run_steps(self, *step_lines):
        prog, steps, errors = program(*step_lines)
        self.assertEqual(errors, [])
        return w.execute_plan_v4(w.plan_run(prog, steps), prog)

    def test_failed_require_stops_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "r.json", REPORT)
            res = self.run_steps(f"require {p}:G0.median < 0.01 why=\"gate\"", f"require {p}:G0.median <= 0.5")
            self.assertEqual([r["status"] for r in res], ["guard_failed", "skipped_by_guard"])
            self.assertEqual(res[0]["result"]["observed"], 0.03)
            self.assertEqual(res[0]["result"]["why"], "gate")
            self.assertEqual(w.run_exit_code(res), 1)

    def test_passing_require_lets_the_rest_run(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "r.json", REPORT)
            res = self.run_steps(f"require {p}:G0.median <= 0.5", f"require {p}:energy.*.retained >= 0.5")
            self.assertEqual([r["status"] for r in res], ["ok", "ok"])
            self.assertEqual(w.run_exit_code(res), 0)

    def test_require_on_missing_file_at_run_is_guard_failed(self):
        with tempfile.TemporaryDirectory() as d:
            res = self.run_steps(f"require {Path(d) / 'gone.json'}:a >= 1")
            self.assertEqual(res[0]["status"], "guard_failed")
            self.assertIn("gone.json", res[0]["result"]["reason"])


class TestRegistered(unittest.TestCase):
    def test_require_is_a_read_only_implemented_op(self):
        self.assertIn("require", w.OPS_ALLOWED)
        self.assertIn("require", w.READ_ONLY_OPS)
        self.assertEqual(w.run_exit_code([{"status": "guard_failed"}]), 1)


if __name__ == "__main__":
    unittest.main()
