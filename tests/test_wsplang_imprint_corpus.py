"""Contract for the `imprint_corpus` op: the .wspl run must reproduce the direct Python measurement
(lab.meta.interference.score_config) on a real NanoWSP-3, and must stay `simulated` when nothing is written."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.nanowsp3_runtime import NanoWSP3  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402
from lab.meta import interference as itf  # noqa: E402
from lab.meta.grid import make_task  # noqa: E402

SPEC = {"old_layer": 1, "n_old": 20, "n_new": 20, "seed": 301}
CFG = (1, 1.0, 1e-4)

PROGRAM = """experiment imprint_corpus_contract:
    model: nanowsp3
    budgets:
        allow_write = {allow}
    steps:
        step imprint_corpus facts={old} layer={old_layer} eta=3.5 reg_lambda=0.0001 model={base} out={mid}
        step imprint_corpus facts={new} probe={old} layer={layer} eta={eta} reg_lambda={reg} model={mid} out={fin}
"""


def run(code, dry_run):
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    assert not errors, errors
    return execute_plan_v4(plan_run(prog, steps), prog, dry_run=dry_run)


class TestImprintCorpusContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        old = make_task(SPEC["n_old"], 0, SPEC["seed"])
        new = [(p.replace("MEM", "NEW"), t) for p, t in make_task(SPEC["n_new"], 0, SPEC["seed"] + 1)]
        (d / "old.json").write_text(json.dumps(old))
        (d / "new.json").write_text(json.dumps(new))
        torch.manual_seed(0)
        NanoWSP3().save_checkpoint(d / "base.safetensors")
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def _code(self, allow, cfg=CFG):
        d = self.d
        return PROGRAM.format(allow=allow, old=(d / "old.json").as_posix(), new=(d / "new.json").as_posix(),
                              base=(d / "base.safetensors").as_posix(), mid=(d / "mid.safetensors").as_posix(),
                              fin=(d / "fin.safetensors").as_posix(), old_layer=SPEC["old_layer"],
                              layer=cfg[0], eta=cfg[1], reg=cfg[2])

    def test_wspl_run_matches_direct_python_measurement(self):
        scores = []
        for cfg in (CFG, (3, 8.0, 1e-2), (5, 2.0, 1e-4)):     # different outcomes, so a match is not a coincidence
            with self.subTest(cfg=cfg):
                res = run(self._code("true", cfg), dry_run=False)
                self.assertEqual([r["status"] for r in res], ["ok", "ok"], res)
                last = res[1]["result"]
                self.assertIs(last["weights_written"], True)
                self.assertEqual(res[1]["execution_state"], "executed")
                wspl_score = min(last["recall_facts_after"], last["recall_probe_after"])
                direct = itf.score_config(SPEC, cfg)
                self.assertAlmostEqual(wspl_score, direct, places=9)
                self.assertEqual(last["probe_pairs"], SPEC["n_old"])
                scores.append(direct)
        self.assertGreater(len(set(scores)), 1)

    def test_without_write_permission_nothing_is_written_and_state_is_simulated(self):
        res = run(self._code("false"), dry_run=False)
        self.assertEqual(res[0]["status"], "ok", res[0])
        self.assertEqual(res[0]["execution_state"], "simulated")
        self.assertIs(res[0]["result"]["weights_written"], False)
        self.assertFalse((self.d / "mid.safetensors").exists())

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_without_allow_write_the_source_checkpoint_is_untouched_but_the_measurement_is_real(self):
        before = self._sha(self.d / "base.safetensors")
        res = run(self._code("false"), dry_run=False)
        r = res[0]["result"]
        self.assertEqual(self._sha(self.d / "base.safetensors"), before)
        self.assertEqual(sorted(p.name for p in self.d.iterdir()), ["base.safetensors", "new.json", "old.json"])
        self.assertIs(r["persisted"], False)
        self.assertEqual((r["recall_facts_before"], r["recall_facts_after"]), (0.0, 1.0))   # real forward passes
        self.assertIn("real forward passes", r["measurement"])

    def test_dry_run_never_writes(self):
        before = self._sha(self.d / "base.safetensors")
        res = run(self._code("true"), dry_run=True)
        self.assertEqual(res[0]["execution_state"], "simulated")
        self.assertEqual(self._sha(self.d / "base.safetensors"), before)
        self.assertFalse((self.d / "mid.safetensors").exists())

    def test_out_equal_to_the_source_does_not_overwrite_it(self):
        base = self.d / "base.safetensors"
        before = self._sha(base)
        code = self._code("true").replace(f"out={(self.d / 'mid.safetensors').as_posix()}", f"out={base.as_posix()}", 1)
        res = run(code, dry_run=False)
        self.assertEqual(self._sha(base), before)
        self.assertEqual(res[0]["status"], "error", res[0])

    def test_chaining_with_explicit_model_uses_distinct_checkpoints(self):
        res = run(self._code("true"), dry_run=False)
        base, mid, fin = (self.d / n for n in ("base.safetensors", "mid.safetensors", "fin.safetensors"))
        self.assertEqual(Path(res[0]["result"]["base_checkpoint"]), base)
        self.assertEqual(Path(res[1]["result"]["base_checkpoint"]), mid)
        self.assertEqual(len({self._sha(base), self._sha(mid), self._sha(fin)}), 3)
        # step 2 started from step 1's memory: the old facts were recalled before it edited anything
        self.assertEqual(res[1]["result"]["recall_probe_before"], 1.0)

    def test_bad_facts_file_is_an_error_not_a_pass(self):
        (self.d / "old.json").write_text(json.dumps([["only-one"]]))
        res = run(self._code("true"), dry_run=False)
        self.assertEqual(res[0]["status"], "error")
        self.assertIn("imprint_corpus", res[0]["error"])


if __name__ == "__main__":
    unittest.main()
