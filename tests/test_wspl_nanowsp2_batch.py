"""programs/create_nanowsp2_batch.wspl runs end to end inside its 100 delta budget and the model recalls the facts."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.nanowsp2_runtime import NanoWSP2  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402

PROGRAM = ROOT / "programs" / "create_nanowsp2_batch.wspl"
ORIGINAL = ROOT / "programs" / "create_nanowsp2.wspl"


class TestNanoWSP2Batch(unittest.TestCase):
    def test_runs_in_budget_and_recalls_every_prompt(self):
        code = PROGRAM.read_text(encoding="utf-8")
        self.assertIn("max_delta_norm = 100.0", code)             # the safety budget is not loosened
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "model.safetensors"
            code = code.replace("out=models/nanowsp2/model.safetensors", f"out={out.as_posix()}")
            prog = parse_args_lines(code)
            steps, errors = validate(prog)
            self.assertEqual(errors, [])
            res = execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)
            self.assertEqual([r["status"] for r in res], ["ok", "ok"], res)
            m = NanoWSP2()
            m.load_checkpoint(out)
            pairs = re.findall(r'prompt="([^"]+)" target="([^"]+)"', ORIGINAL.read_text(encoding="utf-8"))
            hits = [m.generate(p, max_new_tokens=len(t), enable_titans_monitor=False).endswith(t) for p, t in pairs]
        self.assertEqual(len(pairs), 11)
        self.assertTrue(all(hits), hits)

    def test_pair_file_is_the_original_programs_targets_expanded_per_character(self):
        pairs = re.findall(r'prompt="([^"]+)" target="([^"]+)"', ORIGINAL.read_text(encoding="utf-8"))
        expect = []
        for p, t in pairs:
            cur = p
            for ch in t:
                expect.append({"prompt": cur, "target": ch})
                cur += ch
        self.assertEqual(json.loads((ROOT / "data" / "nanowsp2_pairs.json").read_text()), expect)


if __name__ == "__main__":
    unittest.main()
