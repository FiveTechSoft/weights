"""engine/sealed_set.py: a confirmation set sealed before use (GSM8K 100 + MATH-500 50), disjoint from every item the
transplant work has looked at (GSM8K first 100, MATH-500 select_indices(2026, 50)), with a content hash that load()
checks, and bench_runner.run on explicit item lists so the sealed set is run by the same code as the dev set."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

import engine.sealed_set as ss  # noqa: E402
import engine.bench_runner as br  # noqa: E402
import math500 as m5  # noqa: E402


class TestMake(unittest.TestCase):
    def test_disjoint_from_dev_items_and_deterministic(self):
        a, b = ss.make(seed=7), ss.make(seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a["gsm8k"]), 100)
        self.assertEqual(len(a["math500"]), 50)
        self.assertTrue(all(i >= 100 for i in a["gsm8k"]))
        self.assertFalse(set(a["math500"]) & set(m5.select_indices(2026, 50, 500)))
        self.assertEqual(len(set(a["gsm8k"])), 100)

    def test_load_checks_the_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps(ss.make(seed=7)))
            self.assertEqual(len(ss.load(p)["gsm8k"]), 100)
            bad = json.loads(p.read_text())
            bad["gsm8k"][0] = 0                      # an item swapped after sealing
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                ss.load(p)


class TestRunOnExplicitItems(unittest.TestCase):
    def test_run_asks_exactly_the_given_items(self):
        fake = {"content": "\\boxed{1}", "reasoning": "", "finish": "stop", "tokens": 5, "tok_s": 30.0}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "ask", return_value=fake):
            rep = br.run("t", 1, parallel=1, out_dir=d, log=lambda s: None, gsm_idx=[500, 700], math_idx=[3])
        got = sorted((r["bench"], r["i"]) for r in rep["items"])
        self.assertEqual(got, [("gsm8k", 500), ("gsm8k", 700), ("math500", 3)])


if __name__ == "__main__":
    unittest.main()
