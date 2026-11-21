"""lab/probes/cc/chain_guard.py: a chain that skips steps when their output files exist must also know the outputs
belong to the same model, budget, scale and code (outside review 2026-11-15 of chain.sh). First call writes
<dir>/config.json; later calls compare and refuse on any difference."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.probes.cc import chain_guard as cg  # noqa: E402


class TestChainGuard(unittest.TestCase):
    def test_first_call_writes_config_and_same_call_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cg.check(d, {"gguf": "e574", "budget": "1024"}), "written")
            self.assertEqual(json.loads((Path(d) / "config.json").read_text())["budget"], "1024")
            self.assertEqual(cg.check(d, {"gguf": "e574", "budget": "1024"}), "same")

    def test_changed_or_missing_key_is_refused_and_named(self):
        with tempfile.TemporaryDirectory() as d:
            cg.check(d, {"gguf": "e574", "budget": "1024"})
            with self.assertRaises(cg.ConfigMismatch) as cm:
                cg.check(d, {"gguf": "e574", "budget": "2048"})
            self.assertIn("budget", str(cm.exception))
            self.assertIn("1024", str(cm.exception))
            with self.assertRaises(cg.ConfigMismatch) as cm:
                cg.check(d, {"gguf": "e574"})
            self.assertIn("budget", str(cm.exception))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cg.main([d, "gguf=e574", "scale=0.5"]), 0)
            self.assertEqual(cg.main([d, "gguf=e574", "scale=0.5"]), 0)
            self.assertEqual(cg.main([d, "gguf=e574", "scale=1.0"]), 1)


if __name__ == "__main__":
    unittest.main()
