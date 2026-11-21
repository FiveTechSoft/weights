"""Sealed confirmation set 3 (outside review 2026-11-13, point 1: sets 1 and 2 are spent): 200 GSM8K test + 100 MATH-500
items, disjoint from dev, sealed sets 1 and 2, the cc split, and every (bench, i) already present under lab/runs/.
Single use.

The committed data/sealed_confirm3.json is the source of truth: it was made once and spent (hash and the
spend record live in that file). Re-running sealed3.make() would re-derive from current lab/runs/, which now
contains the spending runs themselves — so the test verifies the committed file, not the generator."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PATH = ROOT / "data" / "sealed_confirm3.json"


class Sealed3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from engine import sealed_set as ss
        cls.ss = ss
        cls.made = json.loads(PATH.read_text(encoding="utf-8"))

    def test_sizes_seed_and_digest(self):
        m = self.made
        self.assertEqual((len(m["gsm8k"]), len(m["math500"])), (200, 100))
        self.assertEqual(m["seed"], 20261113)
        self.assertEqual(m["sha256"], self.ss.digest(m["gsm8k"], m["math500"]))
        self.assertEqual(m["gsm8k"], sorted(set(m["gsm8k"])))
        self.assertEqual(m["math500"], sorted(set(m["math500"])))

    def test_disjoint_from_dev_sealed_sets_and_cc_split(self):
        sys.path.insert(0, str(ROOT / "rig"))
        import math500 as m5
        m = self.made
        g, mm = set(m["gsm8k"]), set(m["math500"])
        s1 = self.ss.load()
        s2 = json.loads((ROOT / "data" / "sealed_confirm2.json").read_text())
        split = json.loads((ROOT / "lab" / "runs" / "cc" / "split.json").read_text())
        self.assertFalse(g & set(range(self.ss.DEV_GSM_N)))
        self.assertFalse(g & set(s1["gsm8k"]) or g & set(s2["gsm8k"]))
        dev_m = set(m5.select_indices(self.ss.DEV_MATH[0], self.ss.DEV_MATH[1], len(m5.load_items())))
        self.assertFalse(mm & dev_m)
        self.assertFalse(mm & set(s1["math500"]) or mm & set(s2["math500"]))
        self.assertFalse(mm & set(split["discovery_math"]) or mm & set(split["screen_math"]))

    def test_spent_record_is_present(self):
        use = self.made.get("use", "")
        self.assertTrue("spent" in use or "single use" in use)


if __name__ == "__main__":
    unittest.main()
