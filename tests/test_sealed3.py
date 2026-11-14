"""Sealed confirmation set 3 (outside review 2026-11-13, point 1: sets 1 and 2 are spent): 200 GSM8K test + 100 MATH-500
items, disjoint from dev, sealed sets 1 and 2, the cc split and every (bench, i) already present under lab/runs/.
Single use."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Sealed3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lab.probes.sealed import sealed3
        cls.s3 = sealed3
        cls.made = sealed3.make()

    def test_sizes_seed_and_digest(self):
        from engine import sealed_set as ss
        m = self.made
        self.assertEqual((len(m["gsm8k"]), len(m["math500"])), (200, 100))
        self.assertEqual(m["seed"], self.s3.SEED)
        self.assertEqual(m["sha256"], ss.digest(m["gsm8k"], m["math500"]))
        self.assertEqual(m["gsm8k"], sorted(set(m["gsm8k"])))
        self.assertEqual(m["math500"], sorted(set(m["math500"])))

    def test_disjoint_from_dev_sealed_sets_split_and_runs(self):
        from engine import sealed_set as ss
        sys.path.insert(0, str(ROOT / "rig"))
        import math500 as m5
        m = self.made
        g, mm = set(m["gsm8k"]), set(m["math500"])
        s1 = ss.load()
        s2 = json.loads((ROOT / "data" / "sealed_confirm2.json").read_text())
        split = json.loads((ROOT / "lab" / "runs" / "cc" / "split.json").read_text())
        used = self.s3.used()
        self.assertFalse(g & set(range(ss.DEV_GSM_N)))
        self.assertFalse(g & set(s1["gsm8k"]) or g & set(s2["gsm8k"]) or g & used["gsm8k"])
        dev_m = set(m5.select_indices(ss.DEV_MATH[0], ss.DEV_MATH[1], len(m5.load_items())))
        self.assertFalse(mm & dev_m)
        self.assertFalse(mm & set(s1["math500"]) or mm & set(s2["math500"]) or mm & used["math500"])
        self.assertFalse(mm & set(split["discovery_math"]) or mm & set(split["screen_math"]))

    def test_deterministic(self):
        again = self.s3.make()
        self.assertEqual(again["sha256"], self.made["sha256"])


if __name__ == "__main__":
    unittest.main()
