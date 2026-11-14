"""lab/probes/cc/comp.py: composition of the conclude vector with the Jackrong adapter (lab/probes/cc/PLAN.md,
"Composition"). Pure parts: the K1 classification and the control band from per-benchmark comparisons."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "cc"))

import comp  # noqa: E402


def cmp(arm, base, win, loss, p):
    return {"arm": arm, "base": base, "win": win, "loss": loss, "p": p}


class TestK1(unittest.TestCase):
    def test_adds_when_better_on_one_and_not_below_on_the_other(self):
        k = {"gsm8k": cmp(95, 90, 6, 1, 0.03), "math500": cmp(35, 34, 2, 1, 0.6)}
        self.assertEqual(comp.k1(k), "adds")

    def test_interferes_when_significantly_below_on_either(self):
        k = {"gsm8k": cmp(90, 90, 3, 3, 1.0), "math500": cmp(27, 34, 0, 7, 0.016)}
        self.assertEqual(comp.k1(k), "interferes")

    def test_redundant_within_noise(self):
        k = {"gsm8k": cmp(89, 90, 2, 3, 1.0), "math500": cmp(33, 34, 1, 2, 1.0)}
        self.assertEqual(comp.k1(k), "redundant")

    def test_better_on_one_but_significantly_worse_on_the_other_is_interference(self):
        k = {"gsm8k": cmp(96, 90, 7, 1, 0.02), "math500": cmp(26, 34, 0, 8, 0.008)}
        self.assertEqual(comp.k1(k), "interferes")


class TestControl(unittest.TestCase):
    def test_within_three_and_not_significant_on_both(self):
        k = {"gsm8k": cmp(88, 90, 2, 4, 0.69), "math500": cmp(34, 34, 1, 1, 1.0)}
        self.assertEqual(comp.control_ok(k), {"gsm8k": True, "math500": True})

    def test_outside_band(self):
        k = {"gsm8k": cmp(84, 90, 1, 7, 0.07), "math500": cmp(34, 34, 1, 1, 1.0)}
        self.assertEqual(comp.control_ok(k)["gsm8k"], False)


if __name__ == "__main__":
    unittest.main()
