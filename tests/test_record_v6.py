"""The committed v6 record must pass the real engine/record.py schema and agree with its own criteria."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import record as rec  # noqa: E402

PATH = ROOT / "lab" / "meta" / "records" / "e_v6_adaptive.json"


class TestRecordV6(unittest.TestCase):
    def setUp(self):
        self.r = json.loads(PATH.read_text())

    def test_passes_the_real_schema(self):
        self.assertEqual(rec.validate(self.r), [])
        self.assertEqual(rec.validate_tier_rules(self.r), [])

    def test_conclusion_follows_the_pre_registered_criteria(self):
        m = self.r["resultados"]["metricas"]
        for k, v in m.items():                       # holds <=> the 95% interval lies entirely below 0
            self.assertEqual(v["holds"], v["ci95"][1] < 0, k)
        claim = m["k"]["holds"] and m["l"]["holds"]
        self.assertEqual(self.r["conclusion"], "SUPPORTED" if claim else "INCONCLUSIVE")
        self.assertEqual(self.r["veredicto"], "keep" if claim else "sin_verificar")
        self.assertIs(self.r["placebo_required"], True)

    def test_a_causal_claim_without_placebo_is_rejected_by_the_schema_rules(self):
        bad = dict(self.r, placebo_required=False)
        self.assertTrue(rec.validate_tier_rules(bad))


if __name__ == "__main__":
    unittest.main()
