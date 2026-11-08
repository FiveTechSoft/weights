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


class TestProvenance(unittest.TestCase):
    def test_record_links_protocol_data_code_and_states_its_limits(self):
        pv = json.loads(PATH.read_text())["evidence"]["provenance"]
        self.assertEqual(pv["protocol_commit"], "6ba876c")
        for k in ("protocol_sha256_at_commit", "protocol_sha256_now"):
            self.assertRegex(pv[k], r"^[0-9a-f]{64}$")
        self.assertTrue(pv["data_sha256"] and pv["code_sha256"] and pv["results_sha256"])
        self.assertIn("do not seal", pv["limits"])

    def test_sha_at_commit_is_stable_and_missing_files_fail(self):
        from lab.meta.provenance import sha256_at_commit
        a = sha256_at_commit(ROOT, "6ba876c", "lab/meta/interference_v6.py")
        self.assertEqual(a, sha256_at_commit(ROOT, "6ba876c", "lab/meta/interference_v6.py"))
        with self.assertRaises(RuntimeError):
            sha256_at_commit(ROOT, "6ba876c", "lab/meta/does_not_exist.py")
