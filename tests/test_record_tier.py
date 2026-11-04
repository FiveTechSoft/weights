"""test_record_tier: audit 2026-11-04 — surgery_tier/claim_class/placebo_required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.record import (infer_tier, validate, validate_tier_rules,  # noqa: E402
                          SURGERY_TIERS, CLAIM_CLASSES)
import unittest


class TestTierInference(unittest.TestCase):

    def test_exact_tier(self):
        self.assertEqual(infer_tier({"trained": 1.0, "candidate": 1.0, "placebo": 0.08}), "EXACT")

    def test_unreachable_from_chance(self):
        self.assertEqual(infer_tier({"trained": 0.84, "candidate": 0.50, "placebo": 0.49,
                                     "chance": 0.50}, chance=0.50), "UNREACHABLE")

    def test_unreachable_placebo_kill(self):
        self.assertEqual(infer_tier({"trained": 0.9, "candidate": 0.4, "placebo": 0.4}), "UNREACHABLE")

    def test_approx_tier(self):
        self.assertEqual(infer_tier({"trained": 0.86, "candidate": 0.82, "placebo": 0.64}), "APPROX")

    def test_tier_rules_placebo_required(self):
        bad = {"claim_class": "causal_learn", "placebo_required": False}
        self.assertTrue(validate_tier_rules(bad))
        good = {"claim_class": "causal_learn", "placebo_required": True}
        self.assertEqual(validate_tier_rules(good), [])

    def test_tier_rules_pipeline_check_ok_sin_placebo(self):
        self.assertEqual(validate_tier_rules({
            "claim_class": "pipeline_check", "surgery_tier": "EXACT"}), [])

    def test_full_record_validate(self):
        rec = {
            "experiment_id": "e2026.114.t2s",
            "model_id": "TinyAssociator",
            "objective": "sustitucion de training lookup por cirugia exacta",
            "baseline": {"nombre": "w0"}, "edicion": {"program": "closed-form"},
            "ejecucion": {"estado": "ok"}, "resultados": {"acc": 1.0},
            "hypothesis": "cirugia exacta == training en lookup",
            "surgery_tier": "EXACT", "tier_inferred": True,
            "claim_class": "causal_learn", "placebo_required": True,
            "conclusion": "SUPPORTED",
        }
        self.assertEqual(validate(rec), [])

    def test_full_record_causal_sin_placebo_rechazado(self):
        rec = {
            "experiment_id": "e2026.115.m1p",
            "model_id": "MLP-8-32-1",
            "objective": "ridge surgery",
            "baseline": {"nombre": "w0"}, "edicion": {"program": "ridge"},
            "ejecucion": {"estado": "ok"}, "resultados": {"acc": 0.5},
            "surgery_tier": "UNREACHABLE", "tier_inferred": True,
            "claim_class": "causal_learn", "placebo_required": False,
            "conclusion": "REJECTED",
        }
        errs = validate(rec)
        self.assertTrue(any("placebo_required" in e for e in errs), str(errs))

    def test_enums(self):
        self.assertEqual(set(SURGERY_TIERS), {"EXACT", "APPROX", "UNREACHABLE"})
        self.assertEqual(set(CLAIM_CLASSES), {"pipeline_check", "behavior_fix", "causal_learn"})


if __name__ == "__main__":
    unittest.main()
