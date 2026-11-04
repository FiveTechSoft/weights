"""tests compile_from_trace: deteccion de tier determinista + record válido + ediciones emitidas."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest
from compile_from_trace import detect_tier, compile_row, experiment_from_trace  # noqa: E402


class TestCompiler(unittest.TestCase):

    def test_rule_exact_parity(self):
        tier, rule = detect_tier({"last_linear_share": 0.9}, "parity8")
        self.assertEqual((tier, rule), ("EXACT", "R1-compose_circuit"))

    def test_rule_approx_last_linear(self):
        tier, rule = detect_tier({"last_linear_share": 0.925}, "recall")
        self.assertEqual((tier, rule), ("APPROX", "R2-ridge-readout"))

    def test_rule_unreachable(self):
        tier, rule = detect_tier({"last_linear_share": 0.2}, "recall")
        self.assertEqual(tier, "UNREACHABLE")

    def test_program_json_clean(self):
        p = compile_row("recall", {"last_linear_share": 0.9})
        self.assertEqual(p["rule_id"], "R2-ridge-readout")
        json.dumps(p)

    def test_experiment_from_trace_valid_record(self):
        tr_path = Path(__file__).resolve().parent.parent / "lab" / "runs" / "finetune_recall.npz"
        if not tr_path.exists():
            self.skipTest("trace de finetune no construido aun")
        rec, errs, program = experiment_from_trace(str(tr_path))
        self.assertEqual(errs, [], errs)
        self.assertTrue(rec["tier_inferred"])
        self.assertIn(rec["surgery_tier"], ("EXACT", "APPROX", "UNREACHABLE"))
        # placebo exigido para claim causal
        self.assertTrue(rec["placebo_required"])


if __name__ == "__main__":
    unittest.main()
