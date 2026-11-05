"""tests/test_training_to_surgery_benchmarks.py — Verification of v1, v2, v3 benchmarks."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import benchmarks.training_to_surgery_v1 as b_v1
import benchmarks.training_to_surgery_v2 as b_v2
import benchmarks.training_to_surgery_v3_meta_transfer as b_v3


class TestTrainingToSurgeryBenchmarks(unittest.TestCase):

    def test_v1_pipeline(self):
        out_dir = ROOT / "runs" / "test_v1"
        args = b_v1.argparse.Namespace(
            device="cpu",
            seed=7,
            n_symbols=16,
            n_train_symbols=12,
            n_labels=4,
            hidden=24,
            lr=0.08,
            steps=50,
            out=str(out_dir),
        )
        b_v1.run(args)
        report_p = out_dir / "report.json"
        self.assertTrue(report_p.exists())
        data = json.loads(report_p.read_text())
        self.assertEqual(data["steps"], 50)
        self.assertIn("trained_model", data["scores"])
        self.assertIn("direct_delta_replay", data["scores"])

    def test_v2_closed_form(self):
        out_dir = ROOT / "runs" / "test_v2"
        args = b_v2.argparse.Namespace(
            device="cpu",
            seed=7,
            n_symbols=16,
            n_train_symbols=12,
            n_labels=4,
            hidden=24,
            lr=0.08,
            steps=50,
            out=str(out_dir),
        )
        b_v2.run(args)
        report_p = out_dir / "report_v2.json"
        self.assertTrue(report_p.exists())
        data = json.loads(report_p.read_text())
        self.assertEqual(data["metrics"]["arm2_wsplang_closed_form_surgery"]["backprop_steps"], 0)
        self.assertTrue(data["verdict"]["placebo_eliminated"])

    def test_v3_meta_transfer(self):
        out_dir = ROOT / "runs" / "test_v3"
        args = b_v3.argparse.Namespace(
            device="cpu",
            seed=42,
            n_meta_tasks=3,
            n_test_tasks=2,
            n_symbols=16,
            n_train_symbols=12,
            n_labels=4,
            hidden=24,
            lr=0.08,
            steps=50,
            lambda_meta=0.05,
            out=str(out_dir),
        )
        b_v3.run_benchmark(args)
        report_p = out_dir / "report_v3.json"
        self.assertTrue(report_p.exists())
        data = json.loads(report_p.read_text())
        self.assertEqual(data["metrics_summary"]["arm2_meta_learned_surgery"]["backprop_steps"], 0)
        self.assertFalse(data["metrics_summary"]["arm2_meta_learned_surgery"]["access_to_target_sgd_delta"])
        self.assertTrue(data["verdict"]["zero_delta_access_verified"])
        self.assertGreater(data["metrics_summary"]["meta_transfer_fidelity"]["mean_cosine_similarity"], 0.95)


if __name__ == "__main__":
    unittest.main()
