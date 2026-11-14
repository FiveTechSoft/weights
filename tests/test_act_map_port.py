"""lab/probes/act_map/port.py (PLAN.md gate 3): a contrast measured in one model, written into another through the
gate 1 maps, and the gate 3 decision rule."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "act_map"))

import act_map as am  # noqa: E402
import port as pt  # noqa: E402


class TestPortDiff(unittest.TestCase):
    def test_mapped_difference_equals_difference_of_mapped_states(self):
        g = torch.Generator().manual_seed(0)
        X = torch.randn(200, 6, generator=g, dtype=torch.float64)
        Y = X @ torch.randn(6, 4, generator=g, dtype=torch.float64) + 3.0
        src = am.Source(X)
        W, my = src.solve(Y, 1e-3)
        a, b = X[:50].mean(0), X[50:].mean(0)
        want = src.predict(a[None], W, my)[0] - src.predict(b[None], W, my)[0]
        D_src = torch.stack([torch.zeros(6), a - b])                     # source states 0, 1
        got = pt.port_diff(D_src, {1: W, 2: W}, {1: 1, 2: 0}, n_dst=3)
        torch.testing.assert_close(got[1], want)
        torch.testing.assert_close(got[2], torch.zeros(4, dtype=torch.float64))   # from source state 0 (zero diff)
        torch.testing.assert_close(got[0], torch.zeros(4, dtype=torch.float64))   # state 0 is never steered

    def test_each_target_state_reads_its_own_source_state(self):
        D_src = torch.eye(3, dtype=torch.float64)
        I = torch.eye(3, dtype=torch.float64)
        got = pt.port_diff(D_src, {1: I, 2: I}, {1: 2, 2: 1}, n_dst=3)
        torch.testing.assert_close(got[1], D_src[2])
        torch.testing.assert_close(got[2], D_src[1])

    def test_increments_leave_out_the_last_layer(self):
        # HF's last hidden state is after the final norm: its difference with the state before is not a residual
        # increment (0.8B: norm 44.9 against <= 1 for the others). Steered layers: 1 .. n_layers - 2.
        D = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2) ** 2            # 5 layers + embeddings
        inc = pt.residual_increments(D)
        self.assertEqual(sorted(inc), [1, 2, 3])
        torch.testing.assert_close(torch.as_tensor(sum(inc.values())), D[4] - D[1])

    def test_cosines_per_state_skip_state_zero(self):
        A = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        B = torch.tensor([[0.0, 0.0], [2.0, 0.0], [-1.0, -1.0]])
        self.assertEqual([round(c, 6) for c in pt.cos_by_state(A, B)], [1.0, -1.0])


def stats(tone, lex=3.0, coh=40, tp=1e-4, lp=1e-4):
    return {"tone_diff": tone, "tone_sign_p": tp, "lex_diff": lex, "lex_sign_p": lp, "coherent": coh, "n": 40}


class TestDecide(unittest.TestCase):
    # amendment 2: coherence relative to the base's own (judge calls only 5/10 0.8B base answers coherent)
    def good(self):
        return {"own": stats(1.6, coh=24), "port": stats(1.2, coh=24), "embport": stats(0.5), "placebo": stats(0.1)}

    def test_pass_and_p6_read_separately(self):
        d = pt.decide(self.good(), base_test_coherent=26)
        self.assertEqual(d["verdict"], "PASS")
        self.assertTrue(d["P6 port beats embport by >= 0.5"])
        s = self.good()
        s["embport"] = stats(1.0)
        d = pt.decide(s, base_test_coherent=26)
        self.assertEqual(d["verdict"], "PASS")                   # P6 does not decide the gate
        self.assertFalse(d["P6 port beats embport by >= 0.5"])

    def test_coherence_is_relative_to_the_base(self):
        s = self.good()
        self.assertEqual(pt.decide(s, base_test_coherent=28)["verdict"], "PASS")     # 24 >= 28 - 4
        s["port"] = stats(1.2, coh=23)
        self.assertEqual(pt.decide(s, base_test_coherent=28)["verdict"], "FAIL")     # 23 < 24

    def test_no_own_direction_is_not_decided(self):
        s = self.good()
        s["own"] = stats(0.8)
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "NOT DECIDED: own fails C1-C4")
        s["own"] = stats(1.6, coh=21)
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "NOT DECIDED: own fails C1-C4")
        s["own"] = None                                          # no dev scale was coherent
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "NOT DECIDED: own fails C1-C4")

    def test_each_port_criterion_can_fail(self):
        for k, v in {"P1": stats(0.9), "P1 p": stats(1.2, tp=0.02), "P2": stats(1.2, lex=1.0),
                     "P2 p": stats(1.2, lp=0.02), "P4": stats(1.2, coh=21)}.items():
            s = self.good()
            s["port"] = v
            self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "FAIL", k)
        s = self.good()
        s["own"], s["port"] = stats(2.2), stats(1.05)            # P1 holds (1.05 >= 1.0), P5 needs >= 1.1
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "FAIL", "P5")
        s = self.good()
        s["placebo"] = stats(0.5)
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "FAIL", "P3")
        s = self.good()
        s["port"] = None
        self.assertEqual(pt.decide(s, base_test_coherent=26)["verdict"], "FAIL", "no port scale")

    def test_dev_scale_rule_is_relative_to_the_base(self):
        coh = {0.125: 6, 0.25: 4, 0.5: 6, 1.0: 2, 2.0: 0, 4.0: 0}
        self.assertEqual(pt.choose_scale(coh, base_dev_coherent=5), 0.5)       # largest with >= 5 - 1
        self.assertEqual(pt.choose_scale(coh, base_dev_coherent=10), None)     # none with >= 9
        self.assertEqual(pt.choose_scale({0.125: 4, 0.25: 7}, base_dev_coherent=5), 0.25)


class TestHappyRunsOption(unittest.TestCase):
    def test_runs_directory_can_be_redirected(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, HAPPY_RUNS=d)
            out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); import happy; "
                                  "print(happy.RUNS)" % (ROOT / "lab" / "probes" / "happy")],
                                 env=env, capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(Path(out), Path(d))
            env.pop("HAPPY_RUNS")
            out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); import happy; "
                                  "print(happy.RUNS)" % (ROOT / "lab" / "probes" / "happy")],
                                 env=env, capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(Path(out), ROOT / "lab" / "runs" / "happy")


class TestArmStats(unittest.TestCase):
    def test_paired_differences_against_base(self):
        base = [{"tone": 2, "lex": 0.0, "coherent": True}] * 10
        arm = [{"tone": 4, "lex": 3.0, "coherent": True}] * 9 + [{"tone": None, "lex": 3.0, "coherent": False}]
        s = pt.arm_stats(arm, base)
        self.assertAlmostEqual(s["tone_diff"], (9 * 2 - 2) / 10)    # unparsed tone counts as 0, as happy.verdict
        self.assertEqual(s["coherent"], 9)
        self.assertAlmostEqual(s["lex_diff"], 3.0)
        self.assertLess(s["lex_sign_p"], 0.01)


if __name__ == "__main__":
    unittest.main()
