"""test_wspx: integridad de los datasets exactos (idempotencia, particion, oracle tags)."""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab"))
from wspx_datasets import BUILDERS, fs4, recall, mod13, parity8, noecho  # noqa: E402


class TestWspxDatasets(unittest.TestCase):

    def test_all_builders_idempotent(self):
        for name, fn in BUILDERS.items():
            a, b = fn(), fn()
            self.assertEqual(a["sha256"], b["sha256"], name)
            self.assertEqual(a["universe"], b["universe"], name)

    def test_partitions_disjoint(self):
        for name, fn in BUILDERS.items():
            ds = fn()
            tr = {x["prompt"] for x in ds["train"]}
            ho = {x["prompt"] for x in ds["holdout"]}
            self.assertFalse(tr & ho, name)
            self.assertEqual(tr | ho, {x["prompt"] for x in ds["universe"]}, name)

    def test_fsm4_oracle(self):
        ds = fs4()
        by_p = {r["prompt"]: r["target"] for r in ds["universe"]}
        self.assertEqual(by_p["FSM: a ="], "1")
        self.assertEqual(by_p["FSM: b ="], "2")
        self.assertEqual(by_p["FSM: ab ="], "3")
        self.assertEqual(by_p["FSM: abb ="], "1")
        self.assertEqual(len(by_p), 62)

    def test_parity8_oracle(self):
        ds = parity8()
        by_p = {r["prompt"]: r["target"] for r in ds["universe"]}
        self.assertEqual(by_p["P8: 00000000 ="], "0")
        self.assertEqual(by_p["P8: 11111111 ="], "0")
        self.assertEqual(by_p["P8: 10000000 ="], "1")
        self.assertEqual(by_p["P8: 11111110 ="], "1")

    def test_mod13_targets(self):
        ds = mod13()
        by_p = {r["prompt"]: r["target"] for r in ds["universe"]}
        self.assertEqual(by_p["M13: 12+3 ="], "2")
        self.assertEqual(by_p["M13: 4*5 ="], "7")
        self.assertEqual(by_p["M13: 12*12 ="], "1")
        self.assertEqual(len(by_p), 338)

    def test_recall_200_unique_keys(self):
        ds = recall()
        ks = [r["prompt"].split()[1] for r in ds["universe"]]
        self.assertEqual(len(ks), 200)
        self.assertEqual(len(set(ks)), 200)
        for r in ds["universe"]:
            self.assertEqual(len(r["target"]), 1)

    def test_noecho_all_x(self):
        ds = noecho()
        self.assertTrue(all(r["target"] == "x" for r in ds["universe"]))
        self.assertEqual(ds["n_universe"], 500)

    def test_answer_len_le2(self):
        for name, fn in BUILDERS.items():
            for r in fn()["universe"]:
                self.assertLessEqual(len(r["target"]), 2, (name, r))

    def test_certify_selftest_passes(self):
        import importlib
        certify = importlib.import_module("certify")
        for name in BUILDERS:
            ds = BUILDERS[name]()
            self.assertEqual(certify.oracle_selftest(ds), "OK", name)

    def test_universe_eval_verdicts(self):
        import importlib
        certify = importlib.import_module("certify")
        ds = BUILDERS["fsm4"]()
        perfect = {r["prompt"]: r["target"] for r in ds["universe"]}
        res = certify.universe_eval(ds, perfect)
        self.assertEqual(res["verdict"], "EXACT")
        res2 = certify.universe_eval(ds, {r["prompt"]: "9" for r in ds["universe"]})
        self.assertEqual(res2["verdict"], "FAILED")
        half = dict(list(perfect.items())[: len(perfect) // 2 + 1])
        res3 = certify.universe_eval(ds, half)
        self.assertEqual(res3["verdict"], "APPROX")


if __name__ == "__main__":
    unittest.main()
