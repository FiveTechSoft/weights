"""lab/probes/close_probe/reward.py: IWIBench's single score for an edit, on fast_eval results.

gain = max(0, (edited target success - base target success) / (1 - base target success))
retention = share of the controls the base got right that the edited model still gets right
reward = gain * retention          (a no-op scores 0)"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


def run(rows):
    return {"items": [{"grp": g, "i": i, "ok": ok} for g, i, ok in rows]}


BASE = run([("A", 1, False), ("A", 2, False), ("A", 3, False), ("A", 4, True),
            ("ok", 10, True), ("ok", 11, True), ("ok", 12, False)])


class TestReward(unittest.TestCase):
    def test_no_op_scores_zero(self):
        import reward
        r = reward.score(BASE, BASE)
        self.assertEqual((r["gain"], r["retention"], r["reward"]), (0.0, 1.0, 0.0))

    def test_gain_times_retention(self):
        import reward
        ed = run([("A", 1, True), ("A", 2, True), ("A", 3, False), ("A", 4, True),
                  ("ok", 10, True), ("ok", 11, False), ("ok", 12, True)])
        r = reward.score(BASE, ed)
        self.assertAlmostEqual(r["gain"], (3 / 4 - 1 / 4) / (1 - 1 / 4))      # 2/3
        self.assertAlmostEqual(r["retention"], 1 / 2)                          # control 12 was not base-correct
        self.assertAlmostEqual(r["reward"], 1 / 3)
        self.assertEqual((r["target"], r["controls"]), ("3/4 (base 1/4)", "1/2 kept (+1 newly right)"))

    def test_worse_than_base_is_zero_gain_not_negative(self):
        import reward
        ed = run([("A", 1, False), ("A", 2, False), ("A", 3, False), ("A", 4, False),
                  ("ok", 10, True), ("ok", 11, True), ("ok", 12, False)])
        self.assertEqual(reward.score(BASE, ed)["gain"], 0.0)

    def test_without_controls_reward_is_none(self):
        import reward
        ed = run([("A", 1, True), ("A", 2, False), ("A", 3, False), ("A", 4, True)])
        r = reward.score(BASE, ed)
        self.assertIsNone(r["retention"])
        self.assertIsNone(r["reward"])

    def test_items_must_match(self):
        import reward
        with self.assertRaises(ValueError):
            reward.score(BASE, run([("A", 99, True)]))


if __name__ == "__main__":
    unittest.main()
