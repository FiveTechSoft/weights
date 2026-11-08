"""Logic tests for lab/meta/meta_learner.py (no torch, synthetic score tables)."""
import itertools
import random
import unittest

from lab.meta.meta_learner import (
    contextual_order, random_expected_trials, shuffled_experience, static_order,
    trials_to_success,
)

CONFIGS = [(l, e, 0.01) for l, e in itertools.product((1, 3), (1.0, 2.0, 3.0))]


def table(best):
    return {c: (1.0 if c == best else 0.1) for c in CONFIGS}


class TestMetaLearner(unittest.TestCase):
    def test_trials_to_success_and_censoring(self):
        s = table(CONFIGS[2])
        self.assertEqual(trials_to_success(CONFIGS, s, 0.95, 6), 3)
        self.assertEqual(trials_to_success(CONFIGS, s, 0.95, 2), 3)  # censored at budget+1
        self.assertEqual(trials_to_success(CONFIGS, s, 2.0, 6), 7)   # never succeeds

    def test_random_expectation_matches_brute_force(self):
        s = {c: (1.0 if i in (1, 4) else 0.0) for i, c in enumerate(CONFIGS)}
        brute = [trials_to_success(p, s, 0.95, 4) for p in itertools.permutations(CONFIGS)]
        self.assertAlmostEqual(random_expected_trials(s, 0.95, 4), sum(brute) / len(brute), places=9)

    def test_contextual_uses_the_nearest_task_static_does_not(self):
        exp = [((1.0, 0.0), table(CONFIGS[0])), ((9.0, 0.0), table(CONFIGS[5]))]
        self.assertEqual(contextual_order(exp, (8.5, 0.0), CONFIGS, k=1)[0], CONFIGS[5])
        self.assertEqual(contextual_order(exp, (1.5, 0.0), CONFIGS, k=1)[0], CONFIGS[0])
        self.assertEqual(static_order(exp, CONFIGS)[0], CONFIGS[0])  # tie -> list order

    def test_shuffled_control_keeps_scores_but_moves_labels(self):
        exp = [((float(i), 0.0), table(CONFIGS[i])) for i in range(6)]
        sh = shuffled_experience(exp, random.Random(3))
        self.assertEqual(sorted(f for f, _ in sh), sorted(f for f, _ in exp))
        self.assertEqual([s for _, s in sh], [s for _, s in exp])
        self.assertNotEqual([f for f, _ in sh], [f for f, _ in exp])

    def test_empty_experience_returns_input_order(self):
        self.assertEqual(contextual_order([], (1.0, 0.0), CONFIGS), CONFIGS)
        self.assertEqual(static_order([], CONFIGS), CONFIGS)


if __name__ == "__main__":
    unittest.main()


class TestRelativeEncoding(unittest.TestCase):
    def test_relative_rule_transfers_to_an_unseen_old_layer(self):
        from lab.meta.meta_learner import relative_contextual_order
        cfgs = [(l, 1.0, 0.01) for l in (1, 2, 3, 4, 5)]
        # Rule in the experience: the good layer equals the old layer (offset 0).
        def tab(old):
            return {c: (1.0 if c[0] == old else 0.0) for c in cfgs}
        exp = [((1 / 2, 5.0, 5.0), tab(1)), ((3 / 2, 5.0, 5.0), tab(3)), ((5 / 2, 5.0, 5.0), tab(5))]
        # Old layer 2 and 4 never appear in the experience.
        self.assertEqual(relative_contextual_order(exp, (2 / 2, 5.0, 5.0), cfgs)[0], (2, 1.0, 0.01))
        self.assertEqual(relative_contextual_order(exp, (4 / 2, 5.0, 5.0), cfgs)[0], (4, 1.0, 0.01))
        # The absolute proposer cannot express it: it proposes a layer from a neighbouring task.
        self.assertNotEqual(contextual_order(exp, (2 / 2, 5.0, 5.0), cfgs, k=1)[0], (2, 1.0, 0.01))

    def test_relative_without_experience_keeps_input_order(self):
        from lab.meta.meta_learner import relative_contextual_order
        cfgs = [(1, 1.0, 0.01), (2, 1.0, 0.01)]
        self.assertEqual(relative_contextual_order([], (0.5, 1.0, 1.0), cfgs), cfgs)
