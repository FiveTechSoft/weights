"""Contract tests for the LLM-ness gate (lab/llmness.py).

The gate is only trustworthy if known-good and known-bad models land where they should,
so these tests use small controls instead of real checkpoints.
"""
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab"))

import llmness as lm  # noqa: E402
import nano3lib as ng  # noqa: E402


def markov_text(n=30000, seed=3):
    rng = random.Random(seed)
    words = ["the", "king", "said", "and", "queen", "went", "to", "her", "house", "with", "a", "sword"]
    out = []
    while sum(map(len, out)) + len(out) < n:
        out.append(rng.choice(words))
    return " ".join(out)[:n]


class NextCharOracle(lm.ByteCodec):
    """Knows the periodic corpus exactly: predicts the true next char, from `lookback` chars back."""

    def __init__(self, period_text, lookback=None):
        self.text = period_text
        self.period = len(period_text)
        self.lookback = lookback

    def logits(self, x):
        B, S = x.shape
        out = torch.full((B, S, 256), -20.0)
        for b in range(B):
            for t in range(S):
                if self.lookback is None:
                    # next char = char following the current one in the unique-char period
                    nxt = self.text[(self.text.index(chr(int(x[b, t]))) + 1) % self.period]
                    out[b, t, ord(nxt)] = 20.0
                elif t + 1 >= self.lookback:
                    out[b, t, int(x[b, t + 1 - self.lookback])] = 20.0
                else:
                    out[b, t] = 0.0
        return out


class ValueCopier(lm.ByteCodec):
    """Copies a value from context while ignoring the key: must NOT count as in-context learning."""

    def logits(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], 256)
        for b in range(x.shape[0]):
            text = self.decode(x[b].tolist())
            out[b, -1, ord(text.splitlines()[0].split(": ")[1][0])] = 10.0
        return out


class TestAlignment(unittest.TestCase):
    def test_perfect_next_token_predictor_scores_near_zero(self):
        period = "abcdefghij"
        nll = lm.score_nll(NextCharOracle(period), period * 40, seq=32, batch=4)
        self.assertLess(nll, 1e-3)

    def test_two_step_ahead_predictor_scores_high(self):
        """A model that is right only about t+2 is what the old off-by-one rewarded."""
        period = "abcdefghij"

        class TwoAhead(lm.ByteCodec):
            def logits(self, x):
                out = torch.full((x.shape[0], x.shape[1], 256), -20.0)
                target = (x - ord("a") + 2) % 10 + ord("a")
                return out.scatter(2, target.unsqueeze(-1), 20.0)

        self.assertGreater(lm.score_nll(TwoAhead(), period * 40, seq=32, batch=4), 5.0)


class TestNanoLibLoading(unittest.TestCase):
    def _fake_trace(self, names, shapes, n_heads):
        import json
        flat = np.random.default_rng(0).standard_normal(sum(int(np.prod(s)) for s in shapes.values()), dtype=np.float32)
        return {"W0_flat": flat, "snapshots_flat": flat[None], "names": json.dumps(json.dumps(names)),
                "shapes": json.dumps(json.dumps(shapes)), "meta": json.dumps({"n_heads": n_heads})}

    def test_build_from_trace_accepts_nanowsp4_names(self):
        names = ["embed.weight", "q.0.weight"]
        tr = self._fake_trace(names, {"embed.weight": [4, 2], "q.0.weight": [2, 2]}, 1)
        P = ng.build_from_trace(tr, -1)
        self.assertEqual(set(P), {"embed.weight", "q.0.weight"})

    def test_build_from_trace_keeps_nanowsp3_legacy_names(self):
        import json
        tr = {"W0_flat": np.zeros(4, dtype=np.float32), "names": json.dumps(["q.0"]), "shapes": json.dumps({"q.0": [2, 2]})}
        self.assertEqual(list(ng.build_from_trace(tr)), ["q.0.weight"])

    def test_explicit_head_count_overrides_legacy_guess(self):
        P = {"embed.weight": torch.zeros(1, 384), "q.0.weight": torch.zeros(1), "min_.0.weight": torch.zeros(1024, 384)}
        self.assertEqual(ng._resolve_cfg(P)[2], 12)  # legacy d_model // 32
        self.assertEqual(ng._resolve_cfg(P, n_heads=6)[2:4], (6, 64))


class TestLanguageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = markov_text()
        cls.train, cls.test_text = text[:-4000], text[-4000:]

    def test_baseline_order_ngram_fails_l1_by_construction(self):
        l1 = lm.check_l1(lm.NGramModel(self.train, order=3), self.train, self.test_text, baseline_order=3)
        self.assertEqual(l1["verdict"], "FAILS")

    def test_baseline_is_scored_with_the_model_window(self):
        model = lm.NGramModel(self.train, order=3)
        model.max_seq = 16
        l1 = lm.check_l1(model, self.train, self.test_text, baseline_order=3)
        self.assertEqual(l1["nll"], l1["baseline_nll"])

    def test_unigram_fails_l1(self):
        l1 = lm.check_l1(lm.NGramModel(self.train, order=1), self.train, self.test_text, baseline_order=3)
        self.assertEqual(l1["verdict"], "FAILS")

    def test_higher_order_table_beats_baseline_in_l1(self):
        l1 = lm.check_l1(lm.NGramModel(self.train, order=5), self.train, self.test_text, baseline_order=3)
        self.assertEqual(l1["verdict"], "PASSES")

    def test_lookup_table_has_no_long_context_gain(self):
        model = lm.NGramModel(self.train, order=5)
        res = lm.check_long_context(model, self.test_text, short=8, long=32, n=100)
        self.assertEqual(res["mean_gain_nats"], 0.0)
        self.assertEqual(res["verdict"], "FAILS")

    def test_long_memory_model_has_long_context_gain(self):
        rng = random.Random(5)
        period = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(20))
        res = lm.check_long_context(NextCharOracle(period, lookback=20), period * 30, short=8, long=32, n=100)
        self.assertGreater(res["ci95_low"], 0.0)
        self.assertEqual(res["verdict"], "PASSES")

    def test_position_only_model_has_no_long_context_gain(self):
        """Position embeddings alone must not look like use of long context."""

        class PositionOnly(lm.ByteCodec):
            def logits(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], 256)
                out[:, 32:, ord(" ")] = 3.0
                return out

        res = lm.check_long_context(PositionOnly(), self.test_text)
        self.assertEqual(res["mean_gain_nats"], 0.0)
        self.assertEqual(res["verdict"], "FAILS")

    def test_random_nanogpt_has_no_long_context_gain(self):
        """Seeds 3, 5 and 9 were false positives when L2 truncated the context on Shakespeare."""
        sys.path.insert(0, str(ROOT / "lab" / "reference" / "karpathy_nanogpt"))
        from model import GPT, GPTConfig
        chars = sorted(set(self.test_text))
        for seed in (3, 5, 9):
            torch.manual_seed(seed)
            net = GPT(GPTConfig(vocab_size=len(chars), block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0))
            res = lm.check_long_context(lm.NanoGPTCharModel(net, chars), self.test_text)
            self.assertEqual(res["verdict"], "FAILS", f"seed {seed}: {res}")

    def test_generation_check_ignores_prompt(self):
        """Natural prompt must not rescue a model that only emits one repeated char."""

        class Stuck(lm.ByteCodec):
            def logits(self, x):
                out = torch.full((x.shape[0], x.shape[1], 256), -20.0)
                out[:, :, ord("z")] = 20.0
                return out

        res = lm.check_generation(Stuck(), prompt=self.test_text[:200], train_text=self.train, n_new=100)
        self.assertEqual(res["verdict"], "FAILS")
        self.assertLessEqual(res["distinct_3grams_ratio"], 0.05)

    def test_generation_check_passes_real_vocabulary(self):
        model = lm.NGramModel(self.train, order=5)
        res = lm.check_generation(model, prompt=self.test_text[:50], train_text=self.train, n_new=200)
        self.assertGreaterEqual(res["known_word_rate"], 0.5)


class TestCorpusSplit(unittest.TestCase):
    TEXT = "".join(chr(97 + i % 26) for i in range(1000))

    def test_baseline_excludes_whole_holdout_not_just_eval_slice(self):
        train, test = lm.split_corpus(self.TEXT, holdout_frac=0.2, eval_chars=50)
        self.assertEqual(train, self.TEXT[:800])
        self.assertEqual(test, self.TEXT[-50:])

    def test_eval_slice_must_fit_inside_holdout(self):
        with self.assertRaises(ValueError):
            lm.split_corpus(self.TEXT, holdout_frac=0.01, eval_chars=50)


class TestICL(unittest.TestCase):
    def test_cases_bind_query_only_in_treatment_arm(self):
        for c in lm.icl_cases(100, seed=1):
            self.assertIn(f"{c['key']}: {c['target']}\n", c["with_binding"])
            self.assertNotIn(f"{c['key']}: ", c["without_binding"].rsplit("\n", 1)[0])
            self.assertTrue(c["with_binding"].endswith(f"{c['key']}: "))
            self.assertTrue(c["without_binding"].endswith(f"{c['key']}: "))

    def test_both_arms_show_same_values(self):
        for c in lm.icl_cases(100, seed=1):
            vals = lambda s: sorted(l.split(": ")[1] for l in s.split("\n")[:-1])  # noqa: E731
            self.assertEqual(vals(c["with_binding"]), vals(c["without_binding"]))

    def test_cases_only_use_shakespeare_char_alphabet(self):
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij: \n")
        for c in lm.icl_cases(50, seed=2):
            self.assertLessEqual(set(c["with_binding"]) | set(c["without_binding"]), allowed)

    def test_copy_oracle_passes(self):
        res = lm.check_icl(lm.CopyOracle(), n=200)
        self.assertEqual(res["acc_with_binding"], 1.0)
        self.assertEqual(res["verdict"], "PASSES")

    def test_key_blind_copier_fails(self):
        self.assertEqual(lm.check_icl(ValueCopier(), n=200)["verdict"], "FAILS")

    def test_uniform_model_fails(self):
        class Uniform(lm.ByteCodec):
            def logits(self, x):
                return torch.zeros(x.shape[0], x.shape[1], 256)

        self.assertEqual(lm.check_icl(Uniform(), n=200)["verdict"], "FAILS")

    def test_exact_sign_test(self):
        self.assertAlmostEqual(lm.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(lm.sign_test_p(10, 0), 2 * 0.5 ** 10)
        self.assertAlmostEqual(lm.sign_test_p(5, 5), 1.0)


class TestVerdict(unittest.TestCase):
    P, F = {"verdict": "PASSES"}, {"verdict": "FAILS"}

    def test_any_language_check_failing_is_fails(self):
        self.assertEqual(lm.overall_verdict(self.F, self.P, self.P, self.P), "FAILS")
        self.assertEqual(lm.overall_verdict(self.P, self.F, self.P, self.P), "FAILS")
        self.assertEqual(lm.overall_verdict(self.P, self.P, self.F, self.P), "FAILS")

    def test_language_model_without_icl(self):
        self.assertEqual(lm.overall_verdict(self.P, self.P, self.P, self.F), "LANGUAGE_MODEL")

    def test_language_model_with_icl(self):
        self.assertEqual(lm.overall_verdict(self.P, self.P, self.P, self.P), "LANGUAGE_MODEL_ICL")


class TestNgramControl(unittest.TestCase):
    def test_predictive_distribution_sums_to_one(self):
        m = lm.NGramModel("abcabcabd", order=3)
        x = torch.tensor([m.encode("abc")])
        probs = torch.exp(torch.log_softmax(m.logits(x), -1))
        self.assertAlmostEqual(float(probs[0, -1].sum()), 1.0, places=5)
        self.assertTrue(math.isfinite(float(m.logits(x).sum())))


if __name__ == "__main__":
    unittest.main()
