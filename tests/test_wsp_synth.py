"""Contract tests for closed-form weight synthesis: n-gram tables compiled into nanoGPT without gradients."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "lab" / "reference" / "karpathy_nanogpt")]

from engine import wsp_synth, wsp_train  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

ARCH = ROOT / "programs" / "arch" / "nanogpt.wsparch"
INPUT = ROOT / "lab" / "data" / "shakespeare_char" / "input.txt"
TEXT = INPUT.read_text(encoding="utf-8")
DATA = wsp_train.char_dataset(TEXT)
SHAPE = dict(block=64, layers=1, heads=6, d=384, dropout=0.0, bias=False)
LOGIT_TOLERANCE = 0.01


def table_and_model(order, **shape):
    table = wsp_synth.WittenBellTable(DATA.train, len(DATA.chars), order)
    model = wsp_synth.compile_ngram(ARCH, table, overrides={**SHAPE, **shape}, seed=1)
    return table, model.eval()


def max_logprob_error(table, model, order, windows=8, length=64):
    ids = DATA.val
    worst = 0.0
    for w in range(windows):
        x = torch.tensor(ids[w * length:w * length + length][None])
        with torch.no_grad():
            logp = torch.log_softmax(model(x, x)[0][0].double(), -1).numpy()
        for t in range(order - 1, length):
            context = tuple(int(v) for v in ids[w * length + t - order + 2:w * length + t + 1])
            worst = max(worst, np.abs(logp[t] - table.log_prob(context)).max())
    return worst


class TestWittenBell(unittest.TestCase):
    IDS = np.array([0, 1, 2, 0, 1, 0, 1, 2], dtype=np.int64)

    def test_rows_are_distributions(self):
        table = wsp_synth.WittenBellTable(self.IDS, 4, 3)
        for context in [(), (0,), (0, 1), (3,), (3, 3)]:
            self.assertAlmostEqual(np.exp(table.log_prob(context)).sum(), 1.0)

    def test_known_value(self):
        """After (0,1): counts {2: 2, 0: 1}, so n=3 and T=2 followers."""
        table = wsp_synth.WittenBellTable(self.IDS, 4, 3)
        low = np.exp(table.log_prob((1,)))
        self.assertAlmostEqual(np.exp(table.log_prob((0, 1)))[2], (2 + 2 * low[2]) / 5)

    def test_unseen_context_backs_off(self):
        table = wsp_synth.WittenBellTable(self.IDS, 4, 3)
        self.assertTrue(np.array_equal(table.log_prob((3, 1)), table.log_prob((1,))))


class TestCompilation(unittest.TestCase):
    def test_bigram_matches_table(self):
        table, model = table_and_model(2)
        self.assertLess(max_logprob_error(table, model, 2), LOGIT_TOLERANCE)

    def test_trigram_matches_table(self):
        table, model = table_and_model(3)
        self.assertLess(max_logprob_error(table, model, 3), LOGIT_TOLERANCE)

    def test_order_four_matches_table_through_attention(self):
        table, model = table_and_model(4)
        self.assertLess(max_logprob_error(table, model, 4), LOGIT_TOLERANCE)

    def test_attention_is_actually_used(self):
        """Without attention output the model cannot know previous tokens."""
        table, model = table_and_model(3)
        with torch.no_grad():
            model.get_parameter("transformer.h.0.attn.c_proj.weight").zero_()
        self.assertGreater(max_logprob_error(table, model, 3), 1.0)

    def test_previous_token_changes_prediction(self):
        _, model = table_and_model(3)
        a = torch.tensor([[DATA.chars.index("q"), DATA.chars.index("u")]])
        b = torch.tensor([[DATA.chars.index("t"), DATA.chars.index("u")]])
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(a)[0], model(b)[0], atol=0.1))

    def test_too_many_positions_or_orders_are_rejected(self):
        table = wsp_synth.WittenBellTable(DATA.train[:5000], len(DATA.chars), 8)
        with self.assertRaises(wsp_synth.SynthesisError):
            wsp_synth.compile_ngram(ARCH, table, overrides=SHAPE, seed=1)

    def test_vocab_larger_than_head_code_is_rejected(self):
        table = wsp_synth.WittenBellTable(DATA.train[:5000], len(DATA.chars), 2)
        with self.assertRaises(wsp_synth.SynthesisError):
            wsp_synth.compile_ngram(ARCH, table, overrides={**SHAPE, "heads": 8, "d": 128}, seed=1)


class TestRealNanoGPTShape(unittest.TestCase):
    """At upstream's 4*d MLP width the weights must run unchanged in Karpathy's own model.py."""

    @classmethod
    def setUpClass(cls):
        cls.table = wsp_synth.WittenBellTable(DATA.train, len(DATA.chars), 3)
        cls.model = wsp_synth.compile_ngram(ARCH, cls.table, overrides={**SHAPE, "layers": 2, "ff": 1536}, seed=1).eval()

    def test_upstream_model_runs_these_weights(self):
        upstream = GPT(GPTConfig(vocab_size=len(DATA.chars), block_size=64, n_layer=2, n_head=6, n_embd=384,
                                 dropout=0.0, bias=False)).eval()
        upstream.load_state_dict(self.model.export_state_dict(), strict=True)
        x = torch.tensor(DATA.val[:64][None])
        with torch.no_grad():
            self.assertTrue(torch.equal(upstream(x, x)[0], self.model(x, x)[0]))
        self.assertLess(max_logprob_error(self.table, upstream, 3), LOGIT_TOLERANCE)

    def test_capacity_keeps_most_frequent_contexts(self):
        table = wsp_synth.WittenBellTable(DATA.train, len(DATA.chars), 4)
        model = wsp_synth.compile_ngram(ARCH, table, overrides={**SHAPE, "layers": 2, "ff": 1536}, seed=1)
        self.assertLess(model.synthesis_info["units_used"], table.num_units())
        self.assertEqual(model.synthesis_info["units_used"], 2 * 1536)


class TestNoGradients(unittest.TestCase):
    def test_guard_blocks_every_gradient_entry_point(self):
        x = torch.ones(1, requires_grad=True)
        loss = (x * 2).sum()
        with wsp_synth.forbid_gradients():
            with self.assertRaises(wsp_synth.SynthesisError):
                loss.backward()
            with self.assertRaises(wsp_synth.SynthesisError):
                torch.autograd.backward(loss)
            with self.assertRaises(wsp_synth.SynthesisError):
                torch.autograd.grad(loss, x)
        loss.backward()  # guard restored

    def test_compiled_parameters_never_received_a_gradient(self):
        _, model = table_and_model(2)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


def run_step(args, allow_write=True, budget=100000000):
    code = f"""experiment t:
model: nanoGPT
budgets:
    max_modified_weights = {budget}
    allow_write = {str(allow_write).lower()}
steps:
    step synthesize {args}
"""
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    assert not errors, errors
    return execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)[0]


class TestWSPLangOp(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.out = Path(d.name) / "m.safetensors"
        self.base = (f"method=ngram order=2 arch=programs/arch/nanogpt.wsparch data=lab/data/shakespeare_char/input.txt "
                     f"out={self.out.as_posix()} block=64 layers=1 heads=6 d=384 dropout=0.0 bias=false seed=1")

    def test_step_writes_checkpoint_with_provenance(self):
        res = run_step(self.base)
        self.assertEqual(res["status"], "ok", res)
        with safe_open(str(self.out), framework="pt") as f:
            meta = f.metadata()
        self.assertEqual(meta["provenance"], "closed_form")
        self.assertEqual(meta["backward_calls"], "0")
        self.assertEqual(res["result"]["backward_calls"], 0)
        self.assertIn("transformer.wte.weight", load_file(str(self.out)))

    def test_step_needs_no_training_budget_but_respects_write_budget(self):
        res = run_step(self.base, allow_write=False)
        self.assertEqual(res["execution_state"], "simulated")
        self.assertFalse(self.out.exists())

    def test_parameter_budget_is_enforced(self):
        res = run_step(self.base, budget=100)
        self.assertEqual(res["status"], "error")
        self.assertIn("max_modified_weights", res["error"])

    def test_unknown_method_is_an_error(self):
        res = run_step(self.base.replace("method=ngram", "method=magic"))
        self.assertEqual(res["status"], "error")
        self.assertIn("magic", res["error"])


if __name__ == "__main__":
    unittest.main()
