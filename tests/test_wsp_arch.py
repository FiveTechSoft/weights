"""Contract tests for the architecture language (.wsparch) and its compiler (engine/wsp_arch.py)."""
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "lab" / "reference" / "karpathy_nanogpt")]

from model import GPT, GPTConfig  # noqa: E402
from engine import wsp_arch  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402

NANOGPT = ROOT / "programs" / "arch" / "nanogpt.wsparch"
SMALL = dict(vocab=65, block=32, layers=2, heads=2, d=32)


def upstream_cfg(dropout=0.0, bias=True):
    return GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=dropout, bias=bias)


def compiled(seed=1337, **overrides):
    return wsp_arch.load_arch_file(NANOGPT, overrides={**SMALL, **overrides}, seed=seed)


def upstream(seed=1337, **kw):
    torch.manual_seed(seed)
    return GPT(upstream_cfg(**kw))


class TestUpstreamParity(unittest.TestCase):
    def test_same_seed_gives_identical_w0_and_parameter_order(self):
        theirs, ours = dict(upstream().named_parameters()), dict(compiled().named_parameters())
        self.assertEqual(list(ours), list(theirs))
        for name, p in ours.items():
            self.assertTrue(torch.equal(p, theirs[name]), name)

    def test_w0_matches_for_the_full_shakespeare_config(self):
        torch.manual_seed(1337)
        theirs = GPT(GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0))
        ours = wsp_arch.load_arch_file(NANOGPT, seed=1337)
        for name, p in ours.named_parameters():
            self.assertTrue(torch.equal(p, dict(theirs.named_parameters())[name]), name)

    def test_bias_false_w0_and_logits(self):
        theirs, ours = upstream(bias=False).eval(), compiled(bias=False).eval()
        self.assertEqual(list(dict(ours.named_parameters())), list(dict(theirs.named_parameters())))
        x = torch.randint(0, 65, (2, 16))
        self.assertTrue(torch.equal(theirs(x, x)[0], ours(x, x)[0]))

    def test_weights_are_tied_by_identity(self):
        ours = compiled()
        self.assertIs(ours.get_parameter("transformer.wte.weight"), ours.get_parameter("lm_head.weight"))

    def test_logits_and_loss_with_targets(self):
        theirs, ours = upstream().eval(), compiled().eval()
        x, y = torch.randint(0, 65, (3, 20)), torch.randint(0, 65, (3, 20))
        lt, lost = theirs(x, y)
        lo, loso = ours(x, y)
        self.assertTrue(torch.equal(lt, lo))
        self.assertEqual(lost.item(), loso.item())

    def test_inference_returns_only_last_position(self):
        theirs, ours = upstream().eval(), compiled().eval()
        x = torch.randint(0, 65, (3, 20))
        lt, lo = theirs(x)[0], ours(x)[0]
        self.assertEqual(lo.shape, (3, 1, 65))
        self.assertTrue(torch.equal(lt, lo))
        self.assertIsNone(ours(x)[1])

    def test_ignore_index_matches_upstream(self):
        theirs, ours = upstream().eval(), compiled().eval()
        x, y = torch.randint(0, 65, (2, 10)), torch.randint(0, 65, (2, 10))
        y[0, :5] = -1
        self.assertEqual(theirs(x, y)[1].item(), ours(x, y)[1].item())

    def test_dropout_stream_matches_in_train_mode(self):
        theirs, ours = upstream(dropout=0.2).train(), compiled(dropout=0.2).train()
        x, y = torch.randint(0, 65, (2, 16)), torch.randint(0, 65, (2, 16))
        torch.manual_seed(5)
        lt = theirs(x, y)[0]
        torch.manual_seed(5)
        lo = ours(x, y)[0]
        self.assertTrue(torch.equal(lt, lo))

    def test_training_trajectory_is_identical(self):
        theirs, ours = upstream().train(), compiled().train()
        ot = torch.optim.AdamW(theirs.parameters(), lr=1e-3, betas=(0.9, 0.99))
        oo = torch.optim.AdamW(ours.parameters(), lr=1e-3, betas=(0.9, 0.99))
        torch.manual_seed(11)
        for _ in range(25):
            x, y = torch.randint(0, 65, (4, 16)), torch.randint(0, 65, (4, 16))
            for model, opt in ((theirs, ot), (ours, oo)):
                opt.zero_grad(set_to_none=True)
                model(x, y)[1].backward()
                opt.step()
        for (name, pt), (_, po) in zip(theirs.named_parameters(), ours.named_parameters()):
            self.assertTrue(torch.equal(pt, po), name)

    def test_sampled_generation_matches_upstream_generate(self):
        theirs, ours = upstream().eval(), compiled().eval()
        x = torch.randint(0, 65, (1, 8))
        torch.manual_seed(9)
        gt = theirs.generate(x.clone(), 40, temperature=0.8, top_k=10)  # crosses block_size=32
        torch.manual_seed(9)
        go = ours.generate(x.clone(), 40, temperature=0.8, top_k=10)
        self.assertEqual(gt.tolist(), go.tolist())

    def test_greedy_generation_is_deterministic_argmax(self):
        ours = compiled().eval()
        x = torch.randint(0, 65, (1, 8))
        self.assertEqual(ours.generate(x.clone(), 5, temperature=0).tolist(),
                         ours.generate(x.clone(), 5, temperature=0).tolist())

    def test_sequence_longer_than_block_is_rejected(self):
        with self.assertRaises(ValueError):
            compiled()(torch.zeros(1, 33, dtype=torch.long))


class TestParityDetectsMutations(unittest.TestCase):
    """Parity only means something if a wrong description cannot pass it."""
    SOURCE = NANOGPT.read_text(encoding="utf-8")
    TIE = "        tie transformer.wte.weight = lm_head.weight\n"
    # torch's normal_ and uniform_ consume the RNG identically for sizes that are multiples of 16,
    # so d=32 cannot tell them apart. d=34 keeps every tensor size away from that coincidence.
    UNALIGNED = dict(vocab=65, block=32, layers=2, heads=2, d=34)
    W0_MUTATIONS = {
        "wrong construction draws": ("linear.weight = kaiming_uniform(sqrt(5))", "linear.weight = normal(0, 0.02)"),
        "no bias draws": ("linear.bias = fan_in_uniform()", "linear.bias = zeros()"),
        "no embedding draws": ("embedding.weight = normal(0, 1)", "embedding.weight = zeros()"),
        "wrong std": ("linear.weight = normal(0, 0.02)", "linear.weight = normal(0, 0.03)"),
        "no residual scaling": ("0.02 / sqrt(2 * layers)", "0.02 / sqrt(layers)"),
        "no tie": (TIE, ""),
    }
    GRAPH_MUTATIONS = {
        "no gelu": ("m = gelu(m)", "m = m"),
        "no residual": ("x = add(x, m)", "x = m"),
        "swapped norm": ("a = layernorm(ln_1, x, d, bias)", "a = x"),
        "reversed qkv": ("q, k, v = split(qkv, 3)", "v, k, q = split(qkv, 3)"),
        "no positions": ("x = add(tok, pos)", "x = add(tok, tok)"),
    }

    def _mutate(self, old, new):
        self.assertIn(old, self.SOURCE, "mutation no longer applies to the source")
        return wsp_arch.compile_source(self.SOURCE.replace(old, new), overrides=SMALL, seed=1337)

    def _unaligned_upstream(self):
        torch.manual_seed(1337)
        return dict(GPT(GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=34,
                                  dropout=0.0)).named_parameters())

    def test_unmutated_source_matches_with_unaligned_sizes(self):
        theirs = self._unaligned_upstream()
        ours = dict(wsp_arch.compile_source(self.SOURCE, overrides=self.UNALIGNED, seed=1337).named_parameters())
        self.assertEqual(list(ours), list(theirs))
        for name in ours:
            self.assertTrue(torch.equal(ours[name], theirs[name]), name)

    def test_init_mutations_break_w0_parity(self):
        theirs = self._unaligned_upstream()
        for label, (old, new) in self.W0_MUTATIONS.items():
            with self.subTest(label):
                self.assertIn(old, self.SOURCE, "mutation no longer applies to the source")
                ours = dict(wsp_arch.compile_source(self.SOURCE.replace(old, new), overrides=self.UNALIGNED,
                                                    seed=1337).named_parameters())
                same = list(ours) == list(theirs) and all(torch.equal(ours[n], theirs[n]) for n in ours)
                self.assertFalse(same)

    def test_tie_position_before_second_pass_is_unobservable(self):
        """Known equivalence: same draws, and pass 2 overwrites them. W0 parity cannot pin this order."""
        theirs = self._unaligned_upstream()
        src = self.SOURCE.replace(self.TIE, "").replace("    init:\n", "    init:\n" + self.TIE)
        ours = dict(wsp_arch.compile_source(src, overrides=self.UNALIGNED, seed=1337).named_parameters())
        for name in ours:
            self.assertTrue(torch.equal(ours[name], theirs[name]), name)

    def test_graph_mutations_break_output_parity(self):
        theirs = upstream().eval()
        x = torch.randint(0, 65, (2, 16))
        for label, (old, new) in self.GRAPH_MUTATIONS.items():
            with self.subTest(label):
                ours = self._mutate(old, new).eval()
                self.assertFalse(torch.equal(theirs(x, x)[0], ours(x, x)[0]))


class TestNoPythonClassBehindIt(unittest.TestCase):
    def test_compiling_nanogpt_imports_no_model_definition(self):
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from engine import wsp_arch
            m = wsp_arch.load_arch_file({str(NANOGPT)!r}, seed=1)
            assert sum(p.numel() for p in m.parameters()) == 10770816
            print([n for n in sys.modules if n in ("model", "engine.nanogpt_runtime", "nanogpt_runtime")])
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_compiled_model_has_no_task_specific_classes(self):
        kinds = {type(m).__name__ for m in compiled().modules()}
        self.assertLessEqual(kinds, {"CompiledArch", "ParamScope"})


class TestCheckpoint(unittest.TestCase):
    def test_saved_checkpoint_loads_strictly_into_upstream(self):
        ours = compiled().eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.safetensors"
            ours.save(path)
            theirs = GPT(upstream_cfg()).eval()
            theirs.load_state_dict(load_file(str(path)), strict=True)
        x = torch.randint(0, 65, (2, 12))
        self.assertTrue(torch.equal(theirs(x, x)[0], ours(x, x)[0]))

    def test_checkpoint_round_trip_keeps_tying(self):
        ours = compiled()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.safetensors"
            ours.save(path)
            other = compiled(seed=2)
            other.load(path)
        self.assertIs(other.get_parameter("transformer.wte.weight"), other.get_parameter("lm_head.weight"))
        self.assertTrue(torch.equal(other.get_parameter("lm_head.weight"), ours.get_parameter("lm_head.weight")))

    def test_loading_a_checkpoint_with_missing_tensor_fails(self):
        from safetensors.torch import save_file
        ours = compiled()
        sd = ours.export_state_dict()
        del sd["transformer.h.0.mlp.c_fc.weight"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.safetensors"
            save_file(sd, str(path))
            with self.assertRaises(RuntimeError):
                ours.load(path)


MINIMAL = """
arch mini:
    config:
        vocab = 10
        d = 8
    graph(idx):
        h = embedding(emb, idx, vocab, d)
        logits = linear(head, h, d, vocab, false)
        output logits
    init:
        pass:
            embedding.weight = normal(0, 1)
            linear.weight = normal(0, 1)
"""


class TestLanguageIsNotNanoGPTOnly(unittest.TestCase):
    def test_attention_free_model_compiles_and_learns(self):
        m = wsp_arch.compile_source(MINIMAL, seed=0)
        self.assertEqual(set(dict(m.named_parameters())), {"emb.weight", "head.weight"})
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        x = torch.tensor([[1, 2, 3, 4]]); y = torch.tensor([[2, 3, 4, 5]])
        first = m(x, y)[1].item()
        for _ in range(60):
            opt.zero_grad()
            m(x, y)[1].backward()
            opt.step()
        self.assertLess(m(x, y)[1].item(), first / 4)

    def test_graph_changes_change_the_computation(self):
        """Post-norm differs from pre-norm only in graph text: same parameters, other function."""
        pre = NANOGPT.read_text(encoding="utf-8")
        post = (pre.replace("a = layernorm(ln_1, x, d, bias)", "a = x")
                   .replace("x = add(x, y)", "x = add(x, y)\n                    x = layernorm(ln_1, x, d, bias)"))
        self.assertNotEqual(pre, post)
        a = wsp_arch.compile_source(pre, overrides=SMALL, seed=1).eval()
        b = wsp_arch.compile_source(post, overrides=SMALL, seed=1).eval()
        self.assertEqual(sorted(dict(a.named_parameters())), sorted(dict(b.named_parameters())))
        x = torch.randint(0, 65, (1, 8))
        self.assertFalse(torch.equal(a(x)[0], b(x)[0]))

    def test_ff_override_changes_mlp_width(self):
        m = compiled(ff=64)
        self.assertEqual(m.get_parameter("transformer.h.0.mlp.c_fc.weight").shape, (64, 32))

    def test_derived_config_follows_overrides(self):
        self.assertEqual(compiled().get_parameter("transformer.h.0.mlp.c_fc.weight").shape, (128, 32))


class TestRejection(unittest.TestCase):
    def assertRejected(self, source, fragment, **kw):
        with self.assertRaises(wsp_arch.ArchError) as ctx:
            wsp_arch.compile_source(source, **kw)
        self.assertIn(fragment, str(ctx.exception))

    def test_code_injection_in_expression(self):
        for expr in ("__import__('os')", "(1).__class__", "open('x')", "[1]", "lambda: 1"):
            self.assertRejected(MINIMAL.replace("vocab = 10", f"vocab = {expr}"), "not allowed")

    def test_unknown_operation(self):
        self.assertRejected(MINIMAL.replace("linear(head,", "exec(head,"), "unknown operation")

    def test_undefined_variable(self):
        self.assertRejected(MINIMAL.replace("linear(head, h,", "linear(head, nope,"), "undefined")

    def test_dimension_mismatch_reports_line(self):
        self.assertRejected(MINIMAL.replace("linear(head, h, d,", "linear(head, h, 9,"), "line 8")

    def test_attention_heads_must_divide_dim(self):
        self.assertRejected(NANOGPT.read_text(encoding="utf-8"), "heads", overrides={**SMALL, "heads": 3})

    def test_duplicate_tensor_name(self):
        src = MINIMAL.replace("logits = linear(head, h, d, vocab, false)",
                              "h = linear(head, h, d, d, false)\n        logits = linear(head, h, d, vocab, false)")
        self.assertRejected(src, "duplicate")

    def test_uninitialized_tensor(self):
        self.assertRejected(MINIMAL.replace("linear.weight = normal(0, 1)", ""), "uninitialized")

    def test_tie_shape_mismatch(self):
        src = MINIMAL.replace("linear(head, h, d, vocab, false)", "linear(head, h, d, 7, false)")
        src += "        tie emb.weight = head.weight\n"
        self.assertRejected(src, "tie")

    def test_tie_unknown_tensor(self):
        self.assertRejected(MINIMAL + "        tie emb.weight = nope.weight\n", "nope.weight")

    def test_missing_output(self):
        self.assertRejected(MINIMAL.replace("output logits", ""), "output")

    def test_unknown_override(self):
        self.assertRejected(MINIMAL, "zzz", overrides={"zzz": 1})

    def test_parameter_limit_before_allocation(self):
        self.assertRejected(MINIMAL, "limit", overrides={"vocab": 10 ** 6, "d": 10 ** 6})

    def test_repeat_limit(self):
        src = MINIMAL.replace("h = embedding(emb, idx, vocab, d)",
                              "h = embedding(emb, idx, vocab, d)\n        repeat i in 100000000:\n            h = gelu(h)")
        self.assertRejected(src, "limit")

    def test_variable_may_not_shadow_config(self):
        self.assertRejected(MINIMAL.replace("h = embedding", "d = embedding"), "shadow")

    def test_tabs_or_bad_indent(self):
        self.assertRejected(MINIMAL.replace("    config:", "\tconfig:"), "indent")


def run_step(args, budget=100000000):
    code = f"""experiment t:
model: nanoGPT
budgets:
    max_modified_weights = {budget}
    allow_write = true
steps:
    step init_architecture {args}
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
        self.base = (f"arch=programs/arch/nanogpt.wsparch vocab=65 block=32 layers=2 heads=2 d=32 "
                     f"seed=1337 out={self.out.as_posix()}")

    def test_step_writes_upstream_w0(self):
        res = run_step(self.base)
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["result"]["compiled_from"], "programs/arch/nanogpt.wsparch")
        self.assertEqual(len(res["result"]["source_sha256"]), 64)
        sd = load_file(str(self.out))
        for name, p in upstream().named_parameters():
            self.assertTrue(torch.equal(sd[name], p), name)

    def test_budget_is_enforced(self):
        res = run_step(self.base, budget=1000)
        self.assertEqual(res["status"], "error")
        self.assertIn("max_modified_weights", res["error"])
        self.assertFalse(self.out.exists())

    def test_language_errors_surface_as_step_errors(self):
        res = run_step(self.base + " heads=3")
        self.assertEqual(res["status"], "error")
        self.assertIn("heads", res["error"])

    def test_override_values_cannot_run_code(self):
        res = run_step(self.base + " layers=__import__('os')")
        self.assertEqual(res["status"], "error")
        self.assertFalse(self.out.exists())

    def test_checkpoint_records_source_hash(self):
        from safetensors import safe_open
        res = run_step(self.base)
        with safe_open(str(self.out), framework="pt") as f:
            self.assertEqual(f.metadata()["wsparch_sha256"], res["result"]["source_sha256"])

    def test_arch_file_must_use_wsparch_extension(self):
        res = run_step(f"arch=README.md out={self.out.as_posix()}")
        self.assertEqual(res["status"], "error")
        self.assertIn(".wsparch", res["error"])


if __name__ == "__main__":
    unittest.main()
