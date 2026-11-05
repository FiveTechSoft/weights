"""Contract tests for the declarative training op; parity runs upstream's real train.py as a subprocess."""
import json
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "lab" / "reference" / "karpathy_nanogpt"
sys.path[:0] = [str(ROOT)]

from engine import wsp_arch, wsp_train  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402

ARCH = ROOT / "programs" / "arch" / "nanogpt.wsparch"
TEXT = (ROOT / "lab" / "data" / "shakespeare_char" / "input.txt").read_text(encoding="utf-8")[:60000]
MODEL = dict(layers=2, heads=2, d=32, block=16, dropout=0.1, bias=False)


def build(data, seed=1337, **model):
    return wsp_arch.load_arch_file(ARCH, overrides={**MODEL, **model, "vocab": len(data.chars)}, seed=seed)


def run_upstream(workdir, **overrides):
    """Runs lab/reference train.py on `TEXT`; returns its checkpoint and printed evals."""
    data = wsp_train.char_dataset(TEXT)
    root = Path(workdir) / "data" / "tiny"
    root.mkdir(parents=True)
    data.train.astype(np.uint16).tofile(root / "train.bin")
    data.val.astype(np.uint16).tofile(root / "val.bin")
    (root / "meta.pkl").write_bytes(pickle.dumps({"vocab_size": len(data.chars)}))
    shutil.copy(REF / "configurator.py", workdir)
    args = dict(dataset="tiny", device="cpu", compile=False, dtype="float32", out_dir="out", log_interval=1000,
                n_layer=MODEL["layers"], n_head=MODEL["heads"], n_embd=MODEL["d"], block_size=MODEL["block"],
                dropout=MODEL["dropout"], bias=MODEL["bias"], batch_size=4, gradient_accumulation_steps=1,
                max_iters=12, eval_interval=12, eval_iters=3, learning_rate=1e-3, min_lr=1e-4, warmup_iters=4,
                lr_decay_iters=12, beta2=0.99, always_save_checkpoint=True)
    args.update(overrides)
    cmd = [sys.executable, str(REF / "train.py")] + [f"--{k}={v}" for k, v in args.items()]
    done = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    evals = re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", done.stdout)
    ckpt = torch.load(Path(workdir) / "out" / "ckpt.pt", map_location="cpu", weights_only=False)
    return ckpt, evals


def ours_config(**overrides):
    cfg = dict(device="cpu", dtype="float32", batch_size=4, gradient_accumulation_steps=1, max_iters=12,
               eval_interval=12, eval_iters=3, learning_rate=1e-3, min_lr=1e-4, warmup_iters=4,
               lr_decay_iters=12, beta2=0.99, always_save_checkpoint=True, log_interval=0)
    cfg.update(overrides)
    return wsp_train.TrainConfig(**cfg)


class TestUpstreamTrainParity(unittest.TestCase):
    def _compare(self, model_kw=None, **train_kw):
        model_kw = model_kw or {}
        with tempfile.TemporaryDirectory() as work:
            upstream_args = {{"layers": "n_layer", "heads": "n_head", "d": "n_embd", "block": "block_size"}.get(k, k): v
                             for k, v in model_kw.items()}
            theirs, printed = run_upstream(work, **upstream_args, **train_kw)
            data = wsp_train.char_dataset(TEXT)
            out = Path(work) / "ours.safetensors"
            result = wsp_train.train(build(data, **model_kw), data, ours_config(**train_kw), out)
            ours = load_file(str(out))
        self.assertEqual(sorted(ours), sorted(k for k in theirs["model"]))
        for name, tensor in theirs["model"].items():
            self.assertTrue(torch.equal(ours[name], tensor), name)
        self.assertEqual(result["saved_iter"], theirs["iter_num"])
        mine = [(str(e["iter"]), f"{e['train']:.4f}", f"{e['val']:.4f}") for e in result["evals"]]
        self.assertEqual(mine, printed)
        return result

    def test_weights_and_losses_match_real_train_py(self):
        self._compare()

    def test_gradient_accumulation_matches(self):
        self._compare(gradient_accumulation_steps=2)

    def test_no_grad_clip_and_no_lr_decay_match(self):
        self._compare(grad_clip=0.0, decay_lr=False)

    def test_best_only_checkpoint_policy_matches(self):
        self._compare(always_save_checkpoint=False, eval_interval=4)

    def test_bias_and_no_dropout_config_matches(self):
        self._compare(model_kw=dict(bias=True, dropout=0.0))


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.cfg = wsp_train.TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_iters=10, lr_decay_iters=110)

    def test_warmup_is_linear(self):
        self.assertAlmostEqual(wsp_train.get_lr(0, self.cfg), 1 / 11)
        self.assertAlmostEqual(wsp_train.get_lr(9, self.cfg), 10 / 11)

    def test_cosine_midpoint(self):
        self.assertAlmostEqual(wsp_train.get_lr(60, self.cfg), 0.55)

    def test_after_decay_is_min_lr(self):
        self.assertEqual(wsp_train.get_lr(500, self.cfg), 0.1)

    def test_decay_disabled_is_constant(self):
        cfg = wsp_train.TrainConfig(learning_rate=0.5, decay_lr=False)
        self.assertEqual(wsp_train.get_lr(3, cfg), 0.5)


class TestDataset(unittest.TestCase):
    def test_split_and_encoding_follow_prepare_py(self):
        data = wsp_train.char_dataset("abcabcabca")
        self.assertEqual(data.chars, "abc")
        self.assertEqual(data.train.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual(data.val.tolist(), [0])

    def test_text_shorter_than_block_is_rejected(self):
        data = wsp_train.char_dataset(TEXT[:200])
        with self.assertRaises(ValueError):
            wsp_train.train(build(data, block=64), data, ours_config(), Path(tempfile.gettempdir()) / "unused.safetensors")


class TestOptimizerGroups(unittest.TestCase):
    def test_only_matrices_are_weight_decayed(self):
        data = wsp_train.char_dataset(TEXT)
        groups = wsp_train.build_optimizer(build(data, bias=True), ours_config(weight_decay=0.1)).param_groups
        self.assertEqual(groups[0]["weight_decay"], 0.1)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertTrue(all(p.dim() >= 2 for p in groups[0]["params"]))
        self.assertTrue(all(p.dim() < 2 for p in groups[1]["params"]))


class TestProvenance(unittest.TestCase):
    def test_checkpoint_declares_it_came_from_gradient_descent(self):
        data = wsp_train.char_dataset(TEXT)
        with tempfile.TemporaryDirectory() as work:
            out = Path(work) / "m.safetensors"
            result = wsp_train.train(build(data), data, ours_config(gradient_accumulation_steps=2), out)
            with safe_open(str(out), framework="pt") as f:
                meta = f.metadata()
            trace = json.loads(out.with_suffix(".train.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["provenance"], "gradient_descent")
        self.assertEqual(int(meta["backward_calls"]), 24)
        self.assertEqual(result["backward_calls_at_save"], 24)
        self.assertEqual(meta["vocab_chars"], json.dumps(data.chars))
        self.assertEqual(trace["provenance"], "gradient_descent")
        self.assertEqual(trace["evals"], result["evals"])


def run_step(args, allow_training=True, allow_write=True, dry_run=False, budget=100000000):
    code = f"""experiment t:
model: nanoGPT
budgets:
    max_modified_weights = {budget}
    allow_write = {str(allow_write).lower()}
    allow_training = {str(allow_training).lower()}
steps:
    step train {args}
"""
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    assert not errors, errors
    return execute_plan_v4(plan_run(prog, steps), prog, dry_run=dry_run)[0]


class TestWSPLangOp(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.out = self.dir / "m.safetensors"
        data = self.dir / "text.txt"
        data.write_text(TEXT, encoding="utf-8")
        self.base = (f"arch=programs/arch/nanogpt.wsparch data={data.as_posix()} out={self.out.as_posix()} "
                     "layers=2 heads=2 d=32 block=16 dropout=0.1 bias=false device=cpu dtype=float32 batch_size=4 "
                     "max_iters=12 eval_interval=12 eval_iters=3 warmup_iters=4 lr_decay_iters=12 "
                     "always_save_checkpoint=true")

    def test_step_reproduces_real_train_py(self):
        res = run_step(self.base + " gradient_accumulation_steps=1 learning_rate=1e-3 min_lr=1e-4 beta2=0.99 seed=1337")
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["result"]["provenance"], "gradient_descent")
        with tempfile.TemporaryDirectory() as work:
            theirs, _ = run_upstream(work)
        ours = load_file(str(self.out))
        for name, tensor in theirs["model"].items():
            self.assertTrue(torch.equal(ours[name], tensor), name)

    def test_training_needs_explicit_budget(self):
        res = run_step(self.base, allow_training=False)
        self.assertEqual(res["status"], "security_rejection")
        self.assertIn("allow_training", res["error"])
        self.assertFalse(self.out.exists())

    def test_dry_run_trains_nothing(self):
        res = run_step(self.base, dry_run=True)
        self.assertEqual(res["execution_state"], "simulated")
        self.assertFalse(self.out.exists())

    def test_parameter_budget_is_enforced(self):
        res = run_step(self.base, budget=100)
        self.assertEqual(res["status"], "error")
        self.assertIn("max_modified_weights", res["error"])

    def test_vocab_disagreeing_with_data_is_an_error(self):
        res = run_step(self.base + " vocab=7")
        self.assertEqual(res["status"], "error")
        self.assertIn("vocab", res["error"])

    def test_unknown_argument_is_an_error(self):
        res = run_step(self.base + " max_iterz=3")
        self.assertEqual(res["status"], "error")
        self.assertIn("max_iterz", res["error"])

    def test_missing_arch_is_an_error(self):
        res = run_step(f"data=x.txt out={self.out.as_posix()}")
        self.assertEqual(res["status"], "error")
        self.assertIn("arch", res["error"])

    @unittest.skipIf(torch.cuda.is_available(), "needs a machine without CUDA")
    def test_requesting_missing_cuda_fails_instead_of_falling_back(self):
        res = run_step(self.base.replace("device=cpu", "device=cuda"))
        self.assertEqual(res["status"], "error")
        self.assertIn("cuda", res["error"])


if __name__ == "__main__":
    unittest.main()
