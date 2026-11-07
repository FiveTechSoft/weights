"""Arm A of als_criterion.md: AdamW on the small nanoGPT shape, then full-val NLL and gate."""
import json
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "lab"), str(R / "lab" / "probes")]

from engine import wsp_arch, wsp_train  # noqa: E402
from als_common import ARCH, SHAPE, measure  # noqa: E402

text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
data = wsp_train.char_dataset(text)
torch.manual_seed(1337)
model = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=len(data.chars)), seed=1337)
cfg = wsp_train.TrainConfig(batch_size=64, gradient_accumulation_steps=1, max_iters=5000, eval_interval=500,
                            eval_iters=50, log_interval=0, learning_rate=1e-3, warmup_iters=100,
                            lr_decay_iters=5000, min_lr=1e-4, weight_decay=0.1, device="cuda", dtype="float32")
out = R / "lab/runs/als_adam_small.pt"
info = wsp_train.train(model, data, cfg, out)
result = measure("A_adam", model, data, text, "cuda")
result["train"] = {"backward_calls": info["backward_calls_at_save"], "seconds": info["seconds"],
                   "tokens_seen": cfg.max_iters * cfg.batch_size * SHAPE["block"]}
(R / "lab/runs/als_adam_small.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
