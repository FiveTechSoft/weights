"""Declarative gradient training for compiled architectures, following nanoGPT's train.py.

The loop reproduces lab/reference/karpathy_nanogpt/train.py including its RNG order, so the same seed
gives the same weights on CPU. This is the control for later work: checkpoints record that they came
from gradient descent and how many backward passes produced them, so they can never pass as synthesis.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

PROVENANCE = "gradient_descent"


@dataclass
class TrainConfig:
    """Same names and defaults as train.py, except device-dependent dtype, resolved in train()."""
    batch_size: int = 12
    gradient_accumulation_steps: int = 40
    max_iters: int = 600000
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 1
    always_save_checkpoint: bool = True
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    device: str = "cuda"
    dtype: str = "auto"


@dataclass
class CharDataset:
    chars: str
    train: np.ndarray
    val: np.ndarray


def char_dataset(text):
    """Character encoding and 90/10 split of nanoGPT's data/shakespeare_char/prepare.py."""
    chars = "".join(sorted(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    n = len(text)
    ids = np.fromiter((stoi[c] for c in text), dtype=np.int64, count=n)
    cut = int(n * 0.9)
    return CharDataset(chars, ids[:cut], ids[cut:])


def get_lr(it, cfg):
    """Cosine decay with linear warmup."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def build_optimizer(model, cfg):
    """AdamW decaying only tensors with 2+ dimensions; fused on CUDA."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    extra = {"fused": True} if "cuda" in cfg.device else {}
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), **extra)


def _resolve_device(cfg):
    if "cuda" in cfg.device and not torch.cuda.is_available():
        raise ValueError(f"device={cfg.device} requested but cuda is not available in this torch build")
    if cfg.dtype == "auto":
        supported = "cuda" in cfg.device and torch.cuda.is_bf16_supported()
        cfg.dtype = "bfloat16" if supported else "float16" if "cuda" in cfg.device else "float32"
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]


def train(model, data, cfg, out_path, on_log=print):
    """Trains in place and writes a checkpoint plus `<out>.train.json`; continues the current RNG."""
    block = model.config.block_size
    for name, split in (("train", data.train), ("val", data.val)):
        if len(split) <= block:
            raise ValueError(f"{name} split has {len(split)} tokens, needs more than block size {block}")
    if cfg.eval_interval > cfg.max_iters:
        raise ValueError(f"eval_interval={cfg.eval_interval} > max_iters={cfg.max_iters}: nothing would be saved")

    ptdtype = _resolve_device(cfg)
    use_cuda = "cuda" in cfg.device
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if use_cuda else nullcontext()
    splits = {"train": data.train, "val": data.val}

    def get_batch(split):
        source = splits[split]
        ix = torch.randint(len(source) - block, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(source[int(i):int(i) + block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(source[int(i) + 1:int(i) + 1 + block].astype(np.int64)) for i in ix])
        if use_cuda:
            return x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(cfg.device, non_blocking=True)
        return x.to(cfg.device), y.to(cfg.device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    losses[k] = model(x, y)[1].item()
            out[split] = float(losses.mean())
        model.train()
        return out

    model.to(cfg.device)
    model.train()
    scaler = torch.amp.GradScaler("cuda" if use_cuda else "cpu", enabled=cfg.dtype == "float16")
    optimizer = build_optimizer(model, cfg)
    out_path = Path(out_path)

    evals, backward_calls, saved_iter, saved_calls = [], 0, None, None
    best_val = 1e9
    t0 = time.time()
    X, Y = get_batch("train")
    iter_num = 0
    while True:
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            evals.append({"iter": iter_num, "train": losses["train"], "val": losses["val"], "lr": lr,
                          "seconds": round(time.time() - t0, 1)})
            on_log(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val or cfg.always_save_checkpoint:
                best_val = losses["val"]
                if iter_num > 0:
                    saved_iter, saved_calls = iter_num, backward_calls
                    model.save(out_path, metadata={"provenance": PROVENANCE, "backward_calls": str(backward_calls),
                                                   "iter_num": str(iter_num), "val_loss": str(losses["val"]),
                                                   "vocab_chars": json.dumps(data.chars)})

        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                loss = model(X, Y)[1] / cfg.gradient_accumulation_steps
            X, Y = get_batch("train")
            scaler.scale(loss).backward()
            backward_calls += 1
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if cfg.log_interval and iter_num % cfg.log_interval == 0:
            on_log(f"iter {iter_num}: loss {loss.item() * cfg.gradient_accumulation_steps:.4f}")
        iter_num += 1
        if iter_num > cfg.max_iters:
            break

    if saved_iter is None:
        raise ValueError("training finished without saving: no evaluation improved after iteration 0")
    result = {"provenance": PROVENANCE, "arch_sha256": model.source_sha256, "config": asdict(cfg),
              "evals": evals, "best_val": best_val, "saved_iter": saved_iter,
              "backward_calls_at_save": saved_calls, "seconds": round(time.time() - t0, 1),
              "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if use_cuda else None}
    out_path.with_suffix(".train.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def split_step_args(named, parse_value):
    """Separates step arguments into a TrainConfig and architecture config overrides."""
    types = {f.name: f.type for f in fields(TrainConfig)}
    train_args, overrides = {}, {}
    for key, value in named.items():
        if key in ("arch", "data", "out", "seed", "model", "format"):
            continue
        if key in types:
            train_args[key] = str(value) if key in ("device", "dtype") else parse_value(value)
        else:
            overrides[key] = parse_value(value)
    return TrainConfig(**train_args), overrides
