#!/usr/bin/env python3
"""Training-to-surgery laboratory, v1.

A deliberately small, reproducible experiment:
  1) train a tiny symbol-association MLP with ordinary SGD;
  2) save the full training trace and before/after state_dicts;
  3) compare trained weights, direct delta replay, and a norm-matched placebo;
  4) evaluate on held-out phrasing templates for seen associations.

Important: delta replay is an oracle/reference baseline, NOT a learned WSPLang
compiler. This harness establishes the experimental protocol before adding a
trace-to-surgery learner.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


TEMPLATES_TRAIN = [
    "lookup {symbol}",
    "what is {symbol}",
    "map {symbol}",
]
TEMPLATES_TEST = [
    "tell me the value for {symbol}",
    "which color belongs to {symbol}",
]


class TinyAssociator(nn.Module):
    def __init__(self, n_symbols: int, n_labels: int, hidden: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, hidden)
        self.proj = nn.Linear(hidden, n_labels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(ids))


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_examples(n_symbols: int, n_labels: int, seed: int):
    rng = random.Random(seed)
    # A deterministic, randomly permuted association map.
    labels = list(range(n_labels))
    rng.shuffle(labels)
    mapping = {i: labels[i % n_labels] for i in range(n_symbols)}
    return mapping


def accuracy(model: nn.Module, mapping: Dict[int, int], ids: List[int], device) -> float:
    model.eval()
    x = torch.tensor(ids, dtype=torch.long, device=device)
    y = torch.tensor([mapping[i] for i in ids], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
    return float((pred == y).float().mean().item())


def clone_state(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def delta_state(before, after):
    return {k: after[k] - before[k] for k in before}


def state_norm(delta) -> float:
    return float(torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values())).item())


def apply_delta(model: nn.Module, before, delta, scale: float = 1.0) -> None:
    state = {k: before[k] + scale * delta[k] for k in before}
    model.load_state_dict(state)


def make_placebo(delta, seed: int):
    """Random tensor perturbation with the same global L2 norm as the true delta."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = {k: torch.randn(v.shape, generator=gen, dtype=v.dtype) for k, v in delta.items()}
    norm = state_norm(noise)
    target = state_norm(delta)
    if norm == 0:
        return {k: torch.zeros_like(v) for k, v in delta.items()}
    return {k: v * (target / norm) for k, v in noise.items()}


def run(args):
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mapping = build_examples(args.n_symbols, args.n_labels, args.seed)
    train_ids = list(range(args.n_train_symbols))
    test_ids = train_ids[:]  # association knowledge is tested on held-out query templates
    model = TinyAssociator(args.n_symbols, args.n_labels, args.hidden).to(device)
    before = clone_state(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    trace = []
    # Train on repeated labeled associations; template variation is logged as
    # query metadata, while the tiny model intentionally isolates symbol IDs.
    for step in range(args.steps):
        random.shuffle(train_ids)
        x = torch.tensor(train_ids, dtype=torch.long, device=device)
        y = torch.tensor([mapping[i] for i in train_ids], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_norm = float(torch.sqrt(sum(
            (p.grad.detach().float() ** 2).sum()
            for p in model.parameters() if p.grad is not None
        )).item())
        optimizer.step()
        trace.append({
            "step": step + 1,
            "loss": float(loss.item()),
            "grad_l2": grad_norm,
            "lr": args.lr,
            "optimizer": "SGD",
        })

    after = clone_state(model)
    delta = delta_state(before, after)
    trained_score = accuracy(model, mapping, test_ids, device)

    # Surgical baseline: directly apply the observed training delta to the
    # original parameters. It is a replay upper bound, not a learned method.
    surgery = TinyAssociator(args.n_symbols, args.n_labels, args.hidden).to(device)
    surgery.load_state_dict(before)
    apply_delta(surgery, before, delta)
    surgery_score = accuracy(surgery, mapping, test_ids, device)

    placebo_delta = make_placebo(delta, args.seed + 1)
    placebo = TinyAssociator(args.n_symbols, args.n_labels, args.hidden).to(device)
    placebo.load_state_dict(before)
    apply_delta(placebo, before, placebo_delta)
    placebo_score = accuracy(placebo, mapping, test_ids, device)

    torch.save({
        "initial_state": before,
        "trained_state": after,
        "delta": delta,
        "placebo_delta": placebo_delta,
        "mapping": mapping,
        "config": vars(args),
    }, out / "weights.pt")
    (out / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    report = {
        "benchmark": "training_to_surgery_v1_symbol_association",
        "seed": args.seed,
        "device": str(device),
        "n_symbols": args.n_symbols,
        "n_labels": args.n_labels,
        "n_train_symbols": args.n_train_symbols,
        "steps": args.steps,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "delta_l2": state_norm(delta),
        "scores": {
            "trained_model": trained_score,
            "direct_delta_replay": surgery_score,
            "norm_matched_random_placebo": placebo_score,
        },
        "interpretation": (
            "Direct delta replay should match the trained model up to numerical "
            "precision. This validates the comparison pipeline only; it does not "
            "show that WSPLang learned a generalized surgery."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-symbols", type=int, default=16)
    p.add_argument("--n-labels", type=int, default=4)
    p.add_argument("--n-train-symbols", type=int, default=12)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", default="runs/training_to_surgery_v1")
    args = p.parse_args()
    if not (1 <= args.n_train_symbols <= args.n_symbols):
        p.error("--n-train-symbols must be in [1, --n-symbols]")
    if args.n_labels < 2:
        p.error("--n-labels must be at least 2")
    return args


if __name__ == "__main__":
    run(parse_args())
