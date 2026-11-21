"""Last option 1 probe: a soft table. Budget of the exact shape: 6 * 1536 = 9216 units on 18 retrieved positions.

Units are templates sampled from real train contexts and fire on partial, position-weighted matches of SVD token
embeddings. Output weights solve a convex softmax regression by Newton-style steps using one fixed Gram matrix.
No backward pass and no representation is trained. Step sizes use a train-tail holdout; val is read for reporting.
"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R)]
from engine import wsp_train

dev = "cuda"
text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
data = wsp_train.char_dataset(text)
V = len(data.chars)
tr = data.train
ids = np.concatenate([tr, data.val])
cut = len(tr)
OFFSETS, R_DIM, UNITS = 18, 16, 9216


def svd_embedding(r):
    co = np.zeros((V, V), dtype=np.float64)
    for gap in (1, 2, 3, 4):
        np.add.at(co, (tr[:-gap], tr[gap:]), 1.0 / gap)
    co = co + co.T
    pr, pc = co.sum(1, keepdims=True), co.sum(0, keepdims=True)
    ppmi = np.maximum(np.log(np.maximum(co * co.sum() / (pr * pc), 1e-12)), 0)
    u, s, _ = np.linalg.svd(ppmi)
    e = u[:, :r] * np.sqrt(s[:r])
    return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


emb = torch.tensor(svd_embedding(R_DIM), device=dev)
all_ids = torch.tensor(ids, device=dev)
decay = torch.tensor(0.8 ** np.arange(OFFSETS), dtype=torch.float32, device=dev)
decay = decay / decay.sum()


def context(i):
    """N x OFFSETS x R_DIM position-weighted embeddings of the tokens at i, i-1, ..."""
    idx = (i[:, None] - torch.arange(OFFSETS, device=dev)[None]).clamp(min=0)
    return emb[all_ids[idx]] * decay[None, :, None] ** 0.5


g = torch.Generator(device="cpu").manual_seed(0)
fit_idx = torch.randperm(int(cut * 0.9) - OFFSETS, generator=g)[:600000].to(dev) + OFFSETS
hold_idx = torch.arange(int(cut * 0.9), cut - 1, device=dev)
val_idx = torch.arange(cut, len(ids) - 1, device=dev)
centers = context(torch.randperm(int(cut * 0.9) - OFFSETS, generator=g)[:UNITS].to(dev) + OFFSETS).flatten(1)
sample = context(fit_idx[:4000]).flatten(1)
dots = sample @ centers.T
theta = dots.quantile(0.95, dim=0)
scale = (dots.std(0) + 1e-6)
del dots, sample
targets = all_ids[1:]


def phi(i):
    return torch.nn.functional.gelu((context(i).flatten(1) @ centers.T - theta) / scale)


def apply(W, idx):
    out = []
    for c in idx.split(20000):
        out.append(phi(c) @ W)
    return torch.cat(out)


xtx = torch.zeros(UNITS + 1, UNITS + 1, device=dev, dtype=torch.float64)
for c in fit_idx.split(20000):
    P = torch.cat([phi(c), torch.ones(len(c), 1, device=dev)], 1)
    xtx += (P.T @ P).double()
print("gram done", flush=True)
L_chol = torch.linalg.cholesky(xtx + 3e2 * torch.eye(UNITS + 1, device=dev, dtype=torch.float64))
del xtx
W = torch.zeros(UNITS + 1, V, device=dev)
E = torch.eye(V, device=dev)


def logits_of(idx):
    out = []
    for c in idx.split(20000):
        out.append(torch.cat([phi(c), torch.ones(len(c), 1, device=dev)], 1) @ W)
    return torch.cat(out)


def nll(idx):
    total = 0.0
    for c in idx.split(20000):
        lg = torch.cat([phi(c), torch.ones(len(c), 1, device=dev)], 1) @ W
        total += torch.nn.functional.cross_entropy(lg, targets[c], reduction="sum").item()
    return total / len(idx)


for it in range(14):
    xty = torch.zeros(UNITS + 1, V, device=dev, dtype=torch.float64)
    for c in fit_idx.split(20000):
        P = torch.cat([phi(c), torch.ones(len(c), 1, device=dev)], 1)
        xty += P.double().T @ (E[targets[c]] - torch.softmax(P @ W, -1)).double()
    D = torch.cholesky_solve(xty, L_chol).float()
    before = W
    best = (nll(hold_idx), 0.0)
    for s in (1.0, 2.0, 4.0, 8.0):
        W = before + s * D
        h = nll(hold_idx)
        if h < best[0]:
            best = (h, s)
    W = before + best[1] * D
    print(f"iter {it}: step {best[1]} hold {best[0]:.4f} val {nll(val_idx):.4f}", flush=True)
    if best[1] == 0.0:
        break
