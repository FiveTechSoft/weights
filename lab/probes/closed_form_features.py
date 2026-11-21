"""Option 1 probe, no nanoGPT weights: can closed-form learned features beat fixed histograms?

Table logits come from tables blind to each fold, so residuals over almost all of train are honest. Features are
expanded by random gelu units, fit by ridge on the softmax residual, step chosen on the last train fold, val read last.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R)]
from engine import wsp_synth, wsp_train

dev = "cuda" if torch.cuda.is_available() else "cpu"
text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
data = wsp_train.char_dataset(text)
V = len(data.chars)
tr, va = data.train, data.val
ids = np.concatenate([tr, va])
ORDER, FOLDS, WIDTH = 4, 5, 3072
cut = len(tr)
fold_edges = np.linspace(0, cut, FOLDS + 1).astype(int)


def lookup(table, positions):
    out = np.zeros((len(positions), V), dtype=np.float32)
    cache = {}
    for n, i in enumerate(positions):
        ctx = tuple(ids[max(0, i - ORDER + 2):i + 1].tolist())
        row = cache.get(ctx)
        if row is None:
            row = table.log_prob(ctx).astype(np.float32)
            cache[ctx] = row
        out[n] = row
    return out


L = np.zeros((len(ids) - 1, V), dtype=np.float32)
for f in range(FOLDS):
    lo, hi = fold_edges[f], fold_edges[f + 1]
    print("fold", f, flush=True)
    t = wsp_synth.WittenBellTable(np.concatenate([tr[:lo], tr[hi:]]), V, ORDER)
    L[lo:min(hi, len(L))] = lookup(t, np.arange(lo, min(hi, len(L))))
full = wsp_synth.WittenBellTable(tr, V, ORDER)
L[cut:] = lookup(full, np.arange(cut, len(ids) - 1))
L = torch.tensor(L, device=dev)
targets = torch.tensor(ids[1:], device=dev)
E = torch.eye(V, device=dev)
fit_idx = torch.arange(0, fold_edges[FOLDS - 1], device=dev)
hold_idx = torch.arange(fold_edges[FOLDS - 1], cut - 1, device=dev)
val_idx = torch.arange(cut, len(ids) - 1, device=dev)


def nll(logits, idx):
    return torch.nn.functional.cross_entropy(logits, targets[idx]).item()


print(f"table order {ORDER} (no unit limit) val {nll(L[val_idx], val_idx):.4f}; cross-fit train hold {nll(L[hold_idx], hold_idx):.4f}", flush=True)


def windows_of(emb, edges):
    """Average of emb over older positions t-b+1 .. t-a for each (a, b)."""
    x = emb[ids[:-1]]
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0, dtype=np.float64)])
    out = []
    t = np.arange(len(x)) + 1
    for a, b in edges:
        end = np.maximum(t - a, 0)
        start = np.maximum(t - b, 0)
        out.append(((c[end] - c[start]) / (b - a)).astype(np.float32))
    return np.concatenate(out, 1)


def svd_embedding(r):
    """PPMI of char with its neighbours within 4 positions; closed form, fit on train only."""
    co = np.zeros((V, V), dtype=np.float64)
    for gap in (1, 2, 3, 4):
        np.add.at(co, (tr[:-gap], tr[gap:]), 1.0 / gap)
    co = co + co.T
    pr, pc = co.sum(1, keepdims=True), co.sum(0, keepdims=True)
    pmi = np.log(np.maximum(co * co.sum() / (pr * pc), 1e-12))
    ppmi = np.maximum(pmi, 0)
    u, s, _ = np.linalg.svd(ppmi)
    return (u[:, :r] * np.sqrt(s[:r])).astype(np.float32)


def class_embedding():
    m = np.zeros((V, len(wsp_synth.CHAR_CLASSES)), dtype=np.float32)
    for j, members in enumerate(wsp_synth.CHAR_CLASSES):
        for ch in members:
            if ch in data.chars:
                m[data.chars.index(ch), j] = 1
    return m


def word_features(dim=32):
    """Upper bound: SVD embeddings of the previous two complete words. nanoGPT may be unable to compute this."""
    words = re.findall(r"[A-Za-z']+", text[:int(len(text) * 0.9)].lower())
    vocab = {w: i for i, w in enumerate(sorted({w for w in set(words)}))}
    top = {w for w, _ in __import__("collections").Counter(words).most_common(2000)}
    keep = {w: i for i, w in enumerate(sorted(top))}
    co = np.zeros((len(keep), len(keep)), dtype=np.float64)
    for a, b in zip(words[:-1], words[1:]):
        if a in keep and b in keep:
            co[keep[a], keep[b]] += 1
    co = co + co.T
    pr, pc = co.sum(1, keepdims=True), co.sum(0, keepdims=True)
    ppmi = np.maximum(np.log(np.maximum(co * co.sum() / np.maximum(pr * pc, 1e-12), 1e-12)), 0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = (u[:, :dim] * np.sqrt(s[:dim])).astype(np.float32)
    zero = np.zeros(dim, dtype=np.float32)
    out = np.zeros((len(ids) - 1, 2 * dim), dtype=np.float32)
    w1, w2, cur = zero, zero, []
    isword = np.array([ch.isalpha() or ch == "'" for ch in data.chars])
    for i, tok in enumerate(ids[:-1]):
        if isword[tok]:
            cur.append(data.chars[tok].lower())
        elif cur:
            word = "".join(cur)
            w2, w1 = w1, emb[keep[word]] if word in keep else zero
            cur = []
        out[i, :dim] = w1
        out[i, dim:] = w2
    return out


def fit(name, raw, blocks=2):
    Z = torch.tensor(raw, device=dev)
    mean, std = Z[fit_idx[:200000]].mean(0), Z[fit_idx[:200000]].std(0) + 1e-6
    logits = L.clone()
    g = torch.Generator().manual_seed(0)
    for b in range(blocks):
        A = torch.randn(Z.shape[1], WIDTH, generator=g).to(dev) * 0.7 / (Z.shape[1] ** 0.5) ** 0.5
        tok = torch.randn(V, WIDTH, generator=g).to(dev)
        off = torch.randn(WIDTH, generator=g).to(dev) * 0.7
        cur = torch.tensor(ids[:-1], device=dev)

        def phi(i):
            return torch.nn.functional.gelu((Z[i] - mean) / std @ A + tok[cur[i]] + off)

        xtx = torch.zeros(WIDTH, WIDTH, device=dev, dtype=torch.float64)
        xty = torch.zeros(WIDTH, V, device=dev, dtype=torch.float64)
        for chunk in fit_idx.split(100000):
            P = phi(chunk)
            res = E[targets[chunk]] - torch.softmax(logits[chunk], -1)
            xtx += (P.T @ P).double()
            xty += P.double().T @ res.double()
        W = torch.linalg.solve(xtx + 1e3 * torch.eye(WIDTH, device=dev, dtype=torch.float64), xty).float()
        H = phi(hold_idx) @ W
        hold_before = nll(logits[hold_idx], hold_idx)
        step = min((0.0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12), key=lambda s: nll(logits[hold_idx] + s * H, hold_idx))
        for chunk in torch.arange(len(logits), device=dev).split(200000):
            logits[chunk] += step * (phi(chunk) @ W)
        print(f"{name:34s} block {b}: step {step} hold {hold_before:.4f}->{nll(logits[hold_idx], hold_idx):.4f} val {nll(logits[val_idx], val_idx):.4f}", flush=True)


EDGES = [(3, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
fit("A fixed classes x5 windows", windows_of(class_embedding(), EDGES))
fit("B SVD chars r=12 x5 windows", windows_of(svd_embedding(12), EDGES))
fit("C SVD chars r=32 x5 windows", windows_of(svd_embedding(32), EDGES))
# position-resolved: one-hot-ish token at each offset 4..12, beyond the table's reach
emb = svd_embedding(12)
pos = np.concatenate([emb[np.concatenate([np.zeros(k, dtype=np.int64), ids[:-1 - k]])] for k in range(3, 12)], 1)
fit("D SVD chars at offsets 3..11", pos)
fit("E previous 2 words (upper bound)", word_features())
