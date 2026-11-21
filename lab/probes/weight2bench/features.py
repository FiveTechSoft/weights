"""Weight-only features of an edit and the paired benchmark targets (lab/probes/weight2bench/PLAN.md).
No forward pass anywhere in this module. Edits are represented as per-layer residual-space directions D (L x d):
control vectors (llama.cpp GGUF `direction.N`, times the served scale), compiled rank-1 writes (v_l * ||u_l||) and
column edits (top left singular vector of the block, scaled to its Frobenius norm)."""
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- artefact loaders
def load_cvec(path, scale=1.0):
    """llama.cpp control vector -> {layer: direction * scale} (float32)."""
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    out = {}
    for t in r.tensors:
        if t.name.startswith("direction."):
            out[int(t.name.split(".")[1])] = np.asarray(t.data, dtype=np.float32).ravel() * float(scale)
    return out


def compiled_direction(u, v):
    """rank-1 dW = outer(v, u) adds (u . h) v to the residual: direction v, magnitude ||v|| ||u||."""
    u = np.asarray(u, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    return (v * np.linalg.norm(u)).astype(np.float32)


def load_compiled(fit_pt, scale=1.0):
    import torch
    d = torch.load(str(fit_pt), map_location="cpu", weights_only=False)
    out = {}
    for L in d["u"]:
        u = d["u"][L].detach().cpu().float().numpy(); v = d["v"][L].detach().cpu().float().numpy()
        out[int(L)] = compiled_direction(u, v) * float(scale)
    return out


def cols_direction(V):
    """column block V (rows x cols): top left singular vector, sign toward the mean column, norm = ||V||_F."""
    V = np.asarray(V, dtype=np.float64)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    u = U[:, 0]
    if float(u @ V.mean(axis=1)) < 0:
        u = -u
    return (u * np.linalg.norm(V)).astype(np.float32)


def load_cols(npz_path, layers, scale=1.0):
    """ediciones npz (cols_i / vec_i in the order of `layers`) -> {layer: direction}."""
    out = {}
    with np.load(str(npz_path), allow_pickle=False) as z:
        for i, L in enumerate(layers):
            out[int(L)] = cols_direction(z[f"vec_{i}"]) * float(scale)
    return out


def stack(dirs):
    layers = sorted(dirs)
    return np.stack([np.asarray(dirs[L], dtype=np.float64) for L in layers]), layers


# ----------------------------------------------------------------------------- features
def mag_features(D):
    D = np.asarray(D, dtype=np.float64)
    norms = np.linalg.norm(D, axis=1)
    L = D.shape[0]
    p2 = float((norms ** 2).sum()); p4 = float((norms ** 4).sum())
    s = np.linalg.svd(D, compute_uv=False)
    p = s / s.sum() if s.sum() > 0 else np.ones_like(s) / len(s)
    p = p[p > 0]
    return {"mag_total": float(np.sqrt(p2)), "mag_mean": float(norms.mean()), "mag_max": float(norms.max()),
            "argmax_depth": float((int(norms.argmax()) + 1) / L),
            "participation": float(p2 ** 2 / (L * p4)) if p4 > 0 else 0.0,
            "eff_rank": float(np.exp(-(p * np.log(p)).sum()))}


def readout(head_row, norm_w):
    """readout direction of a token in residual space: lm_head row times the final RMSNorm weight."""
    return np.asarray(head_row, dtype=np.float64) * np.asarray(norm_w, dtype=np.float64)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def cos_features(D, r_close, r_open, late=8):
    D = np.asarray(D, dtype=np.float64)
    cc = np.array([_cos(d, r_close) for d in D]); co = np.array([_cos(d, r_open) for d in D])
    k = min(late, len(cc))
    return {"cos_close_mean": float(cc.mean()), "cos_close_late8": float(cc[-k:].mean()), "cos_close_last": float(cc[-1]),
            "cos_open_mean": float(co.mean()), "cos_open_late8": float(co[-k:].mean()), "cos_open_last": float(co[-1]),
            "cos_gap": float(cc.mean() - co.mean())}


def cos_between(A, B):
    """mean cosine over the layers two direction dicts share."""
    common = sorted(set(A) & set(B))
    if not common:
        return float("nan")
    return float(np.mean([_cos(np.asarray(A[L], dtype=np.float64), np.asarray(B[L], dtype=np.float64)) for L in common]))


# ----------------------------------------------------------------------------- statistics (scipy-free)
def _ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(scores, labels):
    """Mann-Whitney AUC of scores for label 1 over label 0; ties count 0.5."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- targets
def _n_correct(s, key="acc"):
    return int(round(s[key] * s["n"]))


def targets(arm, base):
    """arm / base: bench_runner report dicts with `summary`. -> deltas (arm - base) per bench, tokens %, cuts, n."""
    sa, sb = arm["summary"], base["summary"]
    out = {"d_gsm": None, "d_gsm_g2": None, "d_math": None, "d_math_g2": None}
    n = 0; tok_a = tok_b = 0.0; cut = 0
    for bench, key in (("gsm8k", "gsm"), ("math500", "math")):
        if bench in sa and bench in sb:
            out[f"d_{key}"] = _n_correct(sa[bench]) - _n_correct(sb[bench])
            out[f"d_{key}_g2"] = _n_correct(sa[bench], "acc_independent") - _n_correct(sb[bench], "acc_independent")
            n += sa[bench]["n"]
            tok_a += sa[bench]["mean_tokens"] * sa[bench]["n"]; tok_b += sb[bench]["mean_tokens"] * sb[bench]["n"]
            cut += sa[bench]["cut_at_max"] - sb[bench]["cut_at_max"]
    out["d_tok"] = float(100.0 * (tok_a / tok_b - 1.0)) if tok_b > 0 else None
    out["d_cut"] = int(cut)
    out["n"] = int(n)
    return out


def target_row(arm_path, base_path):
    arm = json.loads(Path(arm_path).read_text(encoding="utf-8"))
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    return targets(arm, base)
