"""Ternary weights for WSP-Lang (lab/probes/ternary/PLAN.md): ternarise a matrix under ggml's TQ2_0 rule or BitNet
b1.58's absmean rule, and measure how much of a small edit dW survives when the edited matrix is ternarised again
(`requant`: own scales, "edit in float, then convert") or re-coded on the base's scales (`fixed`: "edit the ternary
model in place"). Read-only: the WSP-Lang op `ternary_survival` writes one JSON report, never weights.

Rules
  tq2_0   ggml: blocks of 256 along the last axis, scale = absmax of the block (stored as fp16), code = round(w / absmax)
          with round-half-away-from-zero; dequant matches gguf.quants.dequantize(TQ2_0) (tests).
  absmean BitNet b1.58 with the scale per output row: gamma = mean |w| over the row, code = clip(round(w / gamma), -1, 1).

Readings per scale k (D = deq(T(W0 + k dW)) - deq(T(W0)))
  flip_touched  share of entries with dW != 0 whose code changed      flip_all  share over all entries
  cos           <D, k dW> / (|D| |k dW|)                               alpha     <D, k dW> / |k dW|^2
  noise         |D - alpha k dW| / |alpha k dW|                        k70       smallest k with cos >= 0.7, else "> max k"
"""
import json
from pathlib import Path

import numpy as np

RULES = ("tq2_0", "absmean")
MODES = ("requant", "fixed")
BLOCK = 256
DEFAULT_NAME = "model.language_model.layers.{L}.mlp.down_proj.weight"


def _roundf(x):
    """round half away from zero (ggml's roundf), as gguf.quants does."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _check_rule(rule):
    if rule not in RULES:
        raise ValueError(f"ternary rule must be one of {RULES}, got {rule!r}")


def _blocks(W):
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2 or W.shape[1] % BLOCK:
        raise ValueError(f"tq2_0 needs a 2-D matrix with columns divisible by {BLOCK}, got {W.shape}")
    return W.reshape(W.shape[0], W.shape[1] // BLOCK, BLOCK)


def ternarize(W, rule):
    """-> codes int8 in {-1, 0, 1} (same shape as W), scales float32 (rows,) for absmean, (rows, blocks) for tq2_0."""
    _check_rule(rule)
    W = np.asarray(W, dtype=np.float32)
    if rule == "absmean":
        g = np.abs(W).mean(axis=1)
        inv = np.where(g == 0, 0.0, 1.0 / np.where(g == 0, 1.0, g))
        codes = np.clip(_roundf(W * inv[:, None]), -1, 1).astype(np.int8)
        return codes, g.astype(np.float32)
    B = _blocks(W)
    d = np.abs(B).max(axis=-1)                                  # (rows, blocks)
    inv = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1.0, d))
    codes = np.clip(_roundf(B * inv[:, :, None]), -1, 1).astype(np.int8).reshape(W.shape)
    return codes, d.astype(np.float16).astype(np.float32)        # ggml stores the block scale as fp16


def ternarize_fixed(W, scales, rule):
    """Codes of W on given scales (the base's), scales returned unchanged."""
    _check_rule(rule)
    W = np.asarray(W, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32)
    inv = np.where(s == 0, 0.0, 1.0 / np.where(s == 0, 1.0, s))
    if rule == "absmean":
        codes = np.clip(_roundf(W * inv[:, None]), -1, 1).astype(np.int8)
    else:
        codes = np.clip(_roundf(_blocks(W) * inv[:, :, None]), -1, 1).astype(np.int8).reshape(W.shape)
    return codes, s


def dequant(codes, scales, rule):
    _check_rule(rule)
    c = np.asarray(codes, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32)
    if rule == "absmean":
        return c * s[:, None]
    return (c.reshape(c.shape[0], c.shape[1] // BLOCK, BLOCK) * s[:, :, None]).reshape(c.shape)


def overlap(D, dW):
    """-> (alpha, noise, cos) of D against dW; all zero when dW or D is zero."""
    D = np.asarray(D, dtype=np.float64).ravel()
    E = np.asarray(dW, dtype=np.float64).ravel()
    ee, dd, de = float(E @ E), float(D @ D), float(D @ E)
    if ee == 0 or dd == 0:
        return 0.0, 0.0, 0.0
    alpha = de / ee
    cos = de / np.sqrt(dd * ee)
    kept = alpha * E
    kn = float(np.linalg.norm(kept))
    noise = float(np.linalg.norm(D - kept) / kn) if kn > 0 else 0.0
    return float(alpha), noise, float(cos)


def survival(W0, dW, rule, mode, scales=(1, 2, 5, 10, 20)):
    _check_rule(rule)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    W0 = np.asarray(W0, dtype=np.float32)
    dW = np.asarray(dW, dtype=np.float32)
    if dW.shape != W0.shape:
        raise ValueError(f"dW shape {dW.shape} != W0 shape {W0.shape}")
    c0, s0 = ternarize(W0, rule)
    D0 = dequant(c0, s0, rule)
    touched = dW != 0
    n_touched = int(touched.sum())
    cells, k70 = [], None
    for k in scales:
        W1 = W0 + float(k) * dW
        c1, s1 = ternarize(W1, rule) if mode == "requant" else ternarize_fixed(W1, s0, rule)
        D = dequant(c1, s1, rule) - D0
        changed = c1 != c0
        alpha, noise, cos = overlap(D, float(k) * dW)
        cells.append({"k": k, "flip_touched": float(changed[touched].mean()) if n_touched else 0.0,
                      "flip_all": float(changed.mean()), "cos": cos, "alpha": alpha, "noise": noise})
        if k70 is None and cos >= 0.7:
            k70 = k
    w0n = float(np.linalg.norm(W0))
    return {"rule": rule, "mode": mode, "shape": list(W0.shape), "touched": n_touched,
            "rel_norm_dW": float(np.linalg.norm(dW) / w0n) if w0n else 0.0,
            "base_nonzero": float((c0 != 0).mean()), "cells": cells,
            "k70": k70 if k70 is not None else f"> {max(scales)}"}


def delta_from_npz(z, i, shape):
    """ediciones npz (cols_i, vec_i) -> dense dW of `shape` (rows, cols): dW[:, cols_i] = vec_i."""
    cols = np.asarray(z[f"cols_{i}"], dtype=int)
    V = np.asarray(z[f"vec_{i}"], dtype=np.float32)
    dW = np.zeros(shape, dtype=np.float32)
    if V.shape != (shape[0], cols.size):
        raise ValueError(f"vec_{i} shape {V.shape} != ({shape[0]}, {cols.size})")
    dW[:, cols] = V
    return dW


def delta_from_fit(fit, layer):
    """compile fit (u per neuron, v per output dim) -> dW = outer(v, u), as engine.cc_compile.merge_q6k writes it."""
    return np.outer(_np(fit["v"][layer]), _np(fit["u"][layer])).astype(np.float32)


def report(base, delta, layers, rule="tq2_0", mode="requant", scales=(1, 2, 5, 10, 20), out=None, name=None):
    """base: HF dir (safetensors); delta: .npz (tensors numbered in the order of `layers`) or a compile fit .pt."""
    from engine import weight_diff as wd
    name = name or DEFAULT_NAME
    reader = wd.LocalSafetensors(base)
    delta = str(delta)
    z = fit = None
    if delta.endswith(".npz"):
        z = np.load(delta, allow_pickle=False)
    else:
        import torch
        fit = torch.load(delta, map_location="cpu", weights_only=False)
    tensors = {}
    try:
        for i, L in enumerate(layers):
            tname = name.format(L=L)
            W0 = reader.get(tname).float().numpy()
            dW = delta_from_npz(z, i, W0.shape) if z is not None else delta_from_fit(fit, L)
            tensors[tname] = survival(W0, dW, rule, mode, scales)
    finally:
        if z is not None:
            z.close()
    rep = {"base": str(base), "delta": delta, "rule": rule, "mode": mode, "scales": list(scales),
           "layers": list(layers), "name": name, "tensors": tensors}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep
