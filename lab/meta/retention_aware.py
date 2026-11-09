"""Mechanism probe: why the learned configuration is eta ~ 1 and why writing at the old layer matters.

imprint_thematic_corpus computes  delta = eta * (V - W0 K) (K^T K + lam I)^-1 K^T  from the NEW keys only.
- eta = 1 (lam -> 0) is the exact least-squares write, so "eta near 1" is what the algebra says, not a discovery.
- Old facts are disturbed because their keys k_old are not orthogonal to the new keys: delta k_old != 0. The formula
  has no term for them. Adding keep-keys to the solve with zero residual (MEMIT-style) removes that cross-talk:
      delta = eta * R_all (K_all^T K_all + lam I)^-1 K_all^T,  R_all = [V - W0 K | 0].
This module implements that variant outside the engine, to measure it on development tasks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def keys_values(model, facts, layer):
    ks, vs = [], []
    for p, t in facts:
        ids = model.tokenizer.encode(p, add_bos=True)
        _, tr = model.forward(ids, return_hidden=True)
        k = tr[f"layer_{layer}_mlp_hidden"][0, -1, :].clone()
        ks.append(k / max(float(torch.norm(k)), 1e-6))
        tid = model.tokenizer.encode(t, add_bos=False)[0]
        v = model.weights["output.weight"][tid].clone()
        vs.append(v / max(float(torch.norm(v)), 1e-6))
    return torch.stack(ks, 1), torch.stack(vs, 1)


def imprint_keep(model, facts, layer, eta, lam, keep_facts=None):
    """Closed-form write of `facts` that also constrains the response to `keep_facts` keys to stay unchanged."""
    key = f"layers.{layer}.mlp_out.weight"
    W0 = model.weights[key]
    K, V = keys_values(model, facts, layer)
    R = V - W0 @ K
    if keep_facts:
        Kk, _ = keys_values(model, keep_facts, layer)
        K = torch.cat([K, Kk], 1)
        R = torch.cat([R, torch.zeros(R.shape[0], Kk.shape[1])], 1)
    gram = K.t() @ K + lam * torch.eye(K.shape[1])
    model.weights[key] += eta * R @ torch.linalg.inv(gram) @ K.t()
