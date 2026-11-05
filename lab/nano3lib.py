"""nano3lib: builder batched del NanoWSP-3 desde trace npz (W0 o snapshot) + fwd reutilizable.
Compartido por los runners del curriculum (M-2b, finetune_dataset, compile_from_trace)."""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path(r"C:\weights")

def load_trace(path=None):
    p = Path(path or R / "lab" / "runs" / "nanowsp3_trace.npz")
    return np.load(p, allow_pickle=False)

def _keys_from(names_list):
    keys = []
    for nm in names_list:
        if "." in nm and nm.split(".")[1].isdigit():
            k_, idx = nm.split(".")
            keys.append(f"{k_}.{idx}.weight")
        else:
            keys.append(nm)
    return keys

def build_from_trace(tr, snap_idx=None):
    names = json.loads(str(tr["names"]))
    if isinstance(names, str):
        names = json.loads(names)
    shapes_raw = json.loads(str(tr["shapes"]))
    shapes = shapes_raw if isinstance(shapes_raw, dict) else json.loads(shapes_raw)
    keys = _keys_from(names)
    flat = tr["W0_flat"] if snap_idx is None else tr["snapshots_flat"][snap_idx]
    parts, off = {}, 0
    for nm, key in zip(names, keys):
        n = int(np.prod(shapes[nm]))
        parts[key] = torch.tensor(flat[off:off + n].reshape(shapes[nm]), dtype=torch.float32)
        off += n
    assert off == len(flat)
    return parts

D_MODEL, N_LAYERS, N_HEADS, D_HEAD, D_FF = 256, 6, 8, 32, 512

def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g

def rope(x):
    b, s, h, d = x.shape
    theta = 10000.0
    inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(s, dtype=torch.float32)
    ang = torch.outer(t, inv)
    fcs = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))
    xout = torch.view_as_real(xc * fcs.view(1, s, 1, d // 2)).flatten(3)
    return xout.type_as(x)

def _resolve_cfg(P):
    d_model = P["embed.weight"].shape[1]
    heads = d_model // 32
    layers = max(int(k.split(".")[1]) for k in P if k.startswith("q.")) + 1
    d_ff = P["min_.0.weight"].shape[0]
    return d_model, layers, heads, d_model // heads, d_ff

def fwd(x, P):
    B, S = x.shape
    D, L, H, DH, DFF = _resolve_cfg(P)
    h = F.embedding(x, P["embed.weight"])
    for l in range(L):
        hn = rms(h, P[f"an.{l}.weight"])
        q = (hn @ P[f"q.{l}.weight"].T).view(B, S, H, DH)
        k = (hn @ P[f"k.{l}.weight"].T).view(B, S, H, DH)
        v = (hn @ P[f"v.{l}.weight"].T).view(B, S, H, DH)
        q, k = rope(q), rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        sc = q @ k.transpose(-2, -1) / math.sqrt(DH)
        mask = torch.triu(torch.full((S, S), float("-inf")), 1)
        att = torch.softmax(sc + mask, dim=-1)
        h = h + (att @ v).transpose(1, 2).contiguous().view(B, S, D) @ P[f"o.{l}.weight"].T
        hm = rms(h, P[f"mn.{l}.weight"])
        h = h + F.gelu(hm @ P[f"min_.{l}.weight"].T) @ P[f"mout.{l}.weight"].T
    hf = rms(h, P["fn.weight"])
    return hf @ P["out.weight"].T, hf

def encode(prompt):
    return torch.tensor(list(prompt.encode("ascii", "ignore")), dtype=torch.long)

def prep(rows):
    xs = [encode(r["prompt"]) for r in rows]
    ys = torch.tensor([ord(r["target"][0]) for r in rows], dtype=torch.long)
    L = max(len(x) for x in xs)
    X = torch.zeros(len(xs), L, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, L - len(x):] = x
    return X, ys
