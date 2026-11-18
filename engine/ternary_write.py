"""Ternary checkpoints for NanoWSP-class models and the WSP-Lang ops `ternarize` / `imprint_ternary`
(lab/probes/ternary_nanowsp/PLAN.md). Rules and readings come from engine/ternary.py; this module adds the exact
per-row solve W_row ≈ s · T_row (T ∈ {−1, 0, +1}, one float scale per output row), a checkpoint format that keeps the
existing runtime working (float materialisation beside the codes), and the closed-form associative write projected
back to ternary with TTF-style accumulator refinement.

Checkpoint layout (a directory): model.safetensors (float materialisation s·T for the ternary tensors, the float
tensors unchanged), ternary.safetensors (<name>.codes int8, <name>.scales float32), manifest.json. Ternary tensors are
the 2-D projections inside the layers; embeddings, output head and norms stay float.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from engine import ternary as tn

TERNARY_PATTERNS = (r"^layers\.\d+\.attn_(q|k|v|out)\.weight$", r"^layers\.\d+\.mlp_(in|out)\.weight$")


def is_ternary_tensor(name, patterns=None):
    pats = patterns or TERNARY_PATTERNS
    return any(re.match(p, name) for p in pats)


def _globs_to_patterns(spec):
    """tensors=a.*.b,c -> regex list (shell-style * over the full name)."""
    out = []
    for g in str(spec).split(","):
        g = g.strip()
        if g:
            out.append("^" + re.escape(g).replace(r"\*", ".*") + "$")
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def solve_rows(W, rule="absmean", iters=3, history=False):
    """Per-row alternating solve of min ||W_row - s T_row||^2: T = clip(round(W/s)), s = <W,T>/<T,T>.
    -> codes int8, scales float32 (rows,), residual per row (float64); with history=True also the per-iteration
    residual list (iteration 0 = the rule's initial scale)."""
    W = np.asarray(W, dtype=np.float32)
    if rule == "absmean":
        s = np.abs(W).mean(axis=1).astype(np.float64)
    elif rule == "tq2_0":
        s = np.abs(W).max(axis=1).astype(np.float64) if W.shape[1] % tn.BLOCK else np.abs(W).max(axis=1).astype(np.float64)
    else:
        raise ValueError(f"rule must be absmean or tq2_0, got {rule!r}")
    W64 = W.astype(np.float64)
    hist = []

    def codes_for(scale):
        inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1.0), 0.0)
        return np.clip(np.sign(W64 * inv[:, None]) * np.floor(np.abs(W64 * inv[:, None]) + 0.5), -1, 1)

    T = codes_for(s)
    hist.append(((W64 - s[:, None] * T) ** 2).sum(axis=1))
    for _ in range(int(iters)):
        nnz = (T != 0).sum(axis=1)
        s_new = np.where(nnz > 0, (W64 * T).sum(axis=1) / np.where(nnz > 0, nnz, 1), s)
        T_new = codes_for(s_new)
        r_new = ((W64 - s_new[:, None] * T_new) ** 2).sum(axis=1)
        # exact minimisers per variable: never worse than the previous iterate (guard against float ties)
        keep = r_new <= hist[-1] + 1e-12
        s = np.where(keep, s_new, s)
        T = np.where(keep[:, None], T_new, T)
        hist.append(np.where(keep, r_new, hist[-1]))
    codes = T.astype(np.int8)
    scales = s.astype(np.float32)
    resid = np.sqrt(hist[-1])
    return (codes, scales, resid, hist) if history else (codes, scales, resid)


def deq(codes, scales):
    return np.asarray(codes, dtype=np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


def _check_out(out, inputs):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; ternary checkpoints are never overwritten")
    o = out.resolve()
    for p in inputs:
        p = Path(p).resolve()
        root = p if p.is_dir() else p.parent
        if o == root or root in o.parents:
            raise ValueError(f"refusing to write {out} inside the input {p}")
    return out


def _ckpt_file(model):
    model = Path(model)
    return model if model.is_file() else model / "model.safetensors"


def ternarize_tensors(weights, rule="absmean", patterns=None, iters=3):
    """weights: dict name -> torch tensor. -> (float dict, ternary dict with .codes/.scales, per-tensor report)."""
    flo, tern, rep = {}, {}, {}
    for name, w in weights.items():
        if w.ndim == 2 and is_ternary_tensor(name, patterns):
            W = w.detach().cpu().float().numpy()
            codes, scales, resid = solve_rows(W, rule=rule, iters=iters)
            D = deq(codes, scales)
            flo[name] = torch.from_numpy(D.copy())
            tern[name + ".codes"] = torch.from_numpy(codes.copy())
            tern[name + ".scales"] = torch.from_numpy(scales.copy())
            wn = float(np.linalg.norm(W))
            rep[name] = {"shape": list(W.shape), "density": float((codes != 0).mean()),
                         "rel_err": float(np.linalg.norm(W - D) / wn) if wn else 0.0,
                         "scale_median": float(np.median(scales))}
        else:
            flo[name] = w.detach().cpu().float().contiguous()
    return flo, tern, rep


def write_checkpoint(out, flo, tern, manifest):
    from safetensors.torch import save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in flo.items()}, str(out / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in tern.items()}, str(out / "ternary.safetensors"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def ternarize_checkpoint(model, out, rule="absmean", tensors=None, iters=3):
    """model: NanoWSP checkpoint file or dir. Writes out/{model,ternary}.safetensors + manifest.json. -> manifest."""
    from safetensors.torch import load_file
    src = _ckpt_file(model)
    if not src.exists():
        raise FileNotFoundError(src)
    out = _check_out(out, [src])
    patterns = _globs_to_patterns(tensors) if tensors else None
    weights = load_file(str(src))
    flo, tern, rep = ternarize_tensors(weights, rule=rule, patterns=patterns, iters=iters)
    if not rep:
        raise ValueError("no tensor matched the ternary selection")
    manifest = {"op": "ternarize", "source": str(src), "source_sha256": _sha(src), "rule": rule, "iters": iters,
                "ternary_tensors": sorted(rep), "float_tensors": sorted(k for k in weights if k not in rep),
                "tensors": rep}
    write_checkpoint(out, flo, tern, manifest)
    return manifest


def load_ternary_dir(path):
    """-> (float weights dict, ternary dict, manifest) of a ternary checkpoint dir."""
    from safetensors.torch import load_file
    path = Path(path)
    flo = load_file(str(path / "model.safetensors"))
    tern = load_file(str(path / "ternary.safetensors")) if (path / "ternary.safetensors").exists() else {}
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8")) if (path / "manifest.json").exists() else {}
    return flo, tern, man


def _model_from(flo):
    from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig
    d_model = int(flo["token_embd.weight"].shape[1])
    d_ff = int(flo["layers.0.mlp_in.weight"].shape[0])
    n_layers = len({k.split(".")[1] for k in flo if k.startswith("layers.")})
    max_seq = int(flo["pos_embd.weight"].shape[0])
    vocab = int(flo["token_embd.weight"].shape[0])
    cfg = NanoWSPConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers, d_ff=d_ff, max_seq_len=max_seq,
                        n_heads=2, d_head=d_model // 2)
    m = NanoWSP(cfg, seed=0)
    m.weights = {k: v.clone().float() for k, v in flo.items()}
    return m


def keys_values(model, pairs, layer):
    """The write's keys (normalised mlp hidden at the last position) and values (normalised output rows)."""
    keys, vals = [], []
    for p, t in pairs:
        ids = model.tokenizer.encode(p, add_bos=True)
        _, tr = model.forward(ids, return_hidden=True)
        k = tr[f"layer_{layer}_mlp_hidden"][0, -1, :].clone()
        keys.append(k / max(float(torch.norm(k)), 1e-6))
        tid = model.tokenizer.encode(t, add_bos=False)[0]
        v = model.weights["output.weight"][tid].clone()
        vals.append(v / max(float(torch.norm(v)), 1e-6))
    return torch.stack(keys, dim=1), torch.stack(vals, dim=1)      # X [d_ff, M], Y [d_model, M]


def imprint_ternary(model_dir, pairs, layer=1, eta=4.0, rounds=3, out=None, lam=1e-4, rule="absmean", iters=3,
                    holdout=None):
    """Closed-form write of `pairs` into layers.<layer>.mlp_out.weight of a ternary NanoWSP, projected back to
    {-1,0,+1}·s per row, with `rounds` TTF-style refinements of the accumulator on the written keys. Writes a new
    ternary checkpoint at out= (+ report.json). -> report."""
    model_dir = Path(model_dir)
    flo, tern, man = load_ternary_dir(model_dir)
    out = _check_out(out, [model_dir]) if out else None
    model = _model_from(flo)
    name = f"layers.{layer}.mlp_out.weight"
    X, Y = keys_values(model, pairs, layer)
    M = X.shape[1]
    XtX = X.t() @ X + float(lam) * torch.eye(M)
    X_pinv = torch.linalg.solve(XtX, X.t())                         # [M, d_ff]
    W0 = model.weights[name].clone()
    A = W0 + float(eta) * (Y @ X_pinv)                              # float write (accumulator)
    target_out = A @ X                                               # what the float write produces on the keys
    key_res = []

    def project(Acc):
        c, s, _ = solve_rows(Acc.numpy(), rule=rule, iters=iters)
        return torch.from_numpy(deq(c, s)), c, s

    D, codes, scales = project(A)
    key_res.append(float(torch.norm((target_out - D @ X)) / max(float(torch.norm(target_out)), 1e-12)))
    for _ in range(int(rounds)):
        deficit = target_out - D @ X                                 # [d_model, M]
        A = A + deficit @ X_pinv                                     # re-imprint the deficit through the same pinv
        D, codes, scales = project(A)
        key_res.append(float(torch.norm((target_out - D @ X)) / max(float(torch.norm(target_out)), 1e-12)))
    model.weights[name] = D
    recall = model.evaluate_recall(pairs)
    held = model.evaluate_recall(holdout) if holdout else None
    rep = {"op": "imprint_ternary", "tensor": name, "pairs": M, "eta": float(eta), "lambda": float(lam),
           "rounds": int(rounds), "rule": rule, "key_residual": key_res,
           "density": float((codes != 0).mean()), "recall": recall, "holdout_accuracy": held,
           "delta_norm": float(torch.norm(D - W0)), "w0_norm": float(torch.norm(W0))}
    if out is not None:
        flo2 = {k: (D if k == name else v) for k, v in flo.items()}
        tern2 = dict(tern)
        tern2[name + ".codes"] = torch.from_numpy(codes.copy())
        tern2[name + ".scales"] = torch.from_numpy(scales.copy())
        manifest = {"op": "imprint_ternary", "source": str(model_dir), "source_manifest": man,
                    "source_sha256": _sha(model_dir / "model.safetensors"), "report": rep}
        write_checkpoint(out, flo2, tern2, manifest)
        (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        rep["out"] = str(out)
    return rep
