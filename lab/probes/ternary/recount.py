"""Second method for lab/probes/ternary (no shared code with engine/ternary.py): reads the base tensor with the
safetensors library, builds dW itself, ternarises with its own code (ggml TQ2_0: absmax per 256-block, fp16 scale;
absmean: mean |w| per row) and recomputes flip_touched / cos / alpha for k = 1 and the largest k of a report.

  python -I lab/probes/ternary/recount.py lab/runs/ternary/<report>.json [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_tensor(hf_dir, name):
    from safetensors import safe_open
    import torch
    hf_dir = Path(hf_dir)
    idx = json.loads((hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    with safe_open(str(hf_dir / idx[name]), framework="pt") as f:
        t = f.get_tensor(name)
    return t.to(torch.float32).numpy()


def tern(W, rule, scales=None):
    if rule == "absmean":
        s = np.abs(W).mean(axis=1) if scales is None else scales
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s[:, None] > 0, W / s[:, None], 0.0)
    else:
        B = W.reshape(W.shape[0], -1, 256)
        s = np.abs(B).max(axis=-1).astype(np.float16).astype(np.float32) if scales is None else scales
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(s[:, :, None] > 0, B / s[:, :, None], 0.0).reshape(W.shape)
    c = np.clip(np.trunc(q + np.copysign(0.5, q)), -1, 1)   # round half away from zero
    if rule == "absmean":
        d = c * s[:, None]
    else:
        d = (c.reshape(W.shape[0], -1, 256) * s[:, :, None]).reshape(W.shape)
    return c, s, d


def delta(rep, i, L, shape):
    p = ROOT / rep["delta"] if not Path(rep["delta"]).is_absolute() else Path(rep["delta"])
    if str(p).endswith(".npz"):
        with np.load(p) as z:
            dW = np.zeros(shape, dtype=np.float32)
            dW[:, np.asarray(z[f"cols_{i}"], dtype=int)] = z[f"vec_{i}"]
        return dW
    import torch
    fit = torch.load(p, map_location="cpu", weights_only=False)
    return np.outer(fit["v"][L].float().numpy(), fit["u"][L].float().numpy()).astype(np.float32)


def main(paths):
    for path in paths:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        rule, mode = rep["rule"], rep["mode"]
        ks = [rep["scales"][0], rep["scales"][-1]]
        print(Path(path).name)
        for i, L in enumerate(rep["layers"]):
            name = rep["name"].format(L=L)
            W0 = load_tensor(rep["base"], name)
            dW = delta(rep, i, L, W0.shape)
            c0, s0, d0 = tern(W0, rule)
            touched = dW != 0
            mine = {}
            for k in ks:
                c1, s1, d1 = tern(W0 + k * dW, rule, None if mode == "requant" else s0)
                D = (d1 - d0).ravel().astype(np.float64)
                E = (k * dW).ravel().astype(np.float64)
                mine[k] = {"flip_touched": float((c1 != c0)[touched].mean()),
                           "cos": float(D @ E / (np.linalg.norm(D) * np.linalg.norm(E))),
                           "alpha": float(D @ E / (E @ E))}
            theirs = {c["k"]: c for c in rep["tensors"][name]["cells"] if c["k"] in ks}
            for k in ks:
                a, b = mine[k], theirs[k]
                ok = all(abs(a[m] - b[m]) <= 1e-3 * max(1.0, abs(b[m])) for m in ("flip_touched", "cos", "alpha"))
                print(f"  L{L} k={k}: recount flip {a['flip_touched']:.4f} cos {a['cos']:.4f} alpha {a['alpha']:.4f}"
                      f" | op flip {b['flip_touched']:.4f} cos {b['cos']:.4f} alpha {b['alpha']:.4f} -> {'AGREE' if ok else 'DIFFER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
