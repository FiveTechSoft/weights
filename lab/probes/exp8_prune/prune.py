"""Prune the recovered Exp 8 edit (data/exp8_edit_recovered.npz: cols_i (64,), vec_i (1024, 64) per layer) to its k
largest entries by |value| per layer, as alphaXiv / Daniel Kim's family D does (D4k = 4,096 per layer), or to the
complement (everything but those). The output npz keeps the same keys so ediciones/*.json can apply it with the
add-cols-vectors op unchanged. Plan: lab/probes/exp8_prune/PLAN.md. Tests: tests/test_exp8_prune.py.

  python lab/probes/exp8_prune/prune.py <src.npz> <k> <out.npz> [--complement]   -> prints the per-layer stats as JSON

stats per layer: kept (entries), energy_frac (||V*m||^2 / ||V||^2), push_frac (||u^T (V*m)|| / ||u^T V|| with u the
top left singular vector of V: the part of the push along the edit's direction that survives), cos_to_full
(<V*m, V> / (||V*m|| ||V||)).
"""
import json
import sys

import numpy as np


def topk_mask(V, k):
    """Boolean mask of the k largest |entries| of V (all of them if k >= V.size)."""
    V = np.asarray(V)
    k = int(min(k, V.size))
    flat = np.abs(V).ravel()
    idx = np.argpartition(-flat, k - 1)[:k] if k < V.size else np.arange(V.size)
    m = np.zeros(V.size, dtype=bool)
    m[idx] = True
    return m.reshape(V.shape)


def stats(V, m):
    V = np.asarray(V, dtype=np.float64)
    Vm = V * m
    u = np.linalg.svd(V, full_matrices=False)[0][:, 0]
    nV, nVm = np.linalg.norm(V), np.linalg.norm(Vm)
    return {"kept": int(m.sum()),
            "energy_frac": float((nVm / nV) ** 2) if nV else 0.0,
            "push_frac": float(np.linalg.norm(u @ Vm) / (np.linalg.norm(u @ V) + 1e-30)),
            "cos_to_full": float((Vm * V).sum() / (nVm * nV + 1e-30))}


def prune_npz(src, k, out, complement=False):
    z = np.load(src, allow_pickle=False)
    layers = sorted(int(n.split("_")[1]) for n in z.files if n.startswith("vec_"))
    arrays, report = {}, {}
    for i in layers:
        V = np.asarray(z[f"vec_{i}"])
        m = topk_mask(V, k)
        if complement:
            m = ~m
        arrays[f"cols_{i}"] = np.asarray(z[f"cols_{i}"])
        arrays[f"vec_{i}"] = (V * m).astype(V.dtype)
        report[f"layer_{i}"] = stats(V, m)
    np.savez(out, **arrays)
    return report


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    comp = "--complement" in argv
    rep = prune_npz(argv[0], int(argv[1]), argv[2], complement=comp)
    print(json.dumps({"src": argv[0], "k": int(argv[1]), "out": argv[2], "complement": comp, "layers": rep}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
