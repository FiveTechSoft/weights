"""Prototype measurements for lab/probes/discrete_writer/PLAN.md (CPU): the exact threshold writer on real weights.
  python lab/probes/discrete_writer/run.py            -> lab/runs/discrete_writer/{cc9b,exp8}.json and a table on stdout
Second method for cos: a direct numpy expression beside engine.ternary.overlap's value.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine import ternary as tn  # noqa: E402
from engine import weight_diff as wd  # noqa: E402
from lab.probes.discrete_writer import writer as dw  # noqa: E402

OUT = ROOT / "lab/runs/discrete_writer"
NAME = "model.language_model.layers.{L}.mlp.down_proj.weight"
FRACS = (0.01, 0.05, 0.20)


def cell(W0, T, frac):
    lam = dw.lam_for_fraction(W0, T, frac)
    D, chosen, info = dw.discrete_threshold(W0, T, lam)
    alpha, noise, cos = tn.overlap(D, T)
    cos2 = float((D.astype(np.float64) * T).sum() / (np.linalg.norm(D.astype(np.float64)) * np.linalg.norm(T.astype(np.float64)) + 1e-30))
    flips = dw.bit_flips(W0, dw.to_bf16(W0 + D))
    return {"frac_requested": frac, "lam": lam, "changed": info["changed"], "share": info["changed"] / W0.size,
            "attainable_share": float((dw.gains(W0, T) > 0).mean()), "moves": info["moves"],
            "cos": cos, "cos_check": cos2, "alpha": alpha, "noise": noise, "bit_flips": flips}


def run_cc9b():
    import torch
    fit = torch.load(ROOT / "hf_cache/cc_op_v2/vector/fit.pt", map_location="cpu", weights_only=False)
    reader = wd.LocalSafetensors(str(ROOT / "hf_cache/qwen35-9b-hf"))
    rep = {"model": "hf_cache/qwen35-9b-hf", "delta": "hf_cache/cc_op_v2/vector/fit.pt", "layers": {}}
    for L in (1, 16, 31):
        t0 = time.time()
        W0 = reader.get(NAME.format(L=L)).float().numpy()
        dW = tn.delta_from_fit(fit, L)
        assert dW.shape == W0.shape
        per = {"rel_norm_dW": float(np.linalg.norm(dW) / np.linalg.norm(W0)), "doses": {}}
        for k in (1, 2, 5):
            T = (k * dW).astype(np.float32)
            per["doses"][str(k)] = [cell(W0, T, f) for f in FRACS]
        per["seconds"] = round(time.time() - t0, 1)
        rep["layers"][str(L)] = per
        print(f"9B L{L} done {per['seconds']}s", flush=True)
    (OUT / "cc9b.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def run_exp8():
    reader = wd.LocalSafetensors(str(ROOT / "hf_cache/qwen35-08b-hf"))
    rep = {"model": "hf_cache/qwen35-08b-hf", "delta": "data/exp8_edit_recovered.npz", "layers": {}}
    with np.load(ROOT / "data/exp8_edit_recovered.npz") as z:
        for i, L in enumerate((15, 19)):
            W0 = reader.get(NAME.format(L=L)).float().numpy()
            dW = tn.delta_from_npz(z, i, W0.shape)
            per = {"rel_norm_dW": float(np.linalg.norm(dW) / np.linalg.norm(W0)),
                   "touched_share": float((dW != 0).mean()), "doses": {"1": [cell(W0, dW, f) for f in FRACS]}}
            rep["layers"][str(L)] = per
            print(f"0.8B L{L} done", flush=True)
    (OUT / "exp8.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def table(rep, label):
    print(f"\n{label}")
    print("layer k frac_req share(att) lam cos(check) alpha noise bitflips moves")
    for L, per in rep["layers"].items():
        for k, cells in per["doses"].items():
            for c in cells:
                print(f"L{L} k={k} {c['frac_requested']:.2f} {c['share']:.4f}({c['attainable_share']:.4f}) "
                      f"{c['lam']:.2e} {c['cos']:.3f}({c['cos_check']:.3f}) {c['alpha']:.2f} {c['noise']:.2f} "
                      f"{c['bit_flips']} {c['moves']}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    e = run_exp8()
    table(e, "Exp 8 edit on Qwen3.5-0.8B (threshold writer)")
    c = run_cc9b()
    table(c, "compiled conclude dW on Qwen3.5-9B (threshold writer)")
