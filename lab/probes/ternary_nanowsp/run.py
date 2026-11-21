"""Float vs ternary NanoWSP associative memory (lab/probes/ternary_nanowsp/PLAN.md). CPU, minutes.
Writes lab/runs/ternary_nanowsp/capacity.json, compat.json and a few ternary checkpoints for the recount.

  python lab/probes/ternary_nanowsp/run.py [--quick]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine import ternary as tn  # noqa: E402
from engine import ternary_write as tw  # noqa: E402
from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig  # noqa: E402

OUT = ROOT / "lab" / "runs" / "ternary_nanowsp"
TASK = ROOT / "lab" / "data" / "wspx_recall.json"
WIDTHS = (64, 128)
SEEDS = (42, 43, 44)
LOADS = (8, 16, 32, 48, 64, 96, 128, 144)
SAVE_N = (16, 32, 64, 96)
ETA, LAM, LAYER, ROUNDS = 4.0, 1e-4, 1, 3


def cfg(d):
    return NanoWSPConfig(d_model=d, d_ff=2 * d, n_heads=2, d_head=d // 2)


def fresh(d, seed, path):
    m = NanoWSP(cfg(d), seed=seed)
    if path.exists():
        shutil.rmtree(path)
    m.save_checkpoint(path / "model.safetensors")
    return m


def main(quick=False):
    data = json.loads(TASK.read_text(encoding="utf-8"))
    train = [(it["prompt"], it["target"]) for it in data["train"]]
    hold = [(it["prompt"], it["target"]) for it in data["holdout"]]
    widths = WIDTHS[:1] if quick else WIDTHS
    seeds = SEEDS[:1] if quick else SEEDS
    rows = []
    work = OUT / "work"
    for d in widths:
        for seed in seeds:
            base_dir = work / f"d{d}_s{seed}_float"
            fresh(d, seed, base_dir)
            tern_dir = work / f"d{d}_s{seed}_tern"
            if tern_dir.exists():
                shutil.rmtree(tern_dir)
            man = tw.ternarize_checkpoint(base_dir / "model.safetensors", tern_dir, rule="absmean")
            for n in LOADS:
                pairs = train[:n]
                # A: float
                m = NanoWSP(cfg(d), seed=seed)
                ra = m.imprint_associative_batch(pairs, layer_idx=LAYER, eta=ETA, ridge_lambda=LAM)
                rows.append({"d": d, "seed": seed, "n": n, "arm": "float", "recall": m.evaluate_recall(pairs),
                             "holdout": m.evaluate_recall(hold), "cond": ra["condition_number"]})
                # B1: one projection; B3: three refinement rounds
                for arm, rounds in (("tern1", 0), ("tern3", ROUNDS)):
                    save = (seed == seeds[0] and n in SAVE_N and rounds == ROUNDS)
                    out = (work / f"d{d}_s{seed}_written_n{n}") if save else None
                    if out and out.exists():
                        shutil.rmtree(out)
                    rep = tw.imprint_ternary(tern_dir, pairs, layer=LAYER, eta=ETA, rounds=rounds, out=out, lam=LAM,
                                             holdout=hold)
                    rows.append({"d": d, "seed": seed, "n": n, "arm": arm, "recall": rep["recall"],
                                 "holdout": rep["holdout_accuracy"], "density": rep["density"],
                                 "key_residual": rep["key_residual"], "saved": str(out) if out else None})
                print(f"d{d} s{seed} n{n}: " + " ".join(f"{r['arm']} {r['recall']:.2f}" for r in rows[-3:]), flush=True)
            rows.append({"d": d, "seed": seed, "arm": "base_density", "density": {k: v["density"] for k, v in man["tensors"].items()}})
    summary = capacity(rows, widths, seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capacity.json").write_text(json.dumps({"rows": rows, "summary": summary, "loads": LOADS, "eta": ETA,
                                                   "lambda": LAM, "layer": LAYER, "rounds": ROUNDS}, indent=1), encoding="utf-8")
    comp = compat(train, hold, work)
    (OUT / "compat.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(json.dumps({k: v["pass"] for k, v in comp.items()}, indent=1))
    return 0


def capacity(rows, widths, seeds):
    out = {}
    for d in widths:
        out[str(d)] = {}
        for arm in ("float", "tern1", "tern3"):
            med = {}
            for n in LOADS:
                vals = [r["recall"] for r in rows if r.get("d") == d and r.get("n") == n and r["arm"] == arm]
                med[n] = float(np.median(vals)) if vals else None
            holds = [r["holdout"] for r in rows if r.get("d") == d and r["arm"] == arm and r.get("holdout") is not None]
            ncap = max([n for n in LOADS if med[n] is not None and med[n] >= 0.95], default=0)
            out[str(d)][arm] = {"median_recall": {str(k): v for k, v in med.items()}, "n_cap": ncap,
                                "holdout_max": max(holds) if holds else None,
                                "holdout_mean": float(np.mean(holds)) if holds else None}
        nf = out[str(d)]["float"]["n_cap"]
        out[str(d)]["ratio_tern1"] = out[str(d)]["tern1"]["n_cap"] / nf if nf else None
        out[str(d)]["ratio_tern3"] = out[str(d)]["tern3"]["n_cap"] / nf if nf else None
        dens = [r["density"] for r in rows if r.get("d") == d and r["arm"] == "tern3" and isinstance(r.get("density"), float)]
        out[str(d)]["written_density_median"] = float(np.median(dens)) if dens else None
    return out


def compat(train, hold, work):
    """Four cells on the d 64 ternary checkpoint written with n = 32, rounds = 3 (seed 42)."""
    d, seed, n = 64, 42, 32
    wdir = work / f"d{d}_s{seed}_written_n{n}"
    flo, tern, man = tw.load_ternary_dir(wdir)
    name = f"layers.{LAYER}.mlp_out.weight"
    pairs = train[:n]
    res = {}
    # 1. peek-style read
    W = flo[name].numpy()
    codes = tern[name + ".codes"].numpy()
    res["peek_read"] = {"pass": bool(W.shape == codes.shape), "density": float((codes != 0).mean()),
                        "shape": list(W.shape), "criterion": "runs and reports the ternary density"}
    # 2. project_out + re-ternarise
    rng = np.random.default_rng(7)
    u = rng.normal(size=W.shape[0]).astype(np.float32); u /= np.linalg.norm(u)
    e_before = float(np.linalg.norm(u @ W))
    Wp = W - np.outer(u, u @ W)
    c2, s2, _ = tw.solve_rows(Wp)
    e_after = float(np.linalg.norm(u @ tw.deq(c2, s2)))
    res["project_out_reternarize"] = {"pass": bool(e_after <= 0.10 * e_before), "energy_before": e_before,
                                      "energy_after": e_after, "ratio": e_after / e_before,
                                      "criterion": "energy along u after re-ternarisation <= 10 % of before"}
    # 3. float imprint of one new pair on top, then re-ternarise
    m = tw._model_from(flo)
    new_pair = train[n]                                                # the 33rd pair, not yet written
    m.imprint_associative_rule(new_pair[0], new_pair[1], layer_idx=LAYER, eta=ETA)
    c3, s3, _ = tw.solve_rows(m.weights[name].numpy())
    m.weights[name] = torch.from_numpy(tw.deq(c3, s3))
    new_ok = m.evaluate_recall([new_pair])
    old = m.evaluate_recall(pairs)
    res["float_imprint_then_reternarize"] = {"pass": bool(new_ok == 1.0 and old >= 0.90), "new_pair_recall": new_ok,
                                             "old_recall": old, "criterion": "new pair recalled and old recall >= 0.90"}
    # 4. compile-style small rank-1 delta, then re-ternarise
    rng = np.random.default_rng(8)
    uu = rng.normal(size=W.shape[0]); vv = rng.normal(size=W.shape[1])
    dW = np.outer(uu, vv).astype(np.float32)
    dW *= 0.02 * np.linalg.norm(W) / np.linalg.norm(dW)
    c4, s4, _ = tw.solve_rows(W + dW)
    alpha, noise, cos = tn.overlap(tw.deq(c4, s4) - W, dW)
    res["small_delta_reternarize"] = {"pass": bool(cos >= 0.5), "cos": cos, "alpha": alpha, "noise": noise,
                                      "rel_norm": 0.02, "criterion": "cos(deq_after - deq_before, dW) >= 0.5"}
    return res


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
