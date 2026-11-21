"""POST-DATA exploration (written after capacity.json was read; not part of the pre-registered readings of
lab/probes/ternary_nanowsp/PLAN.md). Questions raised by the data: (1) how much of the float write survives one
projection (cos of the realised change to the intended delta); (2) does a damped accumulator refinement converge
where the undamped one diverged; (3) does a larger eta help the one-shot write; (4) compatibility cell 3 at a load the
ternary model actually holds (n = 8). d_model 64, seed 42. Writes lab/runs/ternary_nanowsp/explore.json."""
import json
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
data = json.loads((ROOT / "lab/data/wspx_recall.json").read_text(encoding="utf-8"))
train = [(it["prompt"], it["target"]) for it in data["train"]]
hold = [(it["prompt"], it["target"]) for it in data["holdout"]]
d, seed, layer, lam = 64, 42, 1, 1e-4
name = f"layers.{layer}.mlp_out.weight"
tern_dir = OUT / "work" / f"d{d}_s{seed}_tern"
flo, tern, _ = tw.load_ternary_dir(tern_dir)


def write_and_project(n, eta, rounds, step):
    m = tw._model_from(flo)
    X, Y = tw.keys_values(m, train[:n], layer)
    M = X.shape[1]
    X_pinv = torch.linalg.solve(X.t() @ X + lam * torch.eye(M), X.t())
    W0 = m.weights[name].clone()
    delta = eta * (Y @ X_pinv)
    A = W0 + delta
    target_out = A @ X
    c, s, _ = tw.solve_rows(A.numpy()); D = torch.from_numpy(tw.deq(c, s))
    res = [float(torch.norm(target_out - D @ X) / torch.norm(target_out))]
    for _ in range(rounds):
        A = A + step * ((target_out - D @ X) @ X_pinv)
        c, s, _ = tw.solve_rows(A.numpy()); D = torch.from_numpy(tw.deq(c, s))
        res.append(float(torch.norm(target_out - D @ X) / torch.norm(target_out)))
    alpha, noise, cos = tn.overlap((D - W0).numpy(), delta.numpy())
    m.weights[name] = D
    return {"n": n, "eta": eta, "rounds": rounds, "step": step, "recall": m.evaluate_recall(train[:n]),
            "holdout": m.evaluate_recall(hold), "key_residual": res, "cos_delta": cos, "alpha_delta": alpha,
            "noise_delta": noise, "delta_rel": float(torch.norm(delta) / torch.norm(W0)),
            "delta_entry_over_scale": float(delta.abs().mean() / np.median(s)), "density": float((c != 0).mean())}


out = {"note": "post-data exploration, d 64 seed 42; not pre-registered", "survival_oneshot": [], "damped": [],
       "eta_sweep": [], "compat3_n8": None}
for n in (8, 16, 32, 64):
    out["survival_oneshot"].append(write_and_project(n, 4.0, 0, 0.0))
for n in (16, 32, 64):
    for step in (0.5, 0.25):
        out["damped"].append(write_and_project(n, 4.0, 10, step))
for n in (16, 32, 64):
    for eta in (8.0, 16.0, 32.0):
        out["eta_sweep"].append(write_and_project(n, eta, 0, 0.0))
# compat cell 3 at n = 8: write 8 pairs (rounds 0), float-imprint the 9th, re-ternarise
m = tw._model_from(flo)
X, Y = tw.keys_values(m, train[:8], layer)
X_pinv = torch.linalg.solve(X.t() @ X + lam * torch.eye(8), X.t())
A = m.weights[name] + 4.0 * (Y @ X_pinv)
c, s, _ = tw.solve_rows(A.numpy()); m.weights[name] = torch.from_numpy(tw.deq(c, s))
old_before = m.evaluate_recall(train[:8])
m.imprint_associative_rule(train[8][0], train[8][1], layer_idx=layer, eta=4.0)
c, s, _ = tw.solve_rows(m.weights[name].numpy()); m.weights[name] = torch.from_numpy(tw.deq(c, s))
out["compat3_n8"] = {"old_recall_before": old_before, "new_pair_recall": m.evaluate_recall([train[8]]),
                     "old_recall_after": m.evaluate_recall(train[:8])}
(OUT / "explore.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k in ("survival_oneshot", "damped", "eta_sweep"):
    for r in out[k]:
        print(k, {kk: (round(v, 3) if isinstance(v, float) else ([round(x, 2) for x in v] if isinstance(v, list) else v)) for kk, v in r.items()})
print("compat3_n8", out["compat3_n8"])
