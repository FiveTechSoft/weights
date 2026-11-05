"""LAB M2-delta: cuanta informacion del delta de entrenamiento hace falta para generalizar.

Continuacion directa de M-1 (commit a7a2e7c): en PARITY-8, training generaliza (83.6%
holdout) pero la cirugia ridge readout-only sobre features congeladas == placebo (50%).
M-1 probo una cirugia SIN el delta. M-2 prueba el eje complementario: ediciones que
conservan solo PARTE del delta, verificadas en el holdout (casos no usados para obtenerlas).

Brazos (todos evaluados en train-seen y holdout, con placebo norm-matched por brazo):
  A. replay del delta completo (oraculo; debe reproducir el holdout del training)
  B. delta parcial: snapshot tras fraccion f del entrenamiento (f = 1/8 .. 7/8)
  C. delta de rango bajo: SVD truncado de delta W1 (r = 1,2,3,4,6,8; W2 ya es rango 1)
  D. delta disperso: solo el top-k en magnitud de todos los pesos del delta (k/N en %)
  E. delta escalado: 0.5x y 2.0x (dosis)
  F. descomposicion: W1 entrenada + W2 ridge closed-form sobre train (sin gradiente),
     y W0 congelada + W2 entrenada (donde vive la generalizacion)
  G. placebo aleatorio con la misma norma L2 que el delta completo

Limite declarado: los brazos A-E siguen derivados de OBSERVAR el entrenamiento (oraculo).
No son una regla de cirugia aprendida; miden el minimo de informacion que una edicion
debe transportar. Multi-seed (5) para no fiar de una semilla.
Salida: lab/runs/m2delta_results.json
Errata 2026-11-04: la version inicial uso etiquetas 0/1 con umbral 0 en el brazo
ridge (decomp_trainedW1_ridgeW2), lo que infravaloraba ese brazo (62.2%). Con etiquetas +-1
el valor correcto es ~81.4% (ver M2-readout, lab/train_lab_toy_m2readout.py).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "lab" / "runs"

# importar el dataset wspx sin sus rutas Windows hardcodeadas
spec = importlib.util.spec_from_file_location("wspx_datasets", REPO / "lab" / "wspx_datasets.py")
wspx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wspx)
wspx.R = REPO
wspx.DATA = REPO / "lab" / "data"

D_IN, D_HID = 8, 32
N_STEPS, LR = 4000, 0.02  # identico a M-1


class Net(nn.Module):
    def __init__(self, init_seed=11):
        super().__init__()
        torch.manual_seed(init_seed)
        self.W1 = nn.Parameter(torch.randn(D_IN, D_HID) * 0.7)
        self.b1 = nn.Parameter(torch.zeros(D_HID))
        self.W2 = nn.Parameter(torch.randn(D_HID, 1) * 0.7)


def forward(net, X):
    return torch.tanh(X @ net.W1 + net.b1) @ net.W2


def acc_from_logits(z, Y, mask):
    pred = (z[:, 0] > 0).float()
    return float((pred[mask] == Y[mask, 0]).float().mean())


def run_seed(seed_idx, X, Y, tr_mask):
    torch.manual_seed(2718 + seed_idx)
    init_seed = 11 + seed_idx
    net0 = Net(init_seed)
    init = {k: v.detach().clone() for k, v in net0.state_dict().items()}

    # --- entrenamiento con snapshots (identico protocolo que M-1) ---
    net = Net(init_seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    snap_at = [N_STEPS * f // 8 for f in range(1, 8)]  # 1/8 .. 7/8
    snaps = {}
    for t in range(1, N_STEPS + 1):
        z = forward(net, X[tr_mask])
        loss = nn.functional.binary_cross_entropy_with_logits(z, Y[tr_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if t in snap_at:
            snaps[t] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    final = {k: v.detach().clone() for k, v in net.state_dict().items()}
    delta = {k: final[k] - init[k] for k in init}

    def eval_state(state):
        with torch.no_grad():
            n = Net(init_seed)
            n.load_state_dict(state)
            z = forward(n, X)
        return acc_from_logits(z, Y, tr_mask), acc_from_logits(z, Y, ~tr_mask)

    def apply(d):
        return {k: init[k] + d[k] for k in init}

    def l2(d):
        return float(torch.sqrt(sum((v ** 2).sum() for v in d.values())))

    def placebo_like(d, pseed):
        g = torch.Generator().manual_seed(pseed)
        noise = {k: torch.randn(v.shape, generator=g) for k, v in d.items()}
        nrm = l2(noise)
        tgt = l2(d)
        return {k: v * (tgt / nrm) for k, v in noise.items()} if nrm else d

    res = {}

    # training final
    res["training"] = {"seen": eval_state(final)[0], "holdout": eval_state(final)[1]}
    # A. delta completo
    res["replay_full"] = {"seen": eval_state(apply(delta))[0], "holdout": eval_state(apply(delta))[1]}
    # G. placebo del delta completo
    res["placebo_full"] = {"seen": eval_state(apply(placebo_like(delta, 1000 + seed_idx)))[0],
                           "holdout": eval_state(apply(placebo_like(delta, 2000 + seed_idx)))[1]}
    # B. deltas parciales
    res["partial"] = {}
    for t, st in snaps.items():
        d = {k: st[k] - init[k] for k in init}
        s, h = eval_state(apply(d))
        res["partial"][f"{t}/{N_STEPS}"] = {"seen": s, "holdout": h, "delta_l2": l2(d)}
    # C. rango bajo (SVD de delta W1)
    U, S, Vh = torch.linalg.svd(delta["W1"], full_matrices=False)
    res["lowrank"] = {}
    for r in [1, 2, 3, 4, 6, 8]:
        d = {k: v.clone() for k, v in delta.items()}
        d["W1"] = (U[:, :r] * S[:r]) @ Vh[:r, :]
        s, h = eval_state(apply(d))
        res["lowrank"][f"r{r}"] = {"seen": s, "holdout": h}
    # D. disperso top-k magnitud global
    flat = torch.cat([v.reshape(-1) for v in delta.values()])
    n = flat.numel()
    res["sparse"] = {}
    for pct in [1, 5, 10, 25, 50]:
        k = max(1, int(n * pct / 100))
        thresh = flat.abs().kthvalue(n - k + 1).values
        d = {kk: torch.where(v.abs() >= thresh, v, torch.zeros_like(v)) for kk, v in delta.items()}
        s, h = eval_state(apply(d))
        res["sparse"][f"{pct}pct"] = {"seen": s, "holdout": h}
    # E. escalado
    res["scaled"] = {}
    for f in [0.5, 2.0]:
        d = {k: v * f for k, v in delta.items()}
        s, h = eval_state(apply(d))
        res["scaled"][f"{f}x"] = {"seen": s, "holdout": h}
    # F. descomposicion features/readout
    with torch.no_grad():
        H_tr = torch.tanh(X[tr_mask] @ final["W1"] + final["b1"]).numpy().astype(np.float64)
        y_tr = (Y[tr_mask, 0].numpy() * 2 - 1).astype(np.float64)  # +-1: umbral 0 correcto
        lam = 0.5
        w_ridge = np.linalg.solve(H_tr.T @ H_tr + lam * np.eye(D_HID), H_tr.T @ y_tr)
    st_a = dict(final); st_a["W2"] = torch.tensor(w_ridge, dtype=torch.float32).reshape(D_HID, 1)
    s, h = eval_state(st_a)
    res["decomp_trainedW1_ridgeW2"] = {"seen": s, "holdout": h}
    st_b = dict(init); st_b["W2"] = final["W2"]
    s, h = eval_state(st_b)
    res["decomp_frozenW1_trainedW2"] = {"seen": s, "holdout": h}
    return res


def main():
    ds = wspx.parity8()
    X = torch.tensor([[float(b) for b in r["prompt"].split()[1]] for r in ds["universe"]])
    Y = torch.tensor([[float(r["target"])] for r in ds["universe"]])
    tr_mask = torch.zeros(len(ds["universe"]), dtype=torch.bool)
    for i, r in enumerate(ds["universe"]):
        tr_mask[i] = any(x["prompt"] == r["prompt"] for x in ds["train"])
    print(f"universe {len(X)} | train {int(tr_mask.sum())} | holdout {int((~tr_mask).sum())}")

    per_seed = [run_seed(s, X, Y, tr_mask) for s in range(5)]

    def mean_arm(getter):
        vals = [getter(r) for r in per_seed]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"task": "parity8", "n_universe": len(X),
               "n_train": int(tr_mask.sum()), "n_holdout": int((~tr_mask).sum()),
               "seeds": 5, "steps": N_STEPS, "lr": LR,
               "protocol": "misma particion hash wspx y protocolo que M-1; paths portables",
               "arms": {}}
    def arm(name, g):
        m, sd = mean_arm(g)
        summary["arms"][name] = {"holdout_mean": round(m, 4), "holdout_std": round(sd, 4)}
    arm("training", lambda r: r["training"]["holdout"])
    arm("replay_full_delta", lambda r: r["replay_full"]["holdout"])
    arm("placebo_norm_matched", lambda r: r["placebo_full"]["holdout"])
    for key in per_seed[0]["partial"]:
        arm(f"partial_{key}", lambda r, k=key: r["partial"][k]["holdout"])
    for key in per_seed[0]["lowrank"]:
        arm(f"lowrank_{key}", lambda r, k=key: r["lowrank"][k]["holdout"])
    for key in per_seed[0]["sparse"]:
        arm(f"sparse_{key}", lambda r, k=key: r["sparse"][k]["holdout"])
    for key in per_seed[0]["scaled"]:
        arm(f"scaled_{key}", lambda r, k=key: r["scaled"][k]["holdout"])
    arm("decomp_trainedW1_ridgeW2", lambda r: r["decomp_trainedW1_ridgeW2"]["holdout"])
    arm("decomp_frozenW1_trainedW2", lambda r: r["decomp_frozenW1_trainedW2"]["holdout"])
    summary["notes"] = ("A-E derivan de observar el entrenamiento (oraculo), no son cirugia "
                        "aprendida; miden el minimo de informacion de la edicion. "
                        "Holdout = casos no usados ni para entrenar ni para derivar la edicion.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m2delta_results.json").write_text(
        json.dumps({"summary": summary, "per_seed": per_seed}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
