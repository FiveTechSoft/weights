"""LAB M2-feat: que cambia en las features entre t=0 y t=250 (readout sustituible) + acuerdo
prediccion a prediccion entre readout ridge y readout entrenado (PARITY-8, 5 seeds).

Continuacion aprobada por el propietario del repo (2026-11-04) tras M2-readout. Dos partes:

A. ACUERDO CONDUCTUAL (caveat pendiente de M2-readout): igualdad de accuracy no es identidad
   de predicciones. En cada checkpoint t: % de acuerdo y kappa de Cohen entre las predicciones
   de trained@t y ridge@t (lam=0.5, ajustado SOLO con train), en train-seen y en holdout.

B. DIAGNOSTICO ESTRUCTURAL de features H@t en t = 0,250,...,4000 (todo en train salvo que diga):
   - cka_H0: CKA lineal entre H@t y H@0 (deriva de representaciones)
   - read_s_R2: R2 de ridge que predice s (suma de bits) desde H@t — ¿las features leen la
     cantidad que usa la solucion exacta M-1b (escalera sobre s)?
   - w1_cos_ones: |cos| medio de las filas de W1 con el vector todo-unos (direccion de s)
   - unit_s_R2_max: maximo R2 de preactivacion de cada unidad vs s (¿emergen unidades escalera?)
   - eff_rank: rango efectivo (participacion) del espectro de covarianza de H@t
   - ridge_seen/holdout: separabilidad lineal (repite M2-readout como referencia)
   - norms: ||W1||, ||b1|| (escala/saturacion tanh)

Sin seleccion de hiperparametros con holdout. Salida: lab/runs/m2feat_results.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "lab" / "runs"

spec = importlib.util.spec_from_file_location("wspx_datasets", REPO / "lab" / "wspx_datasets.py")
wspx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wspx)
wspx.R = REPO
wspx.DATA = REPO / "lab" / "data"

D_IN, D_HID = 8, 32
N_STEPS, LR = 4000, 0.02
SNAP_T = [0, 250, 500, 1000, 2000, 4000]
LAM = 0.5


class Net(nn.Module):
    def __init__(self, init_seed=11):
        super().__init__()
        torch.manual_seed(init_seed)
        self.W1 = nn.Parameter(torch.randn(D_IN, D_HID) * 0.7)
        self.b1 = nn.Parameter(torch.zeros(D_HID))
        self.W2 = nn.Parameter(torch.randn(D_HID, 1) * 0.7)


def forward(net, X):
    return torch.tanh(X @ net.W1 + net.b1) @ net.W2


def ridge_w(H, y, lam=LAM):
    H = H.astype(np.float64); y = y.astype(np.float64)
    return np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ y)


def cka(A, B):
    """CKA lineal entre dos matrices n x d (centrada)."""
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    hsic = float(((Ac.T @ Bc) ** 2).sum())
    na = float(((Ac.T @ Ac) ** 2).sum())
    nb = float(((Bc.T @ Bc) ** 2).sum())
    return hsic / (na * nb) ** 0.5 if na * nb else 0.0


def eff_rank(H):
    Hc = H - H.mean(0)
    sv = np.linalg.svd(Hc, compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def kappa(a, b):
    po = float((a == b).mean())
    pa = float((a == 1).mean()); pb = float((b == 1).mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def run_seed(seed_idx, X, Y, tr_mask):
    torch.manual_seed(2718 + seed_idx)
    net = Net(11 + seed_idx)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    snaps = {0: {k: v.detach().clone() for k, v in net.state_dict().items()}}
    for t in range(1, N_STEPS + 1):
        z = forward(net, X[tr_mask])
        loss = nn.functional.binary_cross_entropy_with_logits(z, Y[tr_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if t in SNAP_T:
            snaps[t] = {k: v.detach().clone() for k, v in net.state_dict().items()}

    trn = tr_mask.numpy()
    s_all = X.sum(dim=1).numpy()              # suma de bits (lo que lee la solucion exacta M-1b)
    y_pm = (Y[:, 0] * 2 - 1).numpy()
    ones = np.ones(D_IN); ones = ones / np.linalg.norm(ones)
    out = {}
    H0 = None
    for t in SNAP_T:
        st = snaps[t]
        with torch.no_grad():
            pre = (X @ st["W1"] + st["b1"]).numpy()
            H = np.tanh(pre)
        if H0 is None:
            H0 = H
        row = {}
        # A. acuerdo conductual
        z_trained = (H @ st["W2"].numpy()).reshape(-1)
        p_trained = (z_trained > 0).astype(int)
        w = ridge_w(H[trn], y_pm[trn])
        p_ridge = (H @ w > 0).astype(int)
        row["agreement"] = {
            "seen": float((p_trained[trn] == p_ridge[trn]).mean()),
            "holdout": float((p_trained[~trn] == p_ridge[~trn]).mean()),
            "kappa_seen": kappa(p_trained[trn], p_ridge[trn]),
            "kappa_holdout": kappa(p_trained[~trn], p_ridge[~trn]),
        }
        row["acc_trained_holdout"] = float((p_trained[~trn] == (Y[~tr_mask, 0].numpy() > 0)).mean())
        row["acc_ridge_holdout"] = float((p_ridge[~trn] == (Y[~tr_mask, 0].numpy() > 0)).mean())
        row["acc_ridge_seen"] = float((p_ridge[trn] == (y_pm[trn] > 0)).mean())
        # B. diagnostico estructural (features de train)
        Ht = H[trn]
        row["cka_H0"] = cka(Ht, H0[trn])
        w_s = ridge_w(Ht, s_all[trn])
        pred_s = Ht @ w_s
        ss = s_all[trn]
        row["read_s_R2"] = float(1 - ((pred_s - ss) ** 2).sum() / ((ss - ss.mean()) ** 2).sum())
        W1n = st["W1"].numpy()
        cos = np.abs(W1n.T @ ones) / (np.linalg.norm(W1n, axis=0) + 1e-9)  # columnas = unidades
        row["w1_cos_ones_mean"] = float(cos.mean())
        r2s = []
        for j in range(D_HID):
            u = pre[trn, j]
            A = np.stack([ss, np.ones_like(ss)], 1)
            coef, *_ = np.linalg.lstsq(A, u, rcond=None)
            r = A @ coef
            r2s.append(1 - ((u - r) ** 2).sum() / (((u - u.mean()) ** 2).sum() + 1e-9))
        row["unit_s_R2_max"] = float(max(r2s))
        row["eff_rank"] = eff_rank(Ht)
        row["W1_norm"] = float(np.linalg.norm(W1n))
        row["b1_norm"] = float(np.linalg.norm(st["b1"].numpy()))
        out[str(t)] = row
    return out


def main():
    ds = wspx.parity8()
    X = torch.tensor([[float(b) for b in r["prompt"].split()[1]] for r in ds["universe"]])
    Y = torch.tensor([[float(r["target"])] for r in ds["universe"]])
    tr_mask = torch.zeros(len(ds["universe"]), dtype=torch.bool)
    for i, r in enumerate(ds["universe"]):
        tr_mask[i] = any(x["prompt"] == r["prompt"] for x in ds["train"])

    per_seed = [run_seed(s, X, Y, tr_mask) for s in range(5)]
    keys = list(per_seed[0]["0"].keys())
    sub = [k for k in per_seed[0]["0"]["agreement"].keys()]
    summary = {"task": "parity8", "seeds": 5, "checkpoints": SNAP_T, "per_checkpoint": {}}
    for t in SNAP_T:
        row = {}
        for k in keys:
            if k == "agreement":
                row[k] = {s2: round(float(np.mean([r[str(t)][k][s2] for r in per_seed])), 4) for s2 in sub}
            else:
                row[k] = round(float(np.mean([r[str(t)][k] for r in per_seed])), 4)
        summary["per_checkpoint"][str(t)] = row
    summary["notes"] = ("A: acuerdo prediccion a prediccion trained vs ridge (caveat de M2-readout). "
                        "B: diagnostico sobre features de train; s = suma de bits (estructura de la "
                        "solucion exacta M-1b). Observacional, no intervencion causal.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m2feat_results.json").write_text(
        json.dumps({"summary": summary, "per_seed": per_seed}, indent=2), encoding="utf-8")
    print(f"{'t':>5} {'agree_ho':>8} {'kappa_ho':>8} {'ckaH0':>6} {'read_s_R2':>9} {'cosones':>7} {'unit_sR2':>8} {'effrank':>7} {'accT':>6} {'accR':>6}")
    for t in SNAP_T:
        r = summary["per_checkpoint"][str(t)]
        print(f"{t:>5} {r['agreement']['holdout']:>8.3f} {r['agreement']['kappa_holdout']:>8.3f} "
              f"{r['cka_H0']:>6.3f} {r['read_s_R2']:>9.3f} {r['w1_cos_ones_mean']:>7.3f} "
              f"{r['unit_s_R2_max']:>8.3f} {r['eff_rank']:>7.2f} {r['acc_trained_holdout']:>6.3f} {r['acc_ridge_holdout']:>6.3f}")


if __name__ == "__main__":
    main()
