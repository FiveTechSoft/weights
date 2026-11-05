"""LAB M2-readout: readout closed-form sobre features de checkpoints tempranos (PARITY-8).

Pregunta aprobada por el propietario del repo (2026-11-04): tras 500 pasos de entrenamiento, ¿las features (W1)
ya son suficientes y solo falta ajustar la salida? Se distingue que parte del aprendizaje
sustituye un readout calculado sin gradiente y que parte no.

Protocolo: mismo paciente/particion hash/hiperparametros que M-1 y M2-delta (determinista).
El trace de M-1 (lab/runs/m1_trace.npz) no guarda b1, asi que se reejecuta el mismo
protocolo capturando W1+b1+W2 en t = 0, 250, 500, 1000, 2000, 4000 (17 s en CPU).

Brazos por checkpoint t (todo evaluado en train-seen Y en holdout):
  - trained@t: el modelo tal cual quedo en el paso t (readout entrenado por gradiente)
  - ridge@t (lam=0.5 fija, como M-1): readout ridge closed-form ajustado SOLO con train
    sobre las features congeladas del paso t
  - ridgeCV@t: igual pero lam elegida por 5-fold CV DENTRO de train (grid fijo);
    el holdout no participa en ninguna seleccion de hiperparametros
Notas: ridge@0 == cirugia readout sobre features random (tipo M-1, pero con las 32
features en vez de top-16). Regresion sobre +-1, umbral 0, sin bias de readout
(el paciente tampoco tiene b2). 5 seeds. Salida: lab/runs/m2readout_results.json
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
LAM_FIXED = 0.5
LAM_GRID = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]


class Net(nn.Module):
    def __init__(self, init_seed=11):
        super().__init__()
        torch.manual_seed(init_seed)
        self.W1 = nn.Parameter(torch.randn(D_IN, D_HID) * 0.7)
        self.b1 = nn.Parameter(torch.zeros(D_HID))
        self.W2 = nn.Parameter(torch.randn(D_HID, 1) * 0.7)


def forward(net, X):
    return torch.tanh(X @ net.W1 + net.b1) @ net.W2


def acc_z(z, Y, mask):
    pred = (z[:, 0] > 0).float()
    return float((pred[mask] == Y[mask, 0]).float().mean())


def ridge_w(Htr, ytr, lam):
    H = Htr.astype(np.float64)
    y = ytr.astype(np.float64)
    return np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ y)


def ridge_cv_lam(Htr, ytr, k=5):
    n = len(ytr)
    idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    best_lam, best_acc = None, -1.0
    for lam in LAM_GRID:
        accs = []
        for i in range(k):
            te = folds[i]
            trm = np.setdiff1d(idx, te)
            w = ridge_w(Htr[trm], ytr[trm], lam)
            pred = (Htr[te] @ w > 0).astype(np.float64) * 2 - 1
            accs.append(float((pred == ytr[te]).mean()))
        m = float(np.mean(accs))
        if m > best_acc:
            best_acc, best_lam = m, lam
    return best_lam


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

    y_pm = (Y[:, 0] * 2 - 1).numpy()  # +-1 para ridge
    out = {}
    for t in SNAP_T:
        st = snaps[t]
        with torch.no_grad():
            H = torch.tanh(X @ st["W1"] + st["b1"]).numpy()
        z_trained = H @ st["W2"].numpy()
        row = {"trained": {"seen": acc_z(torch.tensor(z_trained), Y, tr_mask),
                           "holdout": acc_z(torch.tensor(z_trained), Y, ~tr_mask)}}
        Htr, ytr = H[tr_mask.numpy()], y_pm[tr_mask.numpy()]
        for tag, lam in (("ridge_lam0.5", LAM_FIXED), ("ridge_cv", ridge_cv_lam(Htr, ytr))):
            w = ridge_w(Htr, ytr, lam)
            z = H @ w
            row[tag] = {"lam": lam,
                        "seen": acc_z(torch.tensor(z.reshape(-1, 1)), Y, tr_mask),
                        "holdout": acc_z(torch.tensor(z.reshape(-1, 1)), Y, ~tr_mask)}
        out[str(t)] = row
    return out


def main():
    ds = wspx.parity8()
    X = torch.tensor([[float(b) for b in r["prompt"].split()[1]] for r in ds["universe"]])
    Y = torch.tensor([[float(r["target"])] for r in ds["universe"]])
    tr_mask = torch.zeros(len(ds["universe"]), dtype=torch.bool)
    for i, r in enumerate(ds["universe"]):
        tr_mask[i] = any(x["prompt"] == r["prompt"] for x in ds["train"])
    print(f"universe {len(X)} | train {int(tr_mask.sum())} | holdout {int((~tr_mask).sum())}")

    per_seed = [run_seed(s, X, Y, tr_mask) for s in range(5)]

    summary = {"task": "parity8", "n_train": int(tr_mask.sum()),
               "n_holdout": int((~tr_mask).sum()), "seeds": 5,
               "checkpoints": SNAP_T,
               "protocol": ("ridge ajustado SOLO con train; lam fija 0.5 o 5-fold CV interna de train; "
                            "holdout no participa en seleccion de hiperparametros"),
               "per_checkpoint": {}}
    for t in SNAP_T:
        row = {}
        for tag in ("trained", "ridge_lam0.5", "ridge_cv"):
            hs = [r[str(t)][tag]["holdout"] for r in per_seed]
            ss = [r[str(t)][tag]["seen"] for r in per_seed]
            row[tag] = {"seen_mean": round(float(np.mean(ss)), 4),
                        "holdout_mean": round(float(np.mean(hs)), 4),
                        "holdout_std": round(float(np.std(hs)), 4)}
            if tag == "ridge_cv":
                row[tag]["lams_elegidas"] = [r[str(t)][tag]["lam"] for r in per_seed]
        summary["per_checkpoint"][str(t)] = row
    summary["notes"] = ("ridge@0 = readout sobre features random (tipo M-1 con 32 features). "
                        "Comparacion: ridge@t vs trained@t = que parte del readout es sustituible; "
                        "ridge@t vs trained@4000 = que parte del aprendizaje total es sustituible.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m2readout_results.json").write_text(
        json.dumps({"summary": summary, "per_seed": per_seed}, indent=2), encoding="utf-8")
    for t in SNAP_T:
        r = summary["per_checkpoint"][str(t)]
        print(f"t={t:5d} trained {r['trained']['holdout_mean']:.3f}±{r['trained']['holdout_std']:.3f} | "
              f"ridge0.5 {r['ridge_lam0.5']['holdout_mean']:.3f}±{r['ridge_lam0.5']['holdout_std']:.3f} | "
              f"ridgeCV {r['ridge_cv']['holdout_mean']:.3f}±{r['ridge_cv']['holdout_std']:.3f}")


if __name__ == "__main__":
    main()
