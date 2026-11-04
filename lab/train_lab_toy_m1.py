"""LAB M-1: training re-aprende W1+W2 (2 capas) vs cirugia de solo-readout (closed-form sobre
features W0-frozen top-k por influencia). Protocolo 3 brazos + holdout exacta (universo finito de 256 bits).
Tarea: PARITY-8 (ilha exacta del dataset wspx). Paciente: MLP 8->32->1 tanh (+bias).
Reglas cirugia: solo (W0, ejemplos train); prohibido gradiente/W*/optimizer-state.
Salida: lab/runs/m1_results.json + m1_trace.npz."""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

R = Path(r"C:\weights")
OUT = R / "lab" / "runs"
torch.manual_seed(2718)
rng = np.random.default_rng(42)
D_IN, D_HID, N_SAMPLES = 8, 32, 256

# universo exacto: las 256 combinaciones del dataset wspx PARITY8 (postivos/holdout particion del dataset)
ds = torch.tensor(np.load(R / "lab" / "data" / ".." / "data" / "wspx_parity8.json" and 0 or 0), dtype=torch.float32) if False else None
import sys
sys.path.insert(0, str(R / "lab"))
from wspx_datasets import parity8  # noqa: E402
ds = parity8()
X = torch.tensor([[float(b) for b in r["prompt"].split()[1].replace("=", "")] for r in ds["universe"]])
Y = torch.tensor([[float(r["target"])] for r in ds["universe"]])
tr_mask = torch.zeros(N_SAMPLES, dtype=torch.bool)
for i, r in enumerate(ds["universe"]):
    tr_mask[i] = any(x["prompt"] == r["prompt"] for x in ds["train"])
print(f"universe {N_SAMPLES} | train {int(tr_mask.sum())} | holdout {int((~tr_mask).sum())}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11)
        self.W1 = nn.Parameter(torch.randn(D_IN, D_HID) * 0.7)
        self.b1 = nn.Parameter(torch.zeros(D_HID))
        self.W2 = nn.Parameter(torch.randn(D_HID, 1) * 0.7)

def run_train(n_steps=4000, lr=0.02):
    net = Net()
    opt = torch.optim.Adam([net.W1, net.b1, net.W2], lr=lr)
    snaps = []
    for t in range(n_steps):
        h = torch.tanh(X[tr_mask] @ net.W1 + net.b1)
        z = h @ net.W2
        loss = nn.functional.binary_cross_entropy_with_logits(z, Y[tr_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if t % (n_steps // 8) == 0 or t == n_steps - 1:
            snaps.append({"t": t + 1, "loss": float(loss.detach()),
                          "W1": net.W1.detach().numpy().copy(),
                          "W2": net.W2.detach().numpy().copy()})
    return net, snaps

net, snaps = run_train()
def acc(net, mask):
    with torch.no_grad():
        h = torch.tanh(X @ net.W1 + net.b1)
        pred = ((h @ net.W2) > 0).float()
    return float((pred[mask, 0] == Y[mask, 0]).float().mean())

acc_train_tr = acc(net, tr_mask)
acc_hold_post = acc(net, ~tr_mask)
print(f"TRAINING: acc train-parity {acc_train_tr:.3f}, holdout-parity (unseen): {acc_hold_post:.3f}")

# cirugia: readout-only closed-form sobre features W0-frozen (top-k por influencia = varianza de activacion * cov y)
net0 = Net()
with torch.no_grad():
    H0 = torch.tanh(X @ net0.W1 + net0.b1)          # features del W0 (paciente sin entrenar)
var = H0.var(dim=0).numpy()
yv = Y[:, 0].numpy()
Hn = H0.numpy()
cov = np.array([np.corrcoef(Hn[:, j], yv)[0, 1] ** 2 for j in range(D_HID)])
influence = torch.tensor(var * cov)
k = 16
k_idx = torch.topk(influence, k).indices

Phi_all = H0.numpy()[:, k_idx].astype(np.float64)
lam = 0.5
yy = Y[:, 0].numpy().astype(np.float64)
tr_np = tr_mask.numpy()
XtTx = Phi_all[tr_np].T @ Phi_all[tr_np]
w_surg = np.linalg.solve(XtTx + lam * np.eye(k), Phi_all[tr_np].T @ yy[tr_np])
def acc_surg():
    with torch.no_grad():
        h = torch.tanh(X @ net0.W1 + net0.b1)[:, k_idx]
        pred = (h @ torch.tensor(w_surg, dtype=torch.float32)).reshape(-1) > 0
    return float((pred.float() == Y[:, 0]).float().mean())
acc_hold_surg = acc_surg()

# placebo: readout norm-matched,Features aleatorias (mismo ||w||, dir de W2-features random)
w_norm = float(np.linalg.norm(w_surg))
rngp = np.random.default_rng(9)
rnd_dir = rngp.normal(size=k)
rnd_dir /= np.linalg.norm(rnd_dir)
w_plac = torch.tensor(w_norm * rnd_dir, dtype=torch.float32)
def acc_plac():
    with torch.no_grad():
        h = torch.tanh(X @ net0.W1 + net0.b1)[:, k_idx]
        pred = (h @ w_plac).reshape(-1) > 0
    return float((pred.float() == Y[:, 0]).float().mean())
acc_hold_plac = acc_plac()

res = {"task": "parity8", "n_universe": N_SAMPLES,
       "acc_train_seen": acc_train_tr, "acc_holdout_training": acc_hold_post,
       "acc_holdout_surgery_topk16": acc_hold_surg,
       "acc_holdout_placebo": acc_hold_plac,
       "notes": "cirugia = ridge closed-form sobre top-16 features frozen de W0 (sin gradiente/W*/optimizer)"}
print(json.dumps(res, indent=2, ensure_ascii=False))
(OUT / "m1_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
np.savez_compressed(OUT / "m1_trace.npz",
                    snaps_W1=np.stack([s["W1"] for s in snaps]).astype(np.float32),
                    snaps_W2=np.stack([s["W2"] for s in snaps]).astype(np.float32),
                    snap_t=np.array([s["t"] for s in snaps]),
                    snap_loss=np.array([s["loss"] for s in snaps]),
                    k_idx=k_idx.numpy())
print("guardado m1_results.json + m1_trace.npz")
