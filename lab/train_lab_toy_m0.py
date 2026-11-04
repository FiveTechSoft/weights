"""LAB M-0: Training Trace -> Surgery Compiler sobre MLP-minimal.
Brazos: (1) entrenamiento Adam real, (2) cirugia ridge closed-form (usa SOLO W0+ejemplos,
prohibido gradiente/W*/optimizer-state), (3) placebo norm-matched. Eval en holdout sellada.

Salida: lab/runs/m0_trace.npz (snapshots cada 25 pasos, m/v, loss), lab/runs/m0_results.json.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

R = Path(r"C:\weights")
OUT = R / "lab" / "runs"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(2718)
rng = np.random.default_rng(42)
D_IN, D_HID, D_OUT, N_TRAIN, N_HOLD = 8, 16, 1, 400, 200

def synth(n, rng):
    x = rng.choice([-1.0, 1.0], size=(n, D_IN))
    score = 2.0 * x[:, 0] + 1.5 * x[:, 1] - 1.0 * x[:, 3] + 0.5 * x[:, 6] + 0.8 * x[:, 7]
    y = (score > 0).astype(np.float64)
    flip = rng.random(n) < 0.10          # ruido de etiqueta: veneno anti overfit trivial
    y[flip] = 1 - y[flip]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

Xtr, ytr = synth(N_TRAIN, rng)
Xho, yho = synth(N_HOLD, np.random.default_rng(1337))

# paciente: features frozen no-lineales + readout lineal entrenable
torch.manual_seed(11)
P = torch.randn(D_IN, D_HID, dtype=torch.float32) * (D_IN ** -0.5)
def feats(X):
    return torch.tanh(X @ P)                      # [n, D_HID]
Htr, Hho = feats(Xtr), feats(Xho)

w0 = torch.zeros(D_HID, 1)
model_w = w0.clone().requires_grad_(True)
opt = torch.optim.Adam([model_w], lr=0.05)
loss_tr = []
snapshots = []
for t in range(200):
    z = Htr @ model_w
    p = torch.sigmoid(z)
    loss = nn.functional.binary_cross_entropy(p, ytr)
    loss_tr.append(float(loss))
    opt.zero_grad(); loss.backward(); opt.step()
    if t % 25 == 0 or t == 199:
        g = model_w.grad
        st = opt.state.get(model_w, {})
        snapshots.append({
            "t": t + 1, "W": model_w.detach().clone().flatten().numpy(),
            "loss": float(loss), "lr": opt.param_groups[0]["lr"],
            "grad_norm": float(g.norm()) if g is not None else None,
            "exp_avg": st["exp_avg"].flatten().numpy() if "exp_avg" in st else None,
            "exp_avg_sq": st["exp_avg_sq"].flatten().numpy() if "exp_avg_sq" in st else None,
        })
Wstar = model_w.detach().clone()

# ---- brazo 2: SURGERY ridged closed-form (solo W0 + ejemplos; sin gradiente / W* / optimizer) ----
Phi = Htr.numpy().astype(np.float64)
yy = ytr.numpy()
lam = 1.0
w_surg = np.linalg.solve(Phi.T @ Phi + lam * np.eye(D_HID), Phi.T @ (yy.squeeze() - 0.5))
w_surg_t = torch.tensor(w_surg, dtype=torch.float32).reshape(D_HID, 1)

# ---- brazo 3: PLACEBO norm-matched (mismo ||Delta|| que la cirugia, direccion aleatoria) ----
rngp = np.random.default_rng(7)
delta_surg = w_surg_t - w0
rnd = torch.tensor(rngp.normal(size=(D_HID, 1)), dtype=torch.float32)
rnd = rnd / rnd.norm()
w_plac = w0 + rnd * delta_surg.norm()

def acc(w):
    return float((( (torch.sigmoid(Hho @ w) > 0.5).float() == yho).float().mean()))

r = {
    "acc_train_baseline_w0": float(((torch.sigmoid(Htr @ w0) > 0.5).float() == ytr).float().mean()),
    "acc_holdout_w0": acc(w0),
    "acc_holdout_training": acc(Wstar),
    "acc_holdout_surgery": acc(w_surg_t),
    "acc_holdout_placebo": acc(w_plac),
    "norms": {"Wstar-w0": float((Wstar - w0).norm()), "surg-w0": float((w_surg_t - w0).norm()),
              "plac-w0": float((w_plac - w0).norm())},
    "final_train_loss": loss_tr[-1],
}
print(json.dumps(r, indent=2))
np.savez_compressed(OUT / "m0_trace.npz",
                    W0=w0.flatten().numpy(), Wstar=Wstar.flatten().numpy(),
                    snapshots=np.array([s["W"] for s in snapshots]),
                    snap_steps=np.array([s["t"] for s in snapshots]),
                    P=P.numpy(), loss=np.array(loss_tr))
(R / "lab" / "runs" / "m0_results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
print("guardado lab/runs/m0_results.json + m0_trace.npz")
