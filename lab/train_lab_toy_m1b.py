"""LAB M-1b: compose_circuit — cirugia EXACTA de PARITY-8 por construccion cerrada.
Arquitectura paciente = M-1 (MLP 8->32->1, tanh). Construccion:
  unidades 0..7 : u_m = tanh(k*(s - (m+0.5))) con k=10, s = suma de bits 0/1
  readout alterno: v_m = eps*(-1)^m  ->  z0 ~= eps si s impar, ~0 si s par
  unidad 8     : constante h=tanh(+4); v8 = -eps/2/|h|  -> z = z0 - eps/2 (+ s impar, - s par)
  unidades 9..31: muertas (w=0,b=0 -> h=tanh(0)=0, no aportan)
Verificacion: universo exacto wspx PARITY8 (256) + placebo conv norm-matched.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

R = Path(r"C:\weights")
sys.path.insert(0, str(R / "lab"))
from wspx_datasets import parity8  # noqa: E402

OUT = R / "lab" / "runs"
K = 10.0
EPS = 0.05
D_HID = 32

def build_surgery():
    W1 = np.zeros((8, D_HID), dtype=np.float64)
    b1 = np.zeros(D_HID, dtype=np.float64)
    W2 = np.zeros((D_HID, 1), dtype=np.float64)
    for m in range(8):
        W1[:, m] = K
        b1[m] = -K * (m + 0.5)
        W2[m, 0] = EPS * ((-1) ** m)
    # unidad constante: h=tanh(+4) ~ +0.9993; para levantar la corriente hacia -eps/2 (espejo del readout)
    W1[:, 8] = 0.0
    b1[8] = 4.0
    h_const = np.tanh(4.0)
    W2[8, 0] = -(EPS / 2) / abs(h_const)
    return W1, b1, W2, h_const

ds = parity8()
X = torch.tensor([[float(ch) for ch in r["prompt"].split()[1].replace("=", "")] for r in ds["universe"]])
Y = torch.tensor([float(r["target"]) for r in ds["universe"]])

W1, b1, W2, h_c = build_surgery()
with torch.no_grad():
    h = torch.tanh(X @ torch.tensor(W1, dtype=torch.float32) + torch.tensor(b1, dtype=torch.float32))
    assert float(h[:, 8].sub(h_c).abs().max()) < 1e-6
    assert float(h[:, 9:].abs().max()) == 0.0, "unidades muertas deben salir 0"
    z = (h @ torch.tensor(W2, dtype=torch.float32)).reshape(-1)
    pred = (z > 0).float()
    acc = float((pred == Y).float().mean())
    margin = float(z.abs().min())
    # si construction invierte el signo, se eligio directo (verificamos paridad impar>0)
    odd_pos = float((pred[Y == 1].mean()))
    print(f"acc universo exacto: {acc*100:.1f}% ({int(acc*256)}/256), min|z|={margin:.4f}, impar>0: {odd_pos}")

# placebo: mismo ||DeltaW|| total, direccion aleatoria determinista
W1n = np.linalg.norm(W1); b1n = np.linalg.norm(b1); W2n = np.linalg.norm(W2)
tot = (W1n ** 2 + b1n ** 2 + W2n ** 2) ** 0.5
rngp = np.random.default_rng(1337)
def rand_like(norm):
    w = rngp.normal(size=W1.shape); b = rngp.normal(size=b1.shape); v = rngp.normal(size=W2.shape)
    f = norm / np.sqrt((np.linalg.norm(w) ** 2 + np.linalg.norm(b) ** 2 + np.linalg.norm(v) ** 2))
    return torch.tensor(w * f, dtype=torch.float32), torch.tensor(b * f, dtype=torch.float32), torch.tensor(v * f, dtype=torch.float32)
pw1, pb1, pv2 = rand_like(tot)
with torch.no_grad():
    hp = torch.tanh(X @ pw1 + pb1)
    zp = (hp @ pv2).reshape(-1)
    acc_plac = float(((zp > 0).float() == Y).float().mean())
tr = json.loads((OUT / "m1_results.json").read_text(encoding="utf-8"))
res = {
    "task": "parity8-compose-circuit",
    "acc_universe_exact": acc, "min_margin": margin,
    "acc_placebo": acc_plac,
    "acc_training_from_m1": tr["acc_holdout_training"],
    "acc_surgery_m1_readout": tr["acc_holdout_surgery_topk16"],
    "construction": {"k": K, "eps": EPS, "hidden_used": 9, "hidden_total": D_HID},
}
print(json.dumps(res, indent=2))
(OUT / "m1b_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

# certificado wspx si viene: escribe answers json y corre certify
answers = {r["prompt"]: ("1" if float(p) > 0 else "0") for r, p in zip(ds["universe"], pred.tolist())}
outp = R / "lab" / "data" / "answers_parity8_compose.json"
outp.write_text(json.dumps(answers, indent=1), encoding="utf-8")
print("answers guardadas en", outp, "- correr: python lab/certify.py parity8 --answers", outp)
