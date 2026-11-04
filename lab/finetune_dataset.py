"""finetune_dataset: finetune Simulated trace sobre el paciente pre-entrenado por dataset exacto.
USO: python lab/finetune_dataset.py --dataset recall --steps 600 --out lab/runs/finetune_<ds>.npz
Salida: trace npz con formato curriculum (W0_flat, snapshots, snap_steps, per-layer delta norms, loss).
Soportado: recall (target 1-char). NoEcho/MOD13 quedan en el next del curriculum (multi-token/supresion)."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(R / "lab"))
from wspx_datasets import BUILDERS  # noqa: E402
import nano3lib as ng  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="recall", choices=["recall"])
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--snapshot-every", type=int, default=150)
ap.add_argument("--out", default=None)
a = ap.parse_args()

ds = BUILDERS[a.dataset]()
tr = ng.load_trace()
W0p = ng.build_from_trace(tr)
TRAINABLE = ["out.weight", "mout.5.weight", "mn.5.weight", "fn.weight"]
P = {k: v.clone() for k, v in W0p.items()}
for k in TRAINABLE:
    P[k].requires_grad_(True)
opt = torch.optim.Adam([P[k] for k in TRAINABLE], lr=a.lr)

rows = ds["train"]
X, y = ng.prep(rows)
W0f = np.concatenate([v.detach().flatten().numpy() for v in W0p.values()])
snaps, snap_steps, losses = [], [], []
def flat_snap():
    return np.concatenate([P[k].detach().flatten().numpy() for k in
                           ["embed.weight", "out.weight"] + [f"{b}.{l}.weight" for b in ("q","k","v","o","min_","mout","an","mn") for l in range(6)] + ["fn.weight"]])
t0 = time.time()
for t in range(a.steps):
    lg, _ = ng.fwd(X, P)
    loss = F.cross_entropy(lg[:, -1], y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss.detach()))
    if (t + 1) % a.snapshot_every == 0 or t == a.steps - 1:
        snaps.append(flat_snap())
        snap_steps.append(t + 1)

Wf = flat_snap()
delta = (Wf - W0f)
absd = np.abs(delta)
top64 = np.sort(absd)[::-1][:64]
concentration_top64 = float((top64 ** 2).sum() / (absd ** 2).sum() + 1e-30)
readout_share = None
names = list(P.keys())
outb = np.prod(P["out.weight"].shape)
last_linear_share = float(np.linalg.norm(delta[len(W0f) - outb - 131072:]) / (np.linalg.norm(delta) + 1e-30))
res = {"dataset": a.dataset, "steps": a.steps, "lr": a.lr,
       "acc_train_final": float((((ng.fwd(X, P)[0])[:, -1].argmax(-1) == y).float().mean())),
       "ppl_start": round(float(np.exp(losses[0])), 2), "ppl_final": round(float(np.exp(losses[-1])), 2),
       "seconds": round(time.time() - t0, 1),
       "delta_l2": round(float(np.linalg.norm(delta)), 3),
       "last_linear_share": round(last_linear_share, 3),
       "concentration_top64": round(concentration_top64, 4),
       }
print(json.dumps(res, indent=2))
outp = Path(a.out or R / "lab" / "runs" / f"finetune_{a.dataset}.npz")
shapes_json = json.dumps(json.dumps({k: list(v.shape) for k, v in W0p.items()}))
np.savez_compressed(outp, W0_flat=W0f, snapshots_flat=np.stack(snaps), snap_steps=np.array(snap_steps),
                    loss=np.array(losses),
                    names=json.dumps(json.dumps(["embed.weight", "out.weight"] + [f"{b}.{l}" for b in ("q","k","v","o","min_","mout","an","mn") for l in range(6)] + ["fn.weight"])),
                    shapes=shapes_json[:0] or shapes_json, meta=json.dumps(res))
REMP = R / "lab" / "runs" / f"finetune_{a.dataset}.json"
REMP.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("guardado", outp)
