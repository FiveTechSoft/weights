"""Per-direction loop benefit vs natural Fisher cost for a head's output (64-d), direct path only."""
import json, sys, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
U = W["transformer.wte.weight"] * W["transformer.ln_f.weight"]
L, h = int(sys.argv[1]), int(sys.argv[2]); sl = slice(64 * h, 64 * h + 64)
G = U @ W[f"transformer.h.{L}.attn.c_proj.weight"][:, sl]  # 65x64
def load(npz, rows, pos):
    d = np.load(npz); Y = d[f"transformer.h.{L}.attn.c_proj"][rows][:, pos][..., sl].astype(np.float32).reshape(-1, 64)
    lg = d["logits"][rows][:, pos].astype(np.float32).reshape(-1, 65)
    y = d["ids"][rows][:, pos + 1].reshape(-1)
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return Y, p, y
M4 = json.load(open("probe4_texts.json"))["meta"]
loopr = [i for i, m in enumerate(M4) if m["k"] >= 3]
Yl, pl, yl = load("probe4.npz", loopr, np.arange(120, 255))
# natural: probe3 A rows all positions >= 20, probe4 k=1 prefixes positions 20..200
Y3, p3, y3 = load("probe3.npz", np.arange(32, 64), np.arange(20, 255))
nat1 = [i for i, m in enumerate(M4) if m["k"] == 1]
Y4, p4, y4 = load("probe4.npz", nat1, np.arange(20, 200))
Yn, pn, yn = np.concatenate([Y3, Y4]), np.concatenate([p3, p4]), np.concatenate([y3, y4])
# first-order loop: d logp / d y = G^T (onehot - p)
gl = (np.eye(65)[yl] - pl) @ G
Cl = (Yl[:, :, None] * gl[:, None, :]).mean(0); Cl = (Cl + Cl.T) / 2
ev, E = np.linalg.eigh(Cl); E = E[:, ::-1]; ev = ev[::-1]
def fisher(e):
    a = Yn @ e; dl = np.outer(a, G @ e)  # N x 65 logit change per unit
    return 0.5 * ((dl ** 2 * pn).sum(-1) - (dl * pn).sum(-1) ** 2).mean()
# full head fisher
dl = Yn @ G.T; Kfull = 0.5 * ((dl ** 2 * pn).sum(-1) - (dl * pn).sum(-1) ** 2).mean()
gn = (np.eye(65)[yn] - pn) @ G; Cn = (Yn[:, :, None] * gn[:, None, :]).mean(0); Cn = (Cn + Cn.T) / 2
print(f"L{L}H{h} loop first-order total {np.trace(Cl):.4f}  natural first-order total {np.trace(Cn):.4f}  Fisher full {Kfull:.4f}")
rows = []
for i in range(64):
    e = E[:, i]; f = fisher(e); rows.append((ev[i], f, e @ Cn @ e))
for i in range(12):
    print(f" dir {i}: loop benefit {rows[i][0]:+.4f}  nat first-order {rows[i][2]:+.4f}  fisher {rows[i][1]:.5f}  ratio {rows[i][0]/max(rows[i][1],1e-9):.1f}")
print(" sum fisher top8", sum(r[1] for r in rows[:8]), "sum loop top8", sum(r[0] for r in rows[:8]))
np.savez(f"dirs_L{L}H{h}.npz", E=E, ev=ev, fisher=np.array([r[1] for r in rows]), natfo=np.array([r[2] for r in rows]))
