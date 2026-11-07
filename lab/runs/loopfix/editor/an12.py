"""Loop-benefit directions per head from probe5 (+probe4) loops; Fisher cost on natural text (probe3, probe4)."""
import json, sys, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
U = W["transformer.wte.weight"] * W["transformer.ln_f.weight"]
def load(npz, L, sl, rows, pos):
    d = np.load(npz); Y = d[f"transformer.h.{L}.attn.c_proj"][rows][:, pos][..., sl].astype(np.float32).reshape(-1, 64)
    lg = d["logits"][rows][:, pos].astype(np.float32).reshape(-1, 65)
    y = d["ids"][rows][:, pos + 1].reshape(-1)
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return Y, p, y
M4 = json.load(open("probe4_texts.json"))["meta"]; M5 = json.load(open("probe5_texts.json"))["meta"]
d5 = np.load("probe5.npz"); lg5 = d5["logits"].astype(np.float32); ids5 = d5["ids"]
for ph in range(4):
    rows = [i for i, m in enumerate(M5) if m["phrase"] == ph]
    print("phrase", ph, "self-consistency acc (pos 128-254)", (lg5[rows][:, 128:255].argmax(-1) == ids5[rows][:, 129:256]).mean().round(3))
out = {}
for L, h in [(5, h) for h in range(6)] + [(4, h) for h in range(6)] + [(3, h) for h in range(6)]:
    sl = slice(64 * h, 64 * h + 64)
    G = U @ W[f"transformer.h.{L}.attn.c_proj.weight"][:, sl]
    A = load("probe5.npz", L, sl, np.arange(64), np.arange(128, 255))
    B = load("probe4.npz", L, sl, [i for i, m in enumerate(M4) if m["k"] >= 3], np.arange(150, 255))
    Yl, pl, yl = [np.concatenate(z) for z in zip(A, B)]
    N1 = load("probe3.npz", L, sl, np.arange(32, 64), np.arange(20, 255))
    N2 = load("probe4.npz", L, sl, [i for i, m in enumerate(M4) if m["k"] == 1], np.arange(20, 200))
    Yn, pn, yn = [np.concatenate(z) for z in zip(N1, N2)]
    gl = (np.eye(65)[yl] - pl) @ G
    Cl = (Yl.T @ gl) / len(Yl); Cl = (Cl + Cl.T) / 2
    ev, E = np.linalg.eigh(Cl); E = E[:, ::-1]; ev = ev[::-1]
    def fisher(e):
        a = Yn @ e; dl = np.outer(a, G @ e)
        return 0.5 * ((dl ** 2 * pn).sum(-1) - (dl * pn).sum(-1) ** 2).mean()
    fs = np.array([fisher(E[:, i]) for i in range(64)])
    np.savez(f"dirs5_L{L}H{h}.npz", E=E, ev=ev, fisher=fs)
    print(f"L{L}H{h}: top loop benefits {np.round(ev[:4], 4)} fisher {np.round(fs[:4], 5)} ratio {np.round(ev[:4] / fs[:4], 1)}")
