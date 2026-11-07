import json, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
d = np.load("probe1.npz"); M = json.load(open("probe1_texts.json"))["meta"]
ids = d["ids"]; B, T = ids.shape
U = W["transformer.wte.weight"] * W["transformer.ln_f.weight"]
P = 45; c1 = np.arange(127, 127 + P); c2 = c1 + P
mask = np.tril(np.ones((T, T), bool))
np.set_printoptions(precision=3, suppress=True)
gain = np.zeros((6, 6)); attprev = np.zeros((6, 6))
for L in range(6):
    A = d[f"transformer.h.{L}.attn.c_attn"][:8].astype(np.float32)
    Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]; Wp = W[f"transformer.h.{L}.attn.c_proj.weight"]
    Y = d[f"transformer.h.{L}.attn.c_proj"][:8].astype(np.float32)
    q = A @ Wa[:384].T; k = A @ Wa[384:768].T
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        lg = (Y[..., sl] @ Wp[:, sl].T) @ U.T; lg -= lg.mean(-1, keepdims=True)
        corr = np.take_along_axis(lg[:, :-1], ids[:8, 1:, None], -1)[..., 0]
        gain[L, h] = (corr[:, c2] - corr[:, c1]).mean()
        s = np.einsum("btd,bsd->bts", q[..., sl], k[..., sl]) / 8.0
        s = np.where(mask, s, -1e9); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        # attention mass from cycle-2 positions into previous cycle (positions t-P-2 .. t-P+2)
        m = 0
        for off in range(-2, 3):
            m += p[:, c2, c2 - P + 1 + off].mean()
        attprev[L, h] = m
print("DLA gain cycle2-cycle1\n", gain); print("attention to t-P+1+-2 from cycle 2\n", attprev)
