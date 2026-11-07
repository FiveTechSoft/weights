import numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
d = np.load("probe3.npz"); ids = d["ids"]; T = ids.shape[1]
U = W["transformer.wte.weight"] * W["transformer.ln_f.weight"]
lg = d["logits"].astype(np.float32)
lp = lg - lg.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
lpc = np.take_along_axis(lp[:, :-1], ids[:, 1:, None], -1)[..., 0]
j = np.arange(0, 59); pos = 196 + j
Bi, Ai = np.arange(32), np.arange(32, 64)
print("logp B(repeat) %.3f  A(no repeat) %.3f" % (lpc[Bi][:, pos].mean(), lpc[Ai][:, pos].mean()))
for a, b in [(0, 5), (5, 15), (15, 30), (30, 59)]:
    print(f"  j {a}-{b}: B {lpc[Bi][:, pos[a:b]].mean():.3f} A {lpc[Ai][:, pos[a:b]].mean():.3f}")
mask = np.tril(np.ones((T, T), bool))
np.set_printoptions(precision=3, suppress=True)
for L in (3, 4, 5):
    Aa = d[f"transformer.h.{L}.attn.c_attn"].astype(np.float32)
    Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]; Wp = W[f"transformer.h.{L}.attn.c_proj.weight"]
    Y = d[f"transformer.h.{L}.attn.c_proj"].astype(np.float32)
    q = Aa @ Wa[:384].T; k = Aa @ Wa[384:768].T
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        s = np.einsum("btd,bsd->bts", q[:, pos][..., sl], k[..., sl]) / 8.0
        s = np.where(mask[pos][None], s, -1e9); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        # attention to the first S copy at matched phase (t-126) and next (t-125)
        idx = np.arange(len(pos))
        att_same = p[Bi][:, idx, pos - 126].mean(); att_next = p[Bi][:, idx, pos - 125].mean()
        att_S1 = p[Bi][:, :, 70:130].sum(-1).mean(); att_S1A = p[Ai][:, :, 70:130].sum(-1).mean()
        lgh = (Y[:, pos][..., sl] @ Wp[:, sl].T) @ U.T; lgh -= lgh.mean(-1, keepdims=True)
        corr = np.take_along_axis(lgh, ids[:, pos + 1, None], -1)[..., 0]
        print(f"L{L}H{h} att t-126 {att_same:.3f} t-125 {att_next:.3f} | mass on S1 B {att_S1:.3f} A {att_S1A:.3f} | DLA corr B {corr[Bi].mean():.3f} A {corr[Ai].mean():.3f}")
