import json, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
d = np.load("probe4.npz"); ids = d["ids"]; T = 256
M = json.load(open("probe4_texts.json"))["meta"]
U = W["transformer.wte.weight"] * W["transformer.ln_f.weight"]
pos = np.arange(210, 255)
r1 = [i for i, m in enumerate(M) if m["k"] == 1]; r5 = [i for i, m in enumerate(M) if m["k"] == 5]
mask = np.tril(np.ones((T, T), bool))
tot1 = tot5 = 0
for L in (3, 4, 5):
    Aa = d[f"transformer.h.{L}.attn.c_attn"].astype(np.float32)
    Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]; Wp = W[f"transformer.h.{L}.attn.c_proj.weight"]
    Y = d[f"transformer.h.{L}.attn.c_proj"].astype(np.float32)
    q = Aa @ Wa[:384].T; k = Aa @ Wa[384:768].T
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        s = np.einsum("btd,bsd->bts", q[:, pos][..., sl], k[..., sl]) / 8.0
        s = np.where(mask[pos][None], s, -1e9); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        far5 = p[r5][:, :, :166].sum(-1).mean()  # mass on earlier loop cycles (k=5: loop spans 31..255)
        lgh = (Y[:, pos][..., sl] @ Wp[:, sl].T) @ U.T
        corr = np.take_along_axis(lgh, ids[:, pos + 1, None], -1)[..., 0]
        oth = np.sort(lgh, -1)[..., -2:]  # crude
        mar = corr - lgh.mean(-1)
        print(f"L{L}H{h} far-mass k5 {far5:.3f} | DLA(correct-mean) k1 {mar[r1].mean():.3f} k5 {mar[r5].mean():.3f}  diff {mar[r5].mean()-mar[r1].mean():+.3f}")
