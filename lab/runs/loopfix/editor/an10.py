import json, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
chars = sorted(set(open(r"C:\weights\lab\data\shakespeare_char\input.txt", encoding="utf-8").read()))
d = np.load("probe4.npz"); ids = d["ids"]; T = 256
M = json.load(open("probe4_texts.json"))["meta"]
r5 = [i for i, m in enumerate(M) if m["k"] == 5]
L, h = 5, 3
Aa = d[f"transformer.h.{L}.attn.c_attn"].astype(np.float32); Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]
sl = slice(64 * h, 64 * h + 64)
q = Aa @ Wa[:384][sl].T; k = Aa @ Wa[384:768][sl].T
s = np.einsum("btd,bsd->bts", q, k) / 8; s = np.where(np.tril(np.ones((T, T), bool)), s, -1e9)
p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
i = r5[0]; txt = "".join(chars[c] for c in ids[i])
for t in range(212, 256, 3):
    top = np.argsort(-p[i, t])[:5]
    print(repr(txt[t-6:t+1]), "->", [(t - s_, repr(txt[s_]), round(float(p[i, t, s_]), 2)) for s_ in top])
# mean attention by distance bucket, loop (k5 last cycle) vs prefix-natural region (k=1 rows, positions 100-200)
r1 = [i for i, m in enumerate(M) if m["k"] == 1]
def bucket(rows, ts):
    out = np.zeros(6); edges = [0, 1, 8, 32, 64, 128, 256]
    for t in ts:
        dist = t - np.arange(t + 1)
        for b in range(6):
            sel = (dist >= edges[b]) & (dist < edges[b + 1])
            out[b] += p[rows][:, t, :t + 1][:, sel].sum(-1).mean()
    return out / len(ts)
print("dist buckets [0,1,8,32,64,128,256): loop k5", bucket(r5, range(210, 255)).round(3))
print("natural prefix (k1 rows, t 150-200)", bucket(r1, range(150, 200)).round(3))
