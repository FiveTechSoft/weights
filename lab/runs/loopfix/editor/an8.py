import json, numpy as np
chars = sorted(set(open(r"C:\weights\lab\data\shakespeare_char\input.txt", encoding="utf-8").read()))
d = np.load("probe4.npz"); ids = d["ids"]; lg = d["logits"].astype(np.float32)
M = json.load(open("probe4_texts.json"))["meta"]
lp = lg - lg.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
# consistency of the LAST cycle: positions 210..254 predict 211..255
pos = np.arange(210, 255)
for k in range(1, 7):
    rows = [i for i, m in enumerate(M) if m["k"] == k]
    am = (lg[rows][:, pos].argmax(-1) == ids[rows][:, pos + 1])
    lpc = np.take_along_axis(lp[rows][:, pos], ids[rows][:, pos + 1, None], -1)[..., 0]
    full = am.all(1).mean()
    print(f"k={k} acc {am.mean():.3f} logp {lpc.mean():.3f} fully-consistent cycles {full:.2f}  worst-pos {np.argsort(am.mean(0))[:6]}")
