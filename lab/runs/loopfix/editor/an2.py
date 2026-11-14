import json, numpy as np
d = np.load("probe1.npz"); M = json.load(open("probe1_texts.json"))
texts, meta = M["texts"], M["meta"]
lg = d["logits"].astype(np.float32); ids = d["ids"]
lp = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
P = 45
for i in range(8):
    y = ids[i, 1:]
    lpc = np.take_along_axis(lp[i, :-1], y[:, None], -1)[:, 0]
    srt = np.sort(lg[i, :-1], -1); margin = srt[:, -1] - srt[:, -2]
    am = lg[i, :-1].argmax(-1) == y
    cyc = [(127 + c * P, 127 + (c + 1) * P) for c in range(3)]
    print(i, [f"{lpc[a:b].mean():.3f}/{am[a:b].mean():.2f}" for a, b in cyc if b <= 255])
# per-position detail for text 0, cycles 1 and 2
i = 0; y = ids[i, 1:]; srt = np.sort(lg[i, :-1], -1); margin = srt[:, -1] - srt[:, -2]
lpc = np.take_along_axis(lp[i, :-1], y[:, None], -1)[:, 0]
print(repr(texts[0][128:128 + 46]))
for j in range(P):
    t1 = 127 + j; t2 = 127 + P + j
    print(repr(texts[0][t1 + 1]), f"{lpc[t1]:.2f} {lpc[t2]:.2f}  m {margin[t1]:.2f} {margin[t2]:.2f}")
