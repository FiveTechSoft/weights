import json, numpy as np, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
chars = sorted(set(open(r"C:\weights\lab\data\shakespeare_char\input.txt", encoding="utf-8").read()))
d = np.load("probe1.npz"); ids = d["ids"]; lg = d["logits"].astype(np.float32)
P = 45
for i in [0, 3, 5]:
    out = []
    for jj in range(P):
        t = 127 + P + jj
        o = np.argsort(-lg[i, t]); m = lg[i, t, o[0]] - lg[i, t, o[1]]
        if m < 0.5: out.append(f"{chars[ids[i,t]]!r}->{chars[o[0]]!r}|{chars[o[1]]!r} {m:.2f}")
    print(i, out)
