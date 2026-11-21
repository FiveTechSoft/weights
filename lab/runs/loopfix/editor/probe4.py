"""Probe 4: prefix + k cycles of the dominant greedy loop phrase (k=1..5); does lock-in grow with k?"""
import json, random, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
train = harness.TEXT[:int(len(harness.TEXT) * 0.9)]
rng = random.Random(31)
L = "That the strength of the state of the state,\n"
texts, meta = [], []
for k in range(1, 6):
    for r in range(12):
        n = 256 - 45 * k
        a = rng.randrange(len(train) - 400)
        pre = train[a:a + n]
        pre = pre[:-1] + "\n"
        texts.append(pre + L * k); meta.append({"k": k, "start": n})
for r in range(4):
    t = (L * 6)[r * 7: r * 7 + 256]; texts.append(t); meta.append({"k": 6, "start": 0, "offset": r * 7})
json.dump({"texts": texts, "meta": meta}, open(r"C:\weights\lab\runs\loopfix\editor\probe4_texts.json", "w"))
scopes = [f"transformer.h.{k}.attn.c_attn" for k in (3, 4, 5)] + [f"transformer.h.{k}.attn.c_proj" for k in (3, 4, 5)] + ["transformer.h.5.mlp.c_fc", "lm_head"]
harness.forward_probe(harness.BASE, texts, r"C:\weights\lab\runs\loopfix\editor\probe4.npz", scopes, "probe4 loop lock-in vs number of cycles")
