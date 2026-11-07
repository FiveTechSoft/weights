"""Probe 2: X + S + Y + S. Same local context, second S has been seen before."""
import json, random, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
train = harness.TEXT[:int(len(harness.TEXT) * 0.9)]
rng = random.Random(11)
texts, meta = [], []
for _ in range(64):
    a = rng.randrange(len(train) - 400); X = train[a:a + 70]
    b = rng.randrange(len(train) - 400); S = train[b:b + 60]
    c = rng.randrange(len(train) - 400); Y = train[c:c + 66]
    texts.append(X + S + Y + S); meta.append({"s1": 70, "s2": 196, "L": 60})
json.dump({"texts": texts, "meta": meta}, open(r"C:\weights\lab\runs\loopfix\editor\probe2_texts.json", "w"))
scopes = [f"transformer.h.{k}.mlp.c_fc" for k in range(6)] + ["lm_head", "transformer.h.5.attn.c_attn"]
harness.forward_probe(harness.BASE, texts, r"C:\weights\lab\runs\loopfix\editor\probe2.npz", scopes, "probe2 XSYS repeat contrast")
