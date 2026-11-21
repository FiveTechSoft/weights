"""Probe 3: matched pairs. B = X+S+Y+S (repeat), A = X'+S'+Y+S (no repeat). Same last 126 chars context."""
import json, random, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
train = harness.TEXT[:int(len(harness.TEXT) * 0.9)]
rng = random.Random(23)
seg = lambda n: (lambda a: train[a:a + n])(rng.randrange(len(train) - 400))
B, A = [], []
for _ in range(32):
    X, S, Y, X2, S2 = seg(70), seg(60), seg(66), seg(70), seg(60)
    B.append(X + S + Y + S); A.append(X2 + S2 + Y + S)
texts = B + A
json.dump({"texts": texts}, open(r"C:\weights\lab\runs\loopfix\editor\probe3_texts.json", "w"))
scopes = [f"transformer.h.{k}.attn.c_attn" for k in (3, 4, 5)] + [f"transformer.h.{k}.attn.c_proj" for k in (3, 4, 5)] + ["transformer.h.5.mlp.c_fc", "lm_head"]
harness.forward_probe(harness.BASE, texts, r"C:\weights\lab\runs\loopfix\editor\probe3.npz", scopes, "probe3 matched repeat vs no-repeat")
