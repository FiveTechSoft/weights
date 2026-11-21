"""Probe 5: several greedy attractor phrases seen in dev_eval examples, each filling the window at 16 phases."""
import json, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
phrases = ["That the strength of the state of the state,\n",
           "Which the state of the state of the world's son,\n",
           "I would thou wert to the truth of the world.\n\nGLOUCESTER:\n",
           "That the seas that we have seen to the seas,\nAnd the seas of the strength of the world,\n"]
texts, meta = [], []
for pi, ph in enumerate(phrases):
    rep = ph * (600 // len(ph) + 2)
    for r in range(16):
        off = (r * len(ph)) // 16
        texts.append(rep[off:off + 256]); meta.append({"phrase": pi, "off": off, "P": len(ph)})
json.dump({"texts": texts, "meta": meta, "phrases": phrases}, open(r"C:\weights\lab\runs\loopfix\editor\probe5_texts.json", "w"))
scopes = [f"transformer.h.{k}.attn.c_proj" for k in range(6)] + ["lm_head"]
harness.forward_probe(harness.BASE, texts, r"C:\weights\lab\runs\loopfix\editor\probe5.npz", scopes, "probe5 four attractor phrases, attn outputs")
