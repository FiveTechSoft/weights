"""Probe 1: repeated vs natural text, ln_1 inputs and attention outputs for all layers."""
import json, random, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
train = harness.TEXT[:int(len(harness.TEXT) * 0.9)]
rng = random.Random(7)
texts, meta = [], []
loopphrase = "That the strength of the state of the state,\n"
for p in harness.prompts("dev", 8, seed=101):
    t = p + (loopphrase * 6)
    texts.append(t[:256]); meta.append({"kind": "dev+loop", "period": len(loopphrase), "rep_start": 128})
for _ in range(12):
    P = rng.randint(25, 90); s = rng.randrange(len(train) - P)
    seg = train[s:s + P]
    texts.append((seg * 20)[:256]); meta.append({"kind": "rep", "period": P, "rep_start": 0})
for _ in range(12):
    s = rng.randrange(len(train) - 256)
    texts.append(train[s:s + 256]); meta.append({"kind": "nat", "period": 0, "rep_start": 0})
json.dump({"texts": texts, "meta": meta}, open("probe1_texts.json", "w"))
scopes = [f"transformer.h.{k}.attn.c_attn" for k in range(6)] + [f"transformer.h.{k}.attn.c_proj" for k in range(6)] + ["lm_head"]
harness.forward_probe(harness.BASE, texts, r"C:\weights\lab\runs\loopfix\editor\probe1.npz", scopes, "probe1 repeated vs natural, attn inputs/outputs")
