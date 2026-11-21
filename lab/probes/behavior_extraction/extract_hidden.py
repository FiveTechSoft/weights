"""Cache target states at answer positions for the behavior-extraction probe (DSpine-inspired).
For every position t that must predict an answer token, stores at position t-1: the outputs of the decoder
layers in --layers, the final (post-norm) hidden state, the masked mean of the token embeddings over the
prefix 0..t-1 (the no-hidden-states input for arm X), the previous token id and the target token id.
Same position convention as skill_delta/train_down23.py. Memory: 800 examples, ~5 answer tokens each:
~4000 positions x (4 layers + last + mean) x 1024 x bf16 ~= 50 MB.
usage: extract_hidden.py MODEL_DIR OUT.safetensors --layers 3,11,19,23 --n 800 --set train
Writes OUT.safetensors plus OUT.json (layers, set, n, count). Weights and caches are NOT for the repo.
"""
import argparse, json, os, sys, time
import torch
from safetensors.torch import save_file
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--layers", default="3,11,19,23"); ap.add_argument("--n", type=int, default=800)
ap.add_argument("--set", default="train", choices=["train", "selection"])
ap.add_argument("--teacher-labels", default=None,
                help="optional JSON {prompt: generated_answer}; replaces ground-truth answers by the target's "
                     "own generations (DSpine-faithful mimicry variant: the aux learns the target, errors included)")
a = ap.parse_args(); torch.manual_seed(0)
TEACH = json.load(open(a.teacher_labels)) if a.teacher_labels else None
LAYERS = [int(x) for x in a.layers.split(",")]
q = QwenText(a.model); tok = q.tok
cap = {}
for i in LAYERS:
    q.m.layers[i].register_forward_hook(
        lambda m, inp, o, i=i: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach()))
items = make_sets("mul2x2")[a.set][: a.n]
t0 = time.time()
H = {i: [] for i in LAYERS}; HLAST, EMEAN, PREV, TGT = [], [], [], []
PREF, POFF = [], [0]   # prefix token ids 0..t-1 per cached position (for the conventional GRU arm)
for s in range(0, len(items), 5):
    batch = items[s:s + 5]
    full = [it["prompt"] + " " + (TEACH[it["prompt"]] if TEACH else it["answer"]) + "\n" for it in batch]
    enc = tok(full, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    h_last = q.hidden(ids, mask)
    emb = q.embed(ids).float() * mask[..., None]          # (B, L, 1024), padded rows zeroed
    csum = emb.cumsum(1); ccnt = mask.cumsum(1).clamp(min=1).float()   # cumulative non-pad count per position
    L = ids.shape[1]; pad = (mask == 0).sum(1).tolist()
    for r, it in enumerate(batch):
        n_prompt = len(tok(it["prompt"])["input_ids"])
        for t in range(pad[r] + n_prompt, L):
            for i in LAYERS:
                H[i].append(cap[i][r, t - 1].clone())
            HLAST.append(h_last[r, t - 1].clone())
            EMEAN.append((csum[r, t - 1] / ccnt[r, t - 1]).to(torch.bfloat16))
            PREV.append(ids[r, t - 1].item())
            TGT.append(ids[r, t].item())
            PREF.extend(ids[r, pad[r]:t].tolist()); POFF.append(len(PREF))   # non-pad prefix only (GRU arm)
    if s % 100 == 0:
        print("cached", len(TGT), "positions,", round(time.time() - t0, 1), "s", flush=True)
tensors = {"hlast": torch.stack(HLAST).contiguous(), "emean": torch.stack(EMEAN).contiguous(),
           "prev": torch.tensor(PREV), "tgt": torch.tensor(TGT),
           "pref_ids": torch.tensor(PREF), "pref_off": torch.tensor(POFF)}
for i in LAYERS:
    tensors["h%d" % i] = torch.stack(H[i]).contiguous()
save_file(tensors, a.out)
json.dump({"layers": LAYERS, "set": a.set, "n_items": len(items), "positions": len(TGT),
           "seconds": round(time.time() - t0, 1)}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
print("cached positions", len(TGT), "in", round(time.time() - t0, 1), "s")
