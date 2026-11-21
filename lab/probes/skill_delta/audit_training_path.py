"""Audit of the stage-1 training path (no training, one short forward pass). Run from the repo root:
  python lab/probes/skill_delta/audit_training_path.py MODEL_DIR B_RUN_DIR
Checks: (1) token targets = exactly the answer tokens + newline, (2) tokenisation of prompt is a prefix of the full text,
(3) left padding does not change hidden states at real positions (batch vs single), (4) the cached path norm(res + act @ W^T)
equals the real model forward for the same position, with W0 and with the saved bf16 B."""
import sys, torch, json
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets
from safetensors import safe_open
model, run = sys.argv[1], sys.argv[2]
q = QwenText(model); tok = q.tok
train = make_sets("mul2x2")["train"][:5]
full = [it["prompt"] + " " + it["answer"] + "\n" for it in train]
enc = tok(full, return_tensors="pt", padding=True)
pad = (enc["attention_mask"] == 0).sum(1).tolist()
print("--- (1)/(2) targets")
ok = True
for r, it in enumerate(train):
    ids = enc["input_ids"][r, pad[r]:].tolist(); p = tok(it["prompt"])["input_ids"]
    ok &= ids[:len(p)] == p
    print(repr(it["prompt"][-8:]), "targets:", [tok.decode([t]) for t in ids[len(p):]])
print("prompt tokens are a prefix of the full tokenisation:", ok)
print("--- (3) padding")
L23 = q.m.layers[23]; cap = {}
L23.mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("act", i[0]))
L23.post_attention_layernorm.register_forward_pre_hook(lambda m, i: cap.__setitem__("res", i[0]))
hb = q.hidden(enc["input_ids"], enc["attention_mask"]); actb, resb = cap["act"].clone(), cap["res"].clone()
for r in range(len(train)):
    ids = enc["input_ids"][r:r + 1, pad[r]:]
    hs = q.hidden(ids, torch.ones_like(ids))
    d = (hs[0] - hb[r, pad[r]:]).abs().max().item(); rel = d / hs.abs().max().item()
    print("example", r, "pad", pad[r], "max abs diff single vs padded batch", round(d, 4), "rel", round(rel, 4))
print("--- (4) cached path vs model forward (position of the last prompt token of example 0)")
f = safe_open(model + "/model.safetensors", "pt"); W0 = f.get_tensor("model.language_model.layers.23.mlp.down_proj.weight")
Bt = safe_open(run + "/B_down23.safetensors", "pt").get_tensor("layers.23.mlp.down_proj.weight") if False else None
import safetensors.torch as st
Bt = st.load_file(run + "/B_down23.safetensors")["layers.23.mlp.down_proj.weight"]
nrm = q.m.norm
t = pad[0] + len(tok(train[0]["prompt"])["input_ids"]) - 1
for nm, W in (("W0", W0), ("B", Bt)):
    pre = nrm(resb[0, t:t + 1] + (actb[0, t:t + 1] @ W.T))
    L23.mlp.down_proj.weight.data.copy_(W)
    h = q.hidden(enc["input_ids"][:1, :], enc["attention_mask"][:1, :])[0, t:t + 1] if False else None
    hfull = q.hidden(enc["input_ids"], enc["attention_mask"])[0, t:t + 1]
    print(nm, "rel diff cached-path vs real forward", round(((pre - hfull).norm() / hfull.norm()).item(), 5), "dtypes", pre.dtype, hfull.dtype)
    t0 = L23.mlp.down_proj.weight.data  # keep W for the real forward after the loop
