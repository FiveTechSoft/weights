"""Diagnostic (no training): teacher-forced argmax accuracy per answer slot, A (base) vs B (saved bf16 down_proj of layer 23).
Answer tokens are [' ', d1, d2, d3, (d4), '\\n']; slot k is the k-th answer token (so for 3-digit answers slot 4 is the newline, for 4-digit
answers slot 4 is the last digit and slot 5 the newline). A slot counts as right when the argmax token equals the target.
  train: 1200 sampled positions of the 4657 cached training positions (needs only the cache, final norm and head rows)
  hold : the 100 holdout problems, full model forward (QwenText) with A or B in layer 23
usage: diag_digit_accuracy.py MODEL_DIR B_RUN_DIR train|hold OUT.json [A|B]   (the optional arm splits hold into two slices: tool calls last ~2 min)
This diagnoses WHERE B differs from A; it is not a task score (the task score is exact match of free generation)."""
import sys, json, collections, torch
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from safetensors import safe_open
import safetensors.torch as st
model, run, mode, out = sys.argv[1:5]
ONLY = sys.argv[5] if len(sys.argv) > 5 else None
PFX = "model.language_model."
f = safe_open(model + "/model.safetensors", "pt")
W0 = f.get_tensor(PFX + "layers.23.mlp.down_proj.weight"); B = st.load_file(run + "/B_down23.safetensors")["layers.23.mlp.down_proj.weight"]
sl = f.get_slice(PFX + "embed_tokens.weight"); V = sl.get_shape()[0]
ARMS = [x for x in (("A", W0), ("B", B)) if ONLY in (None, x[0])]


def argmax_head(h, chunk=8192):
    best_v = torch.full((h.shape[0],), -1e30); best_i = torch.zeros(h.shape[0], dtype=torch.long)
    for s in range(0, V, chunk):
        E = sl[s:min(V, s + chunk)].to(torch.bfloat16)
        v, i = (h.to(torch.bfloat16) @ E.T).float().max(-1)
        m = v > best_v; best_v = torch.where(m, v, best_v); best_i = torch.where(m, i + s, best_i)
    return best_i


def summarize(ok, slot):
    return {k: [sum(o for o, s in zip(ok, slot) if s == k), sum(1 for s in slot if s == k)] for k in sorted(set(slot))}


if mode == "train":
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    cfg = AutoConfig.from_pretrained(model).text_config
    norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); norm.weight.data = f.get_tensor(PFX + "norm.weight").float(); norm = norm.float()
    ACT, RES, TGT = torch.load(run + "/cache.pt")
    NL = AutoTokenizer.from_pretrained(model)("\n")["input_ids"][-1]
    slot = []; k = 0
    for t in TGT.tolist():
        slot.append(k); k = 0 if t == NL else k + 1
    idx = torch.randperm(len(TGT), generator=torch.Generator().manual_seed(0))[:1200]
    r = {}
    for nm, W in ARMS:
        pred = argmax_head(norm(RES[idx].float() + ACT[idx].float() @ W.float().T))
        r[nm] = summarize((pred == TGT[idx]).tolist(), ["pos%d" % slot[i] for i in idx.tolist()])
else:
    from qwen_text import QwenText
    from data import make_sets
    q = QwenText(model); tok = q.tok; L23 = q.m.layers[23]
    import os
    lo, hi = [int(x) for x in os.environ.get("PART", "0:100").split(":")]
    items = make_sets("mul2x2")["holdout"][lo:hi]; r = {}   # PART=lo:hi slices the holdout (tool calls last ~2 min)
    for nm, W in ARMS:
        L23.mlp.down_proj.weight.data.copy_(W); ok = []; sl_ = []
        for i in range(0, len(items), 10):
            chunk = items[i:i + 10]
            enc = tok([it["prompt"] + " " + it["answer"] + "\n" for it in chunk], return_tensors="pt", padding=True)
            h = q.hidden(enc["input_ids"], enc["attention_mask"]); L = enc["input_ids"].shape[1]
            pad = (enc["attention_mask"] == 0).sum(1).tolist(); rows, tg, ss = [], [], []
            for r_, it in enumerate(chunk):
                p0 = pad[r_] + len(tok(it["prompt"])["input_ids"])
                for j, t in enumerate(range(p0, L)):
                    rows.append(h[r_, t - 1].float()); tg.append(enc["input_ids"][r_, t].item()); ss.append("pos%d" % j)
            pred = argmax_head(torch.stack(rows))          # one pass over the head rows per 10 problems
            ok += [p == t for p, t in zip(pred.tolist(), tg)]; sl_ += ss
        r[nm] = summarize(ok, sl_)
json.dump({"mode": mode, "arms": [a for a, _ in ARMS], "accuracy_by_slot": r}, open(out, "w"), indent=1)
for nm in r: print(mode, nm, {k: "%d/%d" % tuple(v) for k, v in r[nm].items()})
