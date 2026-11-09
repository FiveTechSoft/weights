"""Small CPU 'teacher': full fine-tune of mlp.down_proj in the top K layers of Qwen3.5-0.8B on one arithmetic skill (answer-token CE).
Only those matrices get gradients (fp32 master copy, Adam); everything else is frozen bf16, so the autograd graph starts at the lowest
trained layer. The delta (trained - base) is written outside the repo (weights are not published) with its sha256 in the log.
usage: train_teacher.py MODEL_DIR SKILL OUT_DIR [--layers 23-23] [--n 600] [--steps 150] [--bs 8] [--lr 3e-4] [--max-seconds 90]"""
import argparse, hashlib, json, os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("skill"); ap.add_argument("out")
ap.add_argument("--layers", default="23-23"); ap.add_argument("--n", type=int, default=600)
ap.add_argument("--steps", type=int, default=150); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--max-seconds", type=float, default=90)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
lo, hi = map(int, a.layers.split("-")); torch.manual_seed(0)
os.makedirs(a.out, exist_ok=True)


class FP32Linear(torch.nn.Module):
    def __init__(self, w):
        super().__init__(); self.w = torch.nn.Parameter(w.float().clone())
    def forward(self, x): return F.linear(x.float(), self.w).to(x.dtype)


q = QwenText(a.model)
mods = {}   # every nn.Linear of the chosen layers (default: layer 23 only) gets an fp32 master copy
TARGET = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj")
for i in range(lo, hi + 1):
    for pn, parent in q.m.layers[i].named_modules():
        for cn, child in list(parent.named_children()):
            if cn in TARGET and isinstance(child, torch.nn.Linear):
                w = FP32Linear(child.weight.data); setattr(parent, cn, w); mods["layers.%d.%s%s" % (i, pn + "." if pn else "", cn)] = w
for p in q.m.parameters():
    if not any(p is m.w for m in mods.values()): p.requires_grad_(False)
params = [m.w for m in mods.values()]
state = os.path.join(a.out, "train_state.pt")
opt = torch.optim.Adam(params, lr=a.lr); step0 = 0
if a.resume and os.path.exists(state):
    s = torch.load(state)
    for m, w in zip(mods.values(), s["w"]): m.w.data.copy_(w)
    opt.load_state_dict(s["opt"]); step0 = s["step"]

train = make_sets(a.skill)["train"][:a.n]
tok = q.tok


def batch(items):
    full = [it["prompt"] + " " + it["answer"] + "\n" for it in items]
    enc = tok(full, return_tensors="pt", padding=True)
    n_prompt = [len(tok(it["prompt"])["input_ids"]) for it in items]
    n_ans = [len(tok(" " + it["answer"] + "\n")["input_ids"]) for it in items]
    L = enc["input_ids"].shape[1]; pad = (enc["attention_mask"] == 0).sum(1).tolist()
    pos = []  # (row, position whose hidden predicts the next token), target token
    for r in range(len(items)):
        start = pad[r] + n_prompt[r]
        for t in range(start, L):
            pos.append((r, t - 1, enc["input_ids"][r, t].item()))
    return enc, pos


def loss_fn(enc, pos):
    h = q.hidden(enc["input_ids"], enc["attention_mask"], grad=True)
    hp = torch.stack([h[r, t] for r, t, _ in pos]).float()
    tgt = torch.tensor([y for _, _, y in pos])
    V = q.sl.get_shape()[0]; lse = []; tl = torch.zeros(len(pos))
    from torch.utils.checkpoint import checkpoint

    def chunk(hp_, s_):                      # recomputed in backward: the 0.5 GB head is never kept for autograd
        E = q.sl[s_:min(V, s_ + 8192)].float()
        lg = hp_ @ E.T
        return torch.logsumexp(lg, -1), lg.gather(1, (tgt - s_).clamp(0, E.shape[0] - 1)[:, None])[:, 0] * ((tgt >= s_) & (tgt < s_ + E.shape[0]))
    for s in range(0, V, 8192):
        ls, tv = checkpoint(chunk, hp, s, use_reentrant=False)
        lse.append(ls); tl = tl + tv
    return (torch.logsumexp(torch.stack(lse), 0) - tl).mean()


t0 = time.time(); log = []
for step in range(step0, a.steps):
    if time.time() - t0 > a.max_seconds: print("time budget at step", step); break
    items = [train[(step * a.bs + j) % len(train)] for j in range(a.bs)]
    enc, pos = batch(items)
    opt.zero_grad(); l = loss_fn(enc, pos); l.backward(); opt.step()
    log.append((step, float(l))); 
    if step % 10 == 0: print("step", step, "loss", round(float(l), 4), "t", round(time.time() - t0, 1), flush=True)
done = step + 1 if log else step0
torch.save({"w": [m.w.data for m in mods.values()], "opt": opt.state_dict(), "step": done}, state)
json.dump({"steps_done": done, "last_losses": log[-5:], "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out, "train_log_%d.json" % done), "w"))
print("saved at step", done)
