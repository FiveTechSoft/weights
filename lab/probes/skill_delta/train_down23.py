"""Teacher B: train ONLY layers.23.mlp.down_proj.weight (1024 x 3584) of Qwen3.5-0.8B on 2-digit multiplication, on this CPU.
Exactness of the cache: layer 23 is the last decoder layer. Its MLP output at position t depends only on position t (the MLP is
position-wise), and everything after it (final RMSNorm, tied head) is position-wise too. So caching, for each position that predicts an
answer token, (a) the residual stream entering the MLP block and (b) the MLP activation act = silu(gate) * up, loses nothing about the
trained path: logits_t = head(norm(resid_t + act_t @ W^T)). Gradients flow through the final norm and the full 248k-way head (exact CE).
Only down_proj is trained, nothing else; other tensors are untouched.
usage: train_down23.py MODEL_DIR OUT_DIR --n 800 --steps 300 --pos-bs 128 --lr 1e-4 [--max-seconds 120]
Writes OUT_DIR/B_down23.safetensors (bf16, the tensor B = round-to-bf16 of the fp32 master) and a log; weights are NOT for the repo."""
import argparse, hashlib, json, os, sys, time
import torch
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=800); ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--pos-bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-seconds", type=float, default=120)
a = ap.parse_args(); torch.manual_seed(0); os.makedirs(a.out, exist_ok=True)
NAME = "layers.23.mlp.down_proj.weight"
q = QwenText(a.model); tok = q.tok; L23 = q.m.layers[23]
cap = {}
L23.mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("act", i[0]))
L23.post_attention_layernorm.register_forward_pre_hook(lambda m, i: cap.__setitem__("res", i[0]))
train = make_sets("mul2x2")["train"][:a.n]
t0 = time.time(); ACT, RES, TGT = [], [], []
for i in range(0, len(train), 5):
    items = train[i:i + 5]
    full = [it["prompt"] + " " + it["answer"] + "\n" for it in items]
    enc = tok(full, return_tensors="pt", padding=True)
    q.hidden(enc["input_ids"], enc["attention_mask"])
    L = enc["input_ids"].shape[1]; pad = (enc["attention_mask"] == 0).sum(1).tolist()
    for r, it in enumerate(items):
        for t in range(pad[r] + len(tok(it["prompt"])["input_ids"]), L):
            ACT.append(cap["act"][r, t - 1].clone()); RES.append(cap["res"][r, t - 1].clone()); TGT.append(enc["input_ids"][r, t].item())
ACT, RES, TGT = torch.stack(ACT), torch.stack(RES), torch.tensor(TGT)
print("cached positions", len(TGT), "in", round(time.time() - t0, 1), "s; cache MB", round((ACT.numel() + RES.numel()) * 2 / 1e6), flush=True)
W0 = q.m.layers[23].mlp.down_proj.weight.data.clone()
norm = q.m.norm.float(); del cap, L23
q.m.layers = torch.nn.ModuleList([])   # the trained path needs no decoder layer: free ~1 GB
import gc; gc.collect()
def anon(): return [l.split()[1] for l in open("/proc/self/status") if l.startswith("RssAnon")][0]
print("anon kB after freeing layers", anon(), flush=True)
W = torch.nn.Parameter(W0.float().clone()); opt = torch.optim.Adam([W], lr=a.lr)
V = q.sl.get_shape()[0]


def ce(idx):
    h = norm(RES[idx].float() + ACT[idx].float() @ W.T); tgt = TGT[idx]; lse = []; tv = torch.zeros(len(idx))

    def chunk(h_, s_):
        E = q.sl[s_:min(V, s_ + 8192)].float(); lg = h_ @ E.T
        ok = (tgt >= s_) & (tgt < s_ + E.shape[0])
        return torch.logsumexp(lg, -1), lg.gather(1, (tgt - s_).clamp(0, E.shape[0] - 1)[:, None])[:, 0] * ok
    for s in range(0, V, 8192):
        ls, v = checkpoint(chunk, h, s, use_reentrant=False); lse.append(ls); tv = tv + v
    return (torch.logsumexp(torch.stack(lse), 0) - tv).mean()


g = torch.Generator().manual_seed(0); log = []; t1 = time.time()
for step in range(a.steps):
    if time.time() - t1 > a.max_seconds: print("time budget at step", step); break
    idx = torch.randint(0, len(TGT), (a.pos_bs,), generator=g)
    opt.zero_grad(); l = ce(idx); l.backward(); opt.step(); log.append(float(l.detach()))
    if step < 4: print("  step", step, "anon kB", anon(), flush=True)
    if step % 10 == 0: print("step", step, "loss", round(float(l), 4), "t", round(time.time() - t1, 1), flush=True)
B = W.data.to(torch.bfloat16)
save_file({NAME: B.contiguous()}, os.path.join(a.out, "B_down23.safetensors"))
d = (B.float() - W0.float())
rep = {"steps": len(log), "first5": log[:5], "last5": log[-5:], "seconds": round(time.time() - t1, 1), "cached_positions": len(TGT),
       "delta_fro": float(d.norm()), "delta_rel": float(d.norm() / W0.float().norm()), "changed_entries": int((B != W0).sum()),
       "B_sha256": hashlib.sha256(B.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
json.dump(rep, open(os.path.join(a.out, "train_report.json"), "w"), indent=1); print(rep)
