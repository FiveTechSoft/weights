"""LAB pretrain_infra: NanoWSP-3 REAL training (char-level, batched forward fiel a engine/nanowsp3_runtime).
Trace: snapshots (W0 + cada S pasos), loss/grad_norm/PPL-holdout, Adam exp_avg/exp_avg_sq al cierre.
Guarda checkpoint safetensors compatible con el runtime (mismo mapping de keys).
USO: python lab/train_nanowsp.py --steps 1500   (status: pretrain_infra - no es un experimento WSPLang)
"""
import argparse, json, math, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path(r"C:\weights")
sys_engine = R / "engine"
import sys
sys.path.insert(0, str(sys_engine))
sys.path.insert(0, str(sys_engine.parent / "lab"))
from nanowsp3_runtime import NanoWSP3, NanoWSP3Config  # noqa: E402

OUT = R / "lab" / "runs"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=8e-4)
ap.add_argument("--snapshot-every", type=int, default=300, dest="snapshot_every")
ap.add_argument("--corpus", default=str(R / "lab" / "data" / "corpus.txt"))
ap.add_argument("--threads", type=int, default=10)
a = ap.parse_args()
torch.set_num_threads(a.threads)

corpus_p = Path(a.corpus)
if not corpus_p.exists():
    corpus_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        for url in ("https://www.gutenberg.org/files/11/11-0.txt",
                    "https://www.gutenberg.org/cache/epub/11/pg11.txt"):
            try:
                urllib.request.urlretrieve(url, corpus_p)
                break
            except Exception as e:
                print("dl fail", url, e)
    except Exception as e:
        print("download failed", e)
if not corpus_p.exists():
    text = (R / "results.md").read_text(encoding="utf-8", errors="replace")[:200000]
    corpus_p.write_text(text, encoding="utf-8")
    print("corpus fallback: results.md")

data = corpus_p.read_text(encoding="utf-8", errors="replace").encode("ascii", "ignore").decode()
data = data[15000:15000 + 400000]
ids = np.frombuffer(data.encode("ascii", "ignore"), dtype=np.uint8).astype(np.int64)
n_hold = min(40000, max(len(ids) // 10, 1000))
train_ids, hold_ids = ids[:-n_hold], ids[-n_hold:]
print(f"corpus chars: {len(ids):,} (train {len(train_ids):,}, hold {len(hold_ids):,})")

# ---- modelonable batched (miemo a engine/runtime keys) ----
c = NanoWSP3Config()
engine = NanoWSP3(c)  # para tokenizer + freqs + init base (W0 real-label)
W0 = {k: v.clone() for k, v in engine.weights.items()}

class Par(nn.Module):
    def __init__(self, w): super().__init__(); self.weight = nn.Parameter(w)

def rms(x, g, eps=c.eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g

def params_init():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = Par(W0["embedding.weight"])
            self.out = Par(W0["output.weight"])
            self.q = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.attention.q_proj.weight"]) for l in range(6)])
            self.k = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.attention.k_proj.weight"]) for l in range(6)])
            self.v = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.attention.v_proj.weight"]) for l in range(6)])
            self.o = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.attention.out_proj.weight"]) for l in range(6)])
            self.min_ = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.mlp_in.weight"]) for l in range(6)])
            self.mout = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.mlp_out.weight"]) for l in range(6)])
            self.an = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.attn_norm.weight"]) for l in range(6)])
            self.mn = nn.ParameterList([nn.Parameter(W0[f"layers.{l}.mlp_norm.weight"]) for l in range(6)])
            self.fn = Par(W0["final_norm.weight"])

    return M()

m = params_init()
fc = engine.freqs_cis  # [max_seq, d_head/2] complex64

def fwd(x):  # x [B, S] int64
    B, S = x.shape
    h = F.embedding(x, m.embed.weight)
    ar = torch.arange(S)
    fc = fc[ar]  # [S, d/2]
    for l in range(c.n_layers):
        hn = rms(h, m.an[l].weight)
        q = (hn @ m.q[l].weight.T).view(B, S, c.n_heads, c.d_head)
        k = (hn @ m.k[l].weight.T).view(B, S, c.n_heads, c.d_head)
        v = (hn @ m.v[l].weight.T).view(B, S, c.n_heads, c.d_head)
        q, k = apply_rope_batch(q, fc), apply_rope_batch(k, fc)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        s = q @ k.transpose(-2, -1) / math.sqrt(c.d_head)
        mask = torch.triu(torch.full((S, S), float("-inf")), 1)
        attn = torch.softmax(s + mask, dim=-1)
        h = h + (attn @ v).transpose(1, 2).contiguous().view(B, S, c.d_model) @ m.o[l].weight.T
        hm = rms(h, m.mn[l].weight)
        h = h + (F.gelu(hm @ m.min_[l].weight.T)) @ m.mout[l].weight.T
    return h @ m.out.weight.T

def apply_rope_batch(x, fcs):
    b, s, h, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))
    fr = fcs.view(1, s, 1, d // 2)
    return torch.view_as_real(xc * fr).flatten(3)

hold_in = torch.tensor([hold_ids[i:i + a.seq] for i in range(0, len(hold_ids) - a.seq - 1, a.seq)], dtype=torch.long)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t_: min(1.0, (t_ + 1) / 100))

def ppl_hold():
    m.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, hold_in.shape[0], 8):
            x = hold_in[i:i + 8]
            lg = fwd(x)
            tot += float(F.cross_entropy(lg[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="sum"))
            ntok += (x.shape[1] - 1) * x.shape[0]
    m.train()
    return math.exp(tot / max(ntok, 1))

t0 = time.time()
losses, snapt_ds, snap_steps, ppls = [], [], [], []
ppl0 = ppl_hold()
print(f"holdout PPL inicial: {ppl0:.2f}", flush=True)
for t in range(a.steps):
    ix = torch.randint(0, len(train_ids) - a.seq - 1, (a.batch,))
    x = torch.tensor(np.stack([train_ids[i:i + a.seq] for i in ix]), dtype=torch.long)
    lg = fwd(x)
    loss = F.cross_entropy(lg[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    opt.zero_grad(); loss.backward()
    gn = float(torch.sqrt(sum(p.grad.norm() ** 2.item() if p.grad is not None else 0.0 for p in m.parameters()))) ** 0.5
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(float(loss))
    if t % 50 == 0:
        print(f"step {t}: loss {loss:.4f} ppl_hold={ppl_hold():.2f} t={time.time()-t0:.0f}s", flush=True)
    if (t + 1) % a.snapshot_every == 0 or t == a.steps - 1:
        sd = {k: p.detach().clone() for k, p in m.named_parameters()}
        snapt_ds.append(sd)
        snap_steps.append(t + 1)
        ppls.append(ppl_hold())

def to_flat(sd):
    return np.concatenate([p.flatten().numpy() for p in sd.values()])

W0_flat = to_flat({k: p for k, p in m.named_parameters() for p in [p]})
final_sd = {k: p.detach().clone() for k, p in m.named_parameters()}
trace = {
    "W0_flat": W0_flat,
    "snapshots_flat": np.stack([to_flat(sd) for sd in snapt_ds]),
    "snap_steps": np.array(snap_steps),
    "ppls": np.array(ppls),
    "loss": np.array(losses),
    "names": json.dumps(list(m.state_dict().keys())),
    "shapes": json.dumps(json.dumps({k: list(p.shape) for k, p in final_sd.items()})),
}
opt_state = {}
for k, st in opt.state.items():
    for kk in ("exp_avg", "exp_avg_sq"):
        if kk in st:
            opt_state[kk] = st[kk].detach().flatten().numpy()
np.savez_compressed(OUT / "nanowsp3_trace.npz", **trace, **opt_state)
engine.checkpoint = None
engine.weights = {k: (p.detach() if torch.is_tensor(p) else torch.tensor(p)) for k, p in
                  {kk: v for kk, v in zip(list(m.state_dict().keys()), m.parameters())}.items()}
cp = OUT / "nanowsp3_pretrained.safetensors"
engine.save_checkpoint(cp)
rep = {"corpus": str(corpus_p), "n_chars": int(len(ids)), "steps": a.steps,
       "seq": a.seq, "batch": a.batch, "lr": a.lr,
       "ppl_start": ppl0, "ppl_final": float(ppls[-1]) if ppls else None,
       "loss_start": float(losses[0]), "loss_final": float(losses[-1]),
       "seconds": round(time.time() - t0, 1), "n_snapshots": len(snap_steps)}
(OUT / "nanowsp3_train.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps(rep, indent=2))
