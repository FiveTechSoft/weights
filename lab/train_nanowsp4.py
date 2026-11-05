"""train_nanowsp4 — pretrain paramétrico del paciente LLM-nano REAL (status: pretrain_infra).
Config objetivo L2: 24L x d512 x 8h(d64) x dff1408 (~30M params), byte-level, corpus Gutenberg 3MB.
Uso: python lab/train_nanowsp4.py --layers 24 --d-model 512 --n-heads 8 --d-ff 1408 --steps 3000
Smoke:   python lab/train_nanowsp4.py --layers 2 --d-model 64 --steps 8 --corpus lab/data/corpus_n4.txt
Trace estandarizado (curriculum): W0_flat + snapshots + loss + PPL-holdout + Adam exp_avg final.
"""
import argparse, json, math, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path(r"C:\weights")

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=24)
ap.add_argument("--d-model", type=int, default=512)
ap.add_argument("--n-heads", type=int, default=8)
ap.add_argument("--d-ff", type=int, default=1408)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--snapshot-every", type=int, default=400)
ap.add_argument("--corpus", default=str(R / "lab" / "data" / "corpus_n4.txt"))
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--tag", default="n4")
a = ap.parse_args()
torch.set_num_threads(a.threads)

CORPUS_P = Path(a.corpus)
if not CORPUS_P.exists():
    CORPUS_P.parent.mkdir(parents=True, exist_ok=True)
    urls = ["https://www.gutenberg.org/files/11/11-0.txt",
            "https://www.gutenberg.org/cache/epub/11/pg11.txt"]
    blobs = []
    for url in urls:
        try:
            blobs.append(urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore"))
            break
        except Exception as e:
            print("dl fail", e)
    if not blobs:
        blobs = [(R / "results.md").read_text(encoding="utf-8", errors="replace")]
    CORPUS_P.write_text("\n\n".join(blobs), encoding="utf-8")

data = CORPUS_P.read_text(encoding="utf-8", errors="replace").encode("ascii", "ignore").decode()
print(f"corpus: {len(data):,} chars")
ids = np.frombuffer(data.encode("ascii", "ignore"), dtype=np.uint8).astype(np.int64)
n_hold = max(int(len(ids) * 0.05), 2000)
train_ids, hold_ids = ids[:-n_hold], ids[-n_hold:]

cnt = np.bincount(ids, minlength=256)
p = (cnt + 1) / (cnt.sum() + 256)
EPH = float(-np.sum(p * np.log(p)))

CFG = {"n_layers": a.layers, "d_model": a.d_model, "n_heads": a.n_heads,
       "d_head": a.d_model // a.n_heads, "d_ff": a.d_ff}

def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g

def rope(x):
    b, s, h, d = x.shape
    theta = 10000.0
    inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(s, dtype=torch.float32)
    ang = torch.outer(t, inv)
    fcs = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))
    return torch.view_as_real(xc * fcs.view(1, s, 1, d // 2)).flatten(3).type_as(x)

# --- params: los mismos keys que nano3lib (nanocompat con runner) ---
mk = {"embed.weight": torch.randn(256, CFG["d_model"]) * (256 ** -0.5),
      "out.weight": torch.randn(256, CFG["d_model"]) * (CFG["d_model"] ** -0.5),
      "fn.weight": torch.ones(CFG["d_model"])}
for l in range(CFG["n_layers"]):
    mk[f"q.{l}"] = torch.empty(CFG["d_model"], CFG["d_model"])
    mk[f"k.{l}"] = torch.empty(CFG["d_model"], CFG["d_model"])
    mk[f"v.{l}"] = torch.empty(CFG["d_model"], CFG["d_model"])
    mk[f"o.{l}"] = torch.empty(CFG["d_model"], CFG["d_model"])
    mk[f"min_.{l}"] = torch.empty(CFG["d_ff"], CFG["d_model"])
    mk[f"mout.{l}"] = torch.empty(CFG["d_model"], CFG["d_ff"])
    mk[f"an.{l}"] = torch.ones(CFG["d_model"])
    mk[f"mn.{l}"] = torch.ones(CFG["d_model"])
rngc = torch.Generator().manual_seed(42)
for l in range(CFG["n_layers"]):
    for k in ("q", "k", "v", "o", "min_", "mout"):
        nn.init.orthogonal_(mk[f"{k}.{l}"], generator=rngc)
mk["out.weight"] = mk["embed.weight"].clone()  # tied

P = {k: nn.Parameter(v.clone()) for k, v in mk.items()}
TRAINABLE = [k for k in P if not k.startswith("embed.")]
opt = torch.optim.AdamW([P[k] for k in TRAINABLE], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

def fwd(x):
    B, S = x.shape
    h = F.embedding(x, P["embed.weight"])
    for l in range(CFG["n_layers"]):
        hn = rms(h, P[f"an.{l}"])
        q = (hn @ P[f"q.{l}"].T).view(B, S, CFG["n_heads"], CFG["d_head"])
        k = (hn @ P[f"k.{l}"].T).view(B, S, CFG["n_heads"], CFG["d_head"])
        v = (hn @ P[f"v.{l}"].T).view(B, S, CFG["n_heads"], CFG["d_head"])
        q, k = rope(q), rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        sc = q @ k.transpose(-2, -1) / math.sqrt(CFG["d_head"])
        mask = torch.triu(torch.full((S, S), float("-inf")), 1)
        h = h + (torch.softmax(sc + mask, dim=-1) @ v).transpose(1, 2).contiguous().view(B, S, CFG["d_model"]) @ P[f"o.{l}"].T
        hm = rms(h, P[f"mn.{l}"])
        h = h + F.gelu(hm @ P[f"min_.{l}"].T) @ P[f"mout.{l}"].T
    hf = rms(h, P["fn.weight"])
    return hf @ P["out.weight"].T

def maybe_flatten(flat=0):
    return np.concatenate([v.detach().flatten().numpy() for v in P.values()])

def flat_key_list():
    return list(P.keys())

def flat_shapes():
    return json.dumps(json.dumps({k: list(v.shape) for k, v in P.items()}))

hold_in = torch.tensor(np.array([hold_ids[i:i + a.seq] for i in range(0, len(hold_ids) - a.seq - 1, a.seq)]), dtype=torch.long)

def ppl_hold():
    tot, ntk = 0.0, 0
    for i in range(0, hold_in.shape[0], 8):
        x = hold_in[i:i + 8]
        lg = fwd(x)
        tot += float(F.cross_entropy(lg[:, :-1].flatten(0, 1), x[:, 1:].flatten(), reduction="sum"))
        ntk += (x.shape[1] - 1) * x.shape[0]
    return math.exp(tot / max(ntk, 1))

W0_flat = maybe_flatten()
def flat():
    return maybe_flatten()

losses, snapt, ssteps, pplsn = [], [], [], []
t0 = time.time()
print(f"EPH corpus {EPH:.3f} | init PPL hold {ppl_hold():.2f}", flush=True)
for t in range(a.steps):
    ia = torch.randint(0, len(train_ids) - a.seq - 1, (a.batch,))
    x = torch.tensor(np.array([train_ids[i:i + a.seq] for i in ia]), dtype=torch.long)
    lg = fwd(x)
    loss = F.cross_entropy(lg[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss.detach()))
    if t % 50 == 0:
        print(f"step {t}: loss {loss:.4f} ppl_hold={ppl_hold():.2f} t={time.time()-t0:.0f}s gn={float(P['q.0'].grad.norm() if P['q.0'].grad is not None else -1):.2f}", flush=True)
    if (t + 1) % a.snapshot_every == 0 or t == a.steps - 1:
        snapt.append(flat())
        ssteps.append(t + 1)
        pplsn.append(ppl_hold())

of = flat()
tr = {"W0_flat": W0_flat, "snapshots_flat": np.stack(snapt), "snap_steps": np.array(ssteps),
      "loss": np.array(losses), "names": json.dumps(json.dumps(list(P.keys()))),
      "shapes": flat_shapes(), "meta": json.dumps({**CFG, "dataset": "corpus_n4", "steps": a.steps,
      "ppl_start": float(np.exp(losses[0])), "ppl_final": float(np.exp(losses[-1])),
      "eph": EPH, "seconds": round(time.time() - t0, 1)})}
optst = {}
for kk, st in opt.state.items():
    for sn in ("exp_avg", "exp_avg_sq"):
        if sn in st:
            optst[sn] = st[sn].detach().flatten().numpy()
outp = R / "lab" / "runs" / f"nanowsp4_{a.tag}_trace.npz"
np.savez_compressed(outp, **tr, **optst)
big = min(a.d_ff, 2048) if False else a.d_ff
ck = R / "lab" / "runs" / f"nanowsp4_{a.tag}.safetensors"
from safetensors.torch import save_file as _sf
_sf({k: v.detach() for k, v in P.items()}, str(ck))
rep = {**CFG, "steps": a.steps, "corpus": str(CORPUS_P), "n_chars": int(len(ids)),
       "ppl_start": float(np.exp(losses[0])), "ppl_final": float(np.exp(losses[-1])),
       "eph": EPH, "seconds": round(time.time() - t0, 1), "trace": str(outp), "ckpt": str(ck)}
(R / "lab" / "runs" / f"nanowsp4_{a.tag}_train.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps(rep, indent=2))
