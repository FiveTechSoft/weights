"""Train a behavior-extraction auxiliary on cached states (DSpine-inspired arms, multiplication probe).
Arm H (conditioned, DSpine-style): input = concatenated hidden states of --layers at position t-1;
    logits = head(h_last + aux(states)). The aux predicts an update in the target's post-norm space and the
    frozen tied head is the shared transfer space (DSpine's "unified transfer space"); CE through the full
    248k-way head is exact. No gated adjacent injection: this probe predicts single positions, not blocks.
Arm X (ablation: aux without target representations): input = masked mean of the prefix token embeddings
    (sees the full problem, no decoder hidden states, no order); width scaled to match arm H's parameters.
Arm C (conventional small model, same dataset): GRU over the prefix token embeddings -> linear head input;
    sees order and full problem, no decoder hidden states. Parameter count differs by architecture and is
    reported exactly; it is the strong conventional baseline, not a matched ablation.
All arms share the frozen tokenizer / embedding table / tied head (static lookups, no target computation),
so the only thing that distinguishes H from X/C is access to the target's internal hidden states.
Labels: ground truth by default; with a teacher-labeled cache (extract_hidden.py --teacher-labels) the aux
learns the target's own generations, errors included (DSpine-faithful mimicry; a secondary measurement).
Head rows are read in 8192-row chunks through checkpoint, the measured memory pattern of train_down23.py.
usage: train_aux.py MODEL_DIR CACHE.safetensors OUT_DIR --arm h|x|c [--layers 3,11,19,23] [--steps 150]
Writes OUT_DIR/aux_<arm>.safetensors (bf16) and train_report.json. Files are NOT for the repo.
"""
import argparse, hashlib, json, os, sys, time
import torch
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("cache"); ap.add_argument("out")
ap.add_argument("--arm", choices=["h", "x", "c"], required=True)
ap.add_argument("--layers", default="3,11,19,23")
ap.add_argument("--steps", type=int, default=150); ap.add_argument("--pos-bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--max-seconds", type=float, default=300)
ap.add_argument("--width", type=int, default=1024); ap.add_argument("--gru-hidden", type=int, default=512)
a = ap.parse_args(); torch.manual_seed(0); os.makedirs(a.out, exist_ok=True)
LAYERS = [int(x) for x in a.layers.split(",")]
cache = load_file(a.cache)
TGT = cache["tgt"]; H_LAST = cache["hlast"].float()
q = QwenText(a.model)

if a.arm == "h":
    IN = torch.cat([cache["h%d" % i].float() for i in LAYERS], -1)          # (N, 1024*len(LAYERS))
    width = a.width
    aux = torch.nn.Sequential(torch.nn.Linear(IN.shape[1], width), torch.nn.SiLU(),
                              torch.nn.Linear(width, 1024)).float()
    def feats(idx):
        return IN[idx]
elif a.arm == "x":
    width = max(64, a.width * (1024 * (len(LAYERS) + 1) + 1) // 2049)  # exact parameter match with arm H
    IN = cache["emean"].float()                                        # (N, 1024), embeddings only
    aux = torch.nn.Sequential(torch.nn.Linear(1024, width), torch.nn.SiLU(),
                              torch.nn.Linear(width, 1024)).float()
    def feats(idx):
        return IN[idx]
else:
    width = a.gru_hidden
    PREF = [cache["pref_ids"][cache["pref_off"][i]:cache["pref_off"][i + 1]] for i in range(len(TGT))]
    aux = torch.nn.ModuleDict({"gru": torch.nn.GRU(1024, width, batch_first=True),
                               "out": torch.nn.Linear(width, 1024)}).float()
    def feats(idx):
        lens = [len(PREF[i]) for i in idx]; L = max(lens)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :lens[r]] = PREF[i]
        emb = q.embed(ids)                                             # frozen table lookup
        out_, _ = aux["gru"](emb.float())
        return out_[torch.arange(len(idx)), torch.tensor(lens) - 1]

n_params = sum(p.numel() for p in aux.parameters())
opt = torch.optim.Adam(aux.parameters(), lr=a.lr)
V = q.sl.get_shape()[0]
print("arm", a.arm, "width", width, "params", n_params, "positions", len(TGT), flush=True)


def ce(idx):
    f = feats(idx)
    h = H_LAST[idx] + f if a.arm == "h" else f     # only arm H keeps the target's hidden state in the skip
    tgt = TGT[idx]; lse = []; tv = torch.zeros(len(idx))

    def chunk(h_, s_):
        E = q.sl[s_:min(V, s_ + 8192)].float(); lg = h_ @ E.T
        ok = (tgt >= s_) & (tgt < s_ + E.shape[0])
        return torch.logsumexp(lg, -1), lg.gather(1, (tgt - s_).clamp(0, E.shape[0] - 1)[:, None])[:, 0] * ok
    for s in range(0, V, 8192):
        ls, v = checkpoint(chunk, h, s, use_reentrant=False); lse.append(ls); tv = tv + v
    return (torch.logsumexp(torch.stack(lse), 0) - tv).mean()


g = torch.Generator().manual_seed(0); log = []; t1 = time.time()
for step in range(a.steps):
    if time.time() - t1 > a.max_seconds:
        print("time budget at step", step); break
    idx = torch.randint(0, len(TGT), (a.pos_bs,), generator=g).tolist()
    opt.zero_grad(); l = ce(idx); l.backward(); opt.step(); log.append(float(l.detach()))
    if step % 10 == 0:
        print("step", step, "loss", round(float(l), 4), "t", round(time.time() - t1, 1), flush=True)
state = {("aux." + k): w.data.to(torch.bfloat16).contiguous() for k, w in aux.state_dict().items()}
path = os.path.join(a.out, "aux_%s.safetensors" % a.arm)
save_file(state, path)
rep = {"arm": a.arm, "layers": LAYERS, "width": width, "params": n_params, "steps": len(log),
       "first5": log[:5], "last5": log[-5:], "seconds": round(time.time() - t1, 1),
       "aux_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
json.dump(rep, open(os.path.join(a.out, "train_report.json"), "w"), indent=1); print(rep)
