"""Path 1 probe: alternating least squares on the small nanoGPT shape, zero backward (als_criterion.md).

usage: als_probe.py S|C max_sweeps

Each update: forward pass on a fresh train sample, softmax residual R = onehot - p, a residual-stream target
dx = pinv(readout) R carried unchanged through the skip connections, a hand-made local target for the chosen
weight matrix (pseudo-inverse of the next linear map, linearized gelu / softmax), a ridge solve for the change,
and a step size picked on a holdout from the tail of train. Arm C only solves output-side matrices.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "lab"), str(R / "lab" / "probes")]

from engine import wsp_arch, wsp_synth, wsp_train  # noqa: E402
from als_common import ARCH, SHAPE, measure  # noqa: E402

ARM, MAX_SWEEPS = sys.argv[1], int(sys.argv[2])
dev = "cuda"
PATIENCE, FIT_WINDOWS, CHUNK = 4, 1024, 256
STEPS = (0.0,) + tuple(2.0 ** e for e in range(-12, 7))  # fix 1: targets via pinv can be huge
text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
data = wsp_train.char_dataset(text)
V = len(data.chars)
T, L, NH, D, FF = SHAPE["block"], SHAPE["layers"], SHAPE["heads"], SHAPE["d"], SHAPE["ff"]
HD = D // NH
torch.manual_seed(1337)
model = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=V), seed=1337).to(dev).eval()
P = model.get_parameter
wte, wpe, lnf = P("transformer.wte.weight"), P("transformer.wpe.weight"), P("transformer.ln_f.weight")
assert P("lm_head.weight") is wte
UNTIE = len(sys.argv) > 3 and sys.argv[3] == "untie"  # diagnostic only: breaks nanoGPT's tie
if UNTIE:
    model._set_tensor("lm_head.weight", torch.nn.Parameter(wte.detach().clone()))
head = P("lm_head.weight")
blk = [{k: P(f"transformer.h.{l}.{n}.weight") for k, n in
        (("ln1", "ln_1"), ("qkv", "attn.c_attn"), ("aproj", "attn.c_proj"), ("ln2", "ln_2"),
         ("fc", "mlp.c_fc"), ("mproj", "mlp.c_proj"))} for l in range(L)]

train = torch.tensor(data.train, device=dev)
cut = int(len(train) * 0.9)
hold = train[cut:]
hold_w = hold[:(len(hold) - 1) // T * T + 1]
hold_x = hold_w[:-1].view(-1, T)
hold_y = hold_w[1:].view(-1, T)
gen = torch.Generator(device="cpu").manual_seed(0)
mask = torch.ones(T, T, device=dev).tril().bool()


def ln(x, w):
    return F.layer_norm(x, (D,), w, None, 1e-5)


def forward(x, keep=False):
    h = wte[x] + wpe[:x.shape[1]]
    cache = []
    for b in blk:
        a_in = ln(h, b["ln1"])
        q, k, v = (a_in @ b["qkv"].T).split(D, -1)
        q, k, v = (t.view(*x.shape, NH, HD).transpose(1, 2) for t in (q, k, v))
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~mask[:x.shape[1], :x.shape[1]], -1e9).softmax(-1)
        yc = (att @ v).transpose(1, 2).reshape(*x.shape, D)
        h = h + yc @ b["aproj"].T
        m_in = ln(h, b["ln2"])
        pre = m_in @ b["fc"].T
        g = F.gelu(pre)
        h = h + g @ b["mproj"].T
        if keep:
            cache.append(dict(a_in=a_in, q=q, k=k, v=v, att=att, yc=yc, m_in=m_in, pre=pre, g=g))
    sigma = (h.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    z = ln(h, lnf)
    return z @ head.T, cache, z, sigma


def holdout_nll():
    total = 0.0
    for xs, ys in zip(hold_x.split(CHUNK), hold_y.split(CHUNK)):
        total += F.cross_entropy(forward(xs)[0].flatten(0, 1), ys.flatten(), reduction="sum").item()
    return total / hold_y.numel()


def sample():
    starts = torch.randint(0, cut - T - 1, (FIT_WINDOWS,), generator=gen).to(dev)
    idx = starts[:, None] + torch.arange(T + 1, device=dev)
    w = train[idx]
    return w[:, :-1], w[:, 1:]


def ridge_solve(gram, cross, rel=1e-3):
    lam = rel * gram.diagonal().mean().clamp(min=1e-12)
    return torch.linalg.solve(gram + lam * torch.eye(len(gram), device=dev, dtype=gram.dtype), cross)


def pinv_map(W, target, rel=1e-3):
    """Smallest input change u with u @ W.T ~= target, for W of shape (out, in)."""
    gram = (W @ W.T).double()
    return (target.double() @ ridge_solve(gram, W.double(), rel)).float()


def gelu_slope(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def residual_targets(x, y):
    logits, cache, z, sigma = forward(x, keep=True)
    r = F.one_hot(y, V).float() - logits.softmax(-1)
    dz = pinv_map(head, r)  # logits = z @ head.T
    dx = dz * sigma / lnf
    dx = dx - dx.mean(-1, keepdim=True)
    return r, z, dx, cache


def solve_update(which, layer):
    """Accumulates normal equations over the fresh sample, returns (param, delta)."""
    x_all, y_all = sample()
    grams, crosses = {}, {}

    def add(key, feats, tgt):
        f, t = feats.reshape(-1, feats.shape[-1]).double(), tgt.reshape(-1, tgt.shape[-1]).double()
        grams[key] = grams.get(key, 0) + f.T @ f
        crosses[key] = crosses.get(key, 0) + f.T @ t

    for x, y in zip(x_all.split(CHUNK), y_all.split(CHUNK)):
        r, z, dx, cache = residual_targets(x, y)
        if which == "out":
            add(0, z, r)
            continue
        b, c = blk[layer], cache[layer]
        if which == "mproj":
            add(0, c["g"], dx)
        elif which == "fc":
            s = gelu_slope(c["pre"])
            dpre = pinv_map(b["mproj"], dx) * s / (s * s + 0.05)
            add(0, c["m_in"], dpre)
        elif which == "aproj":
            add(0, c["yc"], dx)
        else:
            dy = pinv_map(b["aproj"], dx).view(*x.shape, NH, HD).transpose(1, 2)  # B,H,T,hd
            for h in range(NH):
                att, q, k, v, a_in = c["att"][:, h], c["q"][:, h], c["k"][:, h], c["v"][:, h], c["a_in"]
                if which == "v":
                    add(h, att @ a_in, dy[:, h])
                else:
                    u = dy[:, h] @ v.transpose(-1, -2)  # B,T,T: dy_i . v_j
                    ds = att * (u - (att * u).sum(-1, keepdim=True))
                    if which == "q":
                        add(h, a_in, ds @ k / math.sqrt(HD))
                    else:
                        add(h, a_in, ds.transpose(-1, -2) @ q / math.sqrt(HD))

    if which == "out":
        return head, ridge_solve(grams[0], crosses[0]).float().T  # V x D
    if which in ("mproj", "fc", "aproj"):
        param = blk[layer][which]
        return param, ridge_solve(grams[0], crosses[0]).float().T
    param = blk[layer]["qkv"]
    delta = torch.zeros_like(param)
    base = {"q": 0, "k": D, "v": 2 * D}[which]
    for h in range(NH):
        delta[base + h * HD:base + (h + 1) * HD] = ridge_solve(grams[h], crosses[h]).float().T
    return param, delta


def apply_with_line_search(param, delta):
    before = param.clone()
    best = (holdout_nll(), 0.0)
    for s in STEPS[1:]:
        param.copy_(before + s * delta)
        h = holdout_nll()
        if h < best[0]:
            best = (h, s)
    param.copy_(before + best[1] * delta)
    return best


def update_wpe():
    x_all, y_all = sample()
    total = torch.zeros(T, D, device=dev)
    for x, y in zip(x_all.split(CHUNK), y_all.split(CHUNK)):
        total += residual_targets(x, y)[2].sum(0)
    return wpe, total / len(x_all)


PLAN = [("out", None)]
for layer in reversed(range(L)):
    PLAN += [("mproj", layer), ("aproj", layer)] if ARM == "C" else \
            [("mproj", layer), ("fc", layer), ("aproj", layer), ("v", layer), ("q", layer), ("k", layer)]
if ARM == "S":
    PLAN.append(("wpe", None))

with wsp_synth.forbid_gradients():
    ref = model(hold_x[:8], hold_y[:8])[0]
    assert (ref - forward(hold_x[:8])[0]).abs().max().item() < 1e-3, "probe forward differs from nanoGPT"
    log, best, stale, t0, tokens = [], holdout_nll(), 0, time.time(), 0
    print(f"{ARM} start hold {best:.4f}", flush=True)
    for sweep in range(MAX_SWEEPS):
        steps = []
        for which, layer in PLAN:
            param, delta = update_wpe() if which == "wpe" else solve_update(which, layer)
            tokens += FIT_WINDOWS * T
            h, s = apply_with_line_search(param, delta)
            steps.append(s)
        log.append({"sweep": sweep, "hold": round(h, 4), "steps": steps, "seconds": round(time.time() - t0, 1)})
        print(f"{ARM} sweep {sweep}: hold {h:.4f} steps {steps} {time.time() - t0:.0f}s", flush=True)
        if h < best - 1e-3:
            best, stale = h, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

model.synthesis_info = {"provenance": "iterative_no_backprop", "backward_calls": 0, "method": f"als_{ARM}",
                        "sweeps": len(log), "fit_tokens_processed": tokens}
result = measure(f"{ARM}_als", model, data, text, dev)
result["log"] = log
(R / f"lab/runs/als_{ARM}{'_untied' if UNTIE else ''}_small.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
