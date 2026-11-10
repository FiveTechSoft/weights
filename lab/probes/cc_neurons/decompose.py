"""The 'conclude' vector in the neuron basis, phase A of PLAN.md (fixed before running).

Layer N's update to the residual stream is attention + MLP, and the MLP output is W_N a (W_N = down_proj, a = its input,
12 288 SwiGLU neurons), so the per-layer increment of the control vector splits exactly:

    v_N = dattn_N + W_N da_N,   da_N = mean_pos(a) - mean_neg(a)

One pass over the same rows as engine/contrast_vector.py (lab/runs/cc/{pos,neg}.jsonl, same template, bf16), hooks on
each layer's attention output, MLP output, down_proj input and layer output. Null: the same selected tokens split at
random into two groups of the same expected sizes (seed 2026).

usage: python decompose.py run [out_dir]      (GPU, one layer on the device at a time; default lab/runs/cc_neurons)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE_HF = ROOT / "hf_cache" / "qwen35-9b-hf"
VEC = ROOT / "hf_cache" / "cc" / "cc.gguf"
POS, NEG = ROOT / "lab" / "runs" / "cc" / "pos.jsonl", ROOT / "lab" / "runs" / "cc" / "neg.jsonl"
MAX_RESP, NULL_SEED = 384, 2026          # MAX_RESP: contrast_vector's default (unused with focus rows, kept identical)
KS = (1, 8, 32, 128, 512)


def contributions(W, da, target):
    """c_j = da_j <w_j, target> / |target|^2: neuron j's part of the projection of W da on target (sums to it)."""
    return da * (W.T @ target) / (target @ target)


def concentration(W, da, target):
    """share of |target|^2 carried by the MLP, and how few neurons carry it (positive contributions, sorted)."""
    c = contributions(W.double(), da.double(), target.double())
    share = float(c.sum())
    pos = torch.sort(c[c > 0], descending=True).values
    cum = torch.cumsum(pos, 0)
    res = {"share": share, "n_positive": int(pos.numel()), "k50": None, "k90": None, "top": {}}
    if share <= 0 or pos.numel() == 0:
        return res
    for name, f in (("k50", 0.5), ("k90", 0.9)):
        hit = torch.nonzero(cum >= f * share - 1e-12)
        res[name] = int(hit[0]) + 1 if hit.numel() else None
    for k in KS:
        res["top"][k] = float(cum[min(k, cum.numel()) - 1] / share)
    return res


def additivity_error(D, attn, mlp):
    """D: (n_layers + 1, d), row k = mean layer output after layer k - 1 (row 0 = embeddings).
    -> {N: |(D[N+1] - D[N]) - (attn_N + mlp_N)| / |D[N+1] - D[N]|}."""
    out = {}
    for N in attn:
        inc = D[N + 1] - D[N]
        out[N] = float((inc - attn[N] - mlp[N]).norm() / inc.norm().clamp_min(1e-30))
    return out


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def run(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine import contrast_vector as cvx
    out = Path(out)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out}/summary.json exists")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    P = cvx._pairs(tok, cvx._rows(POS), MAX_RESP)
    N = cvx._pairs(tok, cvx._rows(NEG), MAX_RESP)
    n_pos, n_neg = sum(s.stop - s.start for *_, s in P), sum(s.stop - s.start for *_, s in N)
    p_a = n_pos / (n_pos + n_neg)
    print(json.dumps({"pos_rows": len(P), "neg_rows": len(N), "pos_tokens": n_pos, "neg_tokens": n_neg}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(BASE_HF), dtype=torch.bfloat16).eval()
    core = model.model
    core = core.language_model if hasattr(core, "language_model") else core
    layers = core.layers
    dev = "cuda"
    hooks = cvx._stream(layers, dev)
    groups = ("pos", "neg", "A", "B")
    S = {g: {k: [None] * len(layers) for k in ("attn", "mlp", "a", "out")} for g in groups}
    ctx = {}

    def acc(g, kind, l, x):
        S[g][kind][l] = x if S[g][kind][l] is None else S[g][kind][l] + x

    def add(kind, l, t):
        t = _first(t)[0, ctx["sel"]].float()                   # (n_sel, width), on whatever device the layer is
        acc(ctx["group"], kind, l, t.sum(0).cpu())
        m = ctx["mask"].to(t.device)
        acc("A", kind, l, t[m].sum(0).cpu())
        acc("B", kind, l, t[~m].sum(0).cpu())

    for l, L in enumerate(layers):
        attn = L.linear_attn if hasattr(L, "linear_attn") else L.self_attn
        hooks.append(attn.register_forward_hook(lambda m, i, o, l=l: add("attn", l, o)))
        hooks.append(L.mlp.register_forward_hook(lambda m, i, o, l=l: add("mlp", l, o)))
        hooks.append(L.mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: add("a", l, i)))
        hooks.append(L.register_forward_hook(lambda m, i, o, l=l: add("out", l, o)))
    g = torch.Generator().manual_seed(NULL_SEED)
    emb = {k: None for k in groups}
    hsum = {"pos": None, "neg": None}
    cnt = {k: 0 for k in groups}
    t0 = time.time()
    try:
        with torch.no_grad():
            for group, pairs in (("pos", P), ("neg", N)):
                for i, (a, b, s) in enumerate(pairs):
                    sel = slice(len(a) + s.start, len(a) + s.stop)
                    mask = torch.rand(s.stop - s.start, generator=g) < p_a
                    ctx.update(sel=sel, group=group, mask=mask)
                    hs = core(input_ids=torch.tensor([a + b]), output_hidden_states=True, use_cache=False).hidden_states
                    v = torch.stack([h[0, sel].float().cpu().sum(0) for h in hs])     # as contrast_vector.layer_means
                    hsum[group] = v if hsum[group] is None else hsum[group] + v
                    e = hs[0][0, sel].float().cpu()
                    for k, x in ((group, e), ("A", e[mask]), ("B", e[~mask])):
                        emb[k] = x.sum(0) if emb[k] is None else emb[k] + x.sum(0)
                    cnt[group] += s.stop - s.start
                    cnt["A"] += int(mask.sum())
                    cnt["B"] += int((~mask).sum())
                    if i % 10 == 0:
                        print(f"[cc_neurons] {group} {i}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()
    mean = {gk: {k: torch.stack(S[gk][k]) / cnt[gk] for k in S[gk]} for gk in groups}
    for gk in groups:
        mean[gk]["D"] = torch.cat([(emb[gk] / cnt[gk])[None], mean[gk]["out"]])
    torch.save({"mean": mean, "count": cnt}, out / "means.pt")

    # check 1: the hidden-state pass reproduces cc.gguf (same rows, same code path as contrast_vector)
    from engine.contrast_vector import increments
    Dh = hsum["pos"] / cnt["pos"] - hsum["neg"] / cnt["neg"]
    inc_h = increments(Dh)
    vec = read_vector(VEC)
    cos_vec = {n: float(torch.nn.functional.cosine_similarity(inc_h[n], vec[n], 0)) for n in vec}
    # check 2: per layer, the increment of the raw layer outputs = attention + MLP
    d = {k: mean["pos"][k] - mean["neg"][k] for k in ("attn", "mlp", "a", "D")}
    add_err = additivity_error(d["D"], {n: d["attn"][n] for n in vec}, {n: d["mlp"][n] for n in vec})
    # check 3: the MLP output is W a (no bias)
    rows, null_rows = {}, {}
    for n in sorted(vec):
        W = layers[n].mlp.down_proj.weight.float()
        lin_err = float((W @ d["a"][n] - d["mlp"][n]).norm() / d["mlp"][n].norm())
        r = concentration(W, d["a"][n], vec[n])
        r.update(layer=n, v_norm=float(vec[n].norm()), cos_vs_gguf=cos_vec[n], additivity_err=add_err[n],
                 linearity_err=lin_err, mlp_share_of_raw_increment=float(
                     (d["mlp"][n] @ (d["D"][n + 1] - d["D"][n])) / (d["D"][n + 1] - d["D"][n]).norm() ** 2))
        rows[n] = r
        da0 = mean["A"]["a"][n] - mean["B"]["a"][n]
        null_rows[n] = concentration(W, da0, W @ da0)
        print(json.dumps({k: r[k] for k in ("layer", "share", "k50", "k90", "cos_vs_gguf", "additivity_err")}
                         | {"null_k50": null_rows[n]["k50"]}), flush=True)
    summ = {"counts": cnt, "rows": rows, "null": null_rows, "seconds": round(time.time() - t0),
            "checks": {"min_cos_vs_gguf": min(cos_vec.values()), "max_additivity_err": max(add_err.values()),
                       "max_linearity_err": max(r["linearity_err"] for r in rows.values())}}
    summ["reading"] = reading(rows, null_rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"checks": summ["checks"], "reading": summ["reading"]}, indent=1))


def reading(rows, null_rows):
    """PLAN rule: layers carrying >= 70 % of sum_N share_N |v_N|^2; median k50 there vs the null's."""
    w = {n: max(r["share"], 0) * r["v_norm"] ** 2 for n, r in rows.items()}
    tot = sum(w.values())
    chosen, acc = [], 0.0
    for n in sorted(w, key=w.get, reverse=True):
        if acc >= 0.7 * tot:
            break
        chosen.append(n)
        acc += w[n]
    big = 10 ** 9
    med = float(np.median([rows[n]["k50"] or big for n in chosen])) if chosen else None
    nmed = float(np.median([null_rows[n]["k50"] or big for n in chosen])) if chosen else None
    if med is None:
        verdict = "no MLP share"
    elif med <= 32 and nmed >= 5 * med:
        verdict = "concentrated"
    elif med >= 512:
        verdict = "spread"
    else:
        verdict = "partial"
    return {"layers": sorted(chosen), "median_k50": med, "null_median_k50": nmed, "verdict": verdict}


def read_vector(path):
    import gguf
    r = gguf.GGUFReader(str(path))
    return {int(t.name.split(".")[1]): torch.tensor(np.array(t.data, dtype=np.float32)) for t in r.tensors}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1] if len(a) > 1 else ROOT / "lab" / "runs" / "cc_neurons")
    else:
        sys.exit(__doc__)
