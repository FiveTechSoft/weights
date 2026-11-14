"""Closed-form behaviour distillation (no gradients): the WSP / MEMIT least-squares write of mlp.down_proj, used to make
the base's residual stream follow a teacher's (MiMo's) on the teacher's own answers, layer by layer, bottom-up.

Per layer l, with both models streamed one layer at a time through the GPU:
  h_t <- teacher layer l (h_t)                       teacher's residual stream on the same tokens
  y_b, K <- base layer l (h_b)                       K = input of base down_proj (the "keys")
  R = y_t - y_b                                      what is missing at this layer's output
  dW = argmin ||K dW^T - R||^2 + mu*mean(diag K^T K)*||dW||^2  = ((K^T K + mu' I)^-1 K^T R)^T
  h_b <- base layer l with down_proj + dW (h_b)      the next layer sees the edited stream
The change to the residual is exactly K dW^T (down_proj output goes straight into the residual), so this is the best
the down_proj of that layer can do for that layer's mismatch. Stats per layer: share of ||R||^2 removed, on the
training sequences and on held-out ones.

Export: PEFT LoRA of each dW by exact SVD (engine.lora_export.lowrank), lora_alpha = r, converted by llama.cpp's
convert_lora_to_gguf.py like every other adapter here.

usage: python lab/probes/close_probe/cf_distill.py <base_hf> <teacher_hf> <data.jsonl> <out_dir>
       [--rank 256] [--mu 1e-2] [--heldout 8] [--max_len 2600]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "model.language_model.layers"


def text_model(model):
    m = model.model
    return m.language_model if hasattr(m, "language_model") else m


def layers_of(model):
    return text_model(model).layers


def solve(G, P, mu, device="cpu"):
    """G = K^T K (in x in), P = K^T R (in x out) -> dW (out x in), ridge mu relative to the mean diagonal of G.
    device "cpu": float64 LU (exact; ~2.4 GB of RAM at 12288, too much next to a 9B model on 32 GB, run cf3 died).
    other devices: float32 Cholesky there (the ridge keeps it positive definite), float64 CPU fallback if it fails."""
    if str(device) != "cpu":
        Gf = G.to(device=device, dtype=torch.float32)
        M = (Gf + Gf.T) / 2
        M.diagonal().add_(mu * M.diagonal().mean())
        L, info = torch.linalg.cholesky_ex(M)
        del M
        if int(info) == 0:
            return torch.cholesky_solve(P.to(device=device, dtype=torch.float32), L).T.contiguous()
    G = G.double().cpu()
    G = (G + G.T) / 2                      # fp32 accumulation leaves it only nearly symmetric / PSD
    M = G + mu * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype)
    return torch.linalg.solve(M, P.double().cpu()).T.float().contiguous()


def residual_error(dW, Gh, Ph, Eh):
    """||R - K dW^T||^2 from the held-out sums Gh = K^T K, Ph = K^T R, Eh = ||R||^2 (no pass over the data)."""
    dW = dW.to(Gh.dtype)
    return Eh - 2 * (dW.T * Ph).sum() + ((dW @ Gh) * dW).sum()


def solve_select(G, P, Gh, Ph, Eh, mus, device="cpu"):
    """Ridge chosen per layer by the held-out error. Returns (dW, mu, {mu: held-out error})."""
    best, errs = None, {}
    for mu in mus:
        dW = solve(G, P, mu, device)
        errs[mu] = float(residual_error(dW, Gh, Ph, Eh))
        if best is None or errs[mu] < errs[best[1]]:
            best = (dW, mu)
        del dW
    return best[0], best[1], errs


class _Stub(torch.nn.Module):
    """Stands in for a decoder layer: records the call's keyword arguments and passes the stream through."""
    def __init__(self, i, rec):
        super().__init__()
        self.i, self.rec = i, rec

    def forward(self, hidden_states, **kw):
        self.rec[self.i] = kw
        if self.i == 0:
            self.rec["h0"] = hidden_states
        return hidden_states


def layer_kwargs(model, ids, device):
    """Per-layer call arguments (masks, rotary embeddings) the text model would pass, and the base embedding,
    without running any layer."""
    tm = text_model(model)
    orig = list(tm.layers)
    rec = {}
    try:
        for i in range(len(orig)):
            tm.layers[i] = _Stub(i, rec)
        with torch.no_grad():
            tm(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for i, m in enumerate(orig):
            tm.layers[i] = m
    h0 = rec.pop("h0")
    memo = {}       # every layer gets the same rotary tensors: one device copy, not one per layer

    def mv(x):
        if torch.is_tensor(x):
            return memo.setdefault(id(x), x.to(device))
        if isinstance(x, (tuple, list)):
            return type(x)(mv(v) for v in x)
        return x
    kws = {i: {k: mv(v) for k, v in kw.items() if k not in ("past_key_values", "use_cache")} for i, kw in rec.items()}
    return kws, h0[0].to(device)


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def attn_out_name(layer):
    """Name of the projection that writes the attention (or linear-attention) output into the residual stream."""
    return "linear_attn.out_proj" if hasattr(layer, "linear_attn") else "self_attn.o_proj"


def attn_out(layer):
    return layer.get_submodule(attn_out_name(layer))


class _Acc:
    """K^T K, K^T R and ||R||^2 sums of one linear write, on the training and the held-out sequences."""
    def __init__(self, n_in, n_out, device, sel):
        z = lambda *s: torch.zeros(*s, device=device)
        self.G, self.P, self.E = z(n_in, n_in), z(n_in, n_out), 0.0
        self.sel = sel
        self.Gh, self.Ph, self.Eh = (z(n_in, n_in), z(n_in, n_out), 0.0) if sel else (None, None, 0.0)

    def add(self, K, R, train):
        if train:
            self.G += K.T @ K
            self.P += K.T @ R
            self.E += float(R.pow(2).sum())
        elif self.sel:
            self.Gh += K.T @ K
            self.Ph += K.T @ R
            self.Eh += float(R.pow(2).sum())

    def solve(self, mu, mus, device):
        mus = mus if self.sel else None
        """(dW fp32 on device, mu, held-out error by mu or None, explained train, explained held-out or None) -
        the explained shares in closed form from the sums (the write is linear in the residual)."""
        if mus:
            dW, mu, errs = solve_select(self.G, self.P, self.Gh, self.Ph, torch.tensor(self.Eh, device=device), mus,
                                        device)
        else:
            dW, errs = solve(self.G, self.P, mu, device), None
        dW = dW.to(device=device, dtype=torch.float32)
        ex = 1 - float(residual_error(dW, self.G, self.P, self.E)) / max(self.E, 1e-30)
        exh = 1 - float(residual_error(dW, self.Gh, self.Ph, self.Eh)) / max(self.Eh, 1e-30) if self.sel else None
        self.G = self.P = self.Gh = self.Ph = None
        return dW, mu, errs, ex, exh


def fit(model, teacher_state, embed_t, seqs, layers, mu=1e-2, device="cuda", heldout=0, log=None, mus=None,
        targets=("down",)):
    """seqs: token id lists. teacher_state(l) -> state dict of the teacher's layer l (layer-relative names).
    Fits the down_proj change of every layer in `layers`; the last `heldout` sequences are only measured, and with
    `mus` they also choose each layer's ridge (closed-form held-out error) - then explained_heldout is optimistic.
    targets ("attn", "down"): first the attention output projection (o_proj / linear_attn.out_proj) is fitted so the
    stream after attention follows the teacher's (input of post_attention_layernorm), then down_proj to what is left
    at the layer output; explained_* are then shares of the layer's original mismatch, attn_* of the post-attention
    one (closed form, training / held-out).
    Returns {"delta": {l: dW (out x in, fp32, CPU)}, "delta_attn": {l: ...}, "stats": {l: {...}},
    "keys_sample": {l: first rows of K}}. The model's weights are not modified."""
    L = layers_of(model)
    fitset = set(layers)
    top = max(layers)
    ntrain = len(seqs) - heldout
    do_attn = "attn" in targets
    kws, hb, ht = [], [], []
    emb_dt = next(model.parameters()).dtype
    for n, s in enumerate(seqs):
        if n % 20 == 0:
            print(json.dumps({"prepared": n, "of": len(seqs)}), flush=True)
        kw, h0 = layer_kwargs(model, s, device)
        kws.append(kw)
        hb.append(h0)
        ht.append(embed_t[torch.tensor(s)].to(device=device, dtype=emb_dt))
    out = {"delta": {}, "delta_attn": {}, "stats": {}, "keys_sample": {}}
    for l in range(top + 1):
        t0 = time.time()
        bl = L[l]
        tl = copy.deepcopy(bl)
        tl.load_state_dict(teacher_state(l), strict=True)
        # base layer to the device by pointer, back by pointer to the untouched CPU tensors: copying it back with
        # .to("cpu") crashed natively (access violation, run cf4 layer 11), as it did in distill_train
        masters = [(p, p.data) for p in bl.parameters()] + \
                  [(m, k, b) for m in bl.modules() for k, b in m._buffers.items() if b is not None]
        for x in masters:
            if len(x) == 2:
                x[0].data = x[1].to(device)
            else:
                x[0]._buffers[x[1]] = x[2].to(device)
        tl.to(device)
        dp = bl.mlp.down_proj
        cap = {}
        delta = None
        dl = {}             # attention write of this layer, once fitted

        def hook(mod, inp, o):
            cap["K"] = inp[0]
            if delta is not None:
                return o + (inp[0].float() @ delta.T).to(o.dtype)

        def hook_a(mod, inp, o):
            cap["Ka"] = inp[0]
            if "a" in dl:
                return o + (inp[0].float() @ dl["a"].T).to(o.dtype)

        def pre(key):
            def f(mod, inp):
                cap[key] = inp[0]
            return f
        hh = [dp.register_forward_hook(hook)]
        if do_attn and l in fitset:
            hh += [attn_out(bl).register_forward_hook(hook_a),
                   bl.post_attention_layernorm.register_forward_pre_hook(pre("h1b")),
                   tl.post_attention_layernorm.register_forward_pre_hook(pre("h1t"))]
        try:
            with torch.no_grad():
                st = {}
                err_orig = None
                yt = None
                if do_attn and l in fitset:
                    # pass 0: teacher output (kept), and the attention write, fitted to the post-attention stream
                    am = attn_out(bl)
                    acc = _Acc(am.in_features, am.out_features, device, heldout > 0)
                    yt, err_orig = [], []
                    for i in range(len(seqs)):
                        y_t = _first(tl(ht[i][None], **kws[i][l]))[0]
                        y_b = _first(bl(hb[i][None], **kws[i][l]))[0]
                        err_orig.append(float((y_t - y_b).float().pow(2).sum()))
                        acc.add(cap["Ka"][0].float(), (cap["h1t"][0] - cap["h1b"][0]).float(), i < ntrain)
                        yt.append(y_t)
                    da, mu_a, errs_a, ex_a, exh_a = acc.solve(mu, mus, device)
                    del acc
                    dl["a"] = da
                    st.update({"attn_explained_train": ex_a, "attn_explained_heldout": exh_a,
                               "attn_rel_norm": float(da.norm() / am.weight.float().norm()), "attn_mu": mu_a,
                               "attn_module": attn_out_name(bl)})
                    if errs_a:
                        st["attn_heldout_err_by_mu"] = {str(k): round(v, 4) for k, v in errs_a.items()}
                    out["delta_attn"][l] = da.cpu() if str(device) == "cpu" else da.to(torch.bfloat16).cpu()
                if l in fitset:
                    n_in, n_out = dp.in_features, dp.out_features
                    G = torch.zeros(n_in, n_in, device=device)
                    P = torch.zeros(n_in, n_out, device=device)
                    sel = bool(mus) and heldout > 0
                    if sel:
                        Gh = torch.zeros(n_in, n_in, device=device)
                        Ph = torch.zeros(n_in, n_out, device=device)
                        Eh = 0.0
                have_yt = yt is not None
                yt = yt if have_yt else []
                err_old = []
                for i in range(len(seqs)):
                    y_t = yt[i] if have_yt else _first(tl(ht[i][None], **kws[i][l]))[0]
                    y_b = _first(bl(hb[i][None], **kws[i][l]))[0]
                    R = (y_t - y_b).float()
                    err_old.append(float(R.pow(2).sum()))
                    if l in fitset:
                        K = cap["K"][0].float()
                        if i < ntrain:
                            G += K.T @ K
                            P += K.T @ R
                        elif sel:
                            Gh += K.T @ K
                            Ph += K.T @ R
                            Eh += err_old[-1]
                        if i == 0:
                            out["keys_sample"][l] = K[:256].cpu()
                    if not have_yt:
                        yt.append(y_t)
                    if l not in fitset:
                        hb[i] = y_b
                if l in fitset:
                    if sel:
                        delta, mu_l, errs = solve_select(G, P, Gh, Ph, torch.tensor(Eh, device=device), mus,
                                                         device)
                        del Gh, Ph
                    else:
                        delta, mu_l, errs = solve(G, P, mu, device), mu, None
                    delta = delta.to(device=device, dtype=torch.float32)
                    del G, P
                    if str(device) != "cpu":
                        torch.cuda.empty_cache()
                    err_new = []
                    for i in range(len(seqs)):
                        y_b = _first(bl(hb[i][None], **kws[i][l]))[0]
                        err_new.append(float((yt[i] - y_b).float().pow(2).sum()))
                        hb[i] = y_b
                    W = dp.weight.float()
                    e0 = err_orig if err_orig is not None else err_old     # the layer's mismatch before any write
                    st.update({"explained_train": 1 - sum(err_new[:ntrain]) / max(sum(e0[:ntrain]), 1e-30),
                               "explained_heldout": (1 - sum(err_new[ntrain:]) / max(sum(e0[ntrain:]), 1e-30))
                               if heldout else None,
                               "rel_norm": float(delta.norm() / W.norm()),
                               "mismatch_rms": (sum(e0) / sum(len(s) for s in seqs) / W.shape[0]) ** 0.5,
                               "mu": mu_l})
                    if errs:
                        st["heldout_err_by_mu"] = {str(k): round(v, 4) for k, v in errs.items()}
                    # bf16 on the CPU for a GPU run: 32 fp32 deltas of a 9B (6.4 GB) helped run cf3 out of RAM
                    out["delta"][l] = delta.cpu() if str(device) == "cpu" else delta.to(torch.bfloat16).cpu()
                    del W
                    out["stats"][l] = st
                ht = yt
        finally:
            for h in hh:
                h.remove()
            for x in masters:
                if len(x) == 2:
                    x[0].data = x[1]
                else:
                    x[0]._buffers[x[1]] = x[2]
            del tl, masters
        rec = {"layer": l, "fit": l in fitset, **out["stats"].get(l, {}), "seconds": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    return out


_READER = None


def teacher_loader(teacher_dir):
    """Reads the teacher tensor by tensor with plain byte-range reads (engine.weight_diff.LocalSafetensors).
    safe_open per tensor crashed natively on Windows here (access violation in storage __getitem__, run cf2)."""
    global _READER
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from engine.weight_diff import LocalSafetensors
    _READER = r = LocalSafetensors(teacher_dir)
    names = r.keys()

    def get(name):
        return r.get(name)

    def state(l):
        pre = f"{PREFIX}.{l}."
        return {k[len(pre):]: get(k) for k in names if k.startswith(pre)}
    return state, get


def main():
    import faulthandler
    faulthandler.enable()          # a native crash (no Python traceback) killed the first 9B run silently
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from engine.lora_export import lowrank
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("teacher")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--mu", type=float, default=1e-2)
    ap.add_argument("--mus", default="0.01,0.1,1,10", help="ridge grid chosen per layer on the held-out sequences")
    ap.add_argument("--heldout", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2600)
    ap.add_argument("--targets", default="down", help="down | attn,down (attention output projection first)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    tok = AutoTokenizer.from_pretrained(a.base)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = [json.loads(x) for x in Path(a.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    seqs = []
    for r in rows:
        if r["ok"]:
            s = tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"] + [end]
            if len(s) <= a.max_len:
                seqs.append(s)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).eval()
    state, get = teacher_loader(a.teacher)
    embed_t = get("model.language_model.embed_tokens.weight")
    n = len(layers_of(model))
    print(json.dumps({"seqs": len(seqs), "tokens": sum(map(len, seqs)), "heldout": a.heldout, "layers": n,
                      "mu": a.mu}), flush=True)
    out.mkdir(parents=True)
    t0 = time.time()
    res = fit(model, state, embed_t, seqs, layers=list(range(n)), mu=a.mu, device="cuda", heldout=a.heldout, mus=[float(x) for x in a.mus.split(",")] if a.mus else None,
              log=out / "fit.jsonl", targets=tuple(a.targets.split(",")))
    names = {l: attn_out_name(m) for l, m in enumerate(layers_of(model))}
    del model
    T, energy = {}, {}
    writes = [(f"{l}.mlp.down_proj", d) for l, d in res["delta"].items()] + \
             [(f"{l}.{names[l]}", d) for l, d in res["delta_attn"].items()]
    for key, d in writes:
        A, B, kept = lowrank(d, a.rank)
        stem = f"base_model.model.{PREFIX}.{key}"
        T[stem + ".lora_A.weight"] = A.to(torch.bfloat16).contiguous()
        T[stem + ".lora_B.weight"] = B.to(torch.bfloat16).contiguous()
        energy[key] = kept
    save_file(T, str(out / "adapter_model.safetensors"))
    mods = ["down_proj"] + (["o_proj", "out_proj"] if res["delta_attn"] else [])
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.rank, "lora_dropout": 0.0,
           "bias": "none", "target_modules": mods, "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "fit_summary.json").write_text(json.dumps({"stats": res["stats"], "kept_at_rank": energy,
                                                      "seconds": round(time.time() - t0)}, indent=1))
    print(json.dumps({"out": str(out), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
