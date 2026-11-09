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


def fit(model, teacher_state, embed_t, seqs, layers, mu=1e-2, device="cuda", heldout=0, log=None, mus=None):
    """seqs: token id lists. teacher_state(l) -> state dict of the teacher's layer l (layer-relative names).
    Fits the down_proj change of every layer in `layers`; the last `heldout` sequences are only measured, and with
    `mus` they also choose each layer's ridge (closed-form held-out error) - then explained_heldout is optimistic.
    Returns {"delta": {l: dW (out x in, fp32, CPU)}, "stats": {l: {...}}, "keys_sample": {l: first rows of K}}.
    The model's weights are not modified."""
    L = layers_of(model)
    fitset = set(layers)
    top = max(layers)
    ntrain = len(seqs) - heldout
    kws, hb, ht = [], [], []
    emb_dt = next(model.parameters()).dtype
    for n, s in enumerate(seqs):
        if n % 20 == 0:
            print(json.dumps({"prepared": n, "of": len(seqs)}), flush=True)
        kw, h0 = layer_kwargs(model, s, device)
        kws.append(kw)
        hb.append(h0)
        ht.append(embed_t[torch.tensor(s)].to(device=device, dtype=emb_dt))
    out = {"delta": {}, "stats": {}, "keys_sample": {}}
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

        def hook(mod, inp, o):
            cap["K"] = inp[0]
            if delta is not None:
                return o + (inp[0].float() @ delta.T).to(o.dtype)
        hh = dp.register_forward_hook(hook)
        try:
            with torch.no_grad():
                if l in fitset:
                    n_in, n_out = dp.in_features, dp.out_features
                    G = torch.zeros(n_in, n_in, device=device)
                    P = torch.zeros(n_in, n_out, device=device)
                    sel = bool(mus) and heldout > 0
                    if sel:
                        Gh = torch.zeros(n_in, n_in, device=device)
                        Ph = torch.zeros(n_in, n_out, device=device)
                        Eh = 0.0
                yt, err_old = [], []
                for i in range(len(seqs)):
                    y_t = _first(tl(ht[i][None], **kws[i][l]))[0]
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
                    st = {"explained_train": 1 - sum(err_new[:ntrain]) / max(sum(err_old[:ntrain]), 1e-30),
                          "explained_heldout": (1 - sum(err_new[ntrain:]) / max(sum(err_old[ntrain:]), 1e-30))
                          if heldout else None,
                          "rel_norm": float(delta.norm() / W.norm()),
                          "mismatch_rms": (sum(err_old) / sum(len(s) for s in seqs) / W.shape[0]) ** 0.5,
                          "mu": mu_l}
                    if errs:
                        st["heldout_err_by_mu"] = {str(k): round(v, 4) for k, v in errs.items()}
                    # bf16 on the CPU for a GPU run: 32 fp32 deltas of a 9B (6.4 GB) helped run cf3 out of RAM
                    out["delta"][l] = delta.cpu() if str(device) == "cpu" else delta.to(torch.bfloat16).cpu()
                    del W
                    out["stats"][l] = st
                ht = yt
        finally:
            hh.remove()
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
              log=out / "fit.jsonl")
    del model
    T, energy = {}, {}
    for l, d in res["delta"].items():
        A, B, kept = lowrank(d, a.rank)
        stem = f"base_model.model.{PREFIX}.{l}.mlp.down_proj"
        T[stem + ".lora_A.weight"] = A.to(torch.bfloat16).contiguous()
        T[stem + ".lora_B.weight"] = B.to(torch.bfloat16).contiguous()
        energy[l] = kept
    save_file(T, str(out / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.rank, "lora_dropout": 0.0,
           "bias": "none", "target_modules": ["down_proj"], "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "fit_summary.json").write_text(json.dumps({"stats": res["stats"], "kept_at_rank": energy,
                                                      "seconds": round(time.time() - t0)}, indent=1))
    print(json.dumps({"out": str(out), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
