"""A llama.cpp control vector compiled into the weights (WSP-Lang op compile_control_vector; plan
lab/probes/cc_weights/OP_PLAN.md; the run it reproduces: results §43.19, code lab/probes/cc_weights/compile.py).

Per edited layer N with increment v_N (llama.cpp adds it at layer N's output): dW_N = v_N u_N^T in mlp.down_proj, whose
output goes straight into the residual stream, so the edit adds v_N (u_N^T x) at every token. u_N = argmin
sum_t (u^T x_t - 1)^2 + lam * mean diag(K^T K) ||u||^2 over the down_proj inputs K of the fit split, taken from the
stream with layers < N already edited. Closed form, no gradients, nothing added to the model.

Every run fits the vector and a same-norm random placebo. G0 on held-out tokens (median over layers of the per-layer
median |u^T x - 1| <= 0.5) decides whether anything but the report is written. Exports: rank-1 PEFT LoRA per arm
(+ GGUF adapter with llama.cpp's converter), merged Q6_K copies of a base GGUF. Benchmarks are not run here.
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LAM, SEED, HELDOUT, MAX_LEN = 1e-3, 20261111, 28, 4096
NOTE = ("Benchmarks are not run by this op. The effect must be shown on a sealed set with the placebo arm "
        "(plan fixed before the run).")


# ---- helpers shared with lab/probes/close_probe/cf_distill.py (copied: engine code does not import lab code) ----
def text_model(model):
    m = model.model
    return m.language_model if hasattr(m, "language_model") else m


def layers_of(model):
    return text_model(model).layers


def solve(G, P, mu, device="cpu"):
    """G = K^T K, P = K^T R -> (K^T K + mu mean(diag) I)^-1 K^T R, transposed. Float32 Cholesky off the CPU (float64
    CPU fallback), float64 LU on the CPU. Same code as cf_distill.solve, so the 9B run is reproduced bit for bit."""
    if str(device) != "cpu":
        Gf = G.to(device=device, dtype=torch.float32)
        M = (Gf + Gf.T) / 2
        M.diagonal().add_(mu * M.diagonal().mean())
        L, info = torch.linalg.cholesky_ex(M)
        del M
        if int(info) == 0:
            return torch.cholesky_solve(P.to(device=device, dtype=torch.float32), L).T.contiguous()
    G = G.double().cpu()
    G = (G + G.T) / 2
    M = G + mu * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype)
    return torch.linalg.solve(M, P.double().cpu()).T.float().contiguous()


class _Stub(torch.nn.Module):
    def __init__(self, i, rec):
        super().__init__()
        self.i, self.rec = i, rec

    def forward(self, hidden_states, **kw):
        self.rec[self.i] = kw
        if self.i == 0:
            self.rec["h0"] = hidden_states
        return hidden_states


def layer_kwargs(model, ids, device):
    """Per-layer call arguments (masks, rotary embeddings) and the embedding, without running any layer."""
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
    memo = {}

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


def _mv(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(_mv(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _mv(v, dev) for k, v in x.items()}
    return x


# ---- pieces ----
def ridge_u(G, P, lam, device="cpu"):
    """u = (G + lam mean(diag G) I)^-1 P for P = K^T 1: the u with u^T x ~ 1 on the rows of K. -> (n_in,)"""
    return solve(G, P, lam, device)[0].float()


def g0(layers):
    """G0 on held-out tokens: median over layers of each layer's median |u^T x - 1|; passes at <= 0.5."""
    errs = sorted(float(v["heldout_median_abs_err"]) for v in layers.values())
    n = len(errs)
    med = errs[n // 2] if n % 2 else (errs[n // 2 - 1] + errs[n // 2]) / 2
    return {"median": med, "worst": errs[-1], "threshold": 0.5, "pass": med <= 0.5}


def split_rows(rows, heldout, seed):
    """Fixed split: shuffle the indices with random.Random(seed); the last `heldout` are held out."""
    if not 0 < heldout < len(rows):
        raise ValueError(f"heldout must be in 1..{len(rows) - 1} for {len(rows)} calibration rows")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_fit = len(rows) - heldout
    return [rows[i] for i in order[:n_fit]], [rows[i] for i in order[n_fit:]]


def calib_ids(tok, rows, template="think", max_len=MAX_LEN):
    """Rows {"prompt", "reasoning"} -> token ids of chat template + the model's own text (thinking template by default)."""
    if template not in ("think", "plain"):
        raise ValueError("template must be think or plain")
    out = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=template == "think")
        out.append(tok(p + r["reasoning"].lstrip("\n"), add_special_tokens=False)["input_ids"][:max_len])
    return out


def read_vector(path):
    import gc
    import gguf
    r = gguf.GGUFReader(str(path))
    v = {int(t.name.split(".")[1]): torch.tensor(np.array(t.data, dtype=np.float32)) for t in r.tensors
         if t.name.startswith("direction.")}
    del r
    gc.collect()                       # the reader memory-maps the file; Windows keeps it locked while mapped
    return v


def down_proj_name(model, l):
    target = layers_of(model)[l].mlp.down_proj
    for n, m in model.named_modules():
        if m is target:
            return n
    raise ValueError(f"no down_proj found for layer {l}")


def check_vector(model, vecs):
    L = layers_of(model)
    for l, v in vecs.items():
        if not 0 <= l < len(L):
            raise ValueError(f"vector has layer {l}; the model has layers 0..{len(L) - 1}")
        d = L[l].mlp.down_proj.out_features
        if v.numel() != d:
            raise ValueError(f"vector layer {l} has {v.numel()} values; the model's down_proj writes {d}")


def fit_layers(model, fit_ids, ho_ids, vecs, lam=LAM, device="cpu", log=None):
    """Layer by layer, bottom-up, each block moved to `device` only for its own pass (as compile.py). The model's
    parameters are left untouched: the edit lives in a forward hook while the stream is carried to the next layer.
    -> ({layer: u}, {layer: stats})"""
    check_vector(model, vecs)
    L = layers_of(model)
    seqs = list(fit_ids) + list(ho_ids)
    n_fit = len(fit_ids)
    kws, hs = [], []
    for s in seqs:
        kw, h0 = layer_kwargs(model, s, "cpu")
        kws.append(kw)
        hs.append(h0)
    U, summ = {}, {}
    for l in range(len(L)):
        t0 = time.time()
        bl = L[l]
        masters = [(p, p.data) for p in bl.parameters()] + \
                  [(m, k, b) for m in bl.modules() for k, b in m._buffers.items() if b is not None]
        for x in masters:
            if len(x) == 2:
                x[0].data = x[1].to(device)
            else:
                x[0]._buffers[x[1]] = x[2].to(device)
        cap, edit, st = {}, {}, {}

        def hook(mod, inp, o):
            cap["K"] = inp[0]
            if "u" in edit:
                c = inp[0].float() @ edit["u"]
                return o + (c[..., None] * edit["v"]).to(o.dtype)
        hh = bl.mlp.down_proj.register_forward_hook(hook)
        try:
            with torch.no_grad():
                if l in vecs:
                    n_in = bl.mlp.down_proj.in_features
                    G = torch.zeros(n_in, n_in, device=device)
                    P = torch.zeros(n_in, 1, device=device)
                    for i in range(n_fit):
                        _first(bl(hs[i].to(device)[None], **_mv(kws[i][l], device)))
                        K = cap["K"][0].float()
                        G += K.T @ K
                        P += K.sum(0)[:, None]
                    u = ridge_u(G, P, lam, device)
                    del G, P
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    edit.update(u=u, v=vecs[l].float().to(device))
                    U[l] = u.cpu()
                c_tr, c_ho = [], []
                for i in range(len(seqs)):
                    y = _first(bl(hs[i].to(device)[None], **_mv(kws[i][l], device)))[0]
                    if l in vecs:
                        c = (cap["K"][0].float() @ edit["u"]).cpu()
                        (c_tr if i < n_fit else c_ho).append(c)
                    hs[i] = y.cpu()
                if l in vecs:
                    ct, ch = torch.cat(c_tr), torch.cat(c_ho)
                    st = {"layer": l, "v_norm": round(float(vecs[l].norm()), 4), "u_norm": float(u.norm()),
                          "rel_norm_dW": float(vecs[l].norm() * u.norm() / bl.mlp.down_proj.weight.float().norm()),
                          "train_median_abs_err": float((ct - 1).abs().median()),
                          "heldout_median_abs_err": float((ch - 1).abs().median()),
                          "heldout_mean_c": float(ch.mean()), "heldout_std_c": float(ch.std()),
                          "heldout_p10_c": float(ch.quantile(0.1)), "heldout_p90_c": float(ch.quantile(0.9))}
                    summ[l] = st
        finally:
            hh.remove()
            for x in masters:
                if len(x) == 2:
                    x[0].data = x[1]
                else:
                    x[0]._buffers[x[1]] = x[2]
        if log:
            log({"layer": l, **st, "seconds": round(time.time() - t0, 1)})
    return U, summ


def lora_tensors(model, U, V):
    """Rank-1 PEFT LoRA on down_proj: A = u^T (1 x in), B = v (out x 1), alpha = r = 1. -> (tensors, adapter_config)"""
    T = {}
    for l, u in U.items():
        stem = "base_model.model." + down_proj_name(model, l)
        T[stem + ".lora_A.weight"] = u[None, :].to(torch.bfloat16).contiguous()
        T[stem + ".lora_B.weight"] = V[l].float()[:, None].to(torch.bfloat16).contiguous()
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1, "lora_dropout": 0.0, "bias": "none",
           "target_modules": ["down_proj"], "fan_in_fan_out": False}
    return T, cfg


def merge_q6k(base_gguf, U, V, dst):
    """Copy of base_gguf with blk.N.ffn_down = Q6_K requant(W + v u^T), written as dst.partial and renamed. Only Q6_K
    ffn_down tensors are accepted (checked before anything is written). -> per-layer stats of the written change."""
    import gguf
    sys.path.insert(0, str(ROOT))
    import wsp
    base_gguf, dst = Path(base_gguf), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; merged models are never overwritten")
    r = gguf.GGUFReader(str(base_gguf))
    tm = {t.name: t for t in r.tensors}
    for l in U:
        t = tm.get(f"blk.{l}.ffn_down.weight")
        if t is None:
            raise ValueError(f"{base_gguf.name} has no blk.{l}.ffn_down.weight")
        if t.tensor_type != gguf.GGMLQuantizationType.Q6_K:
            raise ValueError(f"blk.{l}.ffn_down.weight is {t.tensor_type.name}; only Q6_K is supported")
    tmp = Path(str(dst) + ".partial")
    shutil.copyfile(base_gguf, tmp)
    stats = {}
    with open(tmp, "r+b") as f:
        for l in sorted(U):
            t0 = time.time()
            t = tm[f"blk.{l}.ffn_down.weight"]
            W = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)          # out x in
            dW = np.outer(V[l].float().numpy(), U[l].float().numpy()).astype(np.float32)
            if W.shape != dW.shape:
                raise ValueError(f"layer {l}: ffn_down is {W.shape}, the edit is {dW.shape}")
            nb = W.shape[1] // 256
            new = wsp._requant_blocks_bytes(t, {row: list(range(nb)) for row in range(W.shape[0])}, W + dW)
            assert len(new) == t.n_bytes
            f.seek(t.data_offset)
            f.write(new)
            Wq = gguf.dequantize(np.frombuffer(new, dtype=np.uint8).reshape(t.data.shape), t.tensor_type)
            got = Wq - W
            stats[l] = {"dW_norm": float(np.linalg.norm(dW)),
                        "rel_err_written_vs_dW": float(np.linalg.norm(got - dW) / np.linalg.norm(dW)),
                        "cos_written_dW": float((got * dW).sum() / (np.linalg.norm(got) * np.linalg.norm(dW))),
                        "seconds": round(time.time() - t0, 1)}
    del r, tm
    import gc
    gc.collect()
    tmp.rename(dst)
    return stats


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _arm(model, fit_ids, ho_ids, vecs, d, lam, device, source):
    d.mkdir(parents=True)

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(d / "fit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    t0 = time.time()
    U, summ = fit_layers(model, fit_ids, ho_ids, vecs, lam=lam, device=device, log=log)
    torch.save({"u": U, "v": {l: vecs[l] for l in U}, "vector": str(source)}, d / "fit.pt")
    gate = g0(summ)
    (d / "fit_summary.json").write_text(json.dumps({"vector": str(source), "lam": lam, "fit_seqs": len(fit_ids),
                                                    "heldout_seqs": len(ho_ids), "layers": summ, "G0": gate,
                                                    "seconds": round(time.time() - t0)}, indent=1))
    return U, summ, gate


def _write_lora(model, U, V, d, lora_base):
    from safetensors.torch import save_file
    peft = d / "peft"
    peft.mkdir()
    T, cfg = lora_tensors(model, U, V)
    save_file(T, str(peft / "adapter_model.safetensors"))
    (peft / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    res = {"peft": str(peft)}
    conv = os.environ.get("WSP_LORA_CONVERTER")
    if lora_base and conv:
        out = d / "lora_r1.gguf"
        cmd = [os.environ.get("WSP_CONVERT_PYTHON", sys.executable), conv, str(peft), "--base", str(lora_base),
               "--outtype", "f32", "--outfile", str(out)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode:
            raise RuntimeError(f"convert_lora_to_gguf failed: {cp.stderr[-800:]}")
        res["gguf"] = str(out)
    return res


def _report_md(rep):
    g, lines = rep["G0"], []
    lines.append(f"# compile_control_vector report\n\n{rep['note']}\n")
    lines.append(f"- vector: `{rep['inputs'].get('vector')}`\n- G0 (held-out median of medians |u^T x - 1|): "
                 f"{g['median']:.4f}, worst layer {g['worst']:.4f}, threshold 0.5 -> **{'PASS' if g['pass'] else 'FAIL'}**")
    if "placebo" in rep:
        pg = rep["placebo"]["G0"]
        lines.append(f"- placebo G0: {pg['median']:.4f}, worst {pg['worst']:.4f}")
    lines.append("\n| layer | v norm | ||dW||/||W|| | held-out median err | merge cos | merge rel err |\n|---|---|---|---|---|---|")
    ms = rep["vector"].get("merge", {}).get("layers", {})
    for l, s in sorted(rep["vector"]["layers"].items(), key=lambda kv: int(kv[0])):
        m = ms.get(l) or ms.get(int(l)) or {}
        lines.append(f"| {l} | {s['v_norm']:.3f} | {s['rel_norm_dW']:.4f} | {s['heldout_median_abs_err']:.4f} | "
                     f"{m.get('cos_written_dW', float('nan')):.3f} | {m.get('rel_err_written_vs_dW', float('nan')):.3f} |")
    lines.append("\nFiles: " + ", ".join(f"`{x}`" for x in rep["files"]))
    return "\n".join(lines) + "\n"


def compile_from_ids(model, fit_ids, ho_ids, vecs, out, placebo=None, placebo_seed=2026, lam=LAM, device="cpu",
                     base_gguf=None, lora=True, lora_base=None, meta=None, source="vector"):
    """The pipeline on token ids: fit vector -> G0 -> (stop) | placebo fit, LoRA, merges -> report.json/.md."""
    from engine.wsplang import WriteGuardError
    from engine import contrast_vector as cvx
    out = Path(out)
    if out.exists():
        raise WriteGuardError(f"compile_control_vector: {out} exists; outputs are never overwritten")
    check_vector(model, vecs)
    out.mkdir(parents=True)
    t0 = time.time()
    rep = {"note": NOTE, "inputs": dict(meta or {}), "lam": lam,
           "calib": {"fit": len(fit_ids), "heldout": len(ho_ids), "tokens": sum(map(len, fit_ids)) + sum(map(len, ho_ids))},
           "files": []}
    U, summ, gate = _arm(model, fit_ids, ho_ids, vecs, out / "vector", lam, device, source)
    rep["G0"] = gate
    rep["vector"] = {"layers": summ, "fit": str(out / "vector" / "fit.pt")}
    rep["files"].append(str(out / "vector" / "fit.pt"))
    if gate["pass"]:
        if placebo is None:
            pv = cvx.placebo({l: v.float().numpy() for l, v in vecs.items()}, placebo_seed)
            placebo = {l: torch.tensor(v) for l, v in pv.items()}
            cvx.write_gguf(pv, out / f"placebo{placebo_seed}.gguf", (meta or {}).get("model_hint", "qwen35"))
            rep["placebo_source"] = f"generated, seed {placebo_seed}"
            rep["files"].append(str(out / f"placebo{placebo_seed}.gguf"))
        else:
            rep["placebo_source"] = (meta or {}).get("placebo", "given")
        Up, summ_p, gate_p = _arm(model, fit_ids, ho_ids, placebo, out / "placebo", lam, device, "placebo")
        rep["placebo"] = {"layers": summ_p, "G0": gate_p, "fit": str(out / "placebo" / "fit.pt")}
        rep["files"].append(str(out / "placebo" / "fit.pt"))
        arms = (("vector", U, vecs), ("placebo", Up, placebo))
        if lora:
            for name, Ua, Va in arms:
                rep[name]["lora"] = _write_lora(model, Ua, Va, out / name, lora_base)
                rep["files"] += list(rep[name]["lora"].values())
        if base_gguf:
            for name, Ua, Va in arms:
                dst = out / ("merged-Q6_K.gguf" if name == "vector" else "placebo_merged-Q6_K.gguf")
                st = merge_q6k(base_gguf, Ua, Va, dst)
                rep[name]["merge"] = {"file": str(dst), "layers": st,
                                      "cos_written_dW_min": min(s["cos_written_dW"] for s in st.values()),
                                      "cos_written_dW_median": _median([s["cos_written_dW"] for s in st.values()]),
                                      "rel_err_median": _median([s["rel_err_written_vs_dW"] for s in st.values()])}
                (out / name / "merge_summary.json").write_text(json.dumps(rep[name]["merge"], indent=1))
                rep["files"].append(str(dst))
    rep["seconds"] = round(time.time() - t0)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "report.md").write_text(_report_md(json.loads(json.dumps(rep, default=str))), encoding="utf-8")
    return rep


def compile_control_vector(model_dir, vector, calib, out, placebo=None, placebo_seed=2026, base_gguf=None, lora=True,
                           lam=LAM, heldout=HELDOUT, seed=SEED, max_len=MAX_LEN, template="think", device=None):
    """Load the model and calibration traces, then compile_from_ids. Defaults are the §43.19 run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, vector, calib, out = Path(model_dir), Path(vector), Path(calib), Path(out)
    vecs = read_vector(vector)
    pl = read_vector(placebo) if placebo else None
    rows = [json.loads(x) for x in calib.read_text(encoding="utf-8").splitlines() if x.strip()]
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    fit_ids, ho_ids = split_rows(calib_ids(tok, rows, template, max_len), heldout, seed)
    meta = {"model": str(model_dir), "vector": str(vector), "vector_sha256": sha256(vector), "calib": str(calib),
            "calib_sha256": sha256(calib), "split_seed": seed, "template": template, "max_len": max_len,
            "device": device}
    if placebo:
        meta.update(placebo=str(placebo), placebo_sha256=sha256(placebo))
    if base_gguf:
        meta.update(base_gguf=str(base_gguf), base_gguf_sha256=sha256(base_gguf))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval()
    return compile_from_ids(model, fit_ids, ho_ids, vecs, out, placebo=pl, placebo_seed=placebo_seed, lam=lam,
                            device=device, base_gguf=base_gguf, lora=lora, lora_base=model_dir, meta=meta,
                            source=vector)
