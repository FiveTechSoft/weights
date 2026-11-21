"""Compile a llama.cpp control vector into mlp.down_proj as a rank-1 closed-form write (PLAN.md, fixed before running).

Per layer N with an increment v_N (added by llama.cpp at layer N's output): dW_N = v_N u_N^T, where
u_N = argmin sum_t (u^T x_t - 1)^2 + lam ||u||^2, lam = 1e-3 * mean diag(K^T K), K = inputs of that layer's down_proj on
the calibration tokens, taken from the model with layers < N already edited. down_proj's output goes straight into the
residual stream, so the edit adds v_N * (u_N^T x) at every token: the vector, wherever u_N^T x ~ 1.

subcommands:
  fit <vector.gguf> <out_dir>      -> out_dir/{fit.pt, fit_summary.json, fit.jsonl}; G0 numbers on held-out traces
  lora <out_dir>                   -> out_dir/peft/ (rank-1 PEFT LoRA) and out_dir/lora_r1.gguf (needs WSP_LORA_CONVERTER)
  merge <out_dir> <merged.gguf>    -> copy of A0-Q6_K.gguf with ffn_down = requant(W + dW); requant error per layer
  gate <out_dir>                   -> G0 verdict from fit_summary.json (+ merge stats if present)
"""
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
BASE_HF = ROOT / "hf_cache" / "qwen35-9b-hf"
BASE_GGUF = ROOT / "hf_cache" / "h4" / "A0-Q6_K.gguf"
TRACES = ROOT / "lab" / "runs" / "cc" / "traces.jsonl"
SPLIT_SEED, N_FIT, MAX_LEN, LAM = 20261111, 100, 4096, 1e-3
PREFIX = "model.language_model.layers"


def read_vector(path):
    import gguf
    r = gguf.GGUFReader(str(path))
    return {int(t.name.split(".")[1]): torch.tensor(np.array(t.data, dtype=np.float32)) for t in r.tensors}


def sequences(tok, traces_path=None, n_fit=N_FIT, max_len=MAX_LEN, split_seed=SPLIT_SEED):
    rows = [json.loads(x) for x in (Path(traces_path or TRACES)).read_text(encoding="utf-8").splitlines() if x.strip()]
    seqs = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        ids = tok(p + r["reasoning"].lstrip("\n"), add_special_tokens=False)["input_ids"][:max_len]
        seqs.append(ids)
    order = list(range(len(seqs)))
    random.Random(split_seed).shuffle(order)
    return [seqs[i] for i in order[:n_fit]], [seqs[i] for i in order[n_fit:]]


def _mv(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(_mv(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _mv(v, dev) for k, v in x.items()}
    return x


def fit(vec_path, out, base_hf=None, traces=None, lam=LAM, n_fit=N_FIT, max_len=MAX_LEN,
        split_seed=SPLIT_SEED, device=None):
    import faulthandler
    faulthandler.enable()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cf_distill import layer_kwargs, layers_of, solve, _first
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    vecs = read_vector(vec_path)
    tok = AutoTokenizer.from_pretrained(str(base_hf or BASE_HF))
    tr, ho = sequences(tok, traces, n_fit, max_len, split_seed)
    seqs = tr + ho
    model = AutoModelForCausalLM.from_pretrained(str(base_hf or BASE_HF), dtype=torch.bfloat16).eval()
    L = layers_of(model)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"fit_seqs": len(tr), "heldout_seqs": len(ho), "tokens": sum(map(len, seqs)),
                      "layers_edited": sorted(vecs)}), flush=True)
    kws, hs = [], []
    for s in seqs:
        kw, h0 = layer_kwargs(model, s, "cpu")
        kws.append(kw)
        hs.append(h0)
    U, summ = {}, {}
    t_all = time.time()
    for l in range(len(L)):
        t0 = time.time()
        bl = L[l]
        masters = [(p, p.data) for p in bl.parameters()] + \
                  [(m, k, b) for m in bl.modules() for k, b in m._buffers.items() if b is not None]
        for x in masters:
            if len(x) == 2:
                x[0].data = x[1].to(dev)
            else:
                x[0]._buffers[x[1]] = x[2].to(dev)
        cap, st = {}, {}
        edit = {}

        def hook(mod, inp, o):
            cap["K"] = inp[0]
            if "u" in edit:
                c = inp[0].float() @ edit["u"]                       # (1, n)
                return o + (c[..., None] * edit["v"]).to(o.dtype)
        hh = bl.mlp.down_proj.register_forward_hook(hook)
        try:
            with torch.no_grad():
                if l in vecs:
                    n_in = bl.mlp.down_proj.in_features
                    G = torch.zeros(n_in, n_in, device=dev)
                    P = torch.zeros(n_in, 1, device=dev)
                    for i in range(len(tr)):
                        _first(bl(hs[i].to(dev)[None], **_mv(kws[i][l], dev)))
                        K = cap["K"][0].float()
                        G += K.T @ K
                        P += K.sum(0)[:, None]
                    u = solve(G, P, LAM, dev)[0].float()                 # (n_in,)
                    del G, P
                    torch.cuda.empty_cache()
                    edit.update(u=u, v=vecs[l].to(dev))
                    U[l] = u.cpu()
                c_tr, c_ho = [], []
                for i in range(len(seqs)):
                    y = _first(bl(hs[i].to(dev)[None], **_mv(kws[i][l], dev)))[0]
                    if l in vecs:
                        c = (cap["K"][0].float() @ edit["u"]).cpu()
                        (c_tr if i < len(tr) else c_ho).append(c)
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
        rec = {"layer": l, **st, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with open(out / "fit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    torch.save({"u": U, "v": {l: vecs[l] for l in U}, "vector": str(vec_path)}, out / "fit.pt")
    (out / "fit_summary.json").write_text(json.dumps({"vector": str(vec_path), "lam": lam, "split_seed": split_seed,
                                                      "fit_seqs": len(tr), "heldout_seqs": len(ho),
                                                      "layers": summ, "seconds": round(time.time() - t_all)},
                                                     indent=1))


def lora(out, base_hf=None):
    import os
    import subprocess
    from safetensors.torch import save_file
    out = Path(out)
    d = torch.load(out / "fit.pt")
    peft = out / "peft"
    peft.mkdir()
    T = {}
    for l, u in d["u"].items():
        stem = f"base_model.model.{PREFIX}.{l}.mlp.down_proj"
        T[stem + ".lora_A.weight"] = u[None, :].to(torch.bfloat16).contiguous()
        T[stem + ".lora_B.weight"] = d["v"][l][:, None].to(torch.bfloat16).contiguous()
    save_file(T, str(peft / "adapter_model.safetensors"))
    (peft / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 1, "lora_alpha": 1, "lora_dropout": 0.0, "bias": "none",
         "target_modules": ["down_proj"], "fan_in_fan_out": False}, indent=1))
    cmd = [os.environ.get("WSP_CONVERT_PYTHON", sys.executable), os.environ["WSP_LORA_CONVERTER"], str(peft),
           "--base", str(base_hf or BASE_HF), "--outtype", "f32", "--outfile", str(out / "lora_r1.gguf")]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        raise SystemExit(cp.stderr[-1500:])
    print(json.dumps({"lora": str(out / "lora_r1.gguf")}))


def merge(out, merged, base_gguf=None):
    import gguf
    import wsp
    out, merged = Path(out), Path(merged)
    if merged.exists():
        raise SystemExit(f"{merged} exists")
    d = torch.load(out / "fit.pt")
    r = gguf.GGUFReader(str(base_gguf or BASE_GGUF))
    tm = {t.name: t for t in r.tensors}
    tmp = merged.with_suffix(".partial")
    shutil.copyfile(BASE_GGUF, tmp)
    stats = {}
    with open(tmp, "r+b") as f:
        for l in sorted(d["u"]):
            t0 = time.time()
            t = tm[f"blk.{l}.ffn_down.weight"]
            assert t.tensor_type == gguf.GGMLQuantizationType.Q6_K
            W = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)          # (4096, 12288) = out x in
            dW = np.outer(d["v"][l].numpy(), d["u"][l].numpy()).astype(np.float32)
            assert W.shape == dW.shape, (W.shape, dW.shape)
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
            print(json.dumps({"layer": l, **stats[l]}), flush=True)
    tmp.rename(merged)
    (out / "merge_summary.json").write_text(json.dumps({"merged": str(merged), "layers": stats}, indent=1))


def gate(out):
    out = Path(out)
    s = json.loads((out / "fit_summary.json").read_text())
    errs = sorted(v["heldout_median_abs_err"] for v in s["layers"].values())
    med = errs[len(errs) // 2] if len(errs) % 2 else (errs[len(errs) // 2 - 1] + errs[len(errs) // 2]) / 2
    res = {"median_over_layers_of_heldout_median_abs_err": med, "worst_layer": max(errs), "G0": med <= 0.5}
    m = out / "merge_summary.json"
    if m.exists():
        ms = json.loads(m.read_text())["layers"]
        res["merge_cos_written_dW_min"] = min(v["cos_written_dW"] for v in ms.values())
        res["merge_rel_err_median"] = sorted(v["rel_err_written_vs_dW"] for v in ms.values())[len(ms) // 2]
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    {"fit": lambda: fit(a[1], a[2]), "lora": lambda: lora(a[1]), "merge": lambda: merge(a[1], a[2]),
     "gate": lambda: gate(a[1])}[a[0]]()
