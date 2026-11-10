"""A fine-tune's change as a reusable LoRA adapter (WSPLang op delta_to_lora).

D = W_model - W_base per linear matrix of the text decoder layers, truncated SVD to rank r, written as a PEFT adapter
(lora_alpha = r, so scale 1 means "the rank-r part of the whole change"). llama.cpp's convert_lora_to_gguf.py turns it
into a GGUF adapter that llama-server applies on the base with a per-request scale (--lora-init-without-apply and
"lora": [{"id": 0, "scale": s}]), so one adapter gives every alpha without rebuilding a model.

energy.json records, per matrix, the share of ||D||^2 the rank keeps: read it before evaluating. Measured on Qwen3.5-9B
(results.md §43.4): Jackrong's distill keeps 99.97 % at rank 64 and its rank-16 adapter at scale 0.5 reproduces the
improvement; MiMo's full fine-tune keeps 29 % and its rank-64 adapter does not.

1-D tensors (norms, A_log, dt_bias) and embed / lm_head are not in a LoRA. Names are matched across checkpoints with
weight_diff.text_model_name; the adapter is written with the base's own names (what the converter maps against).
"""
import json
import re
from pathlib import Path

import torch

TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
KINDS = ("mlp", "self_attn", "linear_attn")
PAT = re.compile(r"^model\.layers\.\d+\.(%s)\.(%s)\.weight$" % ("|".join(KINDS), "|".join(TARGETS)))


def _device():
    """cuda when it has room for a 12288 x 4096 SVD (~1.5 GB), else cpu."""
    if torch.cuda.is_available():
        try:
            return "cuda" if torch.cuda.mem_get_info()[0] > 2 * 2**30 else "cpu"
        except RuntimeError:
            return "cpu"
    return "cpu"


RANKS_AT = (1, 4, 16, 64, 256, 512, 1024)


def lowrank(D, r, device=None, curve=None):
    """A (r x in), B (out x r) with B @ A the best rank-r approximation of D (exact SVD); kept = share of ||D||^2.
    curve: optional dict filled with {k: share kept at rank k} for k in RANKS_AT (k <= min(D.shape)).

    Exact, not torch.svd_lowrank: fine-tune deltas have flat spectra and randomized SVD with q = r + 8 missed the true
    rank-16 truncation by 11-16 % on Qwen3.5-9B attention matrices (results §43.6)."""
    r = min(r, min(D.shape))
    X = D.float().to(device or _device())
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    tot = float(S.pow(2).sum())
    if curve is not None:
        c = S.pow(2).cumsum(0)
        curve.update({str(k): round(float(c[k - 1]) / tot, 4) if tot > 0 else 1.0 for k in RANKS_AT if k <= len(S)})
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    kept = float(S.pow(2).sum()) / tot if tot > 0 else 1.0
    sq = S.sqrt()
    return (Vh * sq[:, None]).contiguous().cpu(), (U * sq).contiguous().cpu(), min(kept, 1.0)


def wanted(canonical, kinds=None):
    m = PAT.match(canonical)
    return bool(m) and (kinds is None or m.group(1) in kinds)


def peft_names(original):
    stem = "base_model.model." + original[: -len(".weight")]
    return stem + ".lora_A.weight", stem + ".lora_B.weight"


def export_lora(base_r, model_r, rank, out_dir, kinds=None, base_label=""):
    """base_r / model_r: weight_diff.Renamed readers (canonical names, .original(name) gives the stored name)."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    names = [n for n in sorted(set(base_r.keys()) & set(model_r.keys()))
             if wanted(n, kinds) and tuple(base_r.shape(n)) == tuple(model_r.shape(n))]
    if not names:
        raise ValueError("no linear decoder matrix in common" + (f" for kinds={sorted(kinds)}" if kinds else ""))
    T, stats = {}, {}
    for n in names:
        D = model_r.get(n).float() - base_r.get(n).float()
        curve = {}
        A, B, kept = lowrank(D, rank, curve=curve)
        a, b = peft_names(base_r.original(n))
        T[a], T[b] = A.to(torch.bfloat16), B.to(torch.bfloat16)
        stats[base_r.original(n)] = {"kept": round(kept, 4), "norm": round(float(D.norm()), 4), "kept_at": curve}
    out.mkdir(parents=True)
    save_file(T, str(out / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
           "bias": "none", "target_modules": sorted({n.split(".")[-2] for n in names}),
           "base_model_name_or_path": base_label, "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "energy.json").write_text(json.dumps(stats, indent=1))
    ks = [s["kept"] for s in stats.values()]
    w = [s["norm"] ** 2 for s in stats.values()]
    at = {}
    for k in RANKS_AT:
        rows = [(v["kept_at"][str(k)], v["norm"] ** 2) for v in stats.values() if str(k) in v["kept_at"]]
        if len(rows) == len(stats):
            at[str(k)] = round(sum(a * b for a, b in rows) / sum(b for _, b in rows), 4) if sum(w) else 1.0
    return {"output": str(out), "kept_weighted_at": at, "delta_norm": round(sum(w) ** 0.5, 4), "matrices": len(ks), "rank": rank, "mean_kept": sum(ks) / len(ks),
            "kept_weighted": sum(k * x for k, x in zip(ks, w)) / sum(w) if sum(w) else 1.0, "min_kept": min(ks),
            "bytes": (out / "adapter_model.safetensors").stat().st_size}
