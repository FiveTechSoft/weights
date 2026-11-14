"""A fine-tune's change as a reusable LoRA: D = W_source - W_base per linear matrix of the text layers, truncated SVD
to rank r, written as a PEFT adapter (lora_alpha = r, so scale 1). llama.cpp's convert_lora_to_gguf.py turns it into
a GGUF adapter that llama-server applies on the base with a per-request scale: no model rebuild per variant.
Logs per matrix the share of ||D||^2 kept by rank r. 1-D tensors (norms, A_log, dt_bias) and embed/lm_head are not
in the adapter (LoRA covers linear layers only).

usage: python lab/probes/close_probe/delta_to_lora.py <base_dir> <source_dir> <rank> <out_dir> [kinds, e.g. mlp or mlp,linear_attn]
"""
import json
import re
import sys
from pathlib import Path

import torch

TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
PAT = re.compile(r"^model\.language_model\.layers\.\d+\.(linear_attn|self_attn|mlp)\.(%s)\.weight$" % "|".join(TARGETS))


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engine.lora_export import lowrank  # noqa: E402  exact SVD (randomized one missed by 11-16 %, results §43.6)


def wanted(name, kinds):
    m = PAT.match(name)
    return bool(m) and (kinds is None or m.group(1) in kinds)


def peft_names(name):
    stem = "base_model.model." + name[: -len(".weight")]
    return stem + ".lora_A.weight", stem + ".lora_B.weight"


def main():
    from safetensors import safe_open
    from safetensors.torch import save_file
    base, src, r, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    kinds = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    bmap = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    smap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(n for n in bmap if wanted(n, kinds) and n in smap)
    H = {}

    def get(d, f, n):
        k = (str(d), f)
        if k not in H:
            H[k] = safe_open(str(d / f), "pt")
        return H[k].get_tensor(n)

    T, stats = {}, {}
    for n in names:
        D = get(src, smap[n], n).float() - get(base, bmap[n], n).float()
        A, B, kept = lowrank(D, r)
        a, b = peft_names(n)
        T[a], T[b] = A.to(torch.bfloat16), B.to(torch.bfloat16)
        stats[n] = {"kept": round(kept, 4), "norm": round(float(D.norm()), 4)}
    save_file(T, str(out / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none",
           "target_modules": sorted({n.split(".")[-2] for n in names}), "base_model_name_or_path": str(base), "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "energy.json").write_text(json.dumps(stats, indent=1))
    ks = [s["kept"] for s in stats.values()]
    print(json.dumps({"matrices": len(ks), "rank": r, "mean_kept": sum(ks) / len(ks), "min_kept": min(ks)}), flush=True)


if __name__ == "__main__":
    main()
