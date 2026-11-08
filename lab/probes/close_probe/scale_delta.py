"""Build base + alpha * (source - base) for every tensor the two models share (task-arithmetic scaling of a fine-tune).

Tensors missing in the source (MTP head, F28) keep the base value. Config, tokenizer and chat template come from the
base (avoids the source's older pre-tokenizer, F25). Shards are written with the base's names. Inputs are never
modified; out must be a new directory.

usage: python lab/probes/close_probe/scale_delta.py <base_dir> <source_dir> <alpha> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch


def mix(b, s, alpha):
    if s is None:
        return b
    return (b.float() + alpha * (s.float() - b.float())).to(b.dtype)


def main():
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    base, src, alpha, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.mkdir(parents=True)
    smap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    kept, mixed = [], 0
    for shard in sorted(base.glob("*.safetensors")):
        T = load_file(str(shard))
        for n in sorted(T):
            if n not in smap:
                kept.append(n)
                continue
            f = smap[n]
            if f not in handles:
                handles[f] = safe_open(str(src / f), "pt")
            s = handles[f].get_tensor(n)
            if s.shape != T[n].shape:
                raise SystemExit(f"{n}: shape {tuple(s.shape)} != {tuple(T[n].shape)}")
            T[n] = mix(T[n], s, alpha)
            mixed += 1
        save_file(T, str(out / shard.name), metadata={"format": "pt", "provenance": "scale_delta",
                                                       "source": src.name, "alpha": str(alpha)})
        del T
        print(f"{shard.name} done", flush=True)
    for extra in base.iterdir():
        if extra.is_file() and not extra.name.endswith(".safetensors"):
            shutil.copy2(extra, out / extra.name)
    print(json.dumps({"mixed": mixed, "kept_from_base": len(kept), "alpha": alpha}), flush=True)


if __name__ == "__main__":
    main()
