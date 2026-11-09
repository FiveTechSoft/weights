"""Norm-matched random placebo of a PEFT LoRA adapter: each adapted matrix gets B'A' with the same rank and the same
Frobenius norm as the real B A, in random (Gaussian) directions. The config is copied unchanged, so the runtime scale
is the same. If the placebo moves the benchmark like the real adapter, the effect is "any change of that size".

usage: python lab/probes/close_probe/placebo_lora.py <adapter_dir> <out_dir> [seed=0]
"""
import json
import shutil
import sys
from pathlib import Path

import torch


def make(src, dst, seed=0):
    from safetensors.torch import load_file, save_file
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    T = load_file(str(src / "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(seed)
    out = {}
    for ka in sorted(k for k in T if k.endswith("lora_A.weight")):
        kb = ka.replace("lora_A", "lora_B")
        A, B = T[ka].float(), T[kb].float()
        real = torch.trace((B.T @ B) @ (A @ A.T)).clamp_min(0).sqrt()        # ||B A||_F without forming B A
        A2 = torch.randn(A.shape, generator=g)
        B2 = torch.randn(B.shape, generator=g)
        fake = torch.trace((B2.T @ B2) @ (A2 @ A2.T)).sqrt()
        s = (real / fake).sqrt()
        out[ka] = (A2 * s).to(T[ka].dtype).contiguous()
        out[kb] = (B2 * s).to(T[kb].dtype).contiguous()
    if set(out) != set(T):
        raise SystemExit("adapter has tensors that are not lora_A / lora_B pairs")
    dst.mkdir(parents=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    (dst / "placebo.json").write_text(json.dumps({"source": str(src), "seed": seed}))


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
