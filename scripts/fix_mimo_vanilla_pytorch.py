#!/usr/bin/env python3
"""
fix_mimo_vanilla_pytorch.py — Minimal standalone script to patch MiMo-V2.6-Distill-Qwen-9B
WITHOUT requiring WSPLang. Requires only standard PyTorch and SafeTensors.

Usage:
    python scripts/fix_mimo_vanilla_pytorch.py \
        --model-dir /path/to/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B \
        --out-dir /path/to/MiMo-V2.6-Distill-Qwen-9B-repaired \
        --u-path data/u_loop_L14.npz
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser(description="Vanilla PyTorch fix for MiMo-9B tool loop")
    parser.add_argument("--model-dir", required=True, help="Path to original HF model folder")
    parser.add_argument("--out-dir", required=True, help="Path to write repaired HF model")
    parser.add_argument("--u-path", default="data/u_loop_L14.npz", help="Path to loop direction vector npz")
    parser.add_argument("--alpha", type=float, default=0.05, help="Subtractive projection factor")
    parser.add_argument("--layer", type=int, default=14, help="Layer index (default: 14)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load direction vector
    data = np.load(args.u_path)
    u_np = data[list(data.files)[0]].astype(np.float32)
    u_np = u_np / np.linalg.norm(u_np)
    u = torch.from_numpy(u_np)

    # 2. Find target shard
    target_tensor = f"model.language_model.layers.{args.layer}.mlp.down_proj.weight"
    index_file = model_dir / "model.safetensors.index.json"
    target_shard = None
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            idx = json.load(f)
        target_shard = idx.get("weight_map", {}).get(target_tensor)

    # 3. Process shards
    for f in model_dir.glob("*.safetensors"):
        dst = out_dir / f.name
        if f.name == target_shard or target_shard is None:
            tensors = load_file(str(f))
            if target_tensor in tensors:
                print(f"[*] Patching {target_tensor} in {f.name}...")
                W = tensors[target_tensor]  # [d_model, d_ffn]
                orig_dtype = W.dtype
                W_f = W.to(torch.float32)
                u_f = u.to(W_f.device)

                # Rank-1 projection subtraction
                # u_proj = u^T @ W -> shape [d_ffn]
                u_proj = torch.matmul(u_f, W_f)
                
                # Select top energetic columns
                top_k_cols = torch.topk(torch.abs(u_proj), k=int(W_f.shape[1] * 0.127)).indices
                
                # Apply delta: W[:, cols] -= alpha * (u outer u_proj[cols])
                delta = args.alpha * torch.outer(u_f, u_proj[top_k_cols])
                
                # Sparsify delta to top-50 rows per column
                top_rows = torch.topk(torch.abs(u_f), k=50).indices
                mask = torch.zeros_like(delta)
                mask[top_rows, :] = 1.0
                delta = delta * mask
                
                W_f[:, top_k_cols] -= delta
                tensors[target_tensor] = W_f.to(orig_dtype)
                save_file(tensors, str(dst))
                print(f"[+] Saved repaired shard {dst.name} ({delta.nonzero().shape[0]:,} weights edited)")
                continue

        # Unmodified shards: hard-link or copy
        if not dst.exists():
            try:
                os.link(str(f), str(dst))
            except Exception:
                import shutil
                shutil.copy2(str(f), str(dst))

    # Copy config / tokenizer files
    for meta in model_dir.glob("*"):
        if not meta.name.endswith(".safetensors"):
            dst_meta = out_dir / meta.name
            if not dst_meta.exists():
                try:
                    os.link(str(meta), str(dst_meta))
                except Exception:
                    import shutil
                    if meta.is_file():
                        shutil.copy2(str(meta), str(dst_meta))

    print(f"[✓] Repaired model ready at: {out_dir}")

if __name__ == "__main__":
    main()
