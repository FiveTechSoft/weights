"""apply_wsp_to_safetensors.py — Apply WSP Low-Rank Directional Surgery to Original Hugging Face BF16 / FP16 Models.

Applies the exact continuous low-rank projection:
    W[:, cols] <- W[:, cols] - alpha * u (u^T W[:, cols])
directly on the unquantized SafeTensors files, eliminating quantization block noise.

Usage:
    python scripts/apply_wsp_to_safetensors.py \\
        --model-dir /path/to/huggingface_model \\
        --out-dir /path/to/repaired_model \\
        --layer 14 \\
        --vector data/u_loop_L14.npz \\
        --k 64 \\
        --target-frac 0.04
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def load_vector(npz_path):
    data = np.load(npz_path)
    for k in data.files:
        vec = data[k].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec
    raise ValueError(f"No array found in {npz_path}")


def find_tensor_file(model_dir, target_name):
    """Finds which safetensors file contains target_name using model.safetensors.index.json or scanning."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        if target_name in weight_map:
            return weight_map[target_name]
            
    # Fallback scan all .safetensors
    from safetensors import safe_open
    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            fpath = os.path.join(model_dir, fname)
            with safe_open(fpath, framework="np") as f:
                if target_name in f.keys():
                    return fname
    return None


def apply_wsp_edit(
    model_dir: str,
    out_dir: str,
    layer_idx: int,
    vector_path: str,
    k: int = 64,
    target_frac: float = 0.04,
    tensor_pattern: str = "model.layers.{layer}.mlp.down_proj.weight"
):
    import torch
    from safetensors.torch import load_file, save_file

    model_path = Path(model_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_tensor_name = tensor_pattern.format(layer=layer_idx)
    # Check alternate naming (e.g. Qwen language_model prefix)
    alt_tensor_name = f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"

    print(f"[*] Applying WSP Directional Fix to original Hugging Face model:")
    print(f"    Source:      {model_path}")
    print(f"    Target:      {out_path}")
    print(f"    Target Layer: {layer_idx}")
    print(f"    Vector:      {vector_path} (target_frac={target_frac}, k={k})")

    u_vec = load_vector(vector_path)

    # 1. Locate safetensors file containing target tensor
    target_file = find_tensor_file(model_path, target_tensor_name)
    actual_name = target_tensor_name
    if not target_file:
        target_file = find_tensor_file(model_path, alt_tensor_name)
        if target_file:
            actual_name = alt_tensor_name

    if not target_file:
        print(f"[ERROR] Could not find tensor '{target_tensor_name}' in {model_path}.")
        sys.exit(1)

    print(f"[*] Found '{actual_name}' in shard '{target_file}'")

    # 2. Link or copy all configuration, tokenizer, and unmodified safetensors files
    for item in os.listdir(model_path):
        src_item = model_path / item
        dst_item = out_path / item
        if item == target_file:
            continue  # Will be written modified
        if src_item.is_file():
            if dst_item.exists():
                dst_item.unlink()
            try:
                os.link(src_item, dst_item)
                print(f"    Hard-linked {item} (instant, 0 disk bytes used)...")
            except Exception:
                print(f"    Copying {item}...")
                shutil.copy2(src_item, dst_item)

    # 3. Load shard, apply exact continuous edit, and save
    print(f"[*] Loading shard {target_file} for surgical modification...")
    tensors = load_file(str(model_path / target_file))

    w = tensors[actual_name]  # torch tensor (e.g. [hidden_dim, intermediate_dim], BF16)
    orig_dtype = w.dtype
    w_float = w.to(torch.float32)

    hidden_dim, inter_dim = w_float.shape
    u_torch = torch.from_numpy(u_vec).to(torch.float32)

    # Project each column onto u: v = u^T W
    # If u has shape [hidden_dim], u_torch @ w_float has shape [inter_dim]
    if u_torch.shape[0] != hidden_dim:
        print(f"[!] Warning: vector dimension ({u_torch.shape[0]}) != tensor hidden_dim ({hidden_dim}). Truncating or padding.")
        if u_torch.shape[0] > hidden_dim:
            u_torch = u_torch[:hidden_dim]
            u_torch = u_torch / torch.norm(u_torch)

    v = torch.matmul(u_torch, w_float)  # shape [inter_dim]
    v_abs = torch.abs(v)

    # Select top-k columns with highest directional alignment
    topk_vals, topk_cols = torch.topk(v_abs, k=min(k, inter_dim))
    cols_idx = topk_cols.tolist()

    print(f"[*] Top {k} columns selected (max alignment: {topk_vals[0].item():.5f}, min: {topk_vals[-1].item():.5f})")

    # Apply continuous subtraction: W[:, c] <- W[:, c] - alpha * u * (u^T W[:, c])
    # rank-1 outer product for selected columns
    v_selected = v[topk_cols].unsqueeze(0)  # [1, k]
    u_col = u_torch.unsqueeze(1)            # [hidden_dim, 1]
    delta = target_frac * torch.matmul(u_col, v_selected)  # [hidden_dim, k]

    w_float[:, topk_cols] -= delta

    # Convert back to original dtype (BF16 or FP16)
    tensors[actual_name] = w_float.to(orig_dtype)

    # 4. Save modified shard
    out_target_file = str(out_path / target_file)
    print(f"[*] Saving repaired shard to {out_target_file}...")
    save_file(tensors, out_target_file)

    # 5. Save edit manifest for reproducibility
    manifest = {
        "source_model": str(model_path),
        "target_model": str(out_path),
        "target_tensor": actual_name,
        "shard": target_file,
        "layer": layer_idx,
        "k_columns": k,
        "target_frac": target_frac,
        "vector_path": vector_path,
        "selected_columns": cols_idx,
        "weights_modified": k * hidden_dim,
        "percentage_of_tensor": f"{(k / inter_dim):.4%}",
    }
    manifest_file = out_path / "wsp_edit_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SUCCESS] WSP Fix applied successfully!")
    print(f"    Total parameters modified: {k * hidden_dim:,} ({manifest['percentage_of_tensor']} of layer)")
    print(f"    Manifest written to: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Apply WSP Low-Rank Directional Fix to original Hugging Face SafeTensors")
    parser.add_argument("--model-dir", required=True, help="Directory containing original Hugging Face SafeTensors")
    parser.add_argument("--out-dir", required=True, help="Output directory for repaired model")
    parser.add_argument("--layer", type=int, default=14, help="Layer index (default: 14 for MiMo-9B)")
    parser.add_argument("--vector", default="data/u_loop_L14.npz", help="Path to contrastive direction vector (.npz)")
    parser.add_argument("--k", type=int, default=64, help="Number of columns to project (default: 64)")
    parser.add_argument("--target-frac", type=float, default=0.04, help="Continuous edit fraction (default: 0.04)")
    args = parser.parse_args()

    apply_wsp_edit(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        layer_idx=args.layer,
        vector_path=args.vector,
        k=args.k,
        target_frac=args.target_frac,
    )


if __name__ == "__main__":
    main()
