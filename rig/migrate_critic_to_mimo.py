"""migrate_critic_to_mimo.py — Multi-Layer Permanent Weight Migration of Self-Critique Circuit (u_critic) into MiMo-9B.

Applies the Gaussian multi-layer envelope across layers 10..22:
    W_l[:, cols] <- W_l[:, cols] + w(l) * alpha * u (u^T W_l[:, cols])
Directly on unquantized SafeTensors shards (BF16), keeping model size 100% strictly invariant.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "hf_cache" / "mimo-distill-hf"
OUTPUT_DIR = ROOT / "models" / "mimo-wsp-critic"
VECTOR_PATH = ROOT / "data" / "u_critic_L16.npz"
MANIFEST_PATH = OUTPUT_DIR / "mimo_critic_migration_manifest.json"


def load_vector(npz_path: Path) -> torch.Tensor:
    data = np.load(npz_path)
    u = data["u_critic"].astype(np.float32)
    norm = np.linalg.norm(u)
    if norm > 1e-12:
        u = u / norm
    return torch.from_numpy(u)


def run_migration():
    print("=" * 80)
    print("MIMO-9B: MULTI-LAYER PERMANENT WEIGHT MIGRATION OF SELF-CRITIQUE CIRCUIT (u_critic)")
    print("Method: Coordinated Gaussian Multi-Layer Envelope (Layers 10..22, Peak L16)")
    print("=" * 80)

    t0_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Vector
    print(f"\n[1] Loading u_critic vector from {VECTOR_PATH.relative_to(ROOT)}...")
    u_torch = load_vector(VECTOR_PATH)
    print(f"    Vector dimension: {len(u_torch)}, L2 Norm: {torch.norm(u_torch):.6f}")

    # 2. Compute Envelope Profile
    peak_layer = 16
    bandwidth = 2.5
    alpha = 0.04
    l_start, l_end = 10, 22
    
    envelope_weights = {}
    print(f"\n[2] Computing Gaussian Layer Envelope (Peak={peak_layer}, Sigma={bandwidth}, Alpha={alpha}):")
    for l in range(l_start, l_end + 1):
        w_l = float(np.exp(-0.5 * ((l - peak_layer) / bandwidth) ** 2))
        envelope_weights[l] = round(w_l, 5)
        print(f"    Layer {l:>2}: weight = {w_l:.4f}")

    # 3. Read Shard Index Map
    index_file = SOURCE_DIR / "model.safetensors.index.json"
    with open(index_file, "r", encoding="utf-8") as f:
        idx_data = json.load(f)
    weight_map = idx_data.get("weight_map", {})

    # Map each layer in 10..22 to its shard
    layer_to_shard = {}
    shard_to_layers = {}
    for l in range(l_start, l_end + 1):
        tname = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        shard_name = weight_map.get(tname)
        if not shard_name:
            # Fallback check
            for k, sh in weight_map.items():
                if f".layers.{l}.mlp.down_proj" in k:
                    tname = k
                    shard_name = sh
                    break
        layer_to_shard[l] = (tname, shard_name)
        shard_to_layers.setdefault(shard_name, []).append((l, tname))

    print(f"\n[3] Target Shards to Modify:")
    for sh, layers in shard_to_layers.items():
        print(f"    - {sh}: Layers {[l for l, _ in layers]}")

    # 4. Hard-link all non-modified files (instant, 0 disk bytes used)
    print("\n[4] Hard-linking unmodified configuration, tokenizer, and shard files...")
    shards_to_modify = set(shard_to_layers.keys())
    for item in os.listdir(SOURCE_DIR):
        src_item = SOURCE_DIR / item
        dst_item = OUTPUT_DIR / item
        if item in shards_to_modify:
            continue  # Will be written with modifications
        if src_item.is_file():
            if dst_item.exists():
                dst_item.unlink()
            try:
                os.link(src_item, dst_item)
                print(f"    Hard-linked {item} (instant, 0 disk bytes)...")
            except Exception:
                shutil.copy2(src_item, dst_item)
                print(f"    Copied {item}...")

    # 5. Process each shard surgically
    print("\n[5] Executing surgical multi-layer WSP weight surgery across shards...")
    total_weights_modified = 0
    layer_diagnostics = []

    for shard_name, layers in shard_to_layers.items():
        src_shard = SOURCE_DIR / shard_name
        dst_shard = OUTPUT_DIR / shard_name
        print(f"\n  --> Loading {shard_name} ({src_shard.stat().st_size / (1024**3):.2f} GB)...")
        t0_sh = time.time()
        shard_tensors = load_file(str(src_shard))
        print(f"      Loaded in {time.time() - t0_sh:.2f}s. Modifying {len(layers)} layers...")

        for l, tensor_name in layers:
            w_raw = shard_tensors[tensor_name]
            orig_dtype = w_raw.dtype
            w_float = w_raw.to(torch.float32)

            frob_orig = float(torch.norm(w_float).item())
            hidden_dim, inter_dim = w_float.shape

            # Compute column projections: v = u^T W
            v = torch.matmul(u_torch, w_float)  # [inter_dim]
            w_l = envelope_weights[l]

            # Continuous Rank-1 additive imprint: Delta_W = w(l) * alpha * (u outer v)
            delta = (w_l * alpha) * torch.outer(u_torch, v)
            frob_delta = float(torch.norm(delta).item())
            perturb_pct = (frob_delta / frob_orig) * 100.0

            # In-place update
            w_float += delta
            shard_tensors[tensor_name] = w_float.to(orig_dtype)

            total_weights_modified += (hidden_dim * inter_dim)
            layer_diagnostics.append({
                "layer": l,
                "tensor": tensor_name,
                "shard": shard_name,
                "envelope_weight": w_l,
                "frob_orig": round(frob_orig, 4),
                "frob_delta": round(frob_delta, 5),
                "perturbation_pct": round(perturb_pct, 4)
            })
            print(f"      Layer {l:>2}: w={w_l:.4f} | ||Delta_W||_F={frob_delta:.4f} ({perturb_pct:.4f}%)")

        print(f"      Saving modified shard to {dst_shard.name}...")
        t0_save = time.time()
        save_file(shard_tensors, str(dst_shard))
        print(f"      Saved in {time.time() - t0_save:.2f}s.")

    # 6. Verify Exact Byte-Size Invariance
    print("\n[6] Verifying Exact Byte-Size Invariance:")
    size_matches = True
    for sh in shards_to_modify:
        src_sz = (SOURCE_DIR / sh).stat().st_size
        dst_sz = (OUTPUT_DIR / sh).stat().st_size
        delta_sz = dst_sz - src_sz
        if delta_sz != 0:
            size_matches = False
        print(f"    Shard {sh}:")
        print(f"      Source Size:  {src_sz:,} bytes")
        print(f"      Target Size:  {dst_sz:,} bytes")
        print(f"      Difference:   {delta_sz} bytes (Exact Invariance: {delta_sz == 0})")

    total_time = time.time() - t0_start

    # 7. Export Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "MiMo-v2.6-Distill-9B-WSP-Critic",
        "method": "Multi-Layer Gaussian Envelope Weight Migration",
        "vector_source": str(VECTOR_PATH.relative_to(ROOT)),
        "layers_range": f"{l_start}..{l_end}",
        "peak_layer": peak_layer,
        "bandwidth": bandwidth,
        "alpha_scale": alpha,
        "total_layers_modified": len(layer_diagnostics),
        "total_weights_modified": total_weights_modified,
        "exact_byte_invariance_verified": size_matches,
        "layer_diagnostics": layer_diagnostics,
        "execution_time_sec": round(total_time, 2)
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Multi-layer migration completed in {total_time:.2f} seconds!")
    print(f"    Checkpoints Directory: {OUTPUT_DIR}")
    print(f"    Exact Byte Invariance: {'CONFIRMED (0 extra bytes)' if size_matches else 'MISMATCH'}")
    print(f"    Manifest exported to:  {MANIFEST_PATH.relative_to(ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    run_migration()
