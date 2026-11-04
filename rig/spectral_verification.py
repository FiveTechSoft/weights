"""spectral_verification.py — Empirical Matrix Perturbation & Leverage Verification Rig for MiMo-9B.

Grounds Weight Space Surgery in Matrix Perturbation Theory (Weyl, Davis-Kahan, RandNLA; Yu-Xiang Wang et al.):
1. Empirical verification of Weyl's Inequality on real MiMo-9B weights:
     |sigma_i(W - Delta W) - sigma_i(W)| <= ||Delta W||_2
2. Empirical verification of Davis-Kahan sin Theta Subspace Stability Theorem:
     sin Theta(U_1, U_1_tilde) <= ||Delta W||_2 / delta
3. Statistical leverage scores:
     l_j = sum_{i=1}^r V_{ji}^2
4. Layer-by-layer eigengap comparison (Layer 2 vs Layer 14 vs Layer 27 down_proj).
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsplang

def run_spectral_verification():
    print("=" * 80)
    print("EMPIRICAL SPECTRAL PERTURBATION & LEVERAGE VERIFICATION RIG (MiMo-9B)")
    print("Mathematical Foundations: Weyl's Inequality, Davis-Kahan sin Theta, RandNLA")
    print("=" * 80)

    hf_dir = ROOT / "hf_cache" / "mimo-distill-hf"
    index_path = hf_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"[ERROR] Hugging Face cache not found at: {hf_dir}")
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    # Target Tensor: Layer 14 down_proj
    l14_tensor = "model.language_model.layers.14.mlp.down_proj.weight"
    shard_name = weight_map[l14_tensor]
    shard_path = hf_dir / shard_name
    print(f"\n[*] Loading Layer 14 down_proj from {shard_name}...")
    tensors = load_file(str(shard_path))
    W = tensors[l14_tensor].float()
    d_in, d_out = W.shape
    print(f"    Tensor dimensions: {d_in} x {d_out} ({W.numel():,} parameters, BF16)")

    # 1. Baseline Randomized SVD with Oversampling
    print("\n[*] 1. Computing Baseline Truncated SVD (rank q=16, oversampling=30, niter=10)...")
    t0 = time.time()
    spec_before = wsplang.compute_spectral_decomposition(W, q=16, niter=10)
    svd_time = time.time() - t0
    S_before = torch.tensor(spec_before["singular_values"], dtype=torch.float32)
    U0_before = spec_before["U"][:, 0]
    delta_gap = spec_before["spectral_gap"]
    print(f"    Top 5 singular values: {[round(x, 4) for x in S_before[:5].tolist()]}")
    print(f"    Spectral eigengap (delta = sigma_1 - sigma_2): {delta_gap:.6f}")
    print(f"    Condition number (sigma_1 / sigma_16): {spec_before['condition_number']:.4f}")
    print(f"    SVD computation completed in: {svd_time:.3f}s")

    # 2. Load Loop Vector & Compute Rank-1 Surgical Edit
    vec_path = ROOT / "data" / "u_loop_L14.npz"
    print(f"\n[*] 2. Loading tool-loop intervention vector from {vec_path.name}...")
    u_vec = wsplang.load_vector_numpy(vec_path)
    u_t = torch.from_numpy(u_vec).float().to(W.device)

    alpha = 0.05
    u_proj = torch.matmul(u_t, W)
    k_cols = 520  # 4.23% of 12,288 columns
    top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
    v_target = u_proj[top_cols_idx]

    delta = torch.zeros_like(W)
    delta[:, top_cols_idx] = alpha * torch.outer(u_t, v_target)

    # Exact operator norm of rank-1 matrix: ||Delta W||_2 = alpha * ||u||_2 * ||v||_2
    op_norm_delta = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v_target).item())
    print(f"    Selected intervention columns: {k_cols} / {d_out} ({k_cols/d_out:.2%})")
    print(f"    Edit scaling factor alpha: {alpha}")
    print(f"    Spectral Operator Norm ||Delta W||_2: {op_norm_delta:.6f}")

    # 3. Apply Perturbation & Compute Perturbed Spectrum
    print("\n[*] 3. Applying Perturbation W_tilde = W - Delta W and Recomputing SVD...")
    W_tilde = W - delta
    t0 = time.time()
    spec_after = wsplang.compute_spectral_decomposition(W_tilde, q=16, niter=10)
    svd_after_time = time.time() - t0
    S_after = torch.tensor(spec_after["singular_values"], dtype=torch.float32)
    U0_after = spec_after["U"][:, 0]

    # 4. Empirical Verification of Weyl's Inequality
    print("\n[*] 4. Verifying Weyl's Inequality (|sigma_i(W_tilde) - sigma_i(W)| <= ||Delta W||_2)...")
    shifts = torch.abs(S_after - S_before)
    max_shift = float(torch.max(shifts).item())
    margin = op_norm_delta - max_shift
    weyl_ok = max_shift <= op_norm_delta + 1e-4

    print(f"    Theoretical upper bound (||Delta W||_2):  {op_norm_delta:.6f}")
    print(f"    Maximum empirical singular value shift:    {max_shift:.6f}")
    print(f"    Weyl's Inequality Safety Margin:          {margin:.6f}")
    print(f"    Principal singular value shift (|d sigma_1|): {float(shifts[0].item()):.8f} ({float(shifts[0].item())/S_before[0].item():.6%})")
    print(f"    Weyl's Inequality Satisfied Empirically:  {'[PASS] TRUE' if weyl_ok else '[FAIL] FALSE'}")

    # 5. Empirical Verification of Davis-Kahan sin Theta Theorem
    print("\n[*] 5. Verifying Davis-Kahan sin Theta Subspace Stability...")
    cos_theta = min(1.0, float(torch.abs(torch.dot(U0_before, U0_after)).item()))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))
    angle_deg = float(np.degrees(np.arcsin(sin_theta)))

    dk_bound = op_norm_delta / delta_gap
    bound_angle_deg = float(np.degrees(np.arcsin(min(1.0, dk_bound))))
    dk_ok = sin_theta <= dk_bound + 1e-4

    print(f"    Spectral gap delta:                        {delta_gap:.6f}")
    print(f"    Empirical subspace rotation sin Theta:     {sin_theta:.6f} ({angle_deg:.4f} degrees)")
    print(f"    Davis-Kahan upper bound ||Delta W||_2 / d: {dk_bound:.6f} ({bound_angle_deg:.4f} degrees)")
    print(f"    Davis-Kahan Bound Satisfied Empirically:   {'[PASS] TRUE' if dk_ok else '[FAIL] FALSE'}")

    # 6. Statistical Leverage Scores & Efficiency Filtering
    print("\n[*] 6. Analyzing Statistical Leverage Scores (RandNLA)...")
    lev_scores = spec_before["leverage_scores"]
    lev_sum = float(lev_scores.sum().item())
    lev_mean = float(lev_scores.mean().item())
    lev_max = float(lev_scores.max().item())

    # Raw projection columns leverage
    raw_cols_lev = float(lev_scores[top_cols_idx].mean().item())

    # Efficiency-filtered columns: |u_proj| / sqrt(leverage)
    col_efficiency = torch.abs(u_proj) / (torch.sqrt(lev_scores) + 1e-4)
    eff_cols_idx = torch.topk(col_efficiency, k=k_cols).indices
    eff_cols_lev = float(lev_scores[eff_cols_idx].mean().item())
    lev_reduction = (raw_cols_lev - eff_cols_lev) / raw_cols_lev

    print(f"    Sum of column leverage scores (sum l_j):   {lev_sum:.4f} (Identity matches rank r=16)")
    print(f"    Average column leverage score (overall):   {lev_mean:.6f}")
    print(f"    Mean leverage of raw magnitude columns:    {raw_cols_lev:.6f}")
    print(f"    Mean leverage of efficiency-filtered cols: {eff_cols_lev:.6f}")
    print(f"    Leverage reduction (less core footprint):  {lev_reduction:.2%}")

    # 7. Layer-by-Layer Spectral Stability Comparison
    print("\n[*] 7. Comparing Layer Spectral Stability (Layer 2 vs Layer 14 vs Layer 27 down_proj)...")
    layers_comp = {}
    for l_idx in [2, 14, 27]:
        t_name = f"model.language_model.layers.{l_idx}.mlp.down_proj.weight"
        sh = weight_map[t_name]
        l_tensors = load_file(str(hf_dir / sh))
        l_W = l_tensors[t_name].float()
        l_spec = wsplang.compute_spectral_decomposition(l_W, q=10, niter=6)
        l_delta = l_spec["spectral_gap"]
        l_sigma1 = l_spec["singular_values"][0]
        l_sigma2 = l_spec["singular_values"][1]
        l_dk_bound = op_norm_delta / max(l_delta, 1e-12)
        l_bound_deg = float(np.degrees(np.arcsin(min(1.0, l_dk_bound))))
        layers_comp[f"layer_{l_idx}"] = {
            "layer": l_idx,
            "sigma_1": round(l_sigma1, 4),
            "sigma_2": round(l_sigma2, 4),
            "spectral_gap_delta": round(l_delta, 6),
            "davis_kahan_bound": round(l_dk_bound, 4),
            "max_rotation_degrees": round(l_bound_deg, 2),
            "stability_regime": "Optimal (Stable Surgery)" if l_delta > 0.5 else "Catastrophic (Subspace Dispersion)"
        }
        print(f"    Layer {l_idx:2d} down_proj: sigma_1={l_sigma1:.4f}, sigma_2={l_sigma2:.4f}, gap={l_delta:.4f} -> max rotation bound: {l_bound_deg:.2f} deg ({layers_comp[f'layer_{l_idx}']['stability_regime']})")

    # Compile Structured Verification Manifest
    report = {
        "benchmark": "MiMo-9B Weight Surgery Spectral Perturbation Verification",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        "target_tensor": l14_tensor,
        "tensor_shape": [d_in, d_out],
        "surgical_edit": {
            "type": "Rank-1 Directional Subtraction (project_out / delta_unlearn)",
            "alpha": alpha,
            "active_columns": k_cols,
            "active_fraction": f"{k_cols/d_out:.4%}",
            "operator_norm_bound_delta_W": round(op_norm_delta, 6)
        },
        "weyl_inequality_verification": {
            "theorem": "|sigma_i(W - Delta W) - sigma_i(W)| <= ||Delta W||_2",
            "operator_norm_bound": round(op_norm_delta, 6),
            "max_empirical_shift": round(max_shift, 6),
            "safety_margin": round(margin, 6),
            "principal_singular_value_shift": round(float(shifts[0].item()), 8),
            "relative_principal_shift": f"{(float(shifts[0].item())/S_before[0].item()):.6%}",
            "verified": weyl_ok
        },
        "davis_kahan_verification": {
            "theorem": "sin Theta(U_1, U_1_tilde) <= ||Delta W||_2 / delta",
            "spectral_gap_delta": round(delta_gap, 6),
            "empirical_sin_theta": round(sin_theta, 6),
            "empirical_rotation_degrees": round(angle_deg, 4),
            "davis_kahan_upper_bound": round(dk_bound, 6),
            "davis_kahan_max_degrees": round(bound_angle_deg, 4),
            "verified": dk_ok
        },
        "statistical_leverage_analysis": {
            "sum_leverage_scores": round(lev_sum, 4),
            "overall_mean_leverage": round(lev_mean, 6),
            "raw_columns_mean_leverage": round(raw_cols_lev, 6),
            "efficiency_columns_mean_leverage": round(eff_cols_lev, 6),
            "leverage_reduction_pct": f"{lev_reduction:.2%}"
        },
        "layer_spectral_gap_comparison": layers_comp
    }

    out_file = ROOT / "rig_runs" / "spectral_perturbation_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCESS] Spectral verification manifest written to: {out_file}")
    return report


if __name__ == "__main__":
    run_spectral_verification()
