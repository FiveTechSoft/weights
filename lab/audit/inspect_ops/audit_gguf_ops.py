"""Audit GGUF-backed inspection: peek/semantic areas, spectral/semantic/ssm deltas (Qwen3.5-0.8B Q6_K, read-only)."""
import json
import numpy as np
import torch
import gguf
from harness import run, dump, OUT, ROOT
import wsp
from engine import comparative_drift as cd

R = {}
G = ROOT / "hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf"
GE = ROOT / "servidos/qwen35-08b__exp8-replica.gguf"  # documented edit of blk.15/19 ffn_down

def deq(path, name):
    r = gguf.GGUFReader(str(path))
    t = {x.name: x for x in r.tensors}[name]
    a = gguf.dequantize(t.data, t.tensor_type)
    return np.asarray(a, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else np.asarray(a, np.float32)

# ---- peek (GGUF): numbers vs independent dequant
D = deq(G, "blk.3.attn_output.weight")
p = wsp.peek(G, "blk.3.attn_output.weight", row=100, col=200)
R["peek_gguf_element"] = {"op_value": p["value"], "indep_value": float(D[100, 200]), "op_row_norm": p["row_norm"],
                          "indep_row_norm": float(np.linalg.norm(D[100])), "head": p["architecture"]["head"],
                          "expected_head_col//256": 200 // 256, "dominant": (p.get("semantic_areas") or {}).get("dominant")}
r = run("semantic_areas blk.3.attn_output.weight row=100 col=200 top=3", model=G, model_name="qwen")
R["semantic_areas_op_gguf"] = {"feas_err": r["feasibility_errors"], "status": r["step"]["status"],
                               "dominant": (r["step"].get("result", {}).get("semantic_areas") or {}).get("dominant")}
r = run("semantic_map top=3", model=G, model_name="qwen")
R["semantic_map_gguf_top"] = list(r["step"]["result"]["top_areas"].keys())
R["semantic_map_gguf_note"] = "top_areas sorted by len(seeds) of the hard-coded SEMANTIC_AREAS dict, not by any weight"

# ---- informativeness of semantic labels: real columns vs random vectors
cents = wsp.get_semantic_centroids(G)
areas = [k for k in cents if k in wsp.SEMANTIC_AREAS]
seeds = [k for k in cents if k.startswith("seed_")]
R["centroids"] = {"areas": len(areas), "seeds_resolved": len(seeds), "seeds_total": sum(len(v["seeds"]) for v in wsp.SEMANTIC_AREAS.values())}
C = np.stack([cents[a] for a in areas])
def dom_scores(V):
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ C.T
    return S.max(1), S.argmax(1)
Dd = deq(G, "blk.3.ffn_down.weight")  # (1024, 3584): column j writes to residual
real_max, real_arg = dom_scores(Dd.T)
rng = np.random.default_rng(0)
rnd_max, rnd_arg = dom_scores(rng.standard_normal((3584, 1024)).astype(np.float32))
# column-shuffled control: same marginal stats, structure destroyed
shuf = Dd.T.copy(); [rng.shuffle(row) for row in shuf]
sh_max, _ = dom_scores(shuf)
def q(x): return [round(float(v), 4) for v in np.quantile(x, [0.05, 0.5, 0.95])]
R["semantic_null_test_blk3_ffn_down_cols"] = {
    "real_dominant_score_q05_50_95": q(real_max), "random_gaussian_q05_50_95": q(rnd_max), "shuffled_cols_q05_50_95": q(sh_max),
    "frac_vectors_given_a_label_real": float(np.mean(real_max > -1)), "frac_given_label_random": float(np.mean(rnd_max > -1)),
    "real_label_hist": {areas[i]: int((real_arg == i).sum()) for i in range(len(areas))},
    "random_label_hist": {areas[i]: int((rnd_arg == i).sum()) for i in range(len(areas))}}
sem_rand = wsp.attribute_semantic_area(rng.standard_normal(1024).astype(np.float32), cents)
R["attribute_semantic_area_pure_noise"] = sem_rand["dominant"]

# ---- spectral_delta on the documented exp8 edit vs exact SVD
for ten in ("blk.15.ffn_down.weight", "blk.3.ffn_down.weight"):
    r = run(f"spectral_delta {ten} base={G.as_posix()} k=5", model=GE, model_name="qwen")
    st = r["step"]; res = st.get("result", {})
    dlt = deq(GE, ten) - deq(G, ten)
    S = torch.linalg.svdvals(torch.from_numpy(dlt).double()).numpy()
    fro = float(np.linalg.norm(dlt))
    R[f"spectral_delta_{ten}"] = {"feas_err": r["feasibility_errors"][:1], "status": st["status"], "op_top_sv": res.get("top_singular_values"),
                                  "exact_top_sv": [round(float(s), 4) for s in S[:5]], "op_fro": res.get("frobenius_norm_delta"), "exact_fro": round(fro, 4),
                                  "op_top1_share": res.get("top1_variance_share"), "exact_top1_share": round(float(S[0] ** 2 / max(fro ** 2, 1e-30)), 6) if fro else None,
                                  "op_geometry": res.get("update_geometry"), "nnz_delta": int((dlt != 0).sum())}

# ---- semantic_delta on the same edit, plus a null (random delta)
r = run(f"semantic_delta blk.15.ffn_down.weight base={G.as_posix()} top=3", model=GE, model_name="qwen")
R["semantic_delta_exp8"] = {"status": r["step"]["status"], "out": r["step"].get("result", r["step"].get("error"))}
dlt = deq(GE, "blk.15.ffn_down.weight") - deq(G, "blk.15.ffn_down.weight")
u0 = np.linalg.svd(dlt, full_matrices=False)[0][:, 0]
real = np.abs(C @ u0)
null = np.abs(C @ (rng.standard_normal((1024, 2000)) / np.sqrt(1024)))  # unit-ish random vectors
null_max = null.max(0)
R["semantic_delta_null"] = {"exp8_u0_max_abs_alignment": round(float(real.max()), 4), "exp8_argmax_area": areas[int(real.argmax())],
                            "random_unit_vec_max_alignment_q50_q95_q99": [round(float(np.quantile(null_max, x)), 4) for x in (0.5, 0.95, 0.99)],
                            "percentile_of_exp8_in_null": float((null_max < real.max()).mean())}

# ---- ssm_dynamics_diff: real pair + forced mutation of ssm_a (diagnosis text sensitivity)
r = run(f"ssm_dynamics_diff layer=0 base={G.as_posix()}", model=GE, model_name="qwen")
R["ssm_diff_real_pair"] = {"status": r["step"]["status"], "statuses": {k: v["status"] for k, v in r["step"]["result"]["dynamics_components"].items()},
                           "diagnosis": r["step"]["result"]["structural_diagnosis"][:120]}
orig = cd.decode_tensor_weights
def mutated(path, name):
    W, dt = orig(path, name)
    if Path(path) == GE and name.endswith("ssm_a"):
        W = W * 3.0
    return W, dt
from pathlib import Path
cd.decode_tensor_weights = mutated
res = cd.compute_ssm_dynamics_diff(GE, G, layer_idx=0)
cd.decode_tensor_weights = orig
R["ssm_diff_ssm_a_tripled"] = {"ssm_a_status": res["dynamics_components"]["ssm_a"]["status"], "ssm_a_rel": res["dynamics_components"]["ssm_a"]["relative_drift"],
                               "diagnosis_still_says_frozen": "remain mathematically frozen" in res["structural_diagnosis"]}
r = run(f"ssm_dynamics_diff layer=3 base={G.as_posix()}", model=GE, model_name="qwen")  # layer 3 is full attention: no ssm tensors
R["ssm_diff_attention_layer"] = {"status": r["step"]["status"], "components": r["step"]["result"]["dynamics_components"], "diagnosis_still_says_frozen": "frozen" in r["step"]["result"]["structural_diagnosis"]}

# ---- diff on GGUF pair vs independent dequant
r = run(f"diff blk.15.ffn_down.weight base={G.as_posix()}", model=GE, model_name="qwen")
D1, D2 = deq(G, "blk.15.ffn_down.weight"), deq(GE, "blk.15.ffn_down.weight")
R["diff_gguf_exp8_blk15"] = {"status": r["step"]["status"], "op_changed": r["step"].get("result", {}).get("total_changed"),
                             "exact_changed": int((D1 != D2).sum()), "rel1e-4_changed": int((np.abs(D1 - D2) > 1e-4 * (np.abs(D1) + 1e-30)).sum())}

print(json.dumps(R, indent=1, default=str))
dump("gguf_ops", R)
