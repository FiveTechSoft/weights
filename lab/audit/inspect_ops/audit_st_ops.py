"""Audit inspection ops on small safetensors models, with independent recomputation."""
import json
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from harness import run, dump, MODELS, OUT, ROOT

R = {}
A = MODELS / "rand_1337.safetensors"
B = MODELS / "rand_7.safetensors"
E = MODELS / "rand_1337_edit37.safetensors"
TA = load_file(str(A)); TB = load_file(str(B)); TE = load_file(str(E))
FC = "transformer.h.0.mlp.c_fc.weight"
PROJ = "transformer.h.0.mlp.c_proj.weight"

# ---- 0. ops that are in OPS_ALLOWED but have no runtime branch
for op in ["inspect", "observe", "hypothesize", "localize", "confirm", "record", "select_candidate", "assert_budget"]:
    r = run(f"{op} {FC}", model=A)
    R[f"unimpl_{op}"] = {"feas": r["feasibility_errors"][:1], "status": r["step"]["status"]}

# ---- 1. peek (safetensors branch): stats vs torch, sensitivity
for nm, path, T in (("A", A, TA), ("B", B, TB)):
    r = run(f"peek {FC} row=3 col=5", model=path)["step"]
    W = T[FC].float()
    ref = {"norm": float(W.norm()), "min": float(W.min()), "max": float(W.max()), "mean": float(W.mean()), "element": float(W[3, 5])}
    got = r.get("result", {})
    R[f"peek_{nm}"] = {"status": r["status"], "maxabs_err": max(abs(got[k] - ref[k]) for k in ref), "keys": sorted(got)}
r = run("peek", model=A)["step"]  # default tensor name is a GGUF name
R["peek_default_tensor"] = {"status": r["status"], "error": r.get("error", "")[:120]}
r = run(f"semantic_areas {FC} row=3 col=5", model=A)["step"]
R["semantic_areas_on_safetensors"] = {"status": r["status"], "has_semantic": "semantic_areas" in r.get("result", {}), "keys": sorted(r.get("result", {}))}

# ---- 2. diff vs numpy
def np_diff(pa, pb, pat="*"):
    import fnmatch
    a, b = load_file(str(pa)), load_file(str(pb))
    tot = chg = 0
    for k in sorted(b):
        if not fnmatch.fnmatch(k, pat) or k not in a: continue
        d1, d2 = a[k].float().numpy(), b[k].float().numpy()
        chg += int((d1 != d2).sum()); tot += d2.size
    return chg, tot
for nm, other in (("edit37", E), ("rand7", B), ("self", A)):
    r = run(f"diff * base={A.as_posix()}", model=other)["step"]
    chg, tot = np_diff(A, other)
    R[f"diff_{nm}"] = {"status": r["status"], "op_changed": r.get("result", {}).get("total_changed"), "op_total": r.get("result", {}).get("total_weights"),
                       "numpy_exact_changed": chg, "numpy_total": tot}
# tiny edit (relative 5e-5) is invisible to the 1e-4 relative threshold
T2 = {k: v.clone() for k, v in TA.items()}; T2[FC] = T2[FC] * (1 + 5e-5)
tiny = MODELS / "rand_1337_tiny.safetensors"; save_file(T2, str(tiny))
r = run(f"diff * base={A.as_posix()}", model=tiny)["step"]
R["diff_tiny_scale_5e-5"] = {"op_changed": r["result"]["total_changed"], "numpy_exact_changed": np_diff(A, tiny)[0]}
# default base when omitted = MiMo GGUF
r = run(f"diff {FC}", model=E)
R["diff_no_base"] = {"status": r["step"]["status"], "error": r["step"].get("error", "")[:160]}

# ---- 3. compare / evaluate: just echo a json
j1 = OUT / "eval_a.json"; j1.write_text(json.dumps({"acc": 0.1}))
j2 = OUT / "eval_b.json"; j2.write_text(json.dumps({"acc": 0.9}))
r = run(f"compare {j1.as_posix()} {j2.as_posix()}", model=A)["step"]
R["compare_two_files"] = {"status": r["status"], "result": r.get("result")}
r = run(f"evaluate {j1.as_posix()}", model=A)["step"]
R["evaluate"] = {"status": r["status"], "result": r.get("result"), "note": "no benchmark is run; the JSON is echoed"}

# ---- 4. audit vs torch
r = run(f"audit {PROJ}", model=A)
W = TA[PROJ].float()
R["audit"] = {"status": r["step"]["status"], "feas_err": r["feasibility_errors"][:1],
              "frob_err": abs(r["step"]["result"]["frobenius_norm"] - float(W.norm())),
              "max_err": abs(r["step"]["result"]["max_magnitude"] - float(W.abs().max()))}

# vectors
rng = np.random.default_rng(0)
u64 = OUT / "u64.npz"; np.savez(u64, u=rng.standard_normal(64).astype(np.float32))
u256 = OUT / "u256.npz"; np.savez(u256, u=rng.standard_normal(256).astype(np.float32))

# ---- 5. spectral_audit vs exact SVD
for nm, ten in (("c_proj_64x256", PROJ), ("c_attn_192x64", "transformer.h.0.attn.c_attn.weight"), ("c_fc_256x64", FC)):
    r = run(f"spectral_audit {ten} q=16", model=A)["step"]
    S = torch.linalg.svdvals(TA[ten].float()).numpy()
    got = np.array(r["result"]["top_singular_values"])
    R[f"spectral_audit_{nm}"] = {"status": r["status"], "top8_op": got.tolist(), "top8_exact": np.round(S[:8], 4).tolist(),
                                 "max_rel_err_top8": float(np.max(np.abs(got - S[:8]) / S[:8])),
                                 "gap_op": r["result"]["spectral_gap_delta"], "gap_exact": float(S[0] - S[1]),
                                 "cond_op": r["result"]["condition_number"], "cond_exact_s1_s16": float(S[0] / S[15])}

# ---- 6. assert_weyl: exact check, attempt to force failure
def weyl_exact(W, u, alpha, top=0.127):
    u = u / np.linalg.norm(u)
    up = u @ W; k = max(1, int(W.shape[1] * top)); cols = np.argsort(-np.abs(up))[:k]
    D = np.zeros_like(W); D[:, cols] = alpha * np.outer(u, up[cols])
    s0 = np.linalg.svd(W, compute_uv=False); s1 = np.linalg.svd(W - D, compute_uv=False)
    return float(np.max(np.abs(s1 - s0))), float(np.linalg.norm(D, 2)), D, s0, s1
for alpha in (0.05, 1.0, 50.0):
    r = run(f"assert_weyl {PROJ} vector={u64.as_posix()} alpha={alpha} q=10", model=A)
    st = r["step"]
    mx, opn, *_ = weyl_exact(TA[PROJ].float().numpy().astype(np.float64), np.load(u64)["u"].astype(np.float64), alpha)
    R[f"weyl_alpha{alpha}"] = {"status": st["status"], "feas_err": r["feasibility_errors"][:1], "op": st.get("result", st.get("error")),
                               "exact_max_shift_all_sv": mx, "exact_opnorm_delta": opn}
# Weyl is a theorem for the delta the op itself builds: no user-supplied delta can be checked.
# Only way it "fails" is numerical error of svd_lowrank: try flat spectrum matrix (orthogonal * c) with small q.
Q, _ = torch.linalg.qr(torch.randn(256, 256, generator=torch.Generator().manual_seed(3)))
flat = {"flat.weight": Q[:64, :].contiguous()}  # 64x256 with all singular values = 1
fp = MODELS / "flat.safetensors"; save_file(flat, str(fp))
r = run(f"assert_weyl flat.weight vector={u64.as_posix()} alpha=0.5 q=10", model=fp)["step"]
R["weyl_flat_spectrum"] = {"status": r["status"], "out": r.get("result", r.get("error"))}

# ---- 7. assert_davis_kahan: exact check; max_angle_deg path; search for a violation of ||D||/gap with the op's delta
for alpha in (0.05, 1.0):
    r = run(f"assert_davis_kahan {PROJ} vector={u64.as_posix()} alpha={alpha}", model=A)["step"]
    W = TA[PROJ].float().numpy().astype(np.float64); u = np.load(u64)["u"].astype(np.float64)
    _, opn, D, s0, _ = weyl_exact(W, u, alpha)
    U0 = np.linalg.svd(W)[0][:, 0]; U1 = np.linalg.svd(W - D)[0][:, 0]
    sin_t = float(np.sqrt(max(0, 1 - np.dot(U0, U1) ** 2)))
    R[f"dk_alpha{alpha}"] = {"status": r["status"], "op": r.get("result", r.get("error")),
                             "exact_sin": sin_t, "exact_bound": opn / (s0[0] - s0[1])}
r = run(f"assert_davis_kahan {PROJ} vector={u64.as_posix()} alpha=1.0 max_angle_deg=0.001", model=A)["step"]
R["dk_max_angle_should_fail"] = {"status": r["status"], "error": r.get("error", "")[:160]}
# Construct a matrix where sin(theta) > ||D||/gap with the op's own delta (DK constant-free bound is not a theorem for this gap def.)
best = None
g = np.random.default_rng(1)
for trial in range(4000):
    n = 6
    s = np.sort(g.uniform(0.1, 3, n))[::-1]; s[1] = s[0] - g.uniform(0.01, 0.5)
    Uq = np.linalg.qr(g.standard_normal((n, n)))[0]; Vq = np.linalg.qr(g.standard_normal((n, n)))[0]
    W = Uq @ np.diag(s) @ Vq.T
    u = g.standard_normal(n); u /= np.linalg.norm(u); alpha = g.uniform(0.01, 1.5)
    up = u @ W; k = max(1, int(n * 1.0)); cols = np.argsort(-np.abs(up))[:k]
    D = np.zeros_like(W); D[:, cols] = alpha * np.outer(u, up[cols])
    s0 = np.linalg.svd(W, compute_uv=False)
    bound = np.linalg.norm(D, 2) / (s0[0] - s0[1])
    U0 = np.linalg.svd(W)[0][:, 0]; U1 = np.linalg.svd(W - D)[0][:, 0]
    sin_t = np.sqrt(max(0, 1 - np.dot(U0, U1) ** 2))
    if bound < 1 and sin_t > bound + 1e-3 and (best is None or sin_t - bound > best[0]):
        best = (sin_t - bound, sin_t, bound, W.copy(), u.copy(), alpha)
if best:
    _, sin_t, bound, W, u, alpha = best
    vp = MODELS / "dk_counter.safetensors"; save_file({"m.weight": torch.tensor(W, dtype=torch.float32)}, str(vp))
    up_ = OUT / "dk_u.npz"; np.savez(up_, u=u.astype(np.float32))
    r = run(f"assert_davis_kahan m.weight vector={up_.as_posix()} alpha={alpha} top_cols=1.0", model=vp)["step"]
    R["dk_counterexample"] = {"exact_sin": float(sin_t), "constant_free_bound": float(bound), "op_status": r["status"],
                              "op": r.get("result", r.get("error"))}
else:
    R["dk_counterexample"] = "none found in 4000 trials"

# ---- 8. assert_orthogonal_preservation: model never read
for nm, path in (("A", A), ("nonexistent_model", MODELS / "does_not_exist.safetensors")):
    r = run(f"assert_orthogonal_preservation key={u64.as_posix()}", model=path)["step"]
    R[f"orth_pres_{nm}"] = {"status": r["status"], "result": r.get("result")}

# ---- 9. semantic_map: independent of the model
outs = []
for path in (A, B):
    r = run("semantic_map top=3", model=path)["step"]
    outs.append(json.dumps(r.get("result"), sort_keys=True))
    R[f"semantic_map_{path.stem}"] = {"status": r["status"], "top_areas": list((r.get("result") or {}).get("top_areas", {}).keys()), "err": r.get("error")}
R["semantic_map_identical_for_two_random_models"] = outs[0] == outs[1]
r = run(f"semantic_map tensor={FC} top=3", model=A)["step"]
R["semantic_map_tensor_arg_changes_output"] = json.dumps(r.get("result"), sort_keys=True) != outs[0]

# ---- 10. architectural_drift: cached JSON
outs = []
for base, tgt in ((A, B), (A, E)):
    r = run(f"architectural_drift base={base.as_posix()} top=3", model=tgt)["step"]
    res = r.get("result", {})
    outs.append(json.dumps({k: v for k, v in res.items() if k not in ("base_model", "target_model")}, sort_keys=True))
    R[f"arch_drift_{tgt.stem}"] = {"status": r["status"], "data_source": res.get("data_source"), "total_tensors": res.get("total_tensors_analyzed"),
                                   "top1": (res.get("top_divergent_tensors") or [{}])[0].get("tensor"), "families": list(res.get("family_summary", {}))[:4]}
R["arch_drift_identical_for_different_pairs"] = outs[0] == outs[1]

# ---- 11. spectral/semantic/ssm deltas with safetensors target -> silently swapped for MiMo GGUF
for op in ("spectral_delta", "semantic_delta"):
    r = run(f"{op} {FC} base={A.as_posix()}", model=E)
    R[f"{op}_safetensors"] = {"feas_err": r["feasibility_errors"][:1], "status": r["step"]["status"], "out": str(r["step"].get("result", r["step"].get("error")))[:300]}
r = run(f"ssm_dynamics_diff layer=0 base={A.as_posix()}", model=E)["step"]
R["ssm_dynamics_diff_safetensors"] = {"status": r["status"], "out": r.get("result", r.get("error"))}

# ---- 12. assert_behavioral_drift
for nm, args in (("rand_model_strict", f"corpus=lab/data/shakespeare_char/input.txt max_ppl_increase=0.0001"),
                 ("missing_corpus", "corpus=nope.txt")):
    r = run(f"assert_behavioral_drift {args}", model=A)
    R[f"behav_drift_{nm}"] = {"feas_err": r["feasibility_errors"][:1], "status": r["step"]["status"], "out": r["step"].get("result", r["step"].get("error"))}

print(json.dumps(R, indent=1, default=str)[:20000])
dump("st_ops", R)
