"""Audit decomp/mimo_head.py decompile/judge on Qwen3.5-0.8B (qwen35 arch, read-only)."""
import json
import subprocess
import sys
import numpy as np
from harness import dump, OUT, ROOT, PY
sys.path.insert(0, str(ROOT / "decomp"))
import mimo_head as mh
import wsp

R = {}
G = str(ROOT / "hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf")
geo = mh.attn_geometry(G)
R["geometry"] = {k: geo[k] for k in ("n_head", "n_kv", "head_dim", "q_per_head", "d_model", "rope_dims")} | {"attn_layers": geo["attn_layers"][:8]}

# 1) _tens: "restores logical shape" by reshape. Compare against transpose of the canonical (out,in) decode.
r = mh._reader(G)
for name in ("blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_output.weight"):
    t = mh._rmap(r)[name]
    a = mh._tens(r, name)
    canon = wsp.row_decode(t)  # (ne1=out, ne0=in)
    R[f"tens_{name}"] = {"gguf_shape(ne0,ne1)": [int(x) for x in t.shape], "tens_shape": list(a.shape),
                         "equals_canon_T": bool(np.array_equal(a, canon.T)),
                         "max_abs_diff_vs_canon_T": float(np.abs(a - canon.T).max()),
                         "row0_corr_with_canon_T": float(np.corrcoef(a[:, 0], canon.T[:, 0])[0, 1])}

# 2) render(): hard-coded geometry strings vs this model
H = mh.load_head(G, 3, 0)
txt = mh.render(H)
R["render_hardcoded"] = {"says_of_16": "de 16" in txt, "says_kv_heads_4": "kv-heads 4" in txt, "model_n_head": geo["n_head"], "model_n_kv": geo["n_kv"]}

# 3) judge on the real head and on a pure-noise head with identical shapes
def judge(Hh):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 0.05, (8, geo["d_model"])).astype(np.float32)
    r_ref = mh.ref_attn(Hh, X); r_run = mh.run_prog(mh.program(Hh, X), X)["r"]
    cos = float((r_ref * r_run).sum() / (np.linalg.norm(r_ref) * np.linalg.norm(r_run) + 1e-30))
    H2 = dict(Hh); H2["Wq"] = Hh["Wq"].copy(); H2["Wq"][:, 0] *= 0.9
    d2 = float(np.abs(mh.ref_attn(H2, X) - r_ref).max())
    return {"T2_cos": cos, "T2_pass": cos > 0.9999, "T3_shift": d2, "T3_pass": d2 > 1e-6}
R["judge_real_head"] = judge(H)
g = np.random.default_rng(5)
Hn = dict(H)
for k in ("Wq", "Wgate", "Wk", "Wv", "Wo"):
    Hn[k] = g.standard_normal(H[k].shape).astype(np.float32) * float(H[k].std())
R["judge_noise_head"] = judge(Hn)
cp = subprocess.run([PY, str(ROOT / "decomp/mimo_head.py"), "judge", G, "-l", "3", "-q", "0"], capture_output=True, text=True, cwd=str(ROOT))
R["judge_cli_real"] = cp.stdout[:600] or cp.stderr[-600:]

# 4) is the IR / ref_attn the real computation? compare head output with transposed (correct) weights
Hc = dict(H)
qa = wsp.row_decode(mh._rmap(r)["blk.3.attn_q.weight"]).T; hd = geo["head_dim"]
Hc["Wq"], Hc["Wgate"] = qa[:, 0:hd], qa[:, hd:2 * hd]
ka = wsp.row_decode(mh._rmap(r)["blk.3.attn_k.weight"]).T; Hc["Wk"] = ka[:, 0:hd]
va = wsp.row_decode(mh._rmap(r)["blk.3.attn_v.weight"]).T; Hc["Wv"] = va[:, 0:hd]
oa = wsp.row_decode(mh._rmap(r)["blk.3.attn_output.weight"]).T; Hc["Wo"] = oa[0:hd, :]
X = np.random.default_rng(1).normal(0, 1, (8, geo["d_model"])).astype(np.float32)
a1, a2 = mh.ref_attn(H, X), mh.ref_attn(Hc, X)
R["head_output_tool_vs_transposed_weights"] = {"cos": float((a1 * a2).sum() / (np.linalg.norm(a1) * np.linalg.norm(a2))),
                                               "c1_tool": mh.concentration(mh.bilinear(H))[0], "c1_transposed": mh.concentration(mh.bilinear(Hc))[0]}

# 5) decompile output into audit dir
cp = subprocess.run([PY, str(ROOT / "decomp/mimo_head.py"), "decompile", G, "-l", "3", "-q", "0", "-o", str(OUT / "decomp")], capture_output=True, text=True, cwd=str(ROOT))
R["decompile_cli"] = cp.stdout.strip()[-200:] or cp.stderr[-300:]
print(json.dumps(R, indent=1, default=str))
dump("mimo_head", R)
