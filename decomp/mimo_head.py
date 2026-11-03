#!/usr/bin/env python3
"""WSP decompile (mini #11): traductor de un head de atencion Qwen35/MiMo a pseudo-C.

Arquitectura de un head (de src/models/qwen35.cpp de llama.cpp, verificado):
  attn_q tengo [d, n_head*512]: por head h, cols [0:256] = Q, cols [256:512] = GATE
  attn_k/attn_v [d, n_kv*256], GQA group = h // (n_head//n_kv)
  q/k con RMS-norm per-head (gq/gk) + rope (este tool modela PRE-rope, declarado)
  out = softmax(Q.K/sqrt(256)) @ V, multiplicado por sigmoid(X @ Wgate), luego wo rows [h*256*...]

Modos:
  dims      <gguf>                    -> geometria de atencion
  scan      <gguf> [--layers]         -> por head/layer: concentracion rank-1 de M=Wq^T@Wk
  decompile <gguf> -l L -q Q -o dir   -> npz + pseudo-C + manifiesto del head
  judge     <gguf> -l L -q Q          -> tests mecanicos T1..T4
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import gguf

CJK = {}  # cache de readers

def _reader(p):
    if p not in CJK:
        CJK[p] = gguf.GGUFReader(p)
    return CJK[p]

def _rmap(r):
    return {t.name: t for t in r.tensors}

def _tens(r, name):
    t = _rmap(r)[name]
    return gguf.dequantize(t.data, t.tensor_type).astype(np.float32)

def _meta(r):
    out = {}
    for k, f in r.fields.items():
        vals = []
        for part in f.parts:
            if part.dtype == np.uint8:
                vals.append(bytes(part).decode("utf-8", "ignore"))
            else:
                v = part.tolist()
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                vals.append(v)
        out[k] = vals[0] if len(vals) == 1 else vals
    return out

def attn_layers(gguf_path):
    r = _reader(gguf_path)
    return sorted(int(t.name.split(".")[1]) for t in r.tensors
                  if t.name.endswith("attn_q.weight"))

def attn_geometry(gguf_path):
    m = _meta(_reader(gguf_path))
    A = "qwen35"
    r = _reader(gguf_path)
    names = {t.name for t in r.tensors}
    qname = next(n for n in names if n.endswith("attn_q.weight"))
    q_shape = tuple(r.tensors[[t.name for t in r.tensors].index(qname)].shape)
    names = {t.name for t in r.tensors}
    hd = int(m.get(f"{A}.attention.key_length"))
    return {
        "arch": A,
        "n_head": int(m[f"{A}.attention.head_count"]),
        "n_kv": int(m[f"{A}.attention.head_count_kv"]),
        "head_dim": hd,
        "q_per_head": int(q_shape[1]) // int(m[f"{A}.attention.head_count"]),
        "d_model": int(q_shape[0]),
        "rope_dims": int(m.get(f"{A}.rope.dimension_count") or 0),
        "softcap": float(m.get(f"{A}.attention.softcap") or 0.0),
        "scale": float(1.0 / np.sqrt(hd)),
        "attn_layers": attn_layers(gguf_path),
    }

def load_head(gguf_path, layer, q_head):
    """Tensors de-cuantizados de UN q-head con su grupo kv (semantica qwen35)."""
    g = attn_geometry(gguf_path)
    d, hd, h = g["d_model"], g["head_dim"], q_head
    r = _reader(gguf_path)
    grp = h // (g["n_head"] // g["n_kv"])
    Wqf = _tens(r, f"blk.{layer}.attn_q.weight")[:, h * 2 * hd:(h + 1) * 2 * hd]
    Wq, Wgate = Wqf[:, :hd], Wqf[:, hd:]
    Wk = _tens(r, f"blk.{layer}.attn_k.weight")[:, grp * hd:(grp + 1) * hd]
    Wv = _tens(r, f"blk.{layer}.attn_v.weight")[:, grp * hd:(grp + 1) * hd]
    Wo = _tens(r, f"blk.{layer}.attn_output.weight")[h * hd:(h + 1) * hd, :]
    names = _rmap(r)
    gq = _tens(r, f"blk.{layer}.attn_q_norm.weight") if f"blk.{layer}.attn_q_norm.weight" in names else None
    gk = _tens(r, f"blk.{layer}.attn_k_norm.weight") if f"blk.{layer}.attn_k_norm.weight" in names else None
    return {"d": d, "hd": hd, "q_head": h, "group": grp, "layer": layer,
            "scale": g["scale"], "softcap": g["softcap"],
            "Wq": Wq, "Wgate": Wgate, "Wk": Wk, "Wv": Wv, "Wo": Wo,
            "gq": gq, "gk": gk}

def rms(x, gamma=None):
    n = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    y = x / n
    return y * gamma if gamma is not None else y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ref_attn(H, X):
    """Semantica qwen35 de UN head, PRE-rope. X [T,d] -> contribucion residual [T,d]."""
    q = rms(X @ H["Wq"], H["gq"])
    k = rms(X @ H["Wk"], H["gk"])
    v = X @ H["Wv"]
    gate = sigmoid(X @ H["Wgate"])
    T = X.shape[0]
    s = (q @ k.T) * H["scale"]
    s = np.where(np.triu(np.ones((T, T), bool), 1), -1e30, s)
    if H.get("softcap"):
        s = H["softcap"] * np.tanh(s / H["softcap"])
    e = np.exp(s - s.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    o = (p @ v) * gate
    return o @ H["Wo"]

def bilinear(H):
    """M = (Wq escalado)^T @ (Wk escalado): x_t^T M x_s = logit basal sin rope."""
    Wq, Wk = H["Wq"], H["Wk"]
    if H["gq"] is not None:
        Wq = Wq * H["gq"][None, :]
    if H["gk"] is not None:
        Wk = Wk * H["gk"][None, :]
    Wq = Wq / (np.sqrt((Wq ** 2).sum(1, keepdims=True)) + 1e-6)
    Wk = Wk / (np.sqrt((Wk ** 2).sum(1, keepdims=True)) + 1e-6)
    return Wq.T @ Wk

def concentration(M):
    sig = np.linalg.svd(M, compute_uv=False)
    return float(sig[0] ** 2 / (sig ** 2).sum() + 1e-30), sig[:4].tolist()

# ---------- IR interpretable + ranprogram runs the same order as pseudo-C ----------

def program(H, X):
    steps = [
        ("matmul", "q", ("X", H["Wq"])),
        ("rms", "q", ("q", H["gq"])),
        ("matmul", "k", ("X", H["Wk"])),
        ("rms", "k", ("k", H["gk"])),
        ("matmul", "v", ("X", H["Wv"])),
        ("matmul", "gate", ("X", H["Wgate"])),
        ("sigmoid", "gate", ("gate",)),
        ("dotscale", "s", ("q", "k", H["scale"])),
        ("causal", "s", ("s",)),
        ("softmax", "p", ("s",)),
        ("attnmix", "o", ("p", "v")),
        ("gate_mul", "o", ("o", "gate")),
        ("matmul", "r", ("o", H["Wo"])),
    ]
    return steps

def run_prog(prog, X):
    regs = {"X": X}
    for op, dst, args in prog:
        if op == "matmul":
            regs[dst] = regs[args[0]] @ args[1]
        elif op == "rms":
            regs[dst] = rms(regs[args[0]], args[1])
        elif op == "sigmoid":
            regs[dst] = sigmoid(regs[args[0]])
        elif op == "dotscale":
            regs[dst] = (regs[args[0]] @ regs[args[1]].T) * args[2]
        elif op == "causal":
            s = regs[args[0]]; T = s.shape[0]
            regs[dst] = np.where(np.triu(np.ones((T, T), bool), 1), -1e30, s)
        elif op == "softmax":
            s = regs[args[0]]
            e = np.exp(s - s.max(-1, keepdims=True))
            regs[dst] = e / e.sum(-1, keepdims=True)
        elif op == "attnmix":
            regs[dst] = regs[args[0]] @ regs[args[1]]
        elif op == "gate_mul":
            regs[dst] = regs[args[0]] * regs[args[1]]
    return regs

# ---------- pseudo-C renderer ----------

def render(H):
    L, h, g, hd = H["layer"], H["q_head"], H["group"], H["hd"]
    c0, sig4 = concentration(bilinear(H))
    gate_norm = float(np.linalg.norm(H["Wgate"]))
    q_norm_total = float(np.linalg.norm(H["Wq"]))
    lines = [
        f"// ============ head decompilado ============",
        f"// layer {L}, q-head #{h} de 16 (kv-group {g}, kv-heads 4, GDA ratio 4)",
        f"// dims por head: Q {hd}d, GATE {hd}d sigmoid, K {hd}d, V {hd}d, writeback {hd}x{H['d']}",
        f"// pre-rope model (rope real: {64} dims longitudinisin secciones multimo)",
        f"const float SCALE = {H['scale']:.9f};",
        "",
        "// --- por cada fila t del residual x[t]:",
        "  q_t = RMS_norm( Wq @ x[t] )",
        "  k_s = RMS_norm( Wk @ x[s] )        // compartido por group",
        "  v_s = Wv @ x[s]",
        "  g_t = sigmoid( Wgate @ x[t] )      // gate multiplicative del head",
        "",
        "// --- scores y mezcla (causal)",
        "  score[t][s] = SCALE * dot(q_t, k_s)",
        "  p[t][*] = softmax(score)",
        "  o_t = sum_s p[t][s] * v_s",
        "  o_t *= g_t                          // QWEN35: gate sigmoid",
        "",
        "// --- writeback",
        f"  residual[t] += Wo(row-block head #{h}) @ o_t",
        "",
        f"// --- diagnostico QK (sin rope): M = Wq^T @ Wk  (con normas escaladas)",
        f"//     concentracion rank-1 = {c0:.4f}; sigma_top4 = {[round(s, 3) for s in sig4]}",
        f"//     ||Wq|| vs ||Wgate||: {q_norm_total:.2f} vs {gate_norm:.2f} (ratio gate {gate_norm / (q_norm_total + 1e-30):.3f})",
        f"// {'-> matcher monotipico: 1 direccion en QK domina (candidato matching/copy)' if c0 > 0.5 else '-> QK multieje; matcher no trivial'}",
    ]
    return "\n".join(lines)

# ---------- manifest ----------

def manifest(H, npz_path, gguf_path):
    hh = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hh.update(chunk)
    return {"tool": "mimo_head.py (WSP decompile mini #11)",
            "gguf_size": Path(gguf_path).stat().st_size,
            "layer": H["layer"], "q_head": H["q_head"], "kv_group": H["group"],
            "head_dim": H["hd"], "d_model": H["d"], "scale": H["scale"],
            "pre_rope": True, "semantics": "qwen35.cpp (llama.cpp master)",
            "npz_sha256": hh.hexdigest()}

# ---------- cmds ----------

def cmd_dims(a):
    print(json.dumps(attn_geometry(a.gguf), indent=2))

def cmd_scan(a):
    g = attn_geometry(a.gguf)
    layers = attn_layers(a.gguf)
    if a.layers:
        layers = [int(x) for x in a.layers.split(",") if int(x) in layers]
    rows = []
    for L in layers:
        r = _reader(a.gguf)
        Wq_all = _tens(r, f"blk.{L}.attn_q.weight")
        Wk_all = _tens(r, f"blk.{L}.attn_k.weight")
        Wk_all = Wk_all / (np.sqrt((Wk_all ** 2).sum(axis=0, keepdims=True)) + 1e-6)
        for h in range(g["n_head"]):
            Wq = Wq_all[:, h * 2 * g["head_dim"]:h * 2 * g["head_dim"] + g["head_dim"]]
            if H_g := a and None:
                pass
            grp = h // (g["n_head"] // g["n_kv"])
            M = Wq.T @ Wk_all[:, grp * g["head_dim"]:(grp + 1) * g["head_dim"]]
            c0, _ = concentration(M)
            rows.append({"layer": L, "q_head": h, "group": grp,
                         "c1": round(c0, 4),
                         "norm_M": round(float(np.linalg.norm(M)), 3)})
            sys.stderr.write(f"L{L} h{h} c1={c0:.3f}\n")
    rows.sort(key=lambda x: (-x["c1"], x["norm_M"]))
    out = Path(a.out or "decomp/scan.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("top-10 c1 (matcher monotipico):")
    for r_ in rows[:10]:
        print(f"  L{r_['layer']:>2} q-head {r_['q_head']:>2} grp {r_['group']} c1={r_['c1']} |M|={r_['norm_M']}")
    print(f"escrito {out} ({len(rows)} heads)")

def cmd_decompile(a):
    H = load_head(a.gguf, a.layer, a.q_head)
    outdir = Path(a.o or "decomp"); outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"head_L{a.layer}_q{a.q_head}"
    np.savez_compressed(stem.with_suffix(".npz"),
                        **{k: v for k, v in H.items() if isinstance(v, np.ndarray)},
                        gq=(H["gq"] if H["gq"] is not None else np.zeros(0)),
                        gk=(H["gk"] if H["gk"] is not None else np.zeros(0)))
    (stem.with_suffix(".c")).write_text(render(H), encoding="utf-8")
    mf = manifest(H, stem.with_suffix(".npz"), a.gguf)
    (stem.with_suffix(".manifest.json")).write_text(json.dumps(mf, indent=2), encoding="utf-8")
    print(f"decompilado -> {stem}.npz / .c / .manifest.json")
    print(render(H))

def cmd_judge(a):
    H = load_head(a.gguf, a.layer, a.q_head)
    g = attn_geometry(a.gguf)
    rng = np.random.default_rng(42)
    res = {}
    T1 = (H["Wq"].shape == (g["d_model"], g["head_dim"]) and
          H["Wgate"].shape == (g["d_model"], g["head_dim"]) and
          H["Wk"].shape == (g["d_model"], g["head_dim"]) and
          H["Wv"].shape == (g["d_model"], g["head_dim"]) and
          H["Wo"].shape == (g["head_dim"], g["d_model"]) and
          (H["gq"] is None or H["gq"].shape == (g["head_dim"],)))
    res["T1_shapes"] = bool(T1)
    X = rng.normal(0, 0.05, (8, g["d_model"])).astype(np.float32)
    r_ref = ref_attn(H, X)
    regs = run_prog(program(H, X), X)
    r_run = regs["r"]
    cos = (r_ref * r_run).sum() / (np.linalg.norm(r_ref) * np.linalg.norm(r_run) + 1e-30)
    max_abs = float(np.abs(r_ref - r_run).max())
    res["T2_ir_eq_ref"] = {"cos": round(float(cos), 9), "max_abs": round(max_abs, 9)}
    res["T2_pass"] = bool(cos > 0.9999)
    H2 = dict(H); H2["Wq"] = H["Wq"].copy(); H2["Wq"][0, 0] *= -1.0
    r2 = ref_attn(H2, X)
    d2 = float(np.abs(r2 - r_ref).max())
    res["T3_nulltest"] = {"max_abs_shift": round(d2, 9), "pass": d2 > 1e-6}
    c0, sig4 = concentration(bilinear(H))
    res["T4_diag"] = {"c1_rank(one": None, "c1_rankone": round(c0, 4),
                      "sigma_top4": [round(s, 4) for s in sig4],
                      "gate_ratio": round(float(np.linalg.norm(H["Wgate"]) / (np.linalg.norm(H["Wq"]) + 1e-30)), 4)}
    verdict = "PASS" if (res["T1_shapes"] and res["T2_pass"] and res["T3_nulltest"]["pass"]) else "FAIL"
    print(json.dumps({"verdict": verdict, **res}, indent=2))

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("dims"); p.add_argument("gguf"); p.set_defaults(fn=cmd_dims)
    p = sp.add_parser("scan"); p.add_argument("gguf"); p.add_argument("--layers"); p.add_argument("--out"); p.set_defaults(fn=cmd_scan)
    p = sp.add_parser("decompile"); p.add_argument("gguf"); p.add_argument("-l", type=int, required=True)
    p.add_argument("-q", type=int, required=True); p.add_argument("-o"); p.set_defaults(fn=cmd_decompile)
    p = sp.add_parser("judge"); p.add_argument("gguf"); p.add_argument("-l", type=int, required=True)
    p.add_argument("-q", type=int, required=True); p.set_defaults(fn=cmd_judge)
    a = ap.parse_args()
    a.fn(a)

if __name__ == "__main__":
    main()
