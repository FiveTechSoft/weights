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
    """Dequantiza y restaura la shape logica gguf (gguf-py la guarda invertida)."""
    t = _rmap(r)[name]
    a = gguf.dequantize(t.data, t.tensor_type)
    if len(t.shape) == 1:
        return a.reshape(int(t.shape[0])).astype(np.float32)
    return a.reshape(int(t.shape[0]), int(t.shape[1])).astype(np.float32)

def _meta(r):
    out = {}
    for k, f in r.fields.items():
        if not f.parts:
            continue
        v = f.parts[-1]  # el payload de un ReaderField es su ultima parte
        if v.dtype == np.uint8:
            out[k] = bytes(v).decode("utf-8", "ignore")
        elif len(v) == 1:
            out[k] = v[0].item() if hasattr(v[0], "item") else int(v[0])
        else:
            out[k] = [int(x) for x in v.tolist()]
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

def rope_rot(offset, n_rot, n_total, base=1e7):
    """R_delta [n_total x n_total]: rotacion relativa en dims [0:n_rot], identity fuera."""
    R = np.eye(n_total, dtype=np.float64)
    for i in range(n_rot // 2):
        omega = base ** (-2 * i / n_rot)
        c, s = np.cos(offset * omega), np.sin(offset * omega)
        j = i * 2
        R[j, j], R[j, j + 1] = c, s
        R[j + 1, j], R[j + 1, j + 1] = -s, c
    return R

OFFSETS = [1, 2, 3, 4, 8, 16, 32]

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
        f"// pre-rope model (el rope real rota {64} de {hd} dims; MRoPE multimodal, freq 1e7)",
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
        f"// --- diagnostico QK (pre-rope estatico): M0 = Wq^T @ Wk (normas escaladas)",
        f"//     c1(M0)={c0:.4f}; nota: RoPE ({64} dims de {hd}) altera el score por offset",
        f"//     ||Wq|| vs ||Wgate||: {q_norm_total:.2f} vs {gate_norm:.2f} (ratio gate {gate_norm / (q_norm_total + 1e-30):.3f})",
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
    """Por head/layer: c1 estatica (delta=0) y c1 maxima por offset de RoPE."""
    g = attn_geometry(a.gguf)
    hd = g["head_dim"]
    layers = attn_layers(a.gguf)
    if a.layers:
        layers = [int(x) for x in a.layers.split(",") if int(x) in layers]
    rots = {d: rope_rot(d, g["rope_dims"], g["head_dim"]) for d in OFFSETS}
    rows = []
    for L in layers:
        r = _reader(a.gguf)
        Wq_all = _tens(r, f"blk.{L}.attn_q.weight")
        Wk_all = _tens(r, f"blk.{L}.attn_k.weight")
        cn = lambda W: W / (np.sqrt((W ** 2).sum(axis=0, keepdims=True)) + 1e-6)
        WkN = cn(Wk_all)
        for h in range(g["n_head"]):
            WqN = cn(Wq_all[:, h * 2 * hd:h * 2 * hd + hd])
            grp = h // (g["n_head"] // g["n_kv"])
            WkH = WkN[:, grp * hd:(grp + 1) * hd]
            c_static, _ = concentration(WqN.T @ WkH)
            best = {"c1": c_static, "delta": 0}
            specs = {}
            for d_ in OFFSETS:
                M = WqN.T @ (WkH @ rots[d_].T)
                specs[d_] = M
                c, _ = concentration(M)
                best = {"c1": c, "delta": d_} if c > best["c1"] else best
            Mbest = specs[best["delta"]] if best["delta"] else WqN.T @ WkH
            rows.append({"layer": L, "q_head": h, "group": grp,
                         "c1_static": round(c_static, 4),
                         "c1_max": round(best["c1"], 4), "delta_max": best["delta"],
                         "norm_M": round(float(np.linalg.norm(Mbest)), 3)})
            sys.stderr.write(f"L{L} h{h} stat={c_static:.3f} max={best['c1']:.3f}@d{best['delta']}\n")
    rows.sort(key=lambda x: -x["c1_max"])
    out = Path(a.out or "decomp/scan.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("top-10 c1_max (offset-aware, matcher por desplazamiento):")
    for r_ in rows[:10]:
        print(f"  L{r_['layer']:>2} q-head {r_['q_head']:>2} grp {r_['group']} "
              f"c1_max={r_['c1_max']} @delta {r_['delta_max']} (stat {r_['c1_static']})")
    print(f"escrito {out} ({len(rows)} heads)")

def cmd_decompile(a):
    H = load_head(a.gguf, a.l, a.q)
    outdir = Path(a.o or "decomp"); outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"head_L{a.l}_q{a.q}"
    payload = {k: v for k, v in H.items() if isinstance(v, np.ndarray) and k not in ("gq", "gk")}
    payload["gq"] = H["gq"] if H["gq"] is not None else np.zeros(0)
    payload["gk"] = H["gk"] if H["gk"] is not None else np.zeros(0)
    np.savez_compressed(stem.with_suffix(".npz"), **payload)
    (stem.with_suffix(".c")).write_text(render(H), encoding="utf-8")
    mf = manifest(H, stem.with_suffix(".npz"), a.gguf)
    (stem.with_suffix(".manifest.json")).write_text(json.dumps(mf, indent=2), encoding="utf-8")
    print(f"decompilado -> {stem}.npz / .c / .manifest.json")

def cmd_judge(a):
    H = load_head(a.gguf, a.l, a.q)
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
    # T3: null-test al nivel de una edicion real (QUANT-RUIDO: ediciones >= 0.5% magnitud):
    # escala una columna completa de Wq (un dim q del head) por 0.9 = -10% local
    H2 = dict(H); H2["Wq"] = H["Wq"].copy(); H2["Wq"][:, 0] *= 0.9
    r2 = ref_attn(H2, X)
    d2 = float(np.abs(r2 - r_ref).max())
    res["T3_nulltest"] = {"max_abs_shift": round(d2, 9), "pass": d2 > 1e-6}
    c0, sig4 = concentration(bilinear(H))
    res["T4_diag"] = {"c1_rankone": round(c0, 4),
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
