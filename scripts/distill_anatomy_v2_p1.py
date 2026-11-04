"""Anatomia destilacion 2.0 - paso 1: drift por tensor (base Qwen3.5-9B vs MiMo-9B, GGUF Q6_K).
Paso 2 (mismo script): para los top-N tensores, SVD de delta y proyeccion sobre centroides semanticos.
Salida: data/distill_anatomy_v2.json + resumen en stdout."""
import json
import numpy as np
import gguf
from pathlib import Path

R = Path(r"C:\weights")
BASE = R and Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
DIST = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")

def rmap(p):
    r = gguf.GGUFReader(str(p))
    return {t.name: t for t in r.tensors}

def deq(t):
    a = gguf.dequantize(t.data, t.tensor_type)
    if len(t.shape) == 2:
        return a.reshape(int(t.shape[0]), int(t.shape[1])).astype(np.float32)
    return a.reshape(int(t.shape[0])).astype(np.float32)

tb, td = rmap(BASE), rmap(DIST)
common = sorted(set(tb) & set(td))
print(f"tensores comunes: {len(common)}")

def fam(n):
    if "ffn_down" in n: return "ffn_down"
    if "ffn_up" in n: return "ffn_up"
    if "ffn_gate" in n: return "ffn_gate"
    if "attn_q" in n: return "attn_q"
    if "attn_k" in n: return "attn_k"
    if "attn_v" in n: return "attn_v"
    if "attn_output" in n: return "attn_output"
    if "ssm" in n or "attn_gate" in n or "conv" in n: return "gdn_ssm"
    if "norm" in n: return "norm"
    if "token_embd" in n: return "token_embd"
    if "output" in n: return "output_head"
    return "other"

# bloque-t chunking: rel_l2 por tensor sin cargar f64 completo (chunks de filas)
stats = []
for name in common:
    a, b = deq(tb[name]), deq(td[name])
    if a.size == 0 or a.shape != b.shape:
        continue
    if a.ndim == 1:
        num = float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        den = float(np.sum((a.astype(np.float64)) ** 2))
    else:
        num = den = 0.0
        CH = 256
        for r0 in range(0, a.shape[0], CH):
            x = a[r0:r0 + CH].astype(np.float64)
            y = b[r0:r0 + CH].astype(np.float64)
            d = x - y
            num += float(np.sum(d * d))
            den += float(np.sum(x * x))
    if den == 0:
        continue
    stats.append({"tensor": name, "fam": fam(name), "rel_l2": (num ** 0.5) / (den ** 0.5),
                  "n": int(a.size), "shape": list(a.shape)})
rel = np.array([s["rel_l2"] for s in stats])
print(f"REL MEDIANO: {np.median(rel):.4%} | p90 {np.percentile(rel,90):.3%} | max {rel.max():.3%}")
per_fam = {}
for s in stats:
    per_fam.setdefault(s["fam"], []).append(s["rel_l2"])
print("por familia (media | n | max):")
for f, r in sorted(per_fam.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"  {f:>12}: {np.mean(r):.4%}  n={len(r):>3}  max={np.max(r):.3%}")

top = sorted(stats, key=lambda s: -s["rel_l2"])[:24]
print("\ntop-24 tensores por rel_l2:")
for s in top:
    print(f"  {s['tensor']:<44} {s['rel_l2']:.3%}  n={s['n']:,}")

out = {"meta": {"base": str(BASE), "distill": str(DIST), "n_common": len(common),
                "median_rel": float(np.median(rel))},
       "per_fam": {f: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                   for f, v in per_fam.items()},
       "tensors": stats,
       "top24": top}
(R / "data" / "distill_anatomy_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nguardado data/distill_anatomy_v2.json")
