"""Anatomia destilacion 2.0 - paso 2: direccion dominante del delta (SVD-1) de los
tensores elegidos y su proyeccion semantica (cos vs 10 areas + seeds).
Salida: data/distill_anatomy_v2_semantics.json + tabla stdout."""
import json
import numpy as np
import gguf
from pathlib import Path

R = Path(r"C:\weights")
BASE = Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
DIST = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")

an = json.loads((R / "data" / "distill_anatomy_v2.json").read_text(encoding="utf-8"))
stats = {s["tensor"]: s for s in an["tensors"]}

def pick(pref, k=3):
    cand = sorted((s for s in an["tensors"] if pref in s["tensor"]),
                  key=lambda s: -s["rel_l2"])
    return [s["tensor"] for s in cand[:k]]

chosen = []
chosen += pick("ffn_gate", 3)
chosen += pick("ssm_beta", 3)
chosen += pick("attn_q", 1)
chosen += pick("attn_v", 1)
chosen += pick("attn_output", 1)
chosen = list(dict.fromkeys(chosen))
print("elegidos:", len(chosen))

rb = {t.name: t for t in gguf.GGUFReader(str(BASE)).tensors}
rd = {t.name: t for t in gguf.GGUFReader(str(DIST)).tensors}

cent = np.load(R / "data" / "mimo-v2.6-distill-qwen-9b-Q6_K.semantic_centroids.npz")
AREAS = ["math_logic", "agent_action", "code_software", "abstract_ideas",
         "mind_cognition", "time_sequence", "emotion_pos", "emotion_neg",
         "physical_world", "social_power"]
area_mat = np.stack([cent[a].astype(np.float64) for a in AREAS])
area_mat /= np.linalg.norm(area_mat, axis=1, keepdims=True)
seeds = [k for k in cent.files if k.startswith("seed_")]
seed_mat = np.stack([cent[s].astype(np.float64) for s in seeds])
seed_mat /= np.linalg.norm(seed_mat, axis=1, keepdims=True)

out = {"meta": {"note": "SVD-1 del delta (Distill-Base) por tensor; cos contra centroides semanticos"},
       "tensors": []}
for name in chosen:
    def deq(t, path_lbl):
        a = gguf.dequantize(t.data, t.tensor_type)
        if len(t.shape) == 2:
            return a.reshape(int(t.shape[0]), int(t.shape[1])).astype(np.float32)
        return a.reshape(int(t.shape[0])).astype(np.float32)
    D = deq(rd[name], "d") - deq(rb[name], "b")
    if D.ndim == 1:
        D2 = D[None, :]
    else:
        D2 = D
    G = D2 @ D2.T  # [d_in, d_in]
    w, v = np.linalg.eigh(G.astype(np.float64))
    u = v[:, -1]
    if float(D2.T @ u @ D2.T @ u) < 0:
        pass
    s1 = float(np.sqrt(max(w[-1], 0.0)))
    u = u / (np.linalg.norm(u) + 1e-30)
    cosA = (area_mat @ u)
    cosA_rank = sorted(zip(AREAS, cosA), key=lambda x: -abs(x[1]))
    cosS = seed_mat @ u
    top_seeds = sorted(zip(seeds, cosS), key=lambda x: -abs(x[1]))[:5]
    rec = {"tensor": name, "rel_l2": stats.get(name, {}).get("rel_l2"),
           "shape": list(D2.shape), "sv1": s1,
           "area_cos": {a: round(float(c), 4) for a, c in cosA_rank},
           "top_seeds": {s: round(float(c), 4) for s, c in top_seeds}}
    out["tensors"].append(rec)
    cos_str = " ".join(f"{a}:{c:+.3f}" for a, c in cosA_rank[:3])
    sd_str = " ".join(f"{s.replace('seed_','')}:{c:+.3f}" for s, c in top_seeds[:3])
    print(f"{name:<34} sv1={s1:>9.3f}  {cos_str}   | {sd_str}")

(R / "data" / "distill_anatomy_v2_semantics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nguardado distill_anatomy_v2_semantics.json")
