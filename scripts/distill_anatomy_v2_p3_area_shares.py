"""Anatomia p3: participacion energetica por area semantica: share_a = ||a^T Delta||_F^2 / ||Delta||_F^2.
Si la destilacion fuese 'area-neutra', cada share ~ (||Delta_a||... baseline = 1/10 ajustado por norma del centroide.
Salida: data/distill_anatomy_v3_area_shares.json + tabla."""
import json
import numpy as np
import gguf
from pathlib import Path

R = Path(r"C:\weights")
BASE = Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
DIST = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")

an = json.loads((R / "data" / "distill_anatomy_v2.json").read_text(encoding="utf-8"))
def pick(pref, k=3):
    cand = sorted((s for s in an["tensors"] if pref in s["tensor"]), key=lambda s: -s["rel_l2"])
    return [s["tensor"] for s in cand[:k]]
chosen = list(dict.fromkeys(pick("ffn_gate", 3) + pick("ffn_down", 2) + pick("ssm_beta", 3)
                            + pick("attn_q", 2) + pick("attn_v", 1) + pick("attn_output", 1)))
rb = {t.name: t for t in gguf.GGUFReader(str(BASE)).tensors}
rd = {t.name: t for t in gguf.GGUFReader(str(DIST)).tensors}
cent = np.load(R / "data" / "mimo-v2.6-distill-qwen-9b-Q6_K.semantic_centroids.npz")
AREAS = ["math_logic", "agent_action", "code_software", "abstract_ideas",
         "mind_cognition", "time_sequence", "emotion_pos", "emotion_neg",
         "physical_world", "social_power"]
A = np.stack([cent[a].astype(np.float64) for a in AREAS])
A_list = [a for a in A]
seed_sha = {s: np.linalg.norm(cent[s]) for s in ("seed_bash", "seed_integer", )}

def deq(t):
    a = gguf.dequantize(t.data, t.tensor_type)
    if len(t.shape) == 2:
        return a.reshape(int(t.shape[0]), int(t.shape[1])).astype(np.float32)
    return a.reshape(int(t.shape[0])).astype(np.float32)

# baseline de shares: el centroide medio azarige 1/10 del espacio => share ~ cos^2/norm-sum
# mas util: baseline empirico con 40 direcciones aleatorias unit
rng = np.random.default_rng(42)
out = {"meta": {"metric": "share_a = ||a^T Delta||_2 / ||Delta||_2 ; baseline aleatoria ~ 0.316/unit-norm orientation"}, "tensors": []}
for name in chosen:
    Delta = deq(rd[name]) - deq(rb[name])
    if Delta.ndim == 1:
        Delta = Delta[None, :]
    tot = float(np.linalg.norm(Delta))
    shares = {}
    for ai, a in zip(AREAS, A_list):
        if Delta.shape[0] == a.shape[0]:
            row = a @ Delta                     # proyeccion sobre eje residual (las filas)
        elif Delta.shape[1] == a.shape[0]:
            row = Delta @ a                     # eje residual en columnas (ffn_down-style)
        else:
            shares[ai] = -1.0
            continue
        shares[ai] = round(float(np.linalg.norm(row)) / (tot + 1e-30), 5)
    # baseline: shares medianos de 8 random dirs unit (misma orientacion que areas)
    rnd = rng.normal(size=(8, A.shape[1]))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    base_share = np.mean([float(np.linalg.norm((r @ Delta) if Delta.shape[0] == r.shape[0] else (Delta @ r))) / (tot + 1e-30) for r in rnd])
    top = sorted(shares.items(), key=lambda x: -x[1])[:4]
    out["tensors"].append({"tensor": name, "shares": shares,
                           "random_baseline": round(base_share, 5)})
    print(f"{name:<34} base_rnd={base_share:.3f} | " +
          " ".join(f"{a[:12]}={shares[a]:.3f}" for a, _ in top))
(R / "data" / "distill_anatomy_v3_area_shares.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("guardado v3_area_shares.json")
