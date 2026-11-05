"""Verificacion independiente solicitada por el propietario del repo: texto reservado REAL, copia del corpus, base vs snapshot."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "engine"))
from engine.nanowsp3_runtime import NanoWSP3

z = np.load(R / "lab" / "runs" / "nanowsp3_trace.npz")
names = json.loads(str(z["names"]))
shapes = json.loads(json.loads(str(z["shapes"])))
def sd_from_flat(flat):
    sd, off = {}, 0
    for k in names:
        n = int(np.prod(shapes[k]))
        sd[k] = torch.tensor(flat[off:off+n].reshape(shapes[k])); off += n
    return sd

def load_model(snap_idx):
    m = NanoWSP3()
    m.weights.update({m.load_checkpoint.__self__.config and k: v for k, v in {}})  # noop
    sd = sd_from_flat(z["snapshots_flat"][snap_idx])
    key_map = {"embed.weight": "embedding.weight", "out.weight": "output.weight", "fn.weight": "final_norm.weight"}
    for l in range(m.config.n_layers):
        for a, b in (("q","q_proj"),("k","k_proj"),("v","v_proj"),("o","out_proj")):
            key_map[f"{a}.{l}"] = f"layers.{l}.attention.{b}.weight"
        key_map[f"an.{l}"] = f"layers.{l}.attn_norm.weight"
        key_map[f"mn.{l}"] = f"layers.{l}.mlp_norm.weight"
        key_map[f"min_.{l}"] = f"layers.{l}.mlp_in.weight"
        key_map[f"mout.{l}"] = f"layers.{l}.mlp_out.weight"
    for k, v in sd.items():
        m.weights[key_map.get(k, k)] = v.float()
    return m

raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
data = raw[15000:15000 + 400000].encode("ascii", "ignore").decode()
ids = np.frombuffer(data.encode("ascii", "ignore"), dtype=np.uint8).astype(np.int64)
n_hold = min(40000, max(len(ids) // 10, 1000))
train_txt = data[:-n_hold]; hold_txt = data[-n_hold:]
battery_txt = raw[15000:115000][-20000:]  # lo que la bateria llamo "holdout"

def ppl(m, text, limit=2500):
    tids = [ord(c) for c in text if ord(c) < 256][:limit]
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tids) - 65, 64):
            x = tids[i:i+64]; y = tids[i+1:i+65]
            lg, _ = m.forward(x)
            tot += float(F.cross_entropy(lg[0], torch.tensor(y), reduction="sum").item())
            n += len(y)
    return math.exp(tot / max(n, 1))

def copy_stats(m, prompts, n_chars=90):
    out = []
    for p in prompts:
        g = m.generate(p, max_new_tokens=n_chars, temperature=0.0)
        tail = g[len(p):]
        grams = {tail[i:i+8] for i in range(max(0, len(tail)-7))}
        inc = sum(1 for gr in grams if gr in train_txt)
        # match verbatim mas largo contra corpus train
        best = 0
        for i in range(len(tail)):
            for L in range(best+1, len(tail)-i+1):
                if tail[i:i+L] in train_txt: best = L
                else: break
        out.append({"prompt": p, "tail": tail[:80], "pct_8gram_en_train": round(100*inc/max(len(grams),1),1),
                    "match_verbatim_max_chars": best})
    return out

prompts = ["Alice was ", "The Queen of Hearts ", "Down the rabbit-hole, she ",
           "She said to the ", "Once upon a time ", "The sky was "]
res = {"nota": "modelo BASE (sin cirugia), artefacto reconstruido del trace; evaluacion con runtime actual"}
for idx, tag in ((1, "snapshot_600"), (4, "snapshot_1500_final")):
    m = load_model(idx)
    res[tag] = {
        "ppl_train_slice": round(ppl(m, train_txt), 2),
        "ppl_TRUE_holdout": round(ppl(m, hold_txt), 2),
        "ppl_battery_slice_(es_texto_de_train)": round(ppl(m, battery_txt), 2),
    }
m1500 = load_model(4)
res["copia"] = copy_stats(m1500, prompts)
res["generacion_temp0_80c"] = [{"prompt": p, "tail": m1500.generate(p, max_new_tokens=80, temperature=0.0)[len(p):]} for p in prompts]
print(json.dumps(res, indent=1, ensure_ascii=False))
Path("/tmp/indep_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
