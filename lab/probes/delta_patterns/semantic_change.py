"""Which semantic directions did MiMo change most, and are they new or already in the base? Exploratory, no forward pass.

1. Token rows: relative change |row_mimo - row_base| / |row_base| of embed_tokens (reading) and lm_head (writing).
   Top tokens and the median: a token far above the median is a vocabulary item MiMo re-learned.
2. Per down_proj (writes the residual stream), D = W_mimo - W_base, top-3 left singular directions of D:
   - share of |D|^2 each holds;
   - logit lens: tokens most promoted / suppressed (sign of a singular vector is arbitrary: both ends listed);
   - novelty: fraction of the direction outside the base's own top-64 output subspace of that matrix
     (chance for a random direction: 1 - 64/4096 = 0.984). Low = MiMo rescaled something the base already wrote;
     high = a direction the base barely used.
Logit lens is unreliable in low layers; directions superpose features. Hints, not meanings.

usage: python lab/probes/delta_patterns/semantic_change.py -> lab/runs/delta_patterns/semantic_change.json
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

BASE = ROOT / "hf_cache" / "qwen35-9b-hf"
MIMO = ROOT / "hf_cache" / "mimo-distill-hf"
OUT = ROOT / "lab" / "runs" / "delta_patterns" / "semantic_change.json"
TOP = 64


def row_change(b, m, name, tok, n=30):
    W0, W1 = b.get(name), m.get(name)
    rel = torch.cat([((W1[i:i + 16384].float() - W0[i:i + 16384].float()).norm(dim=1)
                      / W0[i:i + 16384].float().norm(dim=1).clamp_min(1e-6)) for i in range(0, W0.shape[0], 16384)])
    idx = torch.topk(rel, n).indices.tolist()
    return {"median": round(float(rel.median()), 5), "p99": round(float(rel.quantile(0.99)), 5),
            "top": [(tok.decode([i]), round(float(rel[i]), 4)) for i in idx]}


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    b = wd.Renamed(wd.LocalSafetensors(BASE), wd.text_model_name)
    m = wd.Renamed(wd.LocalSafetensors(MIMO), wd.text_model_name)
    rep = {"embed_tokens": row_change(b, m, "model.embed_tokens.weight", tok),
           "lm_head": row_change(b, m, "lm_head.weight", tok)}
    print("embed", rep["embed_tokens"]["median"], rep["embed_tokens"]["top"][:15], flush=True)
    print("lm_head", rep["lm_head"]["median"], rep["lm_head"]["top"][:15], flush=True)
    scale = 1.0 + b.get("model.norm.weight").float()
    U = b.get("lm_head.weight")
    layers = []
    for L in range(32):
        name = f"model.layers.{L}.mlp.down_proj.weight"
        W0 = b.get(name).float()
        D = m.get(name).float() - W0
        tot = float(D.pow(2).sum())
        torch.manual_seed(0)
        Ud, Sd, _ = torch.svd_lowrank(D, q=16, niter=4)
        Ub, _, _ = torch.svd_lowrank(W0, q=TOP + 16, niter=4)
        Ub = Ub[:, :TOP]
        dirs = []
        for j in range(3):
            u = Ud[:, j]
            logits = torch.cat([(U[i:i + 32768].float() * scale) @ u for i in range(0, U.shape[0], 32768)])
            hi, lo = torch.topk(logits, 8).indices.tolist(), torch.topk(-logits, 8).indices.tolist()
            dirs.append({"share": round(float(Sd[j]) ** 2 / tot, 4),
                         "novelty": round(1 - float((Ub.T @ u).pow(2).sum()), 3),
                         "plus": [tok.decode([i]) for i in hi], "minus": [tok.decode([i]) for i in lo]})
        layers.append({"layer": L, "rel_norm": round((tot / float(W0.pow(2).sum())) ** .5, 4), "dirs": dirs})
        d0 = dirs[0]
        print(f"L{L:2d} rel {layers[-1]['rel_norm']} share {d0['share']} nov {d0['novelty']} "
              f"+{d0['plus'][:5]} -{d0['minus'][:5]}", flush=True)
    rep["novelty_median_first_dir"] = round(st.median(x["dirs"][0]["novelty"] for x in layers), 3)
    rep["novelty_chance"] = round(1 - TOP / 4096, 3)
    rep["down_proj"] = layers
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("novelty median (first dir):", rep["novelty_median_first_dir"], "chance", rep["novelty_chance"])


if __name__ == "__main__":
    main()
