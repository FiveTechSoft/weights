"""How much of MiMo's update lies in a small *semantic* subspace: the residual directions that write the loop tokens.

Semantic basis Q: orthonormalised effective unembedding rows u_t = lm_head[t] * (1 + final_norm) of the loop tokens
(Wait, Okay, Hmm, check, Alternatively, </think>, ... as in lens_shared.py), k = number of such tokens.
For every down_proj (4096 x 12288, writes the residual stream), with D = W_mimo - W_base:
  frac_sem    = |Q Q^T D|^2 / |D|^2          (share of the update that writes loop-token directions)
  frac_rand   = same for a random k-dim basis (chance is k/4096)
  frac_top    = same for D's own top-k left singular vectors (the most any k-dim subspace can hold)
Exploratory precheck for H5 (h5_plan.md), no forward pass. Logit lens is unreliable in low layers.

usage: python lab/probes/delta_patterns/semantic_energy.py -> lab/runs/delta_patterns/semantic_energy.json
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import weight_diff as wd  # noqa: E402
from engine.subspace_ops import random_basis  # noqa: E402
from lens_shared import LOOP_WORDS, token_ids  # noqa: E402

BASE = ROOT / "hf_cache" / "qwen35-9b-hf"
MIMO = ROOT / "hf_cache" / "mimo-distill-hf"
OUT = ROOT / "lab" / "runs" / "delta_patterns" / "semantic_energy.json"


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    b = wd.Renamed(wd.LocalSafetensors(BASE), wd.text_model_name)
    m = wd.Renamed(wd.LocalSafetensors(MIMO), wd.text_model_name)
    scale = 1.0 + b.get("model.norm.weight").float()
    loop = token_ids(tok, LOOP_WORDS)
    rows = b.get("lm_head.weight")[list(loop.values())].float() * scale       # (k, 4096)
    Q = torch.linalg.qr(rows.T.double())[0].float()                            # (4096, k)
    k = Q.shape[1]
    R = random_basis(4096, k, 7)
    per = []
    for L in range(32):
        name = f"model.layers.{L}.mlp.down_proj.weight"
        D = m.get(name).float() - b.get(name).float()
        tot = float(D.pow(2).sum())
        U = torch.linalg.svd(D, full_matrices=False)[0][:, :k]
        f = lambda B: float((B.T @ D).pow(2).sum()) / tot                      # noqa: E731
        per.append({"layer": L, "frac_sem": round(f(Q), 5), "frac_rand": round(f(R), 5),
                    "frac_top": round(f(U), 5), "rel_norm": round((tot / float(b.get(name).float().pow(2).sum())) ** .5, 4)})
        print(per[-1], flush=True)
    rep = {"tokens": loop, "k": k, "chance": round(k / 4096, 5)}
    for lo, hi in ((0, 15), (16, 23), (24, 31)):
        sel = [p for p in per if lo <= p["layer"] <= hi]
        rep[f"median_{lo}-{hi}"] = {key: round(st.median(p[key] for p in sel), 5)
                                    for key in ("frac_sem", "frac_rand", "frac_top")}
    rep["layers"] = per
    OUT.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k_: v for k_, v in rep.items() if k_ != "layers"})


if __name__ == "__main__":
    main()
