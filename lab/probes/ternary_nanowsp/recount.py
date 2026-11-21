"""Second method for lab/probes/ternary_nanowsp (no engine.ternary_write code): reloads the written ternary
checkpoints with the plain NanoWSP runtime and recomputes recall / held-out / density; recomputes the float arm by
running the runtime's own imprint_associative_batch; checks the ternary files are exactly codes * scales.

  python lab/probes/ternary_nanowsp/recount.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig  # noqa: E402

OUT = ROOT / "lab" / "runs" / "ternary_nanowsp"


def model_from_dir(path):
    flo = load_file(str(Path(path) / "model.safetensors"))
    d = int(flo["token_embd.weight"].shape[1])
    cfg = NanoWSPConfig(d_model=d, d_ff=int(flo["layers.0.mlp_in.weight"].shape[0]), n_heads=2, d_head=d // 2,
                        max_seq_len=int(flo["pos_embd.weight"].shape[0]))
    m = NanoWSP(cfg, seed=0)
    m.weights = {k: v.clone().float() for k, v in flo.items()}
    return m, flo


def main():
    cap = json.loads((OUT / "capacity.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "lab/data/wspx_recall.json").read_text(encoding="utf-8"))
    train = [(it["prompt"], it["target"]) for it in data["train"]]
    hold = [(it["prompt"], it["target"]) for it in data["holdout"]]
    agree, total = 0, 0
    # ternary arms: saved checkpoints
    for r in cap["rows"]:
        if r.get("saved"):
            m, flo = model_from_dir(r["saved"])
            t = load_file(str(Path(r["saved"]) / "ternary.safetensors"))
            name = f"layers.{cap['layer']}.mlp_out.weight"
            exact = torch.equal(flo[name], t[name + ".codes"].float() * t[name + ".scales"][:, None])
            rec = m.evaluate_recall(train[:r["n"]]); ho = m.evaluate_recall(hold)
            dens = float((t[name + ".codes"] != 0).float().mean())
            ok = exact and abs(rec - r["recall"]) < 1e-9 and abs(ho - r["holdout"]) < 1e-9 and abs(dens - r["density"]) < 1e-6
            agree += ok; total += 1
            print(f"d{r['d']} n{r['n']} {r['arm']}: recount recall {rec:.3f} holdout {ho:.3f} density {dens:.3f} exact_ternary {exact}"
                  f" | run recall {r['recall']:.3f} holdout {r['holdout']:.3f} -> {'AGREE' if ok else 'DIFFER'}")
    # float arm: rerun the runtime's own write for seed 42 at every load
    for d in sorted({r["d"] for r in cap["rows"] if "d" in r}):
        for n in cap["loads"]:
            m = NanoWSP(NanoWSPConfig(d_model=d, d_ff=2 * d, n_heads=2, d_head=d // 2), seed=42)
            m.imprint_associative_batch(train[:n], layer_idx=cap["layer"], eta=cap["eta"], ridge_lambda=cap["lambda"])
            rec = m.evaluate_recall(train[:n])
            ref = [r for r in cap["rows"] if r.get("d") == d and r.get("seed") == 42 and r.get("n") == n and r["arm"] == "float"][0]
            ok = abs(rec - ref["recall"]) < 1e-9
            agree += ok; total += 1
            print(f"d{d} n{n} float: recount {rec:.3f} | run {ref['recall']:.3f} -> {'AGREE' if ok else 'DIFFER'}")
    print(f"agreement {agree}/{total}")
    return 0 if agree == total else 1


if __name__ == "__main__":
    sys.exit(main())
