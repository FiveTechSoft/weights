"""Logit lens of the residual directions MiMo shares with NeoHorse (the H4 A1 subspace), exploratory, no forward pass.

For every layer matrix whose stored signature basis lives in the 4096-dim residual stream (read side: q_proj,
gate/up_proj, DeltaNet in_proj_qkv/z; write side: down_proj), the 16 shared directions are
S = shared_basis(sig_mimo, sig_neohorse). Two readings:
  1. enrichment: for a set of loop-related tokens (Wait, Okay, Hmm, check, </think>, ...), the fraction of each token's
     effective unembedding row u_t = lm_head[t] * (1 + final_norm) that lies in span(S), divided by the same fraction
     averaged over 2000 random tokens (1.0 = no preference; chance fraction 16/4096 for a random subspace).
     The same for MiMo's own top-16 basis and for a random 16-dim basis (control).
  2. top tokens: for each write-side (down_proj) subspace, the tokens most promoted / suppressed by its first shared
     direction (sign of a basis vector is arbitrary, so both ends are listed).
Logit lens is unreliable in low layers and directions can superpose several features: these are hints, not meanings.

usage: python lab/probes/delta_patterns/lens_shared.py  -> lab/runs/delta_patterns/lens_shared.json (+ printed summary)
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402
from engine.subspace_ops import random_basis, shared_basis  # noqa: E402

BASE = ROOT / "hf_cache" / "qwen35-9b-hf"
SIG = ROOT / "lab" / "runs" / "delta_patterns"
LOOP_WORDS = ["Wait", "wait", "Okay", "okay", "Hmm", "check", "Check", "Actually", "Alternatively", "But", "Let",
              "verify", "double", "re", "I'll", "use", "boxed", "</think>"]
K = 16


def token_ids(tok, words):
    ids = {}
    for w in words:
        for v in (w, " " + w):
            e = tok.encode(v, add_special_tokens=False).ids
            if len(e) == 1:
                ids[v] = e[0]
    return ids


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    r = wd.Renamed(wd.LocalSafetensors(BASE), wd.text_model_name)
    scale = 1.0 + r.get("model.norm.weight").float()
    U = r.get("lm_head.weight")                                  # (V, 4096) bf16, kept in bf16
    V = U.shape[0]
    loop = token_ids(tok, LOOP_WORDS)
    g = torch.Generator().manual_seed(0)
    rand_ids = torch.randint(0, 240000, (2000,), generator=g).tolist()
    ids = sorted(set(loop.values()) | set(rand_ids))
    rows = U[ids].float() * scale                               # effective unembedding rows
    rows = rows / rows.norm(dim=1, keepdim=True)
    pos = {t: i for i, t in enumerate(ids)}
    loop_idx = [pos[t] for t in loop.values()]
    rand_idx = [pos[t] for t in rand_ids]

    sm, sn = torch.load(SIG / "sig_mimo.pt"), torch.load(SIG / "sig_neohorse.pt")
    per_tensor, tops = [], []
    for name in sorted(sm):
        a, b = sm[name], sn.get(name)
        if not b or "basis" not in a or "basis" not in b or a["basis"].shape[0] != 4096:
            continue
        shape = a["shape"]
        if wd.layer_of(name) is None or shape[0] == shape[1]:   # layer matrices only; embed / lm_head excluded
            continue                                            # square: basis is on the head side, not residual
        side = "write" if shape[0] == 4096 else "read"
        S, cos = shared_basis(a["basis"], b["basis"], K)
        out = {"tensor": name, "kind": wd.tensor_kind(name), "layer": wd.layer_of(name), "side": side,
               "shared_cos_mean": round(st.mean(cos), 3)}
        for label, B in (("shared", S), ("mimo_top", a["basis"][:, :K].float()),
                         ("random", random_basis(4096, K, 7))):
            frac = (rows @ B.float()).pow(2).sum(1)              # fraction of each unit row inside span(B)
            fl, fr = float(frac[loop_idx].mean()), float(frac[rand_idx].mean())
            out[label] = {"loop_frac": round(fl, 5), "random_tok_frac": round(fr, 5),
                          "enrichment": round(fl / fr, 2) if fr > 0 else None}
        per_tensor.append(out)
        if side == "write":
            s = S[:, 0].float()
            logits = torch.cat([(U[i:i + 32768].float() * scale) @ s for i in range(0, V, 32768)])
            hi, lo = torch.topk(logits, 10).indices.tolist(), torch.topk(-logits, 10).indices.tolist()
            tops.append({"tensor": name, "layer": out["layer"], "cos": round(cos[0], 3),
                         "plus": [tok.decode([i]) for i in hi], "minus": [tok.decode([i]) for i in lo]})

    def summary(sel):
        return {lab: round(st.median(t[lab]["enrichment"] for t in sel), 2) for lab in ("shared", "mimo_top", "random")}
    by_kind = {}
    for t in per_tensor:
        by_kind.setdefault(t["kind"], []).append(t)
    report = {"loop_tokens": loop, "chance_fraction": K / 4096, "n_tensors": len(per_tensor),
              "enrichment_median_all": summary(per_tensor),
              "enrichment_median_by_kind": {k: summary(v) for k, v in sorted(by_kind.items())},
              "enrichment_median_by_depth": {f"{lo}-{lo + 7}": summary([t for t in per_tensor
                                                                        if lo <= t["layer"] <= lo + 7])
                                             for lo in (0, 8, 16, 24)},
              "down_proj_first_direction_tokens": tops, "tensors": per_tensor}
    (SIG / "lens_shared.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("loop tokens:", ", ".join(loop))
    print("tensors:", len(per_tensor), "| enrichment (loop tokens / random tokens), median:", report["enrichment_median_all"])
    for k, v in report["enrichment_median_by_kind"].items():
        print(f"  {k:32s} {v}")
    for k, v in report["enrichment_median_by_depth"].items():
        print(f"  layers {k:6s} {v}")
    for t in tops[-8:]:
        print(f"L{t['layer']:2d} cos {t['cos']}  + {t['plus'][:6]}  - {t['minus'][:6]}")


if __name__ == "__main__":
    main()
