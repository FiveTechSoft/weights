"""Independent (2nd-method) verification of the turbo27b blind-test scoring (plan_turbo27b.md).

P1: changed layers of D_heretic vs predicted 26-63 (Jaccard) + none outside documented 26-55.
P2: over the 60 touched residual-writer tensors: energy_top1 median (claim 0.34) and k90 from the top-16
    spectrum (claim: median 13, with 24/60 needing more than 16).
P3: family profile of D_rest vs the median of the four 9B improvements' by_family stats
    (granularity note: the paper scored per-TYPE; this check is per-FAMILY direction).
CPU only; writes sealed/verify2.json.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(r"C:\weights")
D = ROOT / "lab/runs/delta_patterns/turbo27b"
out = {}

her = json.loads((D / "sealed" / "stats_heretic.json").read_text(encoding="utf-8"))
rest = json.loads((D / "sealed" / "stats_rest.json").read_text(encoding="utf-8"))

# ---------- P1 ----------
changed = sorted({t["layer"] for t in her["tensors"] if t["changed"] > 0})
pred, doc = set(range(26, 64)), set(range(26, 56))
out["P1"] = {"n_layers": len(changed), "first": changed[0], "last": changed[-1],
             "outside_doc": [l for l in changed if l not in doc],
             "jaccard_vs_pred": round(len(set(changed) & pred) / len(set(changed) | pred), 4)}

# ---------- P2 ----------
touched = [t for t in her["tensors"] if t["changed"] > 0]
e1 = sorted(t["energy_top1"] for t in touched)
def k90_from_top16(t):
    # full-spectrum energy is reconstructed as s1^2 / energy_top1 (energy_top1 is over ALL directions)
    s = t["top_singular_values"]
    total = (s[0] ** 2) / t["energy_top1"]
    cum = 0.0
    for i, x in enumerate(s, 1):
        cum += x ** 2
        if cum / total >= 0.9:
            return i
    return None
k90 = [k90_from_top16(t) for t in touched]
known = sorted(k for k in k90 if k is not None)
n_unknown = sum(k is None for k in k90)
# 60-sample median with the >16 unknowns ranked above every known value
n = len(known) + n_unknown
i1, i2 = (n - 1) // 2, n // 2          # 0-based order statistics
v1 = known[i1] if i1 < len(known) else None
v2 = known[i2] if i2 < len(known) else None
med60 = None if (v1 is None or v2 is None) else (v1 + v2) / 2
out["P2"] = {"n_touched": len(touched), "families": sorted({t["family"] for t in touched}),
             "energy_top1_median": round(statistics.median(e1), 3),
             "k90_median_over60": med60,
             "k90_median_known_only": statistics.median(known) if known else None,
             "n_k90_gt16": n_unknown}

# ---------- P3 (per-TYPE granularity, matching the paper's claim) ----------
TYPES = [("embed", ("embed",)), ("head", ("lm_head", "output.weight")), ("norm", ("norm", "layernorm")),
         ("mlp_down", ("down_proj",)), ("mlp_gate", ("gate_proj", "up_proj")),
         ("attn_qkv", ("q_proj", "k_proj", "v_proj", "in_proj_qk", "in_proj_q", "in_proj_k", "in_proj_v")),
         ("attn_o", ("o_proj", "linear_attn.out_proj")),
         ("ssm_zab", ("in_proj_z", "in_proj_a", "in_proj_b", "conv1d")),
         ("ssm_state", ("a_log", "dt_bias", "A_log"))]
def type_of(name):
    n = name.lower()
    for fam, pats in TYPES:
        if any(p in n for p in pats):
            return fam
    if "norm" in n or "layernorm" in n:
        return "norm"
    return "other"
def type_profile(rows):
    acc, cnt = {}, {}
    for r in rows:
        t = type_of(r["tensor"])
        acc[t] = acc.get(t, 0.0) + r["rel_delta"] * r["numel"]
        cnt[t] = cnt.get(t, 0) + r["numel"]
    return {k: acc[k] / cnt[k] for k in acc}
def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
pr = type_profile(jsonl(D / "sealed" / "stats_rest.jsonl"))
profiles = [type_profile(jsonl(ROOT / f"lab/runs/delta_patterns/stats_{n}.jsonl")) for n in ("mimo", "neohorse", "tmax", "fara")]
common = sorted(set(pr) & set.intersection(*(set(p) for p in profiles)))
def rank(d):
    order = sorted(common, key=lambda k: -d[k])
    return {k: i for i, k in enumerate(order)}
ra, rb = rank(pr), rank({k: statistics.median(p[k] for p in profiles) for k in common})
n = len(common)
rho = 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in common) / (n * (n * n - 1))
out["P3"] = {"rest": {k: round(pr[k], 4) for k in common},
             "median9b": {k: round(statistics.median(p[k] for p in profiles), 4) for k in common},
             "spearman_by_type": round(rho, 4), "n_types": n}

(D / "sealed" / "verify2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
