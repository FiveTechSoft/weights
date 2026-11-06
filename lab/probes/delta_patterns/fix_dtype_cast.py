"""Correct stats_<name>.json(l) for the dtype-cast bug (found 2026-11-06, findings F26).

tensor_delta_stats compared F32 base tensors with BF16 copies without rounding both to BF16 first, so a storage
cast (linear_attn.norm, linear_attn.A_log in Qwen3.5) counted as a change of ~0.1-0.24 %. engine/weight_diff.py now
rounds to the less precise dtype. This script recomputes only the tensors whose stored dtype differs between base
and model (read locally, or by HTTP ranges when the download was deleted), rewrites the statistics, and reports
the per-type profile Spearman between models before and after.

usage: python lab/probes/delta_patterns/fix_dtype_cast.py
"""
import json
import struct
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

OUT = ROOT / "lab" / "runs" / "delta_patterns"
BASE = "hf_cache/qwen35-9b-hf"
PAIRS = {  # name: (base, model); local folder if present, else the HF repo
    "mimo": (BASE, "hf_cache/mimo-distill-hf"),
    "neohorse": (BASE, "hf_cache/neohorse-1-9b-hf"),
    "abliterated": (BASE, "huihui-ai/Huihui-Qwen3.5-9B-abliterated"),
    "tmax": (BASE, "allenai/tmax-9b"),
    "fara": (BASE, "microsoft/Fara1.5-9B"),
    "nyx_rp": (BASE, "Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1"),
    "official_post": ("Qwen/Qwen3.5-9B-Base", BASE),
    "wemm_embed": (BASE, "tencent/WeMM-Embedding-9B"),
    "angry_claudius": (BASE, "axiomofmind/Angry-Claudius-9B"),
    "dsv4_distill": (BASE, "hf_cache/dp_keep_dsv4_distill"),
    "dsv4_heretic": ("hf_cache/dp_keep_dsv4_distill", "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"),
}


def reader(spec):
    p = ROOT / spec
    if p.exists():
        return wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(spec), wd.hf_fetch(spec)), wd.text_model_name)


def dtypes(spec):
    out = {}
    p = ROOT / spec
    if p.exists():
        for f in sorted(p.glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                out.update(json.loads(fh.read(n)))
    else:
        fetch = wd.hf_fetch(spec)
        for f in wd.hf_files(spec):
            n = struct.unpack("<Q", fetch(f, 0, 7))[0]
            out.update(json.loads(fetch(f, 8, 8 + n - 1)))
    out.pop("__metadata__", None)
    return {wd.text_model_name(k): v["dtype"] for k, v in out.items() if wd.text_model_name(k)}


def max_ulps(a, b):
    """Largest difference after rounding both to the less precise dtype, in units of its last place (ulp) at the
    base value. 1 ulp can be a rounding-mode difference of the tool that saved the model; > 1 ulp is a change."""
    import torch
    a, b = wd.common_dtype(a, b)
    af, bf = a.float().flatten(), b.float().flatten()
    d = (bf - af).abs()
    if not bool((d > 0).any()):
        return 0.0
    mant_bits = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}.get(a.dtype, 23)
    _, e = torch.frexp(af)
    ulp = torch.ldexp(torch.ones_like(af), e - 1 - mant_bits)
    return round(float((d / ulp).max()), 2)


def profile(rows):
    import statistics as st
    g = {}
    for r in rows:
        g.setdefault(wd.tensor_kind(r["tensor"]), []).append(r["rel_delta"])
    return {k: st.median(v) for k, v in g.items()}


def spearman(p, q):
    """Average ranks for ties (corrected profiles have several exact zeros)."""
    ks = sorted(set(p) & set(q))

    def ranks(d):
        order = sorted(ks, key=lambda k: d[k])
        r, i = {}, 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and d[order[j + 1]] == d[order[i]]:
                j += 1
            for k in order[i:j + 1]:
                r[k] = (i + j) / 2
            i = j + 1
        return r
    a, b = ranks(p), ranks(q)
    ma, mb = sum(a.values()) / len(ks), sum(b.values()) / len(ks)
    num = sum((a[k] - ma) * (b[k] - mb) for k in ks)
    den = (sum((a[k] - ma) ** 2 for k in ks) * sum((b[k] - mb) ** 2 for k in ks)) ** 0.5
    return num / den if den else float("nan")


def main():
    cache, audit, before, after = {}, {}, {}, {}
    for name, (b, t) in PAIRS.items():
        jl = OUT / f"stats_{name}.jsonl"
        rows = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()]
        before[name] = profile(rows)
        db, dt = cache.setdefault(b, dtypes(b)), cache.setdefault(t, dtypes(t))
        cast = sorted(k for k in set(db) & set(dt) if db[k] != dt[k])
        fixed = []
        if cast:
            rb, rt = reader(b), reader(t)
            by = {r["tensor"]: i for i, r in enumerate(rows)}
            for n in cast:
                if n not in by:
                    continue
                old = rows[by[n]]
                ta, tb = rb.get(n), rt.get(n)
                new = wd.tensor_delta_stats(n, ta, tb)
                rows[by[n]] = new
                fixed.append({"tensor": n, "dtypes": f"{db[n]}->{dt[n]}", "rel_before": old["rel_delta"],
                              "rel_after": new["rel_delta"], "changed_before": old["changed"],
                              "changed_after": new["changed"], "max_ulps": max_ulps(ta, tb)})
            jl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            js = OUT / f"stats_{name}.json"
            old_rep = json.loads(js.read_text(encoding="utf-8"))
            rep = wd.summarize(rows, only_in_base=old_rep.get("only_in_base", []),
                               only_in_target=old_rep.get("only_in_target", []),
                               shape_mismatch=old_rep.get("shape_mismatch", []))
            rep["tensors"] = sorted(rep["tensors"], key=lambda r: -r["rel_delta"])
            js.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        after[name] = profile(rows)
        one_ulp = [f for f in fixed if 0 < f["max_ulps"] <= 1]
        real = [f for f in fixed if f["max_ulps"] > 1]
        audit[name] = {"cast_tensors": len(cast), "recomputed": len(fixed),
                       "were_reported_changed": sum(f["changed_before"] > 0 for f in fixed),
                       "one_ulp_only": len(one_ulp), "more_than_one_ulp": [(f["tensor"], f["max_ulps"]) for f in real],
                       "rows": fixed}
        print(f"{name:15s} cast {len(cast):3d} reported-changed {audit[name]['were_reported_changed']:3d} "
              f"after rounding: <=1 ulp {len(one_ulp)}, >1 ulp {len(real)}", flush=True)
    sp = []
    for x, y in combinations(PAIRS, 2):
        sp.append({"pair": f"{x}-{y}", "before": round(spearman(before[x], before[y]), 3),
                   "after": round(spearman(after[x], after[y]), 3)})
    moved = [s for s in sp if abs(s["after"] - s["before"]) >= 0.02]
    print("profile Spearman moved by >= 0.02:", len(moved), "of", len(sp))
    for s in sorted(moved, key=lambda s: -abs(s["after"] - s["before"])):
        print(f"  {s['pair']:32s} {s['before']:.3f} -> {s['after']:.3f}")
    (OUT / "dtype_cast_fix.json").write_text(json.dumps({"audit": audit, "spearman": sp}, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
