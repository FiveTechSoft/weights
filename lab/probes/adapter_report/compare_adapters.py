"""Do LoRA adapters trained with the same recipe on the same base for different harnesses write the same thing?

For two PEFT adapters on the same base, per shared module: exact cosine of the two updates dW = s B A (Frobenius inner
product through the factors: <B1 A1, B2 A2> = sum((B1^T B2) * (A1 A2^T))), and the overlap of their output-side
subspaces (||Q1^T Q2||_F^2 / r, 1 = same r-dim subspace, r/d = chance). Null for both: the same pair of adapters at
mismatched layers (same module kind, layer L vs layer L+1). Reads the factors over HTTP with engine.adapter_report.
Usage: python lab/probes/adapter_report/compare_adapters.py hf:a hf:b [hf:c ...] --out <json>
"""
import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import adapter_report as ar  # noqa: E402


def factors(ad, name):
    m = ad["modules"][name]
    A, B = ad["reader"].get(m["A"]).double(), ad["reader"].get(m["B"]).double()
    return B, A, ad["scale"]


def pair_stats(f1, f2):
    (B1, A1, s1), (B2, A2, s2) = f1, f2
    if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
        return None
    ip = s1 * s2 * float(((B1.T @ B2) * (A1 @ A2.T)).sum())
    n1 = s1 * float(((B1.T @ B1) * (A1 @ A1.T)).sum()) ** 0.5
    n2 = s2 * float(((B2.T @ B2) * (A2 @ A2.T)).sum()) ** 0.5
    Q1, Q2 = torch.linalg.qr(B1)[0], torch.linalg.qr(B2)[0]
    r = min(Q1.shape[1], Q2.shape[1])
    return {"cos": ip / (n1 * n2) if n1 and n2 else 0.0, "overlap": float((Q1.T @ Q2).pow(2).sum()) / r,
            "chance": r / B1.shape[0]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ads = {spec: ar.load_adapter(spec) for spec in a.adapters}
    out = {"adapters": {k: {"r": v["r"], "alpha": v["alpha"], "modules": len(v["modules"])} for k, v in ads.items()},
           "pairs": {}}
    for x, y in itertools.combinations(a.adapters, 2):
        shared = sorted(set(ads[x]["modules"]) & set(ads[y]["modules"]))
        by_kind = {}
        cache = {}

        def get(spec, n):
            if (spec, n) not in cache:
                cache[(spec, n)] = factors(ads[spec], n)
            return cache[(spec, n)]
        for n in shared:
            s = pair_stats(get(x, n), get(y, n))
            if s is None:
                continue
            k = n.split(".", 3)[-1] if ".layers." in n else n
            k = ".".join(n.split(".")[-3:-1])
            by_kind.setdefault(k, {"same": [], "null": []})["same"].append(s)
        # null: x at layer L vs y at the next layer with the same kind
        layer_of = {n: int(n.split(".layers.")[1].split(".")[0]) for n in shared if ".layers." in n}
        for n in shared:
            if n not in layer_of:
                continue
            m = n.replace(f".layers.{layer_of[n]}.", f".layers.{layer_of[n] + 1}.")
            if m in ads[y]["modules"]:
                s = pair_stats(get(x, n), get(y, m))
                if s:
                    by_kind[".".join(n.split(".")[-3:-1])]["null"].append(s)
        summ = {}
        for k, v in by_kind.items():
            f = lambda rows, key: round(st.median(r[key] for r in rows), 4) if rows else None  # noqa: E731
            summ[k] = {"n": len(v["same"]), "cos_median": f(v["same"], "cos"), "overlap_median": f(v["same"], "overlap"),
                       "null_cos_median": f(v["null"], "cos"), "null_overlap_median": f(v["null"], "overlap"),
                       "chance_overlap": f(v["same"], "chance")}
        out["pairs"][f"{x} | {y}"] = summ
        print(x, "|", y)
        for k, v in summ.items():
            print(f"   {k:28s} n {v['n']:3d}  cos {v['cos_median']}  (null {v['null_cos_median']})  overlap {v['overlap_median']} "
                  f"(null {v['null_overlap_median']}, chance {v['chance_overlap']})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
