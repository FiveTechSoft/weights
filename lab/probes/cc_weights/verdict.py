"""Dev verdict (PLAN.md C1-C4): merged / placebo_merged / lora vs base and vs control vector (lab/runs/cc/dev_*.json,
same items and settings, re-used: the setup is deterministic). Exact McNemar, two graders per benchmark.
`verdict.py sealed`: the same C1-C4 on sealed set 2 (lab/runs/sealed_ccw/{base,vec,merged,placebo_merged}.json)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "probes" / "cc"))
from cc import lastnum, mcnemar  # noqa: E402

CC, W = ROOT / "lab" / "runs" / "cc", ROOT / "lab" / "runs" / "cc_weights"
SEALED = ROOT / "lab" / "runs" / "sealed_ccw"


def load(p):
    return {(r["bench"], r["i"]): r for r in json.loads(p.read_text(encoding="utf-8"))["items"]}


G = {"gsm8k": {"g1": lambda r: r["ok"], "lastnum": lastnum}, "math500": {"g1": lambda r: r["ok"], "g2": lambda r: r["ok_independent"]}}


def cmp(X, Y, bench, f):
    keys = sorted(k for k in Y if k[0] == bench)
    a, b = [f(X[k]) for k in keys], [f(Y[k]) for k in keys]
    w = sum(x and not y for x, y in zip(a, b))
    l = sum(y and not x for x, y in zip(a, b))
    return {"arm": sum(a), "ref": sum(b), "win": w, "loss": l, "p": mcnemar(w, l)}


def main(sealed=False):
    if sealed:
        B, V = load(SEALED / "base.json"), load(SEALED / "vec.json")
        arms = {n: load(SEALED / f"{n}.json") for n in ("merged", "placebo_merged") if (SEALED / f"{n}.json").exists()}
    else:
        B, V = load(CC / "dev_base.json"), load(CC / "dev_vec.json")
        arms = {n: load(W / f"dev_{n}.json") for n in ("merged", "placebo_merged", "lora") if (W / f"dev_{n}.json").exists()}
    out = {}
    for n, X in arms.items():
        out[n] = {"vs_base": {b: {g: cmp(X, B, b, f) for g, f in gs.items()} for b, gs in G.items()},
                  "vs_vector": {b: {g: cmp(X, V, b, f) for g, f in gs.items()} for b, gs in G.items()},
                  "cut": {b: sum(r["finish"] == "length" for k, r in X.items() if k[0] == b) for b in G},
                  "mean_tokens": {b: round(sum(r["tokens"] for k, r in X.items() if k[0] == b) /
                                           sum(1 for k in X if k[0] == b)) for b in G}}
    sig = lambda c: c["arm"] > c["ref"] and c["p"] < 0.05
    worse = lambda c: c["arm"] < c["ref"] and c["p"] < 0.05
    if "merged" in out and "placebo_merged" in out:
        m, p = out["merged"], out["placebo_merged"]
        c1 = any(all(sig(c) for c in m["vs_base"][b].values()) for b in G)
        c2 = not any(sig(c) for b in G for c in p["vs_base"][b].values())
        c3 = m["vs_base"]["gsm8k"]["g1"]["arm"] >= m["vs_base"]["gsm8k"]["g1"]["ref"] - 2
        c4 = not any(worse(m["vs_vector"][b]["g1"]) for b in G)
        out["criteria"] = {"C1 merged > base": c1, "C2 placebo not better": c2, "C3 GSM8K no damage": c3,
                           "C4 merged not below vector": c4}
        out["verdict"] = "PASS" if c1 and c2 and c3 and c4 else "FAIL"
    ((SEALED if sealed else W) / "verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sealed=sys.argv[1:2] == ["sealed"])
