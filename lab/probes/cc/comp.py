"""Composition of the conclude vector with the Jackrong adapter (lab/probes/cc/PLAN.md, "Composition"; descriptive).

Arms on the dev set: base (cc_v2 dev_base), C (cc_v2 dev_vec), J (adapter x0.5), J+C, J+P (adapter + cc_v2 placebo).
Readings: K1 J+C vs J (adds / redundant / interferes), K2 tokens, K3 cuts and loop tails, control J+P vs J.

  verdict   -> runs/comp/readings.json     (runs dir: CC_RUNS, default lab/runs/cc_v2; arms J, JC, JP in runs/comp/)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc  # noqa: E402  (mcnemar, compare, tokens; RUNS from CC_RUNS)

RUNS = cc.RUNS
COMP = RUNS / "comp"
BENCHES = ("gsm8k", "math500")
BAND = 3


def k1(k):
    """k: {bench: compare(J+C, J)}. 'interferes' if significantly below on either; 'adds' if significantly above on
    either and not below on the other; else 'redundant'."""
    worse = [b for b in BENCHES if k[b]["arm"] < k[b]["base"] and k[b]["p"] < 0.05]
    better = [b for b in BENCHES if k[b]["arm"] > k[b]["base"] and k[b]["p"] < 0.05]
    if worse:
        return "interferes"
    if better and all(k[b]["arm"] >= k[b]["base"] for b in BENCHES):
        return "adds"
    return "redundant"


def control_ok(k):
    """J+P within +-BAND of J and p >= 0.05, per benchmark."""
    return {b: abs(k[b]["arm"] - k[b]["base"]) <= BAND and k[b]["p"] >= 0.05 for b in BENCHES}


def _load(path):
    return {(r["bench"], r["i"]): r for r in json.loads(Path(path).read_text(encoding="utf-8"))["items"]}


def _cost(arm, bench):
    rows = [r for k, r in arm.items() if k[0] == bench]
    return {"cut": sum(r["finish"] != "stop" for r in rows), "loop": sum(bool(r.get("loop")) for r in rows),
            "mean_tokens": round(cc.tokens(arm, bench), 1)}


def verdict():
    arms = {"base": _load(RUNS / "dev_base.json"), "C": _load(RUNS / "dev_vec.json"), "J": _load(COMP / "J.json"),
            "JC": _load(COMP / "JC.json"), "JP": _load(COMP / "JP.json")}
    g1 = lambda r: r["ok"]
    jc_vs_j = {b: cc.compare(arms["JC"], arms["J"], b, g1) for b in BENCHES}
    jp_vs_j = {b: cc.compare(arms["JP"], arms["J"], b, g1) for b in BENCHES}
    out = {"correct": {a: {b: sum(g1(r) for k, r in X.items() if k[0] == b) for b in BENCHES} for a, X in arms.items()},
           "cost": {a: {b: _cost(X, b) for b in BENCHES} for a, X in arms.items()},
           "JC_vs_J": jc_vs_j, "JP_vs_J": jp_vs_j,
           "J_vs_base": {b: cc.compare(arms["J"], arms["base"], b, g1) for b in BENCHES},
           "K1": k1(jc_vs_j),
           "K2_tokens_JC_over_J": {b: round(cc.tokens(arms["JC"], b) / cc.tokens(arms["J"], b), 3) for b in BENCHES},
           "control_JP_within_band": control_ok(jp_vs_j)}
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "readings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["verdict"]:
        verdict()
    else:
        raise SystemExit(__doc__)
