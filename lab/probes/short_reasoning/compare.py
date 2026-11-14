"""Table + verdict for the short_reasoning arms against the criterion in PLAN.md (fixed before any run).

usage: python lab/probes/short_reasoning/compare.py   (reads lab/runs/short_reasoning/*.json written by bench_runner)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "lab" / "runs" / "short_reasoning"
ARMS = ["mimo_ref", "oss_a050", "oss_a100", "placebo_a100"]


def load(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    rows = {a: load(a) for a in ARMS}
    print(f"{'arm':14} {'GSM ok':>7} {'GSM cut':>8} {'MATH ok':>8} {'MATH cut':>9} {'tok GSM':>8} {'tok MATH':>9}")
    s = {}
    for a, r in rows.items():
        if not r:
            print(f"{a:14} (not run)")
            continue
        g, m = r["summary"]["gsm8k"], r["summary"]["math500"]
        s[a] = {"ok": round(g["acc"] * g["n"]) + round(m["acc"] * m["n"]), "math_cut": m["cut_at_max"], "cut": g["cut_at_max"] + m["cut_at_max"]}
        print(f"{a:14} {round(g['acc'] * g['n']):>7} {g['cut_at_max']:>8} {round(m['acc'] * m['n']):>8} {m['cut_at_max']:>9} "
              f"{g['mean_tokens']:>8} {m['mean_tokens']:>9}")
    if "mimo_ref" not in s:
        sys.exit("mimo_ref missing: no verdict")
    ref = s["mimo_ref"]
    plc = s.get("placebo_a100")
    for a in ("oss_a050", "oss_a100"):
        if a not in s:
            continue
        c1 = ref["math_cut"] - s[a]["math_cut"] >= 5
        c2 = s[a]["ok"] >= ref["ok"] - 2
        c3 = plc is not None and (ref["math_cut"] - plc["math_cut"] < 5)
        verdict = "PASS (go to confirmation set)" if (c1 and c2 and c3) else "FAIL"
        print(f"{a}: C1 math cuts -{ref['math_cut'] - s[a]['math_cut']} (>=5) {c1} | C2 ok {s[a]['ok']} vs {ref['ok']} (>= -2) {c2} | "
              f"C3 placebo does not reach C1 {c3 if plc else 'placebo not run'} -> {verdict}")


if __name__ == "__main__":
    main()
