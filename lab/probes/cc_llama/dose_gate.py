"""Dose gate for the Llama Part C (lab/probes/cc_llama/PLAN.md, Part C; prediction row 40).

Per-scale summary of a dev arm against dev_base (correct, lost = base correct and finished -> arm wrong, gained, cut,
token change) and the pre-registered rule that decides whether sealed set 3 is spent:

  chosen = the scale with the largest (GSM8K correct + MATH correct) on dev (ties: the smaller scale);
  pass   = at the chosen scale GSM8K correct >= base + 3 and lost <= 6 and MATH correct >= base + 4 and
           GSM8K token change <= -20 %.

usage: python lab/probes/cc_llama/dose_gate.py 0.25 0.35   (reads $CC_RUNS/dev_base.json and dev_vec_s{scale}.json,
                                                             writes $CC_RUNS/dose.json)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = Path(os.environ.get("CC_RUNS") or ROOT / "lab" / "runs" / "cc_llama")
if not RUNS.is_absolute():
    RUNS = ROOT / RUNS
BENCHES = ("gsm8k", "math500")
GSM_MIN_GAIN, LOST_MAX, MATH_MIN_GAIN, TOKEN_MAX = 3, 6, 4, -0.20


def load(name):
    d = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    return {(r["bench"], r["i"]): r for r in d["items"]}


def summarize(base, arm):
    out = {}
    for bench in BENCHES:
        keys = sorted(k for k in base if k[0] == bench)
        b = [base[k] for k in keys]
        a = [arm[k] for k in keys]
        tb = sum(r["tokens"] for r in b)
        ta = sum(r["tokens"] for r in a)
        out[bench] = {
            "n": len(keys),
            "ok": sum(bool(r["ok"]) for r in a),
            "ok_independent": sum(bool(r["ok_independent"]) for r in a),
            "base_ok": sum(bool(r["ok"]) for r in b),
            "lost": sum(bool(x["ok"]) and x["finish"] == "stop" and not y["ok"] for x, y in zip(b, a)),
            "gained": sum(bool(y["ok"]) and not x["ok"] for x, y in zip(b, a)),
            "cut": sum(r["finish"] == "length" for r in a),
            "nothink": sum(r.get("reasoning_chars", 1) == 0 for r in a),
            "token_change": round(ta / tb - 1, 4) if tb else 0.0,
        }
    return out


def gate(summ, base_ok):
    scales = sorted(summ, key=float)
    chosen = max(scales, key=lambda s: (summ[s]["gsm8k"]["ok"] + summ[s]["math500"]["ok"], -float(s)))
    c = summ[chosen]
    checks = {
        "gsm_gain": c["gsm8k"]["ok"] >= base_ok["gsm8k"] + GSM_MIN_GAIN,
        "lost": c["gsm8k"]["lost"] <= LOST_MAX,
        "math_gain": c["math500"]["ok"] >= base_ok["math500"] + MATH_MIN_GAIN,
        "tokens": c["gsm8k"]["token_change"] <= TOKEN_MAX,
    }
    return {"chosen": chosen, "checks": checks, "pass": all(checks.values())}


def main(scales):
    base = load("dev_base")
    summ = {s: summarize(base, load(f"dev_vec_s{s}")) for s in scales}
    base_ok = {b: summ[scales[0]][b]["base_ok"] for b in BENCHES}
    g = gate(summ, base_ok)
    out = {"scales": summ, "base_ok": base_ok, **g}
    (RUNS / "dose.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or ["0.25", "0.35"])
