"""Method 2 for Part C: pairwise arm-vs-base counts from per-item records (independent of dose_gate.py and
cc_verdict.py). Usage: python -I pair_llama_c.py <base> <arm> [<arm> ...]  (names relative to lab/runs, no .json)."""
import json
import sys
from math import comb
from pathlib import Path
from statistics import median

RUNS = Path(r"C:\weights\lab\runs")


def load(name):
    d = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    return {(r["bench"], r["i"]): r for r in d["items"]}


def binom_two_sided(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(base, arm):
    out = {}
    for bench in ("gsm8k", "math500"):
        keys = [k for k in base if k[0] == bench and k in arm]
        b = [base[k] for k in keys]
        a = [arm[k] for k in keys]
        for grader, f in (("g1", lambda r: bool(r["ok"])), ("g2", lambda r: bool(r["ok_independent"]))):
            win = sum(f(y) and not f(x) for x, y in zip(b, a))
            loss = sum(f(x) and not f(y) for x, y in zip(b, a))
            out[(bench, grader)] = {"n": len(keys), "base": sum(map(f, b)), "arm": sum(map(f, a)), "win": win,
                                    "loss": loss, "p": round(binom_two_sided(win, loss), 4)}
        lost = [(x, y) for x, y in zip(b, a) if x["ok"] and x["finish"] == "stop" and not y["ok"]]
        gained = sum((not x["ok"]) and y["ok"] for x, y in zip(b, a))
        tb, ta = sum(r["tokens"] for r in b), sum(r["tokens"] for r in a)
        out[(bench, "extra")] = {
            "lost_base_ok_stop": len(lost), "gained": gained,
            "cut_base": sum(r["finish"] == "length" for r in b), "cut_arm": sum(r["finish"] == "length" for r in a),
            "nothink_arm": sum(r.get("reasoning_chars", 1) == 0 for r in a),
            "tokens_base": tb, "tokens_arm": ta, "token_change": round(ta / tb - 1, 4),
            "lost_finish_stop": sum(y["finish"] == "stop" for _, y in lost),
            "lost_reasoning_base_med": median([x["reasoning_chars"] for x, _ in lost]) if lost else None,
            "lost_reasoning_arm_med": median([y["reasoning_chars"] for _, y in lost]) if lost else None,
            "arm_correct_reasoning_med": median([r["reasoning_chars"] for r in a if r["ok"]]) if any(r["ok"] for r in a) else None,
        }
    return out


if __name__ == "__main__":
    base = load(sys.argv[1])
    for name in sys.argv[2:]:
        arm = load(name)
        print(f"== {name} vs {sys.argv[1]}")
        for k, v in compare(base, arm).items():
            print(" ", k, json.dumps(v))
