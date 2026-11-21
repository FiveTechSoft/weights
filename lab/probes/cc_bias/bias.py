"""Constant `</think>` logit bias as the control for the conclude vector (lab/probes/cc_bias/PLAN.md, roadmap gap 18).

  python lab/probes/cc_bias/bias.py pick      -> writes lab/runs/cc_bias/pick.json (bias whose pooled mean tokens is
                                                 closest to dev_vec's; ties to the smaller bias)
  python lab/probes/cc_bias/bias.py compare   -> writes lab/runs/cc_bias/compare.json for the picked arm (and every arm
                                                 against dev_base), R1 / R2 verdict
Pure functions (pick, pooled_tokens, compare, r1, close_think_id) are unit-tested in tests/test_cc_bias.py.
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "lab" / "runs"
BIASES = (2, 4, 6, 8)
NOTHINK_CHARS = 50


def close_think_id(hf_dir):
    """Token id of `</think>` from a Hugging Face tokenizer.json (added token first, plain vocab as fallback)."""
    t = json.loads((Path(hf_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    for a in t.get("added_tokens", []):
        if a.get("content") == "</think>":
            return int(a["id"])
    v = t.get("model", {}).get("vocab", {})
    if "</think>" in v:
        return int(v["</think>"])
    raise KeyError("</think> not in tokenizer.json")


def pooled_tokens(summary):
    """n-weighted mean tokens over the benches of a bench_runner summary."""
    n = sum(summary[b]["n"] for b in summary)
    return sum(summary[b]["n"] * summary[b]["mean_tokens"] for b in summary) / n


def pick(arm_tokens, vec_tokens):
    """arm_tokens: {bias: pooled mean tokens}. Closest to vec_tokens; ties to the smaller bias."""
    return min(sorted(arm_tokens), key=lambda b: (abs(arm_tokens[b] - vec_tokens), b))


def binom_two_sided(win, loss):
    n = win + loss
    if n == 0:
        return 1.0
    k = min(win, loss)
    p = sum(comb(n, j) for j in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _by_key(items):
    return {(r["bench"], r["i"]): r for r in items}


def compare(base_items, arm_items, vec_items):
    base, arm, vec = _by_key(base_items), _by_key(arm_items), _by_key(vec_items)
    out = {}
    for bench in ("gsm8k", "math500"):
        keys = sorted(k for k in arm if k[0] == bench)
        if not keys:
            continue
        a = [arm[k] for k in keys]
        b = [base.get(k) for k in keys]
        v = [vec.get(k) for k in keys]
        win = sum(1 for x, y in zip(a, v) if y is not None and x["ok"] and not y["ok"])
        loss = sum(1 for x, y in zip(a, v) if y is not None and y["ok"] and not x["ok"])
        wb = sum(1 for x, y in zip(a, b) if y is not None and x["ok"] and not y["ok"])
        lb = sum(1 for x, y in zip(a, b) if y is not None and y["ok"] and not x["ok"])
        out[bench] = {
            "n": len(a),
            "arm_ok": sum(x["ok"] for x in a), "arm_ok2": sum(x["ok_independent"] for x in a),
            "vec_ok": sum(y["ok"] for y in v if y is not None), "base_ok": sum(y["ok"] for y in b if y is not None),
            "arm_finished": sum(x["finish"] == "stop" for x in a), "arm_cut": sum(x["finish"] == "length" for x in a),
            "arm_nothink": sum(x.get("reasoning_chars", 10 ** 9) < NOTHINK_CHARS for x in a),
            "arm_tokens": round(sum(x["tokens"] for x in a) / len(a)),
            "vec_tokens": round(sum(y["tokens"] for y in v if y is not None) / max(1, sum(y is not None for y in v))),
            "lost_base_ok_stop": sum(1 for x, y in zip(a, b) if y is not None and y["ok"] and y["finish"] == "stop"
                                     and not x["ok"]),
            "gained": sum(1 for x, y in zip(a, b) if y is not None and not y["ok"] and x["ok"]),
            "vs_vec": {"win": win, "loss": loss, "p": binom_two_sided(win, loss)},
            "vs_base": {"win": wb, "loss": lb, "p": binom_two_sided(wb, lb)},
        }
    return out


def r1(s):
    g, m = s["gsm8k"], s["math500"]
    return (g["arm_ok"] >= g["vec_ok"] - 3 and m["arm_ok"] >= m["vec_ok"] - 2
            and g["vs_vec"]["p"] >= 0.05 and m["vs_vec"]["p"] >= 0.05)


def _load(rel):
    return json.loads((RUNS / f"{rel}.json").read_text(encoding="utf-8"))


def cmd_pick():
    vec = _load("cc_v2/dev_vec")["summary"]
    arms = {b: pooled_tokens(_load(f"cc_bias/dev_bias_{b}")["summary"]) for b in BIASES}
    chosen = pick(arms, pooled_tokens(vec))
    out = {"vec_tokens": pooled_tokens(vec), "arm_tokens": arms, "picked": chosen}
    (RUNS / "cc_bias" / "pick.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
    print(chosen)
    return 0


def cmd_compare():
    base = _load("cc_v2/dev_base")["items"]
    vec = _load("cc_v2/dev_vec")["items"]
    picked = json.loads((RUNS / "cc_bias" / "pick.json").read_text(encoding="utf-8"))["picked"]
    per_arm = {}
    for b in BIASES:
        p = RUNS / "cc_bias" / f"dev_bias_{b}.json"
        if p.exists():
            per_arm[b] = compare(base, json.loads(p.read_text(encoding="utf-8"))["items"], vec)
    s = per_arm[picked]
    verdict = "R1" if r1(s) else "R2"
    out = {"picked": picked, "verdict": verdict, "picked_summary": s, "arms": per_arm}
    (RUNS / "cc_bias" / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"picked": picked, "verdict": verdict,
                      "gsm8k": {k: s["gsm8k"][k] for k in ("arm_ok", "vec_ok", "base_ok", "arm_tokens", "vs_vec")},
                      "math500": {k: s["math500"][k] for k in ("arm_ok", "vec_ok", "base_ok", "arm_tokens", "vs_vec")}}))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"pick": cmd_pick, "compare": cmd_compare}.get(cmd, lambda: print(__doc__) or 2)())
