"""Name test of the "conclude" direction outside maths (lab/probes/cc/PLAN.md, "Name test"; descriptive).

D1 HumanEval-20 is run by lab/probes/close_probe/he_eval.py (names name_<arm>). D2 is the 120 Horn-logic cases of
rig/logic_probe.make_cases, asked here with thinking as the template does it, greedy, a 2,048-token cap (the probe's own
runner uses 256 tokens and no thinking). Readings N1-N3 and the placebo bands are pre-registered in the plan.

  logic <port> <arm>     -> runs/name/logic_<arm>.json
  verdict                -> runs/name/readings.json (base / vec / placebo of both domains)
Runs dir: CC_RUNS (default lab/runs/cc_v2) + /name.
"""
import concurrent.futures
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rig"))
import logic_probe as lp  # noqa: E402

RUNS = Path(os.environ.get("CC_RUNS") or ROOT / "lab" / "runs" / "cc_v2")
if not RUNS.is_absolute():
    RUNS = ROOT / RUNS
NAME = RUNS / "name"
HE_DIR = ROOT / "lab" / "runs" / "close_probe" / "fast"
CAP = 2048
ARMS = ("base", "vec", "placebo")
TOL_CORRECT = {"he": 2, "logic": 6}


def logic_answer(text):
    """YES / NO read after the think block (whole text if none); '' if cut inside it or self-contradictory."""
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return lp.extract_answer(text)


def summarize(rows):
    toks = sorted(r["tokens"] for r in rows)
    return {"n": len(rows), "correct": sum(bool(r["correct"]) for r in rows),
            "cut": sum(r["finish"] != "stop" for r in rows), "median_tokens": toks[len(toks) // 2] if toks else 0}


def readings(base, vec):
    """N1 shorter (<= 0.85 x base tokens), N2 bounded cost (correct >= base - tol), N3 cuts halved where the base cuts
    >= 10 %; name: holds / concludes at a cost / fails."""
    n1 = {d: vec[d]["median_tokens"] <= 0.85 * base[d]["median_tokens"] for d in base}
    n2 = {d: vec[d]["correct"] >= base[d]["correct"] - TOL_CORRECT[d] for d in base}
    n3 = {d: vec[d]["cut"] <= base[d]["cut"] / 2 for d in base if base[d]["cut"] >= 0.1 * base[d]["n"]}
    name = "fails" if not all(n1.values()) else "holds" if all(n2.values()) else "concludes at a cost"
    return {"N1": n1, "N2": n2, "N3": n3 or "not applicable", "name": name}


def control(base, placebo):
    """Placebo within +-10 % tokens and +-tol correct of the base, per domain."""
    return {d: abs(placebo[d]["median_tokens"] - base[d]["median_tokens"]) <= 0.1 * base[d]["median_tokens"]
            and abs(placebo[d]["correct"] - base[d]["correct"]) <= TOL_CORRECT[d] for d in base}


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def ask(port, case):
    r = _post(port, {"messages": [{"role": "user", "content": case["prompt"]}], "temperature": 0, "seed": 0,
                     "max_tokens": CAP})
    ch = r["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    full = f"<think>{reasoning}</think>\n{content}" if reasoning else content
    if ch.get("finish_reason") != "stop" and reasoning and not content:
        full = f"<think>{reasoning}"                 # cut inside the think block
    ans = logic_answer(full)
    return {"id": case["id"], "expected": case["expected"], "predicted": ans, "correct": ans == case["expected"],
            "finish": ch.get("finish_reason"), "tokens": r["usage"]["completion_tokens"],
            "reasoning_chars": len(reasoning), "tail": content[-200:]}


def logic(port, arm, parallel=2):
    cases = lp.make_cases()
    NAME.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(lambda c: ask(port, c), cases))
    out = {"summary": dict(summarize(rows), arm=arm, cap=CAP), "items": rows}
    (NAME / f"logic_{arm}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"]), flush=True)
    return out["summary"]


def _he(arm):
    s = json.loads((HE_DIR / f"he_name_{arm}.json").read_text(encoding="utf-8"))["summary"]
    return {"n": s["n"], "correct": s["ok"], "cut": s["n"] - s["closed"], "median_tokens": s["median_tokens"]}


def _logic(arm):
    s = json.loads((NAME / f"logic_{arm}.json").read_text(encoding="utf-8"))["summary"]
    return {k: s[k] for k in ("n", "correct", "cut", "median_tokens")}


def verdict():
    by = {arm: {"he": _he(arm), "logic": _logic(arm)} for arm in ARMS}
    out = {"arms": by, "readings": readings(by["base"], by["vec"]), "control": control(by["base"], by["placebo"]),
           "base_cuts_tenth": {d: by["base"][d]["cut"] >= 0.1 * by["base"][d]["n"] for d in ("he", "logic")}}
    (NAME / "readings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "logic":
        logic(int(sys.argv[2]), sys.argv[3])
    elif cmd == "verdict":
        verdict()
    else:
        raise SystemExit(__doc__)
