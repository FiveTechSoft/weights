"""Forced-close probe: does a model already have the answer when it should stop?

For each MATH-500 item of the §39 protocol (same prompt, greedy), stream the model's reasoning and stop at the first
closed, non-empty \\boxed{...} inside it (or at </think>, or at the token cap). Two answers per item:
  A "first box": that first boxed answer, graded as is.
  B "forced close": append FORCE ("</think>" + "Final answer: \\boxed{") after the cut and let the model finish the
    box (<= 40 tokens); that box is graded.
Both graded with rig/math500.equivalent. Uses llama-server /apply-template + /completion (raw prompt).

usage: python lab/probes/close_probe/force_close.py <name> <port> [n=50] [cap=8192] [math500|gsm8k]
"""
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
FORCE = "\n</think>\n\nFinal answer: \\boxed{"
CHECKPOINTS = (512, 1024, 2048)   # extra forced closes inside one generation (before the final cut)


def _m5():
    import math500
    return math500


def first_answer(text):
    """First closed, non-empty \\boxed{...} in text, else None."""
    import re
    m5 = _m5()
    for m in re.finditer(r"\\boxed(?![a-zA-Z])", text):
        tok, _ = m5._grab(text, m.end())
        if tok is not None and tok.strip():
            return tok
    return None


def post(url, body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=7200) if stream else json.loads(urllib.request.urlopen(req, timeout=7200).read())


def one(base, problem, cap):
    prompt = post(base + "/apply-template", {"messages": [{"role": "user", "content": problem + SUFFIX}]})["prompt"]
    body = {"prompt": prompt, "n_predict": cap, "temperature": 0, "seed": 0, "stream": True, "cache_prompt": False}
    text, ntok, why, snaps = "", 0, "cap", {}
    with post(base + "/completion", body, stream=True) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            text += ev.get("content", "")
            ntok += 1
            if ntok in CHECKPOINTS and "</think>" not in text:
                snaps[ntok] = text
            if "</think>" in text:
                why = "think_end"
                break
            if first_answer(text) is not None:
                why = "first_box"
                break
            if ev.get("stop"):
                why = "stop"
                break
    reasoning = text.split("</think>")[0]
    a = first_answer(reasoning)

    def force(rsn):
        tail = post(base + "/completion", {"prompt": prompt + rsn + FORCE, "n_predict": 40,
                                           "temperature": 0, "seed": 0, "cache_prompt": False})
        ok_b, b = _m5().extract_boxed("\\boxed{" + tail["content"])
        return b if ok_b else None

    at = {str(k): force(v) for k, v in sorted(snaps.items())}
    b = force(reasoning)
    return {"tokens_to_cut": ntok, "cut_reason": why, "pred_first_box": a, "pred_forced": b,
            "pred_forced_at": at}


def main():
    name, port = sys.argv[1], int(sys.argv[2])
    math_n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    bench = sys.argv[5] if len(sys.argv) > 5 else "math500"
    m5 = _m5()
    if bench == "gsm8k":   # first n test problems, gold after '####' (engine/bench_runner.py)
        lines = (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()[:math_n]
        items = [{"problem": (q := json.loads(s))["question"],
                  "answer": q["answer"].split("####")[-1].strip().replace(",", "")} for s in lines]
        idx = list(range(len(items)))
    else:
        items = m5.load_items()
        idx = m5.select_indices(2026, math_n, len(items))
    base = f"http://127.0.0.1:{port}"
    out = ROOT / "lab" / "runs" / "close_probe"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def job(i):
        r = one(base, items[i]["problem"], cap)
        gold = items[i]["answer"]
        ok = lambda p: bool(p) and m5.equivalent(p, gold)  # noqa: E731
        r.update(i=i, gold=gold, ok_first_box=ok(r["pred_first_box"]), ok_forced=ok(r["pred_forced"]))
        # budget curve: at a checkpoint past the cut, the answer is the one at the cut
        r["ok_at"] = {str(k): ok(r["pred_forced_at"].get(str(k), r["pred_forced"])) for k in CHECKPOINTS}
        return r

    rows = []
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        for k, r in enumerate(ex.map(job, idx), 1):
            rows.append(r)
            print(f"[{name}] {k}/{len(idx)} i={r['i']} cut={r['cut_reason']}@{r['tokens_to_cut']} "
                  f"A={r['ok_first_box']} B={r['ok_forced']}", flush=True)
    n = len(rows)
    summ = {"n": n, "acc_first_box": sum(r["ok_first_box"] for r in rows) / n,
            "acc_forced": sum(r["ok_forced"] for r in rows) / n,
            "graders_disagree": sum(r["ok_first_box"] != r["ok_forced"] for r in rows),
            "acc_forced_at": {str(k): sum(r["ok_at"][str(k)] for r in rows) / n for k in CHECKPOINTS},
            "median_tokens_to_cut": sorted(r["tokens_to_cut"] for r in rows)[n // 2],
            "seconds": round(time.time() - t0)}
    (out / f"{name}.json").write_text(json.dumps({"model": name, "cap": cap, "summary": summ, "items": rows}, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
