"""Compare chat models served by llama-server (OpenAI API) on GSM8K and MATH-500, same protocol for every model.

Greedy (temperature 0, seed 0), thinking as the model's chat template does it, at most 8192 new tokens, the user
message = problem + "Please reason step by step, and put your final answer within \\boxed{}."
GSM8K: first N test problems. MATH-500: rig/math500.select_indices(2026, N).
Two graders that must agree: rig/math500.equivalent on the last \\boxed{}, and an independent one (GSM8K: integer
value of the last number in the box; MATH-500: sympy-free string compare after a separate minimal normaliser).

usage: python lab/probes/model_compare/eval.py <name> <port> [gsm_n] [math_n] [parallel]
"""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rig"))
import math500 as m5  # noqa: E402

OUT = ROOT / "lab" / "runs" / "model_compare"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_TOKENS = 8192


def ask(port, text):
    body = {"messages": [{"role": "user", "content": text + SUFFIX}], "temperature": 0, "seed": 0,
            "max_tokens": MAX_TOKENS}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    ch = r["choices"][0]
    msg = ch["message"]
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "finish": ch["finish_reason"], "tokens": r["usage"]["completion_tokens"],
            "tok_s": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}


def gsm_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def grade_gsm_independent(content, gold):
    """Second grader: the last number inside the last \\boxed{...}, as a float, against the integer gold."""
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", content)
    if not boxes:
        return False
    nums = re.findall(r"-?\d[\d,]*\.?\d*", boxes[-1])
    if not nums:
        return False
    try:
        return abs(float(nums[-1].replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return False


def simple_norm(s):
    """Second grader for MATH-500: strip spaces, $, \\left/\\right, \\! and text wrappers; compare strings."""
    s = re.sub(r"\\(left|right|!|,|;|displaystyle)", "", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "").replace("^\\circ", "").replace("^{\\circ}", "")
    return re.sub(r"\s+", "", s).lower().rstrip(".")


def run(name, port, gsm_n=100, math_n=50, parallel=2):
    OUT.mkdir(parents=True, exist_ok=True)
    gsm = [json.loads(line) for line in (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()][:gsm_n]
    math_all = m5.load_items()
    math_idx = m5.select_indices(2026, math_n, len(math_all))
    jobs = [("gsm8k", i, it["question"], gsm_gold(it["answer"])) for i, it in enumerate(gsm)]
    jobs += [("math500", i, math_all[i]["problem"], math_all[i]["answer"]) for i in math_idx]
    t0 = time.time()
    partial = OUT / f"{name}.partial.jsonl"  # one line per finished item, so a stopped run resumes
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["bench"], r["i"]) for r in rows}
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"[{name}] resuming with {len(rows)} done, {len(jobs)} to go", flush=True)

    def one(job):
        bench, i, text, gold = job
        r = ask(port, text)
        pred = m5.extract_pred(r["content"])
        ok1 = bool(pred) and m5.equivalent(pred, gold)
        ok2 = grade_gsm_independent(r["content"], gold) if bench == "gsm8k" else (
            bool(pred) and simple_norm(pred) == simple_norm(gold))
        return {"bench": bench, "i": i, "gold": gold, "pred": pred, "ok": ok1, "ok_independent": ok2,
                "finish": r["finish"], "tokens": r["tokens"], "tok_s": r["tok_s"],
                "reasoning_chars": len(r["reasoning"]), "tail": (r["reasoning"] + r["content"])[-300:]}

    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            rows.append(row)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                print(f"[{name}] {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for bench in ("gsm8k", "math500"):
        rs = [r for r in rows if r["bench"] == bench]
        summary[bench] = {"n": len(rs), "acc": round(sum(r["ok"] for r in rs) / len(rs), 3),
                          "acc_independent": round(sum(r["ok_independent"] for r in rs) / len(rs), 3),
                          "cut_at_max": sum(r["finish"] == "length" for r in rs),
                          "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs)),
                          "mean_tok_s": round(sum(r["tok_s"] for r in rs) / len(rs), 1)}
    report = {"model": name, "max_tokens": MAX_TOKENS, "seconds": round(time.time() - t0), "summary": summary,
              "items": rows}
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(name, json.dumps(summary), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], int(a[1]), *(int(x) for x in a[2:]))
