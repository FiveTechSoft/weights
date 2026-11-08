"""Fast closing metric (minutes): MATH-500 items where the base had the answer but never closed (results §43.2),
plus the 20 items the base already solves (damage guard, optional). Natural generation (no forcing), greedy, short cap.
Scores: closed (finish == stop) and correct (rig/math500.equivalent on the last \\boxed of the content).

Variants without touching weights:
  --system TEXT     system prompt
  --bias_think B    logit bias added to the </think> token (and only it)

usage: python lab/probes/close_probe/fast_eval.py <name> <port> [--cap 2048] [--guard] [--system ...] [--bias_think B]
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rig"))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("port", type=int)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--system", default=None)
    ap.add_argument("--bias_think", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--lora", default="", help="id:scale,... of adapters loaded with --lora-init-without-apply")
    a = ap.parse_args()
    import math500 as m5
    base = f"http://127.0.0.1:{a.port}"
    rev = json.loads((ROOT / "lab/runs/model_compare/base_cut_review.json").read_text())["classes"]
    ids = [("A", i) for i in rev["answer_found_cannot_close"]]
    if a.guard:
        own = json.loads((ROOT / "lab/runs/model_compare/own_base.json").read_text(encoding="utf-8"))["items"]
        ids += [("ok", r["i"]) for r in own if r["bench"] == "math500" and r["ok"]]
    items = m5.load_items()
    think = post(base + "/tokenize", {"content": "</think>"})["tokens"]
    if a.bias_think and len(think) != 1:
        raise SystemExit(f"</think> is not one token: {think}")

    def one(job):
        grp, i = job
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
               [{"role": "user", "content": items[i]["problem"] + SUFFIX}]
        body = {"messages": msgs, "temperature": 0, "seed": 0, "max_tokens": a.cap}
        if a.bias_think:
            body["logit_bias"] = [[think[0], a.bias_think]]
        if a.lora:
            body["lora"] = [{"id": int(x.split(":")[0]), "scale": float(x.split(":")[1])} for x in a.lora.split(",")]
        r = post(base + "/v1/chat/completions", body)
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        pred = m5.extract_pred(content)   # same extractor as engine/bench_runner.py (box, else last number)
        return {"grp": grp, "i": i, "closed": ch["finish_reason"] == "stop",
                "ok": bool(pred) and m5.equivalent(pred, items[i]["answer"]),
                "tokens": r["usage"]["completion_tokens"], "pred": pred, "content_tail": content[-300:]}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(one, ids))
    summ = {"name": a.name, "cap": a.cap, "system": a.system, "bias_think": a.bias_think, "lora": a.lora,
            "seconds": round(time.time() - t0)}
    for g in sorted({r["grp"] for r in rows}):
        rs = [r for r in rows if r["grp"] == g]
        summ[g] = {"n": len(rs), "closed": sum(r["closed"] for r in rs), "ok": sum(r["ok"] for r in rs),
                   "median_tokens": sorted(r["tokens"] for r in rs)[len(rs) // 2]}
    out = ROOT / "lab/runs/close_probe/fast"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps({"summary": summ, "items": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
