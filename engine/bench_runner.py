"""engine/bench_runner.py - GSM8K and MATH-500 against a chat server (OpenAI API), one fixed protocol for every model.

Greedy (temperature 0, seed 0), the model's own chat template (thinking as the template does it), at most
max_tokens new tokens; user message = problem + "Please reason step by step, and put your final answer within
\\boxed{}." GSM8K: the first N test problems. MATH-500: rig/math500.select_indices(2026, N).
Two graders: rig/math500.equivalent on the last \\boxed{}, and an independent one (GSM8K: the last number in the
last box, as a float; MATH-500: string compare after a separate minimal normaliser). Disagreements are for a human.
Every finished item is appended to <out_dir>/<name>.partial.jsonl, so a stopped run resumes.
Used by the WSPLang op evaluate_benchmark and by lab/probes/model_compare/eval.py.
"""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TAIL_CHARS = 2000   # saved end of reasoning + answer; 300 until 2026-11-07 (too short to see long loop periods)


def loop_tail(text: str, min_len: int = 8, reps: int = 3) -> bool:
    """True if some line of at least min_len characters (stripped) occurs reps or more times: the end-of-answer
    indecision loops of results.md §39.5 ("Wait, I'll check if I should use \\boxed{64}. I'll use ...")."""
    from collections import Counter
    c = Counter(line.strip() for line in text.splitlines() if len(line.strip()) >= min_len)
    return any(v >= reps for v in c.values())


def _m5():
    sys.path.insert(0, str(ROOT / "rig"))
    import math500
    return math500


def endpoint_url(endpoint) -> str:
    e = str(endpoint)
    return f"http://127.0.0.1:{e}/v1/chat/completions" if e.isdigit() else e


def ask(url, text, max_tokens, lora=None, extra=""):
    """lora: [(id, scale), ...] for adapters loaded with llama-server --lora ... --lora-init-without-apply.
    extra: text appended after SUFFIX (prompt-control arm, lab/probes/cc/PLAN.md); "" sends exactly text + SUFFIX."""
    body = {"messages": [{"role": "user", "content": text + SUFFIX + extra}], "temperature": 0, "seed": 0,
            "max_tokens": max_tokens}
    if lora:
        body["lora"] = [{"id": int(i), "scale": float(s)} for i, s in lora]
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    ch = r["choices"][0]
    msg = ch["message"]
    content, reasoning = split_think(msg.get("content") or "", msg.get("reasoning_content") or "")
    return {"content": content, "reasoning": reasoning,
            "finish": ch["finish_reason"], "tokens": r["usage"]["completion_tokens"],
            "tok_s": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}


def served_info(url):
    """What the server says it serves: llama-server's /props model_path and that file's sha256 (None if the path is not
    a local file). None when /props is unavailable. Recorded in every report so an arm labelled "merged" that served
    the base (results §45.11) shows in the JSON. Control vectors and LoRAs are not in /props: the chain log has them."""
    import hashlib
    try:
        base = url.split("/v1/")[0]
        props = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/props"), timeout=30).read())
    except Exception:
        return None
    path = props.get("model_path")
    if not path:
        return None
    sha = None
    if Path(path).is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        sha = h.hexdigest()
    return {"model_path": path, "sha256": sha}


def split_think(content, reasoning):
    """Templates llama-server does not split (Phi-4-mini-reasoning: "<think>...</think>answer" inside content, no
    reasoning_content; lab/probes/cc_phi/PLAN.md P1) -> (content after </think>, reasoning between the tags). An unclosed
    <think> (cut trace) is all reasoning. When the server already split, or there is no tag, nothing changes."""
    if reasoning or "<think>" not in content:
        return content, reasoning
    head, _, rest = content.partition("<think>")
    think, closed, after = rest.partition("</think>")
    return ((head + after).strip("\n") if closed else ""), think.strip("\n")


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


def run(name, endpoint, gsm_n=100, math_n=50, parallel=2, max_tokens=8192, out_dir=None, log=print, lora=None,
        gsm_idx=None, math_idx=None, extra=""):
    """gsm_idx / math_idx: explicit item lists (e.g. engine/sealed_set.py); otherwise GSM8K first gsm_n and MATH-500
    select_indices(2026, math_n). extra: sentence appended to every prompt after SUFFIX (see ask)."""
    m5 = _m5()
    url = endpoint_url(endpoint)
    out_dir = Path(out_dir) if out_dir else ROOT / "lab" / "runs" / "model_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    gsm_all = [json.loads(line) for line in (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()]
    gsm_idx = list(gsm_idx) if gsm_idx is not None else list(range(min(gsm_n, len(gsm_all))))
    math_all = m5.load_items()
    if math_idx is None:
        math_idx = m5.select_indices(2026, math_n, len(math_all)) if math_n else []
    jobs = [("gsm8k", i, gsm_all[i]["question"], gsm_gold(gsm_all[i]["answer"])) for i in gsm_idx]
    jobs += [("math500", i, math_all[i]["problem"], math_all[i]["answer"]) for i in math_idx]
    t0 = time.time()
    partial = out_dir / f"{name}.partial.jsonl"
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["bench"], r["i"]) for r in rows}
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    log(f"[{name}] resuming with {len(rows)} done, {len(jobs)} to go")

    def one(job):
        bench, i, text, gold = job
        r = ask(url, text, max_tokens, lora=lora, extra=extra)
        pred = m5.extract_pred(r["content"])
        ok1 = bool(pred) and m5.equivalent(pred, gold)
        ok2 = grade_gsm_independent(r["content"], gold) if bench == "gsm8k" else (
            bool(pred) and simple_norm(pred) == simple_norm(gold))
        tail = (r["reasoning"] + r["content"])[-TAIL_CHARS:]
        return {"bench": bench, "i": i, "gold": gold, "pred": pred, "ok": ok1, "ok_independent": ok2,
                "finish": r["finish"], "tokens": r["tokens"], "tok_s": r["tok_s"],
                "reasoning_chars": len(r["reasoning"]), "tail": tail, "loop": loop_tail(tail)}

    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            rows.append(row)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                log(f"[{name}] {k}/{len(jobs)} {time.time() - t0:.0f}s")
    summary = {}
    for bench in ("gsm8k", "math500"):
        rs = [r for r in rows if r["bench"] == bench]
        if not rs:
            continue
        summary[bench] = {"n": len(rs), "acc": round(sum(r["ok"] for r in rs) / len(rs), 3),
                          "acc_independent": round(sum(r["ok_independent"] for r in rs) / len(rs), 3),
                          "graders_disagree": sum(r["ok"] != r["ok_independent"] for r in rs),
                          "cut_at_max": sum(r["finish"] == "length" for r in rs),
                          "loop_tails": sum(bool(r.get("loop")) for r in rs),
                          "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs)),
                          "mean_tok_s": round(sum(r["tok_s"] for r in rs) / len(rs), 1)}
    report = {"model": name, "endpoint": url, "max_tokens": max_tokens, "lora": lora, "extra": extra,
              "served": served_info(url), "seconds": round(time.time() - t0), "summary": summary, "items": rows}
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{name} {json.dumps(summary)}")
    return report
