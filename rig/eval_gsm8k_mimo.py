"""eval_gsm8k_mimo.py — Deterministic GSM8K evaluation for MiMo-9B on port 8081.

Evaluates 50 problems with greedy decoding (temperature=0.0) and \\boxed{} extraction.
Reports accuracy, Wilson 95% confidence intervals, and token stats.
"""
import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / "data" / "gsm8k_test.jsonl"
DIR_OUT = ROOT / "rig_runs"
DIR_OUT.mkdir(parents=True, exist_ok=True)

def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))

def extract_boxed(text: str):
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return m[-1].replace(",", "") if m else None
    depth = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                content = text[start:i].strip()
                content = content.strip("$").replace(",", "").strip()
                m = re.findall(r"-?\d+(?:\.\d+)?", content)
                return m[-1] if m else content
            depth -= 1
    return None

def normalize_ans(val):
    if val is None:
        return None
    val = str(val).strip().replace(",", "")
    try:
        f = float(val)
        return int(f) if f.is_integer() else f
    except ValueError:
        return val

def query_chat(endpoint, messages, temperature=0.0, max_tokens=1024):
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 42
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"] or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="rig_runs/gsm8k_mimo_baseline_n50.json")
    args = ap.parse_args()
    
    questions = []
    with open(DIR_DATA, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            row = json.loads(line)
            questions.append({
                "id": i + 1,
                "problem": row["question"],
                "gold": row["answer"].split("####")[-1].strip().replace(",", "")
            })
            
    print(f"=== EVALUATING MIMO-9B ON GSM8K (N={len(questions)}, temp={args.temperature}) ===")
    results = []
    correct_count = 0
    t0 = time.time()
    
    for q in questions:
        prompt_content = q["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        msgs = [{"role": "user", "content": prompt_content}]
        
        t_start = time.time()
        try:
            resp = query_chat(args.endpoint, msgs, temperature=args.temperature)
            pred = extract_boxed(resp)
            norm_pred = normalize_ans(pred)
            norm_gold = normalize_ans(q["gold"])
            is_ok = (norm_pred == norm_gold) and (norm_pred is not None)
            if is_ok:
                correct_count += 1
            dt = time.time() - t_start
            print(f"[{q['id']}/{args.n}] time={dt:.1f}s | pred={norm_pred} gold={norm_gold} -> {'PASS' if is_ok else 'FAIL'}", flush=True)
            results.append({
                "id": q["id"],
                "gold": norm_gold,
                "pred": norm_pred,
                "correct": is_ok,
                "time": round(dt, 2),
                "tail": resp[-200:]
            })
        except Exception as e:
            print(f"[{q['id']}/{args.n}] ERROR: {e}", flush=True)
            results.append({"id": q["id"], "error": str(e), "correct": False})
            
    total_time = time.time() - t0
    acc = correct_count / args.n
    low, high = wilson_interval(correct_count, args.n)
    
    summary = {
        "n": args.n,
        "correct": correct_count,
        "acc": round(acc, 4),
        "wilson_95": [round(low, 4), round(high, 4)],
        "total_seconds": round(total_time, 1),
        "seconds_per_problem": round(total_time / args.n, 2)
    }
    
    print("\n==========================================")
    print(f"RESUMEN GSM8K (N={args.n}):")
    print(f"  Aciertos: {correct_count}/{args.n} ({acc:.1%})")
    print(f"  Intervalo Wilson 95%: [{low:.1%}, {high:.1%}]")
    print(f"  Tiempo total: {total_time:.1f} s ({summary['seconds_per_problem']} s/problema)")
    print("==========================================\n")
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Guardado en {out_path}")

if __name__ == "__main__":
    main()
