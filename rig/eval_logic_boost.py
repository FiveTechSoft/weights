"""eval_logic_boost.py — Targeted GSM8K evaluation for MiMo-9B logic edits.

Evaluates the 5 deterministic baseline failure problems (IDs: 4, 14, 22, 38, 47)
plus 5 control passing problems (IDs: 1, 2, 3, 5, 6) for regression checking.
Uses the exact same 1-based IDs as rig_runs/gsm8k_mimo_baseline_n50.json.
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / "data" / "gsm8k_test.jsonl"
DIR_OUT = ROOT / "rig_runs"
DIR_OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_IDS = [4, 14, 22, 38, 47, 1, 2, 3, 5, 6]
FAIL_IDS = {4, 14, 22, 38, 47}

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
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"] or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--ids", default="4,14,22,38,47,1,2,3,5,6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--alias", default="eval_logic_boost")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    
    target_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    max_id = max(target_ids)
    
    id_to_data = {}
    with open(DIR_DATA, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            curr_id = i + 1
            if curr_id in target_ids:
                row = json.loads(line)
                id_to_data[curr_id] = {
                    "id": curr_id,
                    "problem": row["question"],
                    "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
                    "is_failure_baseline": curr_id in FAIL_IDS
                }
            if curr_id >= max_id:
                break
                
    questions = [id_to_data[qid] for qid in target_ids if qid in id_to_data]
    print(f"=== EVALUATING {args.alias} ON {len(questions)} GSM8K PROBLEMS (temp={args.temperature}) ===")
    
    results = []
    repaired_count = 0
    regressed_count = 0
    failures_total = sum(1 for q in questions if q["is_failure_baseline"])
    controls_total = sum(1 for q in questions if not q["is_failure_baseline"])
    
    t0 = time.time()
    for q in questions:
        tag = "[TARGET FAILURE]" if q["is_failure_baseline"] else "[CONTROL PASS]"
        print(f"Testing #{q['id']} {tag} (gold={q['gold']})...", end=" ", flush=True)
        t_start = time.time()
        
        prompt_content = q["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        msgs = [{"role": "user", "content": prompt_content}]
        
        try:
            resp = query_chat(args.endpoint, msgs, temperature=args.temperature, max_tokens=args.max_tokens)
            pred = extract_boxed(resp)
            norm_pred = normalize_ans(pred)
            norm_gold = normalize_ans(q["gold"])
            correct = (norm_pred == norm_gold)
            elapsed = time.time() - t_start
            
            if q["is_failure_baseline"] and correct:
                repaired_count += 1
                status = "REPAIRED! (SUCCESS)"
            elif not q["is_failure_baseline"] and not correct:
                regressed_count += 1
                status = "REGRESSED! (FAILURE)"
            elif correct:
                status = "PASS (expected)"
            else:
                status = "FAIL (unchanged)"
                
            print(f"{status} pred={pred} in {elapsed:.2f}s")
            results.append({
                "id": q["id"],
                "is_failure_baseline": q["is_failure_baseline"],
                "gold": norm_gold,
                "pred": norm_pred,
                "correct": correct,
                "time": round(elapsed, 2),
                "tail": resp[-250:] if len(resp) > 250 else resp
            })
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"ERROR: {e}")
            results.append({
                "id": q["id"],
                "is_failure_baseline": q["is_failure_baseline"],
                "gold": normalize_ans(q["gold"]),
                "pred": None,
                "correct": False,
                "error": str(e),
                "time": round(elapsed, 2)
            })
            
    total_time = time.time() - t0
    out_file = args.out or str(DIR_OUT / f"{args.alias}.json")
    
    report = {
        "alias": args.alias,
        "endpoint": args.endpoint,
        "total_problems": len(questions),
        "failures_targeted": failures_total,
        "failures_repaired": repaired_count,
        "controls_tested": controls_total,
        "controls_regressed": regressed_count,
        "total_seconds": round(total_time, 2),
        "results": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n=== SUMMARY FOR {args.alias} ===")
    print(f"Repaired Baseline Failures: {repaired_count} / {failures_total}")
    print(f"Regressed Controls:         {regressed_count} / {controls_total}")
    print(f"Report saved to: {out_file}")

if __name__ == "__main__":
    main()
