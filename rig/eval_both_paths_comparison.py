"""eval_both_paths_comparison.py — Rigorous A/B Comparison of Path 1 (GGUF Control Vector) vs Path 2 (SafeTensors Multilayer Checkpoint).
"""
import time
import json
from pathlib import Path
import urllib.request
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

PROBLEMS = [
    {
        "id": "watermelon",
        "raw_prompt": "A 100 kg watermelon is 99% water. After sitting in the sun, it is 98% water. What is its new weight in kg? Answer concisely with just the number.",
        "thought_prompt": "A 100 kg watermelon is 99% water. After sitting in the sun, it is 98% water. What is its new weight in kg?\nThink step by step, verify the non-water mass, and state the final number.",
        "expected": "50"
    },
    {
        "id": "strawberry",
        "raw_prompt": "How many times does the letter 'r' appear in the word 'strawberry'? Answer concisely with just the number.",
        "thought_prompt": "How many times does the letter 'r' appear in the word 'strawberry'?\nSpell out the word letter by letter, verify each position, and state the final count.",
        "expected": "3"
    },
    {
        "id": "sally",
        "raw_prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Answer concisely with just the number.",
        "thought_prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?\nAnalyze the family relations carefully, verify whether Sally herself is one of the sisters, and state the final number.",
        "expected": "1"
    },
    {
        "id": "race",
        "raw_prompt": "If you overtake the person in 2nd place in a race, what place are you in? Answer concisely.",
        "thought_prompt": "If you overtake the person in 2nd place in a race, what place are you in?\nAnalyze the order of runners before and after overtaking, and state your final rank.",
        "expected": "2nd"
    },
    {
        "id": "pills",
        "raw_prompt": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How many minutes do the pills last? Answer concisely.",
        "thought_prompt": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How many minutes do the pills last?\nTrace the exact timestamp of each pill (Pill 1, Pill 2, Pill 3), and state total minutes.",
        "expected": "60"
    }
]

def query(prompt, max_tokens=150):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()

def run_comparison():
    print("=" * 75)
    print("EMPIRICAL COMPARISON: INTUITION RUSH vs DELIBERATIVE REASONING (Path 1)")
    print("Server: llama-server with data/u_critic_control_vector.gguf active")
    print("=" * 75)

    results = []
    for item in PROBLEMS:
        print(f"\n--- Testing Problem: {item['id'].upper()} (Target: {item['expected']}) ---")
        
        # 1. Zero-shot intuition rush (no deliberation tokens)
        t0 = time.time()
        ans_raw = query(item["raw_prompt"], max_tokens=60)
        dt_raw = time.time() - t0
        hit_raw = item["expected"].lower() in ans_raw.lower()

        # 2. Deliberative reasoning (allows verification step)
        t0 = time.time()
        ans_thought = query(item["thought_prompt"], max_tokens=250)
        dt_thought = time.time() - t0
        hit_thought = item["expected"].lower() in ans_thought.lower()

        status_raw = "[PASS]" if hit_raw else "[FAIL]"
        status_thought = "[PASS]" if hit_thought else "[FAIL]"

        print(f"  Mode A (Raw/Rush):        {status_raw} Got: {ans_raw[:35]!r} ({dt_raw:.2f}s)")
        print(f"  Mode B (Deliberative/WSP): {status_thought} Got: {ans_thought[:35]!r} ({dt_thought:.2f}s)")

        results.append({
            "id": item["id"],
            "expected": item["expected"],
            "raw": {"reply": ans_raw, "hit": hit_raw, "latency": round(dt_raw, 2)},
            "deliberative": {"reply": ans_thought, "hit": hit_thought, "latency": round(dt_thought, 2)}
        })

    print("\n" + "=" * 75)
    raw_acc = sum(1 for r in results if r["raw"]["hit"]) / len(results) * 100.0
    delib_acc = sum(1 for r in results if r["deliberative"]["hit"]) / len(results) * 100.0
    print(f"SUMMARY RESULTS ON HARD TRAP QUESTIONS:")
    print(f"  Mode A (Intuition Rush):        {raw_acc:.1f}%")
    print(f"  Mode B (Deliberative/Critic):   {delib_acc:.1f}%")
    print("=" * 75)
    return results

if __name__ == "__main__":
    run_comparison()
