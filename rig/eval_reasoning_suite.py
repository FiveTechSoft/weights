"""eval_reasoning_suite.py — Evaluate MiMo-2.6 on Reasoning, Trap, and Reflection Problems.
"""
import urllib.request
import json
import time

ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

REASONING_BENCHMARK = [
    {
        "id": "watermelon",
        "question": "A 100 kg watermelon is 99% water. After sitting in the sun, it is 98% water. What is its new weight in kg? Respond concisely with just the final number.",
        "expected": "50",
        "category": "arithmetic_trap"
    },
    {
        "id": "strawberry",
        "question": "How many times does the letter 'r' appear in the word 'strawberry'? Respond concisely with just the final number.",
        "expected": "3",
        "category": "counting_trap"
    },
    {
        "id": "sheep",
        "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Respond concisely with just the final number.",
        "expected": "9",
        "category": "linguistic_trap"
    },
    {
        "id": "race",
        "question": "If you overtake the person in 2nd place in a race, what place are you in? Respond concisely with just the final rank (e.g. 1st, 2nd, 3rd).",
        "expected": "2nd",
        "category": "logic_trap"
    },
    {
        "id": "sally",
        "question": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Respond concisely with just the final number.",
        "expected": "1",
        "category": "family_logic"
    },
    {
        "id": "months",
        "question": "Some months have 31 days, others have 30. How many months have 28 days? Respond concisely with just the final number.",
        "expected": "12",
        "category": "riddle_trap"
    },
    {
        "id": "range_step",
        "question": "In Python, what is len(range(0, 10, 2))? Respond concisely with just the final number.",
        "expected": "5",
        "category": "code_invariants"
    },
    {
        "id": "car_speed",
        "question": "A car travels 60 miles at 30 mph, and returns 60 miles at 60 mph. What is the average speed in mph for the round trip? Respond concisely with just the final number.",
        "expected": "40",
        "category": "physics_math"
    },
    {
        "id": "pills",
        "question": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How many minutes do the pills last? Respond concisely with just the final number.",
        "expected": "60",
        "category": "timeline_trap"
    },
    {
        "id": "feathers_rocks",
        "question": "Which weighs more: 1 pound of feathers or 2 pounds of rocks? Respond concisely with just the heavier item.",
        "expected": "2 pounds of rocks",
        "category": "scale_trap"
    },
    {
        "id": "lily_pads",
        "question": "In a lake, a patch of lily pads doubles every day. If it takes 48 days to cover the lake, how many days to cover half? Respond with just the number.",
        "expected": "47",
        "category": "doubling_trap"
    },
    {
        "id": "handshakes",
        "question": "In a room of 5 people, everyone shakes hands with everyone else once. How many total handshakes occur? Respond with just the number.",
        "expected": "10",
        "category": "combinatorics"
    }
]

def run_eval(endpoint=ENDPOINT):
    print("=" * 70)
    print("EVALUATING MIMO-2.6 REASONING & TRAP SUITE (n=12)")
    print("=" * 70)
    results = []
    correct = 0

    for item in REASONING_BENCHMARK:
        payload = {
            "messages": [{"role": "user", "content": item["question"]}],
            "temperature": 0.0,
            "max_tokens": 80
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)
        dt = time.time() - t0
        
        reply = res["choices"][0]["message"]["content"].strip()
        exp = item["expected"].lower()
        
        # Check match
        hit = exp in reply.lower()
        if hit:
            correct += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"{status} {item['id']:<15} | Expected: {item['expected']:<10} | Got: {reply[:35]!r:<38} ({dt:.2f}s)")
        results.append({
            "id": item["id"],
            "category": item["category"],
            "expected": item["expected"],
            "reply": reply,
            "passed": hit,
            "latency": round(dt, 3)
        })

    acc = (correct / len(REASONING_BENCHMARK)) * 100.0
    print("-" * 70)
    print(f"Total Score: {correct}/{len(REASONING_BENCHMARK)} ({acc:.1f}%)")
    print("=" * 70)
    return results, acc

if __name__ == "__main__":
    run_eval()
