"""Unpatched calibration on the CALIBRATION pool only (never the holdout): picks the skill with headroom, per amendment 2.
Resumable: appends one JSON line per batch to lab/runs/closed_form_math/calibration.jsonl and skips batches already done.
usage: calibrate.py MODEL_DIR SKILL [MAX_SECONDS]"""
import json, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets

OUT = "lab/runs/closed_form_math/calibration.jsonl"
skill, budget = sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 90
pool = make_sets(skill)["calibration"]
done = set()
if os.path.exists(OUT):
    done = {(r["skill"], r["i"]) for r in map(json.loads, open(OUT))}
q = QwenText(sys.argv[1]); t0 = time.time()
for i in range(0, len(pool), 5):
    if (skill, i) in done:
        continue
    if time.time() - t0 > budget:
        print("budget reached at", i); break
    gen = q.generate([p["prompt"] for p in pool[i:i + 5]], 6)
    with open(OUT, "a") as f:
        f.write(json.dumps({"skill": skill, "i": i, "prompts": [p["prompt"] for p in pool[i:i + 5]],
                            "answers": [p["answer"] for p in pool[i:i + 5]], "gen": gen}) + "\n")
print("done batches for", skill, round(time.time() - t0, 1), "s")
