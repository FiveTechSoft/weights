"""Unpatched calibration on the CALIBRATION pool only (never the holdout): picks the skill with headroom, per amendment 2."""
import json, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText, rss_mb
from data import make_sets, is_hit, SKILLS

q = QwenText(sys.argv[1])
res = {}
for skill in ["add2", "add3", "add4", "mul2x2"]:
    pool = make_sets(skill)["calibration"]
    t = time.time(); gen = []
    for i in range(0, len(pool), 25):
        gen += q.generate([p["prompt"] for p in pool[i:i + 25]], 6)
    hits = sum(is_hit(g, p["answer"]) for g, p in zip(gen, pool))
    res[skill] = {"hits": hits, "n": len(pool), "exact_match": hits / len(pool), "seconds": round(time.time() - t, 1),
                  "examples": [{"prompt": p["prompt"], "gen": g} for g, p in list(zip(gen, pool))[:3]]}
    print(skill, hits, "/", len(pool), res[skill]["seconds"], "s", "rss", round(rss_mb()), flush=True)
json.dump(res, open("lab/runs/closed_form_math/calibration.json", "w"), indent=1)
