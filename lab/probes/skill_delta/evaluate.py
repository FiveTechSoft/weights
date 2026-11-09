"""Free-generation evaluation of A (base) or B (one tensor replaced). Prints/stores exact-match counts and per-item verdicts.
usage: evaluate.py MODEL_DIR OUT.json SET [SET...] [--tensor FILE.safetensors]   SET in holdout|selection|control|text
Hit = exact match of the text before the first newline (arithmetic) or gold substring in the first 8 generated tokens (text)."""
import json, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets, is_hit

args = sys.argv[1:]; tensor = None
if "--tensor" in args:
    i = args.index("--tensor"); tensor = args[i + 1]; del args[i:i + 2]
model, out, sets = args[0], args[1], args[2:]
import os
LO, HI = map(int, os.environ.get("PART", "0:100000").split(":"))   # run a slice of a set so one call fits the 2-minute tool limit
q = QwenText(model)
if tensor:
    from safetensors.torch import load_file
    for name, w in load_file(tensor).items():
        tgt = dict(q.m.named_parameters())[name]; assert tgt.shape == w.shape and tgt.dtype == w.dtype, name
        tgt.data.copy_(w)
res = {"tensor": tensor}
for s in sets:
    t = time.time()
    if s == "text":
        items = json.load(open("lab/probes/skill_delta/controls_text.json"))[LO:HI]; prompts = [p for p, _ in items]
        gen = [g for i in range(0, len(prompts), 5) for g in q.generate(prompts[i:i + 5], 8)]
        verdict = [gold in g for g, (_, gold) in zip(gen, items)]
    else:
        skill = "add2" if s == "control" else "mul2x2"
        pool = make_sets(skill)["control" if s == "control" else s][LO:HI]
        gen = [g for i in range(0, len(pool), 5) for g in q.generate([p["prompt"] for p in pool[i:i + 5]], 6)]
        verdict = [is_hit(g, p["answer"]) for g, p in zip(gen, pool)]
    res[s] = {"hits": sum(verdict), "n": len(verdict), "verdict": verdict, "gen": gen, "seconds": round(time.time() - t, 1)}
    print(s, sum(verdict), "/", len(verdict), res[s]["seconds"], "s", flush=True)
    json.dump(res, open(out, "w"))
