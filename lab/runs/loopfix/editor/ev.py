"""Budgeted dev_eval wrapper: python ev.py <model path> <note>. Appends result to evals.jsonl."""
import json, sys
sys.path.insert(0, r"C:\weights\lab\probes\loopfix")
import harness
r = harness.dev_eval(sys.argv[1], " ".join(sys.argv[2:]))
r["path"] = sys.argv[1]; r["note"] = " ".join(sys.argv[2:])
open(r"C:\weights\lab\runs\loopfix\editor\evals.jsonl", "a", encoding="utf-8").write(json.dumps(r) + "\n")
print(json.dumps(r, indent=1))
