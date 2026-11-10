"""Sealed confirmation set 2 (PLAN.md): 100 GSM8K test + 50 MATH-500 items not in dev, not in sealed set 1, not in the
cc split, and not present as (bench, i) in any file under lab/runs/. Seed 20261111. Made once, hash recorded.

usage: python lab/probes/cc_weights/sealed2.py make | check"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import sealed_set as ss  # noqa: E402

PATH = ROOT / "data" / "sealed_confirm2.json"
SEED = 20261111


def used():
    seen = {"gsm8k": set(), "math500": set()}

    def walk(o):
        if isinstance(o, dict):
            if o.get("bench") in seen and isinstance(o.get("i"), int):
                seen[o["bench"]].add(o["i"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in (ROOT / "lab" / "runs").rglob("*"):
        if p.suffix not in (".json", ".jsonl") or not p.is_file() or p.stat().st_size > 200 * 2**20:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
            docs = [json.loads(txt)] if p.suffix == ".json" else [json.loads(x) for x in txt.splitlines() if x.strip()]
        except (ValueError, UnicodeDecodeError):
            continue
        for d in docs:
            walk(d)
    return seen


def make():
    sys.path.insert(0, str(ROOT / "rig"))
    import math500 as m5
    n_m, n_g = len(m5.load_items()), len(ss._gsm())
    u = used()
    s1 = ss.load()
    split = json.loads((ROOT / "lab" / "runs" / "cc" / "split.json").read_text())
    bad_g = u["gsm8k"] | set(range(ss.DEV_GSM_N)) | set(s1["gsm8k"])
    bad_m = (u["math500"] | set(m5.select_indices(ss.DEV_MATH[0], ss.DEV_MATH[1], n_m)) | set(s1["math500"])
             | set(split["discovery_math"]) | set(split["screen_math"]))
    r = random.Random(SEED)
    g = sorted(r.sample(sorted(set(range(n_g)) - bad_g), 100))
    m = sorted(r.sample(sorted(set(range(n_m)) - bad_m), 50))
    return {"seed": SEED, "gsm8k": g, "math500": m, "sha256": ss.digest(g, m),
            "excluded": {"gsm8k": len(bad_g), "math500": len(bad_m),
                         "rule": "dev, sealed set 1, cc split, and every (bench, i) in lab/runs/"}}


if __name__ == "__main__":
    if sys.argv[1:] == ["make"]:
        if PATH.exists():
            sys.exit(f"{PATH} exists: a sealed set is made once")
        PATH.write_text(json.dumps(make(), indent=1), encoding="utf-8")
    s = ss.load(PATH)
    print(json.dumps({"ok": True, "sha256": s["sha256"], "excluded": s.get("excluded")}))
