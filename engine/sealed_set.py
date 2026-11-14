"""A confirmation set sealed before use, for transplant verdicts (idea: sealed test sets hashed before any model exists,
matthewgribben.com/blog/written-neural-nets-functional).

The dev set of the transplant work (bench_runner defaults: GSM8K test first 100, MATH-500 select_indices(2026, 50)) has
been looked at many times to choose doses, scales and donors, so a result there is optimistic. The sealed set is 100
other GSM8K test problems and 50 other MATH-500 problems, chosen by a fixed seed and recorded with a sha256 of their
text and answers. Rule (lab/probes/sealed/README.md): run it only on a transplant that already passed its pre-registered
criterion on the dev set, once per arm, and report it whatever it says.

usage: python -m engine.sealed_set make     -> data/sealed_confirm.json (refuses to overwrite)
       python -m engine.sealed_set check    -> verifies the hash
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rig"))
PATH = ROOT / "data" / "sealed_confirm.json"
SEED = 20261109
DEV_GSM_N, DEV_MATH = 100, (2026, 50)


def _gsm():
    return [json.loads(x) for x in (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()]


def digest(gsm_idx, math_idx):
    import math500 as m5
    g, m = _gsm(), m5.load_items()
    blob = json.dumps({"gsm8k": [[i, g[i]["question"], g[i]["answer"]] for i in gsm_idx],
                       "math500": [[i, m[i]["problem"], m[i]["answer"]] for i in math_idx]}, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make(seed=SEED, gsm_n=100, math_n=50):
    import math500 as m5
    total_m = len(m5.load_items())
    dev_m = set(m5.select_indices(DEV_MATH[0], DEV_MATH[1], total_m))
    r = random.Random(seed)
    gsm_idx = sorted(r.sample(range(DEV_GSM_N, len(_gsm())), gsm_n))
    math_idx = sorted(r.sample(sorted(set(range(total_m)) - dev_m), math_n))
    return {"seed": seed, "gsm8k": gsm_idx, "math500": math_idx, "sha256": digest(gsm_idx, math_idx),
            "excluded": "GSM8K test 0-99 and MATH-500 select_indices(2026, 50): the transplant dev set"}


def load(path=PATH):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest(s["gsm8k"], s["math500"]) != s["sha256"]:
        raise ValueError(f"{path}: items do not match the sealed hash")
    return s


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "make":
        if PATH.exists():
            sys.exit(f"{PATH} exists: a sealed set is made once")
        PATH.write_text(json.dumps(make(), indent=1), encoding="utf-8")
        print(json.dumps({"written": str(PATH), "sha256": load()["sha256"]}))
    else:
        print(json.dumps({"ok": True, "sha256": load()["sha256"]}))
