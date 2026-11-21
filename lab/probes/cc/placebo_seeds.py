"""Several same-norm random placebos of one control vector, one per seed (lab/probes/cc/PLAN.md, "Placebo
distribution"). Until now every placebo had one seed (2026). Usage (from the repo root):

  python lab/probes/cc/placebo_seeds.py check                 -> regenerate seed 2026 from cc_v2.gguf, compare with the file
  python lab/probes/cc/placebo_seeds.py make 1 2 3 4 5 6 7 8  -> hf_cache/cc/cc_v2.placebo{seed}.gguf (skips existing)
  python lab/probes/cc/placebo_seeds.py readings 1 2 ... 8    -> lab/runs/cc_v2/placebo_seeds/readings.json

The placebo is contrast_vector.placebo (random direction, the vector's norm per layer), fed with the vector read back
from the GGUF in layer order, which is the order the op used; the check proves the stream is the same."""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import contrast_vector as cv  # noqa: E402

VEC = ROOT / "hf_cache" / "cc" / "cc_v2.gguf"
RUNS = ROOT / "lab" / "runs" / "cc_v2"
BASE, VECTOR = "screen_base", "screen_s1.0"


def read_vecs(path):
    """-> {N: float32 (d,)} in layer order (the op built its dict in this order; placebo() draws in dict order)."""
    import gguf
    r = gguf.GGUFReader(str(path))
    t = {int(x.name.split(".")[1]): np.array(x.data, dtype=np.float32) for x in r.tensors}
    return {N: t[N] for N in sorted(t)}


def model_hint(path):
    import gguf
    r = gguf.GGUFReader(str(path))
    f = r.fields["controlvector.model_hint"]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def placebo_path(vec_path, seed, out_dir=None):
    vec_path = Path(vec_path)
    return Path(out_dir or vec_path.parent) / (vec_path.stem + f".placebo{seed}.gguf")


def make(seeds, vec_path=VEC, out_dir=None):
    vecs = read_vecs(vec_path)
    hint = model_hint(vec_path)
    out = []
    for s in seeds:
        p = placebo_path(vec_path, s, out_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            cv.write_gguf(cv.placebo(vecs, int(s)), p, hint)
        out.append(p)
    return out


def check(vec_path=VEC, placebo_path=None, seed=2026):
    """Regenerate the placebo of `seed` from the vector and compare with the file the op wrote."""
    placebo_path = Path(placebo_path or Path(vec_path).with_name(Path(vec_path).stem + f".placebo{seed}.gguf"))
    vecs = read_vecs(vec_path)
    mine = cv.placebo(vecs, int(seed))
    theirs = read_vecs(placebo_path)
    same = sorted(mine) == sorted(theirs) and all(np.array_equal(mine[N], theirs[N]) for N in mine)
    maxdiff = max(float(np.abs(mine[N] - theirs[N]).max()) for N in mine) if sorted(mine) == sorted(theirs) else None
    return {"vector": str(vec_path), "placebo": str(placebo_path), "seed": int(seed), "layers": len(vecs),
            "identical": bool(same), "max_abs_diff": maxdiff}


def _mcnemar(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def _items(runs, name):
    d = json.loads((Path(runs) / f"{name}.json").read_text(encoding="utf-8"))
    return {(x["bench"], x["i"]): x for x in d["items"]}


def _arm(items, ref=None):
    it = list(items.values())
    out = {"n": len(it), "correct": sum(bool(x["ok"]) for x in it), "correct_g2": sum(bool(x["ok_independent"]) for x in it),
           "cut": sum(x["finish"] == "length" for x in it), "loops": sum(bool(x.get("loop")) for x in it),
           "mean_tokens": round(sum(x["tokens"] for x in it) / max(len(it), 1))}
    if ref is not None:
        w = sum(1 for k, x in items.items() if x["ok"] and not ref[k]["ok"])
        l = sum(1 for k, x in items.items() if not x["ok"] and ref[k]["ok"])
        out["vs_base"] = {"win": w, "loss": l, "p": _mcnemar(w, l)}
    return out


def readings(runs=RUNS, seeds=(), base=BASE, vector=VECTOR, out=None):
    runs = Path(runs)
    B = _items(runs, base)
    V = _items(runs, vector)
    b, v = _arm(B), _arm(V, B)
    per = []
    for s in seeds:
        a = _arm(_items(runs, f"screen_pl{s}"), B)
        a["seed"] = int(s)
        per.append(a)
    corr = [a["correct"] for a in per]
    thr = b["correct"] + math.ceil((v["correct"] - b["correct"]) / 2)
    res = {"base": b, "vector": v, "seeds": per,
           "median_correct": statistics.median(corr) if corr else None,
           "min_correct": min(corr) if corr else None, "max_correct": max(corr) if corr else None,
           "half_gain_threshold": thr,
           "n_seeds_at_or_above_half_gain": sum(c >= thr for c in corr),
           "n_seeds_better_than_base_p05": sum(a["vs_base"]["p"] < 0.05 and a["correct"] > b["correct"] for a in per),
           "n_seeds_worse_than_base_p05": sum(a["vs_base"]["p"] < 0.05 and a["correct"] < b["correct"] for a in per),
           "median_at_or_below_base": (statistics.median(corr) <= b["correct"]) if corr else None,
           "vector_rank": 1 + sum(c >= v["correct"] for c in corr)}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        r = check()
        (RUNS / "placebo_seeds").mkdir(parents=True, exist_ok=True)
        (RUNS / "placebo_seeds" / "check.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r))
        sys.exit(0 if r["identical"] else 1)
    elif cmd == "make":
        for p in make([int(a) for a in args]):
            print(p)
    elif cmd == "readings":
        print(json.dumps(readings(RUNS, [int(a) for a in args], out=RUNS / "placebo_seeds" / "readings.json"), indent=1))
    else:
        sys.exit(__doc__)
