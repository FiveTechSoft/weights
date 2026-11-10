"""'Conclude' direction on Qwen3.5-9B without a donor (PLAN.md, fixed before running).

subcommands:
  split                      -> runs/split.json: discovery / screen MATH-500 items (not dev, not sealed) + GSM8K train 0-63
  traces <port>              base, thinking on, greedy, 4096 cap -> runs/traces.jsonl (full reasoning kept)
  rows                       -> runs/pos.jsonl / runs/neg.jsonl for the contrast_vector op (think + focus rows)
  choose                     scale from runs/screen_*.json (PLAN rule) -> runs/scale.json
  verdict                    C1-C3 on runs/dev_{base,vec,placebo}.json -> runs/verdict.json
"""
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "lab" / "runs" / "cc"
BASE_HF = ROOT / "hf_cache" / "qwen35-9b-hf"
SCALES = (0.25, 0.5, 1.0, 2.0)
POS_TOKENS, NEG_TOKENS = 48, 1536


def split():
    from engine import bench_runner, sealed_set
    m5 = bench_runner._m5()
    n = len(m5.load_items())
    used = set(m5.select_indices(2026, 50, n)) | set(sealed_set.load()["math500"])
    free = [i for i in range(n) if i not in used]
    random.Random(20261110).shuffle(free)
    s = {"discovery_math": free[:64], "screen_math": free[64:128], "discovery_gsm_train": list(range(64))}
    assert not set(s["discovery_math"]) & set(s["screen_math"]) and not set(free[:128]) & used
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "split.json").write_text(json.dumps(s))
    return s


def traces(port):
    import concurrent.futures
    from engine import bench_runner as b
    s = json.loads((RUNS / "split.json").read_text())
    m5 = b._m5()
    items = m5.load_items()
    gsm = [json.loads(x) for x in (ROOT / "data" / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines()]
    jobs = [("math", i, items[i]["problem"]) for i in s["discovery_math"]]
    jobs += [("gsm_train", i, gsm[i]["question"]) for i in s["discovery_gsm_train"]]
    out = RUNS / "traces.jsonl"
    done = {(r["src"], r["i"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())} if out.exists() else set()
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    url = b.endpoint_url(port)

    def one(j):
        r = b.ask(url, j[2], 4096)
        return {"src": j[0], "i": j[1], "prompt": j[2] + b.SUFFIX, "reasoning": r["reasoning"], "content": r["content"],
                "finish": r["finish"], "tokens": r["tokens"]}
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                print(f"[traces] {k}/{len(jobs)}", flush=True)


def finished(r):
    return r["finish"] == "stop" and r["reasoning"].strip() and r["content"].strip()


def rows():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_HF))
    tr = [json.loads(x) for x in (RUNS / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    kept = [r for r in tr if finished(r)]
    pos, neg = [], []
    for r in kept:
        text = r["reasoning"].lstrip("\n")
        off = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(off) < 2 * POS_TOKENS:
            continue
        cut = off[-POS_TOKENS][0]                              # start of the last 48 reasoning tokens
        neg_end = off[min(NEG_TOKENS, len(off) - POS_TOKENS) - 1][1]
        pos.append({"prompt": r["prompt"], "response": text, "think": True, "focus": [cut, len(text)]})
        neg.append({"prompt": r["prompt"], "response": text, "think": True, "focus": [0, neg_end]})
    for n, rr in (("pos", pos), ("neg", neg)):
        (RUNS / f"{n}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rr), encoding="utf-8")
    res = {"traces": len(tr), "finished": len(kept), "rows": len(pos), "enough": len(pos) >= 30}
    print(json.dumps(res))
    return res


def correct(name, grader="ok"):
    return sum(r[grader] for r in json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))["items"])


def choose():
    base = correct("screen_base")
    sc = {s: correct(f"screen_s{s}") for s in SCALES}
    best = max(SCALES, key=lambda s: (sc[s], -s))             # most correct, ties -> smaller scale
    res = {"base": base, "by_scale": sc, "scale": best if sc[best] > base else None}
    (RUNS / "scale.json").write_text(json.dumps(res))
    print(json.dumps(res))
    return res


def mcnemar(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def lastnum(r):
    m = re.findall(r"-?\d[\d,]*\.?\d*", r.get("tail") or "")
    try:
        return bool(m) and abs(float(m[-1].replace(",", "").rstrip(".")) - float(r["gold"].replace(",", ""))) < 1e-6
    except ValueError:
        return False


def compare(arm, base, bench, f):
    keys = sorted(k for k in base if k[0] == bench)
    a = [f(arm[k]) for k in keys]
    b = [f(base[k]) for k in keys]
    win = sum(x and not y for x, y in zip(a, b))
    loss = sum(y and not x for x, y in zip(a, b))
    return {"arm": sum(a), "base": sum(b), "win": win, "loss": loss, "p": mcnemar(win, loss)}


def tokens(arm, bench):
    t = [r["tokens"] for k, r in arm.items() if k[0] == bench]
    return sum(t) / len(t)


def verdict():
    load = lambda n: {(r["bench"], r["i"]): r for r in json.loads((RUNS / f"{n}.json").read_text(encoding="utf-8"))["items"]}
    B, V, P = load("dev_base"), load("dev_vec"), load("dev_placebo")
    g = {"math500": {"g1": lambda r: r["ok"], "g2": lambda r: r["ok_independent"]},
         "gsm8k": {"g1": lambda r: r["ok"], "lastnum": lastnum}}
    out = {arm: {bench: {gn: compare(X, B, bench, f) for gn, f in gs.items()} for bench, gs in g.items()}
           for arm, X in (("vec", V), ("placebo", P))}
    tok = {arm: {bench: round(tokens(X, bench) / tokens(B, bench) - 1, 3) for bench in g} for arm, X in
           (("vec", V), ("placebo", P))}
    out["token_change"] = tok
    sig = lambda c: c["arm"] > c["base"] and c["p"] < 0.05
    v = out["vec"]
    gain = (all(sig(c) for c in v["math500"].values()) or all(sig(c) for c in v["gsm8k"].values()) or
            (all(t <= -0.15 for t in tok["vec"].values()) and
             all(v[b]["g1"]["arm"] >= v[b]["g1"]["base"] - 2 for b in g)))
    pl = out["placebo"]
    placebo_null = (not any(sig(c) for bench in pl.values() for c in bench.values()) and
                    not all(t <= -0.15 for t in tok["placebo"].values()))
    no_damage = v["gsm8k"]["g1"]["arm"] >= v["gsm8k"]["g1"]["base"] - 2
    out["criteria"] = {"C1 gain": gain, "C2 placebo null": placebo_null, "C3 GSM8K not below base by > 2": no_damage}
    out["verdict"] = "PASS" if gain and placebo_null and no_damage else "FAIL"
    (RUNS / "verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    {"split": lambda: print(json.dumps({k: len(v) for k, v in split().items()})), "traces": lambda: traces(a[1]),
     "rows": rows, "choose": choose, "verdict": verdict}[a[0]]()
