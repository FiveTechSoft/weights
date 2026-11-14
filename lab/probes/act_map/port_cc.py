"""Gate 3b of cross-model porting (PLAN.md, addendum "Gate 3b"): does the 9B's "conclude" direction, written into the
0.8B through the gate 1 maps, close the 0.8B's answers at least half as well as the best closing intervention we have
for it (its own contrast over the same texts, or the Exp 8 edit)?

usage (from the repo root):
  python lab/probes/act_map/port_cc.py items      # dev items never used anywhere -> lab/runs/act_map/cc08/dev_items.json
  python lab/probes/act_map/port_cc.py editcheck  # the Exp 8 replica differs from the base in blk.15.ffn_down only
  python lab/probes/act_map/port_cc.py vectors    # D per hidden state in both models over the cc rows, port / embport /
                                                  # placebo, step 0 -> hf_cache/act_map/cvec08/cc_*.gguf, cc08/step0.json
  python lab/probes/act_map/port_cc.py choose <arm>   # dev scale for own / port / embport (PLAN rule) -> prints it or "none"
  python lab/probes/act_map/port_cc.py verdict        # Q1-Q5 mechanically -> cc08/verdict.json
Behavior runs go through engine.bench_runner against the 0.8B (chain script lab/runs/act_map/cc08_chain.sh)."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

OUT = ROOT / "lab" / "runs" / "act_map" / "cc08"
CVEC = ROOT / "hf_cache" / "act_map" / "cvec08"
ROWS = ROOT / "lab" / "runs" / "cc_v2"                 # pos.jsonl / neg.jsonl of the cc_v2 vector
BASE_GGUF = ROOT / "hf_cache" / "qwen35-08b-gguf" / "Qwen_Qwen3.5-0.8B-Q6_K.gguf"
EXP8_GGUF = ROOT / "servidos" / "qwen35-08b__exp8-replica.gguf"
EXP8_PLACEBO_GGUF = ROOT / "servidos" / "qwen35-08b__exp8-placebo.gguf"
ARMS = ("own", "port", "embport")
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
PLACEBO_SEED = 2026
DEV_GSM_N, DEV_MATH_N = 64, 32
TEST_ARMS = ("base", "own", "port", "embport", "placebo", "exp8", "exp8_placebo")


# ---------- pure pieces (tests/test_port_cc.py) ----------

def mcnemar(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def _paired(arm, base, key):
    w = sum(1 for k, x in arm.items() if x[key] and not base[k][key])
    l = sum(1 for k, x in arm.items() if not x[key] and base[k][key])
    return {"win": w, "loss": l, "p": mcnemar(w, l)}


def measures(items, base_items):
    """bench_runner items of one arm and of the base (same (bench, i)) -> per benchmark: correct (grader 1),
    correct2 (grader 2 / last number), finished, cut, loops, and exact McNemar against the base on each."""
    A = {(x["bench"], x["i"]): dict(x, fin=x["finish"] == "stop") for x in items}
    B = {(x["bench"], x["i"]): dict(x, fin=x["finish"] == "stop") for x in base_items}
    out = {}
    for bench in ("gsm8k", "math500"):
        a = {k: v for k, v in A.items() if k[0] == bench}
        b = {k: v for k, v in B.items() if k[0] == bench}
        assert set(a) == set(b), f"{bench}: arm and base items differ"
        out[bench] = {"n": len(a), "correct": sum(bool(x["ok"]) for x in a.values()),
                      "correct2": sum(bool(x["ok_independent"]) for x in a.values()),
                      "finished": sum(x["fin"] for x in a.values()),
                      "cut": sum(x["finish"] == "length" for x in a.values()),
                      "loops": sum(bool(x.get("loop")) for x in a.values()),
                      "vs_base": {"correct": _paired(a, b, "ok"), "correct2": _paired(a, b, "ok_independent"),
                                  "finished": _paired(a, b, "fin")}}
        out[f"mean_tokens_{bench}"] = round(sum(x["tokens"] for x in a.values()) / max(len(a), 1))
    return out


def choose_scale(dev_by_scale, base):
    """dev_by_scale: scale -> {"correct", "finished"} (both benchmarks summed); base: the same for the base arm.
    Most correct if any scale beats the base in correct; else most finished if any beats it (cc08 floor rule); ties ->
    smaller scale; None if nothing beats the base."""
    for key in ("correct", "finished"):
        better = {s: v[key] for s, v in dev_by_scale.items() if v[key] > base[key]}
        if better:
            return min(better, key=lambda s: (-better[s], s))
    return None


def _gain(st, arm, key="correct"):
    return sum(st[arm][b][key] - st["base"][b][key] for b in ("gsm8k", "math500"))


def _beats(st, arm, bench, key="correct"):
    a = st.get(arm)
    return (a is not None and a[bench][key] > st["base"][bench][key] and a[bench]["vs_base"][key]["p"] < 0.05)


def decide(st):
    """st: arm -> measures on the test items (None: the arm had no dev scale). PLAN.md gate 3b rule."""
    ref = None
    for cand in ("own", "exp8"):
        if _beats(st, cand, "gsm8k") or _beats(st, cand, "math500"):
            ref = cand
            break
    out = {"reference": ref}
    if ref is None:
        out["verdict"] = "NOT DECIDED: neither own nor the Exp 8 edit beats the base (p < 0.05) on this test"
        return out
    p = st.get("port")
    pl = st.get("placebo")
    e = st.get("embport")
    pg = _gain(st, "port") if p else 0
    q = {"Q1 closes (GSM8K finished, p<0.05)": _beats(st, "port", "gsm8k", "finished"),
         "Q2 correct above base (both graders, one benchmark)":
             (_beats(st, "port", "gsm8k", "correct") and _beats(st, "port", "gsm8k", "correct2"))
             or (_beats(st, "port", "math500", "correct") and _beats(st, "port", "math500", "correct2")),
         "Q3 placebo flat": (pl is not None and p is not None
                             and not _beats(st, "placebo", "gsm8k") and not _beats(st, "placebo", "math500")
                             and _gain(st, "placebo", "finished") < 0.5 * _gain(st, "port", "finished")),
         "Q4 port gain >= 0.5 x reference gain": p is not None and pg >= 0.5 * _gain(st, ref)}
    out.update(q)
    out["port_gain"], out["reference_gain"] = pg, _gain(st, ref)
    out["verdict"] = "PASS" if all(q.values()) else "FAIL"
    out["Q5 port - embport >= 0.5 x port gain"] = (p is not None and e is not None and pg - _gain(st, "embport") >= 0.5 * pg)
    return out


def free_items(used, gsm_n, math_n, gsm_total, math_total):
    """The first indices of each benchmark not in `used` ({"gsm8k": set, "math500": set})."""
    g = [i for i in range(gsm_total) if i not in used["gsm8k"]][:gsm_n]
    m = [i for i in range(math_total) if i not in used["math500"]][:math_n]
    return {"gsm8k": g, "math500": m}


def differing_tensors(a, b):
    """Names of tensors whose raw bytes differ between two GGUF files (same tensor set expected)."""
    import gguf
    import numpy as np
    ra, rb = gguf.GGUFReader(str(a)), gguf.GGUFReader(str(b))
    tb = {t.name: t for t in rb.tensors}
    out = []
    for t in ra.tensors:
        u = tb.get(t.name)
        if u is None or t.data.shape != u.data.shape or not np.array_equal(np.asarray(t.data).view(np.uint8),
                                                                              np.asarray(u.data).view(np.uint8)):
            out.append(t.name)
    out += [n for n in tb if n not in {t.name for t in ra.tensors}]
    return out


# ---------- run ----------

def items():
    """Dev items never used by any run, sealed set, dev set or the cc split; written once."""
    sys.path.insert(0, str(ROOT / "lab" / "probes" / "sealed"))
    import math500 as m5
    import sealed3
    from engine import sealed_set as ss
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "dev_items.json"
    if p.exists():
        print(p.read_text(encoding="utf-8"))
        return
    u = sealed3.used()
    n_m, n_g = len(m5.load_items()), len(ss._gsm())
    bad_g = set(u["gsm8k"]) | set(range(ss.DEV_GSM_N))
    bad_m = set(u["math500"]) | set(m5.select_indices(ss.DEV_MATH[0], ss.DEV_MATH[1], n_m))
    for f in ("sealed_confirm.json", "sealed_confirm2.json", "sealed_confirm3.json"):
        s = json.loads((ROOT / "data" / f).read_text(encoding="utf-8"))
        bad_g |= set(s["gsm8k"]); bad_m |= set(s["math500"])
    for f in (ROOT / "lab" / "runs" / "cc" / "split.json", ROOT / "lab" / "runs" / "cc_v2" / "split.json"):
        if f.exists():
            sp = json.loads(f.read_text(encoding="utf-8"))
            bad_m |= set(sp.get("discovery_math", [])) | set(sp.get("screen_math", []))
    res = free_items({"gsm8k": bad_g, "math500": bad_m}, DEV_GSM_N, DEV_MATH_N, n_g, n_m)
    res["rule"] = "first indices not in: dev, sealed sets 1-3, cc split, any (bench, i) under lab/runs/"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


def editcheck():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"base": str(BASE_GGUF), "edit": str(EXP8_GGUF), "differing": differing_tensors(BASE_GGUF, EXP8_GGUF)}
    res["placebo_differing"] = differing_tensors(BASE_GGUF, EXP8_PLACEBO_GGUF)
    res["expected"] = ["blk.15.ffn_down.weight", "blk.19.ffn_down.weight"]     # exp8_control/p50c_edit/manifest.json
    res["ok"] = sorted(res["differing"]) == res["expected"]
    (OUT / "editcheck.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res["ok"]


def vectors():
    import torch
    from safetensors.torch import load_file
    import act_map as am
    import port
    from engine import contrast_vector as cv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m9 = json.loads((am.STORE / "q9" / "meta.json").read_text(encoding="utf-8"))
    m08 = json.loads((am.STORE / "q08" / "meta.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    CVEC.mkdir(parents=True, exist_ok=True)
    if (OUT / "D.pt").exists():
        d = torch.load(OUT / "D.pt")
        D9, D08, n_pos, n_neg = d["D9"], d["D08"], d["pairs"][0], d["pairs"][1]
    else:
        D9, n_pos, n_neg = port._means(m9["model"], device, ROWS)
        D08, _, _ = port._means(m08["model"], device, ROWS)
        torch.save({"D9": D9, "D08": D08, "pairs": [n_pos, n_neg]}, OUT / "D.pt")
    n08 = D08.shape[0]
    maps = load_file(str(am.STORE / "maps_q9__q08.safetensors"))
    js = range(1, n08)
    Dport = port.port_diff(D9, {j: maps[f"W.{j}"] for j in js}, {j: int(maps[f"source.{j}"]) for j in js}, n08)
    fit, val, _ = am.split_texts(m08["text_of"], am.N_TEST)
    E = am.load_state("q9", 0)
    src = am.Source(E[fit])
    We = {}
    for j in js:
        Y = am.load_state("q08", j)
        We[j] = am.fit_ridge(src, Y[fit], E[val], Y[val])[0]
    del E, src
    Demb = port.port_diff(D9, We, {j: 0 for j in js}, n08)
    step0 = {"cos_port_own": port.cos_by_state(Dport, D08), "cos_embport_own": port.cos_by_state(Demb, D08),
             "cos_port_embport": port.cos_by_state(Dport, Demb),
             "norm_own": [float(D08[j].norm()) for j in js], "norm_port": [float(Dport[j].norm()) for j in js],
             "norm_embport": [float(Demb[j].norm()) for j in js], "pairs": [n_pos, n_neg]}
    files = {}
    for name, D in (("own", D08), ("port", Dport), ("embport", Demb)):
        vecs = {k: v.float().numpy() for k, v in port.residual_increments(D.float()).items()}
        files[name] = CVEC / f"cc_{name}.gguf"
        cv.write_gguf(vecs, files[name], "qwen35")
        if name == "port":
            files["placebo"] = CVEC / f"cc_port.placebo{PLACEBO_SEED}.gguf"
            cv.write_gguf(cv.placebo(vecs, PLACEBO_SEED), files["placebo"], "qwen35")
    step0["files"] = {k: str(v) for k, v in files.items()}
    (OUT / "step0.json").write_text(json.dumps(step0, indent=1), encoding="utf-8")
    import statistics as s
    print(json.dumps({"median_cos_port_own": s.median(step0["cos_port_own"]),
                      "median_cos_embport_own": s.median(step0["cos_embport_own"]),
                      "median_cos_port_embport": s.median(step0["cos_port_embport"])}), flush=True)


def _items(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))["items"]


def _sum(m):
    return {k: m["gsm8k"][k] + m["math500"][k] for k in ("correct", "finished")}


def choose(arm):
    base = _items("dev_base")
    by = {}
    for sc in SCALES:
        f = OUT / f"dev_{arm}_s{sc}.json"
        if f.exists():
            by[sc] = _sum(measures(_items(f"dev_{arm}_s{sc}"), base))
    sc = choose_scale(by, _sum(measures(base, base)))
    print("none" if sc is None else sc)
    return sc


def verdict():
    base = _items("test_base")
    st = {"base": measures(base, base)}
    for arm in TEST_ARMS[1:]:
        f = OUT / f"test_{arm}.json"
        st[arm] = measures(_items(f"test_{arm}"), base) if f.exists() else None
    out = {"measures": st,
           "scales": json.loads((OUT / "scales.json").read_text()) if (OUT / "scales.json").exists() else None}
    out.update(decide(st))
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "measures"}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "editcheck":
        sys.exit(0 if editcheck() else 1)
    {"items": items, "vectors": vectors, "choose": lambda: choose(a[1]), "verdict": verdict}[a[0]]()
