"""Analysis of the taskvec_mult experiment (criteria preregistered in PREREG.md).

Reads lab/runs/taskvec_mult/fast/*.json. Per candidate prints:
  recovery vs donor (mult), retention vs base (gsm8k + math500, items the base solved), params fraction,
then emits the PREREG verdicts R1 R2 R3. Writes fast/analysis.json.

usage: python lab/probes/taskvec_mult/analyze.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "lab/runs/taskvec_mult/fast"
RANKS = (8, 16, 64, 256)
TOTAL_PARAMS = 0.824e9


def run(name):
    p = FAST / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def acc(name):
    d = run(name)
    if not d:
        return None
    s = d["summary"]["A"]
    return s["closed"] / s["n"]


def base_solved(run_d):
    return {(x["grp"], str(x["i"])) for x in run_d["items"] if x["ok"]}


def screen():
    """Phase A: cheap mult-only screen. Picks the smallest rank that recovers >=90% (or None)."""
    base_m, donor_m = acc("base_mult"), acc("donor_mult")
    if base_m is None or donor_m is None:
        print("MISSING base_mult/donor_mult"); return
    gain = donor_m - base_m
    best = None
    for r in RANKS:
        m = acc(f"adapter_r{r}_mult")
        if m is None:
            print(f"r{r} missing"); continue
        rec = (m - base_m) / gain if gain > 0 else 1.0
        print(f"r{r} mult {m:.3f} recovery {rec:.3f}")
        if rec >= 0.90 and (best is None or r < best):
            best = r
    for r in RANKS:
        p = acc(f"placebo_r{r}_mult")
        print(f"placebo_r{r}", None if p is None else f"{(p - base_m) / gain:.3f}")
    (FAST / "screen.json").write_text(json.dumps({"base": base_m, "donor": donor_m, "best": best},
                                                 indent=1), encoding="utf-8")
    print("BEST", best)


def retention(base_run, cand_run):
    bb, cc = run(base_run), run(cand_run)
    if not bb or not cc:
        return None
    sol = base_solved(bb)
    got = {(x["grp"], str(x["i"])) for x in cc["items"] if x["ok"]}
    kept = len(sol & got)
    return kept / len(sol) if sol else None


def rows_params_frac(r):
    from safetensors.torch import load_file
    p = ROOT / "lab/runs/taskvec_mult" / f"adapter_r{r}" / "adapter_model.safetensors"
    if not p.exists():
        return None
    return sum(t.numel() for t in load_file(str(p)).values()) / TOTAL_PARAMS


def main():
    if "--screen" in sys.argv:
        screen()
        return
    base_m, donor_m = acc("base_mult"), acc("donor_mult")
    gsm_b, math_b = "gsm_base_08", "math_base_08"
    if base_m is None or donor_m is None:
        print("MISSING base_mult/donor_mult runs")
        return
    d_gain = donor_m - base_m
    print(f"mult base {base_m:.3f} donor {donor_m:.3f} donor gain {d_gain:+.3f}")
    rows, best = [], None
    for r in RANKS:
        m = acc(f"adapter_r{r}_mult")
        if m is None:
            continue
        rec = (m - base_m) / d_gain if d_gain > 0 else 1.0
        frac = rows_params_frac(r)
        ret = [retention(gsm_b, f"gsm_adapter_r{r}_08"), retention(math_b, f"math_adapter_r{r}_08")]
        rows.append({"r": r, "mult": m, "recovery": rec, "params_frac": frac,
                     "retention_gsm": ret[0], "retention_math": ret[1]})
        valid = [x for x in ret if x is not None]
        if rec >= 0.9 and (not valid or min(valid) >= 0.9) and (best is None or frac < best["r"]):
            best = {"r": r, "frac": frac}
    for w in rows:
        print(f"r{w['r']:<4} mult {w['mult']:.3f} recovery {w['recovery']:.3f} "
              f"params {'' if w['params_frac'] is None else format(w['params_frac'], '.3%')} "
              f"ret gsm/{fmt_ret(w['retention_gsm'])} math/{fmt_ret(w['retention_math'])}")
    verdict = {}
    if d_gain <= 0:
        print("donor FAILED to improve mult (D criterion)")
        verdict = {"D": False}
    else:
        verdict["D"] = True
        verdict["R1"] = best is not None
        if best:
            g = retention(gsm_b, f"gsm_adapter_r{best['r']}_08")
            m = retention(math_b, f"math_adapter_r{best['r']}_08")
            both = [x for x in (g, m) if x is not None]
            verdict["R2"] = bool(both) and min(both) >= 0.9
        plc_rec = []
        for pr in RANKS:
            p = acc(f"placebo_r{pr}_mult")
            if p is not None:
                plc_rec.append((p - base_m) / d_gain)
        if plc_rec:
            worst = max(plc_rec)
            verdict["R3"] = worst < 0.30
            print("placebo recoveries", [f"{x:.3f}" for x in plc_rec], "-> R3", verdict["R3"])
        print("best adapter", best)
    print("VERDICT", verdict)
    (FAST / "analysis.json").write_text(json.dumps(
        {"rows": rows, "verdict": verdict, "base": base_m, "donor": donor_m, "best": best}, indent=1),
        encoding="utf-8")


def fmt_ret(f):
    return "n/a" if f is None else f"{f:.3f}"


if __name__ == "__main__":
    main()
