"""H8 (h8_plan.md): where the accuracy lives in C_200, and whether learned changes add up. Evaluation only.

Part A: C_200 = P + O with P = <C_200, d> d, d = C_k / |C_k| (k in 10, 25, 50); evaluate W0 + U P and W0 + U O.
Part B: W0 + U (C_i + C_j) for the 21 pairs of the 7 G2 tasks; W0 + U lambda sum(C) for all 7, lambda chosen on dev.
Writes lab/runs/h8/decompose.json and lab/runs/h8/library.json after every evaluation.

usage: python lab/probes/h8/h8.py [--part a|b|all|c]  (c = TIES, addendum a)
"""
import itertools
import json
import statistics as stt
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h7"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))
import tasks as tk  # noqa: E402

H7 = ROOT / "lab" / "runs" / "h7"
KEEP = ROOT / "hf_cache" / "h7"
OUT = ROOT / "lab" / "runs" / "h8"
KS = (10, 25, 50)
LAMBDAS = (0.5, 0.75, 1)


# ---------- pieces with tests ----------

def _dot(A, B):
    return sum(float((A[n].double() * B[n].double()).sum()) for n in A)


def decompose(Ck, CT):
    """C_T = P + O, P along C_k. Returns (P, O, a, |O|^2/|C_T|^2) with a = <C_T, C_k/|C_k|>."""
    nk = _dot(Ck, Ck) ** 0.5
    a = _dot(CT, Ck) / nk
    P = {n: (a / nk) * Ck[n].float() for n in CT}
    O = {n: CT[n].float() - P[n] for n in CT}
    return P, O, a, _dot(O, O) / _dot(CT, CT)


def retention(acc, acc0, acc_own):
    return (acc - acc0) / (acc_own - acc0)


def add_coefs(Cs, lam=1.0):
    return {n: lam * sum(C[n].float() for C in Cs) for n in Cs[0]}


def pick_lambda(dev_sum):
    best = max(dev_sum.values())
    return min(l for l, v in dev_sum.items() if v == best)


def pairs(names):
    return list(itertools.combinations(names, 2))


def ties_merge(Cs, density=1.0):
    """TIES: per task keep the top-density entries by |value| (global over matrices), elect the sign of the summed
    trimmed values per coordinate, average the trimmed values that agree with it (addendum a)."""
    names = sorted(Cs[0])
    X = torch.stack([torch.cat([C[n].float().flatten() for n in names]) for C in Cs])
    if density < 1.0:
        keep = max(1, int(round(density * X.shape[1])))
        for i in range(len(X)):
            idx = X[i].abs().topk(keep).indices
            mask = torch.zeros_like(X[i], dtype=torch.bool)
            mask[idx] = True
            X[i] = torch.where(mask, X[i], torch.zeros_like(X[i]))
    sign = torch.sign(X.sum(0))
    agree = (torch.sign(X) == sign) & (X != 0)
    num = (X * agree).sum(0)
    cnt = agree.sum(0)
    merged = torch.where(cnt > 0, num / cnt.clamp_min(1), torch.zeros_like(num))
    out, i = {}, 0
    for n in names:
        k = Cs[0][n].numel()
        out[n] = merged[i:i + k].reshape(Cs[0][n].shape)
        i += k
    return out


def pick_grid(dev_sum):
    best = max(dev_sum.values())
    return min(k for k, v in dev_sum.items() if v == best)


# ---------- run ----------

def main():
    from safetensors.torch import load_file
    import eval_arith as ea
    import h6f
    import run_task as rt
    import synth
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    ev = {n: json.loads((H7 / n / "evals.json").read_text(encoding="utf-8")) for n in tk.ORDER}
    held = [n for n in tk.ORDER if ev[n]["test"]["200"]["acc"] - ev[n]["test"]["0"]["acc"] >= 0.10]
    acc0 = {n: ev[n]["test"]["0"]["acc"] for n in held}
    accT = {n: ev[n]["test"]["200"]["acc"] for n in held}
    CT = {n: {m: x.float() for m, x in load_file(str(KEEP / n / "C_final.safetensors")).items()} for n in held}
    tok = ea.load_tokenizer()
    model, params = h6f.load_model()
    names = sorted(params)
    shapes = {n: tuple(params[n].shape) for n in names}
    W0 = {n: params[n].detach().clone() for n in names}
    U = {n: u.cuda() for n, u in rt.shared_basis(shapes).items()}
    t0 = time.time()

    @torch.no_grad()
    def set_C(Cd):
        for n in names:
            params[n].copy_(W0[n] + h6f.coef_apply(U[n], Cd[n].cuda(), shapes[n]))

    def test(task):
        return tk.eval_task(model, tok, task, tk.make_splits(task)["test"])

    def say(*a):
        print(*a, f"{time.time() - t0:.0f}s", flush=True)

    # second method for accT: re-evaluate C_200 here and compare with H7's stored number
    check = {}
    for n in held:
        set_C(CT[n])
        check[n] = test(n)["acc"]
        say("check", n, check[n], "h7", accT[n])

    if part in ("a", "all"):
        out = {"held": held, "acc0": acc0, "accT": accT, "accT_recheck": check, "k": {}}
        p = OUT / "decompose.json"
        for k in KS:
            out["k"][str(k)] = {}
            for n in held:
                Ck = {m: x.float() for m, x in load_file(str(KEEP / n / f"C_step{k}.safetensors")).items()}
                P, O, a, share = decompose(Ck, CT[n])
                row = {"a": a, "norm_Ck": _dot(Ck, Ck) ** 0.5, "norm_CT": _dot(CT[n], CT[n]) ** 0.5,
                       "off_share": share}
                masks = {}
                for arm, D in (("par", P), ("off", O)):
                    set_C(D)
                    e = test(n)
                    masks[arm] = e["ok_mask"]
                    row[arm] = {"acc": e["acc"], "R": retention(e["acc"], acc0[n], accT[n])}
                row["mcnemar_par_vs_off"] = synth.mcnemar(masks["par"], masks["off"])
                bk = ev[n]["test"].get(str(k))
                if bk:
                    row["B_k"] = {"acc": bk["acc"], "R": retention(bk["acc"], acc0[n], accT[n])}
                    row["mcnemar_par_vs_Bk"] = synth.mcnemar(masks["par"], bk["ok_mask"])
                out["k"][str(k)][n] = row
                p.write_text(json.dumps(out, indent=1), encoding="utf-8")
                say(f"A k={k} {n} share_off={share:.3f} par R={row['par']['R']:.2f} off R={row['off']['R']:.2f}")
        r25 = out["k"]["25"]
        mp = stt.mean(r25[n]["par"]["R"] for n in held)
        mo = stt.mean(r25[n]["off"]["R"] for n in held)
        out["summary"] = {str(k): {"par": stt.mean(out["k"][str(k)][n]["par"]["R"] for n in held),
                                   "off": stt.mean(out["k"][str(k)][n]["off"]["R"] for n in held),
                                   "off_share": stt.mean(out["k"][str(k)][n]["off_share"] for n in held)}
                          for k in KS}
        out["HA_off_line"] = mp < 0.5 and mo >= 0.5
        out["HA_line_enough"] = mp >= 0.8
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        say("A summary", json.dumps(out["summary"]), "HA", out["HA_off_line"], "HA'", out["HA_line_enough"])

    if part in ("b", "all"):
        geo = json.loads((H7 / "geometry.json").read_text(encoding="utf-8"))
        I, cosf = geo["interference"], geo["cos_final"]
        out = {"held": held, "acc0": acc0, "accT": accT, "accT_recheck": check, "pairs": {}}
        p = OUT / "library.json"
        cells = []
        for i, j in pairs(held):
            set_C(add_coefs([CT[i], CT[j]]))
            row = {"cos": cosf[i][j]}
            for t, o in ((i, j), (j, i)):
                acc = test(t)["acc"]
                pred = I[t][t] + I[o][t] - acc0[t]
                row[t] = {"acc": acc, "r": retention(acc, acc0[t], accT[t]), "additive_pred": pred}
                cells.append({"task": t, "other": o, "r": row[t]["r"], "err_pts": 100 * (acc - pred),
                              "cos": cosf[i][j]})
            out["pairs"][f"{i}+{j}"] = row
            p.write_text(json.dumps(out, indent=1), encoding="utf-8")
            say(f"B1 {i}+{j} r {row[i]['r']:.2f} / {row[j]['r']:.2f}")
        rs = [c["r"] for c in cells]
        worst = min(cells, key=lambda c: c["r"])
        out["B1"] = {"median_r": stt.median(rs), "mean_r": stt.mean(rs), "worst": worst,
                     "additive_mae_pts": stt.mean(abs(c["err_pts"]) for c in cells),
                     "pass": stt.median(rs) >= 0.8}
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        say("B1", json.dumps(out["B1"]))
        dev = {}
        for lam in LAMBDAS:
            set_C(add_coefs([CT[n] for n in held], lam))
            dev[lam] = {n: tk.eval_task(model, tok, n, tk.make_splits(n)["dev"])["acc"] for n in held}
            say("B2 dev", lam, dev[lam])
        lam = pick_lambda({l: sum(d.values()) for l, d in dev.items()})
        set_C(add_coefs([CT[n] for n in held], lam))
        b2 = {"dev": {str(l): d for l, d in dev.items()}, "lambda": lam, "test": {}}
        for n in held:
            acc = test(n)["acc"]
            b2["test"][n] = {"acc": acc, "r": retention(acc, acc0[n], accT[n])}
        b2["retention_nll"] = ea.retention_nll(model, tok)
        b2["mean_r"] = stt.mean(v["r"] for v in b2["test"].values())
        base_nll = ev[held[0]]["retention_nll"]["0"]
        b2["base_nll"] = base_nll
        b2["pass"] = b2["mean_r"] >= 0.7 and b2["retention_nll"] <= base_nll + 0.02
        out["B2"] = b2
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        say("B2", json.dumps({k: b2[k] for k in ("lambda", "mean_r", "retention_nll", "pass")}))

    if part == "c":
        p = OUT / "ties.json"
        b2 = json.loads((OUT / "library.json").read_text(encoding="utf-8"))["B2"]
        out = {"held": held, "acc0": acc0, "accT": accT, "accT_recheck": check, "dev": {}}
        merged = {}
        dev_sum = {}
        for d in (0.2, 0.5, 1.0):
            merged[d] = ties_merge([CT[n] for n in held], d)
            for lam in (1, 2, 3, 5):
                set_C({m: lam * x for m, x in merged[d].items()})
                dv = {n: tk.eval_task(model, tok, n, tk.make_splits(n)["dev"])["acc"] for n in held}
                dev_sum[(d, lam)] = sum(dv.values())
                out["dev"][f"d{d}_l{lam}"] = dv
                p.write_text(json.dumps(out, indent=1), encoding="utf-8")
                say("B3 dev", d, lam, round(dev_sum[(d, lam)], 3), dv)
        d, lam = pick_grid(dev_sum)
        set_C({m: lam * x for m, x in merged[d].items()})
        b3 = {"density": d, "lambda": lam, "test": {}}
        for n in held:
            acc = test(n)["acc"]
            b3["test"][n] = {"acc": acc, "r": retention(acc, acc0[n], accT[n]), "r_B2": b2["test"][n]["r"]}
        b3["retention_nll"] = ea.retention_nll(model, tok)
        b3["mean_r"] = stt.mean(v["r"] for v in b3["test"].values())
        b3["base_nll"] = b2["base_nll"]
        b3["pass"] = b3["mean_r"] >= 0.7 and b3["retention_nll"] <= b2["base_nll"] + 0.02
        out["B3"] = b3
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        say("B3", json.dumps({k: b3[k] for k in ("density", "lambda", "mean_r", "retention_nll", "pass")}))


if __name__ == "__main__":
    main()
