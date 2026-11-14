"""Sealed confirmation 2 verdict (cc_confirm.md): exact McNemar, vector and placebo vs base, two graders per benchmark."""
import json, math, os, re
D = os.environ.get("SEALED_CC_RUNS", "lab/runs/sealed_cc")
def load(n): return {(r["bench"], r["i"]): r for r in json.load(open(f"{D}/{n}.json"))["items"]}
def lastnum(r):
    m = re.findall(r"-?\d[\d,]*\.?\d*", r.get("tail") or "")
    if not m: return False
    try: return abs(float(m[-1].replace(",", "").rstrip(".")) - float(r["gold"].replace(",", ""))) < 1e-6
    except ValueError: return False
def mcn(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)
A = {n: load(n) for n in ("base", "vec", "placebo")}
G = {"gsm8k": {"g1": lambda r: r["ok"], "lastnum": lastnum}, "math500": {"g1": lambda r: r["ok"], "g2": lambda r: r["ok_independent"]}}
R = {}
for bench, graders in G.items():
    keys = sorted(k for k in A["base"] if k[0] == bench)
    for arm in ("vec", "placebo"):
        for g, f in graders.items():
            a = [f(A[arm][k]) for k in keys]; b = [f(A["base"][k]) for k in keys]
            win = sum(x and not y for x, y in zip(a, b)); loss = sum(y and not x for x, y in zip(a, b))
            R[(bench, arm, g)] = c = {"arm": sum(a), "base": sum(b), "win": win, "loss": loss, "p": mcn(win, loss)}
            print(bench, arm, g, c)
sig = lambda c: c["arm"] > c["base"] and c["p"] < 0.05
C1 = all(sig(R[("gsm8k", "vec", g)]) for g in G["gsm8k"]) or all(sig(R[("math500", "vec", g)]) for g in G["math500"])
C2 = not any(sig(R[(b, "placebo", g)]) for b in G for g in G[b])
C3 = R[("gsm8k", "vec", "g1")]["arm"] >= R[("gsm8k", "vec", "g1")]["base"] - 2
for n, X in A.items():
    for bench in G:
        it = [r for k, r in X.items() if k[0] == bench]
        print(n, bench, "ok", sum(r["ok"] for r in it), "cut", sum(r.get("finish") == "length" for r in it),
              "ok_and_cut", sum(r["ok"] and r.get("finish") == "length" for r in it),
              "graders_disagree", sum(r["ok"] != r["ok_independent"] for r in it))
v = {"C1": C1, "C2": C2, "C3": C3, "verdict": "CONFIRMS" if C1 and C2 and C3 else "DOES NOT CONFIRM"}
print(v); open(f"{D}/verdict.txt", "w").write(json.dumps(v) + "\n")
