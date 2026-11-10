"""Sealed confirmation 1 verdict (jackrong_confirm.md): exact McNemar, adapter and placebo vs base, two graders per benchmark."""
import json, math, re
def load(n): return {(r["bench"], r["i"]): r for r in json.load(open(f"lab/runs/sealed/{n}.json"))["items"]}
def lastnum(r):
    m = re.findall(r"-?\d[\d,]*\.?\d*", r.get("tail") or "")
    if not m: return False
    try: return abs(float(m[-1].replace(",", "").rstrip(".")) - float(r["gold"].replace(",", ""))) < 1e-6
    except: return False
def mcn(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)
A = {n: load(n) for n in ("base", "jack_a050", "placebo_a050")}
for bench, graders in (("math500", {"g1": lambda r: r["ok"], "g2": lambda r: r["ok_independent"]}),
                       ("gsm8k", {"g1": lambda r: r["ok"], "lastnum": lastnum})):
    keys = sorted(k for k in A["base"] if k[0] == bench)
    for arm in ("jack_a050", "placebo_a050"):
        for g, f in graders.items():
            a = [f(A[arm][k]) for k in keys]; b = [f(A["base"][k]) for k in keys]
            win = sum(x and not y for x, y in zip(a, b)); loss = sum(y and not x for x, y in zip(a, b))
            print(bench, arm, g, f"arm={sum(a)} base={sum(b)} win={win} loss={loss} p={mcn(win, loss):.4g}")
for n in A:
    s = json.load(open(f"lab/runs/sealed/{n}.json"))["summary"]
    print(n, {b: (s[b]["acc"], s[b]["cut_at_max"], s[b]["mean_tokens"]) for b in s})
