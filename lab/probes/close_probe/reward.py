"""One score per edit, IWIBench's (HCLTech, 2026: hcltech-robotics.github.io/iwibench), on fast_eval.py results:

  gain      = max(0, (edited target success - base target success) / (1 - base target success))
  retention = of the controls the base got right in the same eval, the share the edited model still gets right
  reward    = gain * retention          a no-op scores 0; breaking controls costs in proportion

Targets are group "A" (stuck items), controls group "ok" (guard). Both results must be on the same items, same bench.
Controls the edit newly gets right do not raise retention (reported apart).

usage: python lab/probes/close_probe/reward.py <base.json> <edited.json> [...]
       python lab/probes/close_probe/reward.py --all     every fast/*.json with guard, against its bench's base
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAST = ROOT / "lab/runs/close_probe/fast"
BASES = {"math500": "base_guard.json", "gsm8k": "gsm_base.json"}


def _by(run):
    return {(r["grp"], str(r["i"])): bool(r["ok"]) for r in run["items"]}


def score(base, edited):
    b, e = _by(base), _by(edited)
    if not set(e) <= set(b):
        raise ValueError(f"edited run has items the base run lacks: {sorted(set(e) - set(b))[:5]}")
    tk = [k for k in e if k[0] == "A"]
    if not tk:
        raise ValueError("no target (A) items")
    bt, et = sum(b[k] for k in tk) / len(tk), sum(e[k] for k in tk) / len(tk)
    gain = 0.0 if bt >= 1 else max(0.0, (et - bt) / (1 - bt))
    ck = [k for k in e if k[0] == "ok"]
    kept = [k for k in ck if b[k]]
    out = {"gain": gain, "target": f"{sum(e[k] for k in tk)}/{len(tk)} (base {sum(b[k] for k in tk)}/{len(tk)})",
           "retention": None, "reward": None, "controls": None}
    if ck and kept:
        ret = sum(e[k] for k in kept) / len(kept)
        new = sum(e[k] and not b[k] for k in ck)
        out.update(retention=ret, reward=gain * ret,
                   controls=f"{sum(e[k] for k in kept)}/{len(kept)} kept (+{new} newly right)")
    return out


def main():
    a = sys.argv[1:]
    if a == ["--all"]:
        pairs = []
        for f in sorted(FAST.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            s = d.get("summary", {})
            if "ok" not in s or f.name in BASES.values() or any("grp" not in r for r in d.get("items", [])):
                continue
            pairs.append((FAST / BASES[s.get("bench", "math500")], f))
    else:
        pairs = [(Path(a[0]), Path(p)) for p in a[1:]]
    print(f"{'edited':<34}{'reward':>8}{'gain':>7}{'retain':>8}  target / controls")
    for bp, ep in pairs:
        r = score(json.loads(bp.read_text(encoding="utf-8")), json.loads(ep.read_text(encoding="utf-8")))
        f = lambda x: "  -" if x is None else f"{x:.3f}"
        print(f"{ep.stem:<34}{f(r['reward']):>8}{f(r['gain']):>7}{f(r['retention']):>8}  {r['target']} / {r['controls']}")


if __name__ == "__main__":
    main()
