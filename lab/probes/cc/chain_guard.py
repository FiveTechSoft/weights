"""Configuration guard for resumable chains. A chain that skips a step when its output file exists must know those
outputs came from the same model, budget, scale and code (outside review of chain.sh, 2026-11-15). The first call
writes <dir>/config.json; every later call compares and refuses on any difference, naming the keys.

  python lab/probes/cc/chain_guard.py <dir> key=value [key=value ...]   -> exit 0 same/written, 1 mismatch
"""
import json
import sys
from pathlib import Path


class ConfigMismatch(Exception):
    pass


def check(run_dir, params):
    """-> "written" (first call) | "same"; raises ConfigMismatch listing every key that differs or is missing."""
    p = Path(run_dir) / "config.json"
    params = {str(k): str(v) for k, v in params.items()}
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(params, indent=1, sort_keys=True), encoding="utf-8")
        return "written"
    old = json.loads(p.read_text(encoding="utf-8"))
    diffs = [f"{k}: recorded {old.get(k)!r}, now {params.get(k)!r}" for k in sorted(set(old) | set(params))
             if old.get(k) != params.get(k)]
    if diffs:
        raise ConfigMismatch(f"{p} does not match this run: " + "; ".join(diffs))
    return "same"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    params = dict(a.split("=", 1) for a in argv[1:])
    try:
        print(f"[chain_guard] {check(argv[0], params)}")
        return 0
    except ConfigMismatch as e:
        print(f"[chain_guard] REFUSED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
