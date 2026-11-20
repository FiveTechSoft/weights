"""engine/wsplang_require.py - `step require` (WSP-Lang 0.8.4): a guard on a JSON artefact.

    step require <file>.json:<path> <cmp> <value> [agg=all|any|min|max|mean] [why="..."]
    step require file=<file>.json key=<path> cmp=<cmp> value=<value> [agg=...] [why="..."]

<path> is dotted; a segment is a dict key, a list index, or `*` (every value of a dict or element of a list).
<cmp> is one of >= <= > < == !=; <value> is a number, true / false / null, or a (quoted) string (strings only with
== and !=). With `*` the leaves are judged by `agg`: all (default) / any each leaf, or min / max / mean of the leaves.

The runtime (engine/wsplang.py) evaluates a require at check time when its file exists and no earlier step writes it,
so a doomed program is refused before it spends compute; otherwise when the step is reached. A failed require stops
the run: status guard_failed, every later step skipped_by_guard, exit code 1.
"""
import json
import operator
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CMPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq,
        "!=": operator.ne}
AGGS = ("all", "any", "min", "max", "mean")
_MISSING = object()


def _value(tok: str) -> Any:
    t = tok.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        return t


def parse_require(args: str) -> Dict[str, Any]:
    """The guard a require step's arguments describe. Raises ValueError on a malformed one."""
    from engine.wsplang import parse_named_args  # same tokeniser as every other step
    positional, named = parse_named_args(args)
    if "file" in named or "key" in named:
        missing = [k for k in ("file", "key", "cmp", "value") if k not in named]
        if missing:
            raise ValueError(f"require: named form needs {missing}")
        f, path, cmp, raw = named["file"], named["key"], named["cmp"], named["value"]
        # value= went through the tokeniser unquoted; keep "x" strings as strings
        val = _value(raw if not re.search(r'value="', args) else f'"{raw}"')
    else:
        if len(positional) < 3:
            raise ValueError("require: expected <file>.json:<path> <cmp> <value>")
        m = re.match(r"^(.*?\.json):(.*)$", positional[0], re.I)
        if not m:
            raise ValueError(f"require: first argument must be <file>.json:<path>, got {positional[0]!r}")
        f, path = m.group(1), m.group(2)
        cmp = positional[1]
        rest = args.split(cmp, 1)[1].strip() if cmp in args else positional[2]
        raw_tok = re.match(r'^("[^"]*"|\'[^\']*\'|\S+)', rest)
        val = _value(raw_tok.group(1) if raw_tok else positional[2])
    if not str(f).lower().endswith(".json"):
        raise ValueError(f"require: the file must be a .json artefact, got {f!r}")
    if not path or not path.strip():
        raise ValueError("require: empty path")
    if cmp not in CMPS:
        raise ValueError(f"require: comparison must be one of {list(CMPS)}, got {cmp!r}")
    agg = named.get("agg", "all")
    if agg not in AGGS:
        raise ValueError(f"require: agg must be one of {AGGS}, got {agg!r}")
    if isinstance(val, str) and cmp not in ("==", "!="):
        raise ValueError(f"require: a string value only compares with == or !=, got {cmp}")
    return {"file": f, "path": path.strip(), "cmp": cmp, "value": val, "agg": agg, "why": named.get("why")}


def resolve(obj: Any, path: str) -> List[Any]:
    """Leaves at `path` (dotted, list indices, `*`). A missing key gives a _MISSING leaf."""
    nodes = [obj]
    for seg in path.split("."):
        nxt = []
        for n in nodes:
            if n is _MISSING:
                nxt.append(_MISSING)
            elif seg == "*":
                if isinstance(n, dict):
                    nxt.extend(n.values())
                elif isinstance(n, list):
                    nxt.extend(n)
                else:
                    nxt.append(_MISSING)
            elif isinstance(n, dict):
                nxt.append(n.get(seg, _MISSING))
            elif isinstance(n, list) and re.fullmatch(r"-?\d+", seg) and -len(n) <= int(seg) < len(n):
                nxt.append(n[int(seg)])
            else:
                nxt.append(_MISSING)
        nodes = nxt
    return nodes


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def evaluate(guard: Dict[str, Any], data: Any) -> Dict[str, Any]:
    """Judge `guard` on the loaded JSON `data`."""
    out = {k: guard[k] for k in ("file", "path", "cmp", "value", "agg", "why")}
    leaves = resolve(data, guard["path"])
    if not leaves or any(v is _MISSING for v in leaves):
        out.update(ok=False, observed=None, n_values=0, reason=f"missing: {guard['path']} not in {guard['file']}")
        return out
    fn, want, agg = CMPS[guard["cmp"]], guard["value"], guard["agg"]
    numeric = _num(want) or guard["cmp"] not in ("==", "!=")
    if numeric and not all(_num(v) for v in leaves):
        out.update(ok=False, observed=leaves[0] if len(leaves) == 1 else leaves, n_values=len(leaves),
                   reason="non-numeric value under a numeric comparison")
        return out
    if agg in ("min", "max", "mean"):
        observed = {"min": min, "max": max, "mean": statistics.fmean}[agg](leaves)
        ok = bool(fn(observed, want))
    else:
        per = [bool(fn(v, want)) for v in leaves]
        ok = all(per) if agg == "all" else any(per)
        observed = leaves[0] if len(leaves) == 1 else leaves
    reason = ""
    if not ok:
        if agg in ("all", "any") and len(leaves) > 1:
            bad = [v for v in leaves if not fn(v, want)]
            span = f"min {min(leaves)!r}, max {max(leaves)!r}" if all(_num(v) for v in leaves) else ""
            reason = (f"{guard['path']}: {len(bad)} of {len(leaves)} values fail {guard['cmp']} {want!r} "
                      f"(agg {agg}{'; ' + span if span else ''})")
        else:
            reason = f"{guard['path']} = {observed!r} fails {guard['cmp']} {want!r} (agg {agg})"
    out.update(ok=ok, observed=observed, n_values=len(leaves), reason=reason)
    return out


def _abs(p: str, root: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def check_file(guard: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Load the guard's file and judge it; a missing or unreadable file is a failed guard."""
    p = _abs(guard["file"], root)
    if not p.exists():
        out = {k: guard[k] for k in ("file", "path", "cmp", "value", "agg", "why")}
        out.update(ok=False, observed=None, n_values=0, reason=f"missing file: {guard['file']}")
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        out = {k: guard[k] for k in ("file", "path", "cmp", "value", "agg", "why")}
        out.update(ok=False, observed=None, n_values=0, reason=f"unreadable JSON {guard['file']}: {e}")
        return out
    return evaluate(guard, data)


OUT_KEYS = ("out", "out_dir", "output", "report")


def written_by(steps_before: List[Dict[str, Any]], root: Path) -> List[Path]:
    """Paths (files or dirs) the earlier steps name as outputs."""
    from engine.wsplang import parse_named_args
    paths = []
    for st in steps_before:
        _, named = parse_named_args(st.get("args", ""))
        for k in OUT_KEYS:
            if named.get(k):
                paths.append(_abs(str(named[k]), root).resolve())
    return paths


def static_check(steps: List[Dict[str, Any]], root: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Check-time pass over every require: (errors, judged). A require on a file an earlier step writes is left
    for the run; one on an existing file is judged now; one on a missing file nobody writes is an error."""
    errors, judged = [], []
    for i, st in enumerate(steps):
        if st.get("op") != "require":
            continue
        try:
            g = parse_require(st.get("args", ""))
        except ValueError as e:
            errors.append(f"require malformado: {e}")
            continue
        target = _abs(g["file"], root).resolve()
        produced = written_by(steps[:i], root)
        if any(target == p or p in target.parents for p in produced):
            continue  # deferred: an earlier step writes it
        if not target.exists():
            errors.append(f"require: artefacto inexistente {g['file']} y ningun paso anterior lo escribe")
            continue
        r = check_file(g, root)
        judged.append(r)
        if not r["ok"]:
            why = f" ({g['why']})" if g.get("why") else ""
            errors.append(f"require falla antes de ejecutar{why}: {r['reason']}")
    return errors, judged
