"""wspx — datasets EXACTOS (universo finito + oraculo cerrado) para el curriculum WSPLang.
Reglas: todo deterministic (seed fija), universo enumerable, respuesta <= 2 chars, juez exact-match.
Cada dataset: build() -> {"name", "version", "sha256", "train", "holdout", "universe", "oracle"}
  prompt/tgt son strings ascii; oracle(prompt) -> tgt canónico.
"""
import hashlib, json
from pathlib import Path

R = Path(r"C:\weights")
DATA = R / "lab" / "data"

def _h(s: str) -> int:
    return hashlib.sha256(s.encode("ascii", "ignore")).digest()

def _tally(n_bytes=8):
    return int.from_bytes(_h("wsp-seed"), "big")

def _rng(seed: str):
    import numpy as np
    return np.random.default_rng(int.from_bytes(_h(seed)[:8], "big") % (2 ** 63))

def fs4():
    """FSM-4: 4 estados Moore, alfabeto {a,b}. prompt 'FSM: <w> =' -> estado final '0'..'3'."""
    trans = {("0", "a"): "1", ("0", "b"): "2", ("1", "a"): "0", ("1", "b"): "3",
             ("2", "a"): "3", ("2", "b"): "0", ("3", "a"): "2", ("3", "b"): "1"}
    def oracle(s):
        st = "0"
        for ch in s:
            st = trans[(st, ch)]
        return st
    uni = []
    def rec(prefix, depth):
        if depth == 0:
            return
        for ch in "ab":
            w = prefix + ch
            uni.append(w)
            rec(w, depth - 1)
    rec("", 5)
    universe = [f"FSM: {w} =" for w in uni]
    rows = [{"prompt": p, "target": oracle(p.split(":")[1].split("=")[0].strip())} for p in universe]
    return _pack("FSM4", rows)

def recall():
    """RECALL: 200 pares key 2-chars -> val 1-char. prompt 'KB: XY =' -> val."""
    lit = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    keys, seen = [], set()
    i = 0
    while len(keys) < 200:
        k = lit[_h(f"k{i}")[0] % 26] + lit[_h(f"k{i}")[1] % 26]
        i += 1
        if k in seen:
            continue
        seen.add(k)
        small = "abcdefghij"
        keys.append((k, small[_h(f"v{i}")[0] % 10]))
    rows = [{"prompt": f"KB: {k} =", "target": v} for k, v in keys]
    return _pack("RECALL", rows)

def mod13():
    """MOD13: a,b in 0..12, ops {+,*}: 'M13: a op b =' -> (a op b) % 13."""
    rows = []
    for a in range(13):
        for b in range(13):
            for op, fn in (("+", lambda x, y: (x + y) % 13), ("*", lambda x, y: (x * y) % 13)):
                p = f"M13: {a}{op}{b} ="
                rows.append({"prompt": p, "target": str(fn(a, b))})
    return _pack("MOD13", rows)

def parity8():
    """PARITY8: 'P8: 10101010 =' -> XOR de los booleanos."""
    rows = []
    for v in range(256):
        s = format(v, "08b")
        rows.append({"prompt": f"P8: {s} =", "target": str(s.count("1") % 2)})
    return _pack("PARITY8", rows)

def noecho():
    """NOECHO: 'NE: <palabra> =' -> 'x' SIEMPRE (regla de supresion); la palabra aleatoria del universe.
    El trace de aprendizaje espera que base-echo (mirar train_nanowsp) ein Elian echo; la cirugia
    precedent subtract-direction debe dejar 'x' constante."""
    import numpy as np
    rng = _rng("noecho")
    words = []
    for i in range(500):
        w = "".join("abcdefghijklmnopqrstuvwxyz0123456789"[_h(f"w{i}")[j] % 36] for j in range(_h(f"i{i}")[0] % 4 + 3))
        if w not in words:
            words.append(w)
        else:
            words.append(w + "9")
    rows = [{"prompt": f"NE: {w} =", "target": "x"} for w in words]
    return _pack("NOECHO", rows)

def _pack(name, rows, seed="wsp-seed"):
    """Particion train/holdout determinista: 60/40 por hash del prompt."""
    train, hold = [], []
    for r in rows:
        (train if _h(r["prompt"] + seed)[0] % 100 < 60 else hold).append(r)
    payload = {"name": name, "version": "1", "n_universe": len(rows),
               "n_train": len(train), "n_holdout": len(hold),
               "train": train, "holdout": hold, "universe": rows,
               "oracle_note": "closed-form",
               }
    blob = json.dumps({k: v for k, v in payload.items() if k in ("name", "version", "universe")},
                      sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"wspx_{name.lower()}.json"
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload

BUILDERS = {"fsm4": fs4, "recall": recall, "mod13": mod13, "parity8": parity8, "noecho": noecho}

def build_all():
    outs = {k: fn() for k, fn in BUILDERS.items()}
    print(json.dumps({k: {"n_universe": v["n_universe"], "n_train": v["n_train"],
                           "n_holdout": v["n_holdout"], "sha256": v["sha256"][:16] + "..."}
                      for k, v in outs.items()}, indent=2))
    return outs

if __name__ == "__main__":
    build_all()
