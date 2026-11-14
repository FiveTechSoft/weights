"""Build a WSPLang program from step specs, run it through the CLI, and count changed weights vs the original."""
import subprocess, sys, numpy as np
from pathlib import Path
ED = Path(r"C:\weights\lab\runs\loopfix\editor"); ROOT = Path(r"C:\weights")
PY = r"C:\Users\Anto\.venvs\weights-cuda\Scripts\python.exe"
BASE = ROOT / "models/nanogpt_shakespeare_char/model.safetensors"

def vec(name, arr):
    p = ED / "vecs" / f"{name}.npz"; np.savez(p, v=np.asarray(arr, np.float32)); return p.relative_to(ROOT).as_posix()

def head_rows(L, h, part, scale, tag):
    """Scale rows of c_attn for head h (part q=0,k=1,v=2) by `scale` via one-hot project_out (alpha = 1-scale)."""
    steps = []
    for r in range(384 * part + 64 * h, 384 * part + 64 * h + 64):
        u = np.zeros(1152); u[r] = 1
        steps.append(f"step project_out transformer.h.{L}.attn.c_attn.weight vector={vec(f'onehot1152_{r}', u)} alpha={1-scale:.6g} top_rows=1 top_cols=1.0")
    return steps

def program(name, steps, header=""):
    txt = f"experiment {name}:\n\n    objective:\n        loopfix_{name}\n\n    budgets:\n        max_modified_weights = 107000\n        allow_write = true\n\n    steps:\n"
    if header: txt = header + "\n" + txt
    txt += "".join(f"        {s}\n" for s in steps)
    return txt

def run(name, steps, path=None, header=""):
    path = Path(path) if path else ED / "cands" / f"{name}.wspl"
    path.write_text(program(name, steps, header), encoding="utf-8")
    out = path.parent / name if path.parent.name == "cands" else ED / "edited"
    r = subprocess.run([PY, "engine/wsplang.py", str(path.relative_to(ROOT)), "--run", "--model", str(BASE.relative_to(ROOT)),
                        "--out", str(out.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    (path.with_suffix(".log")).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("wspl failed")
    return out / "model.safetensors"

def count(model):
    from safetensors.numpy import load_file
    a, b = load_file(str(BASE)), load_file(str(model))
    tot = 0; per = {}
    for k in a:
        n = int((a[k] != b[k]).sum())
        if n: per[k] = n; tot += n
    return tot, per

def head_dirs(L, h, E, scales, tag):
    """Scale the V output of head h along orthonormal 64-d directions E[:,i] by scales[i] (rank-1 project_out each)."""
    steps = []
    for i, c in enumerate(scales):
        u = np.zeros(1152); u[768 + 64 * h: 768 + 64 * h + 64] = E[:, i]
        steps.append(f"step project_out transformer.h.{L}.attn.c_attn.weight vector={vec(f'{tag}_L{L}H{h}_d{i}', u)} alpha={1-c:.6g} top_rows=64 top_cols=1.0")
    return steps
