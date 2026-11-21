"""Does llama.cpp accept the control vectors WSPLang writes? (read-only on the model GGUF)

export_control_vector -> direction.0..31 ; layer_envelope -> direction.1..n
Run: C:\\Users\\Anto\\.venvs\\weights-cuda\\Scripts\\python.exe lab/audit/edit_ops/audit_controlvector_llamacpp.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine.wsplang import parse_args_lines, validate, plan_run, execute_plan_v4  # noqa: E402

MODEL = ROOT / "hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf"  # n_embd 1024, read only
EXE = Path(r"C:\Users\Anto\llamacpp\bin\llama-completion.exe")
TMP = Path(tempfile.mkdtemp(prefix="wsp_cv_audit_"))


def run(step):
    code = f"experiment cv:\nbudgets:\n    allow_write = true\nsteps:\n    step {step}\n"
    prog = parse_args_lines(code)
    st, _ = validate(prog)
    return execute_plan_v4(plan_run(prog, st), prog, model_path=None)[0]


vp = TMP / "u1024.npz"
np.savez(vp, u=np.random.RandomState(0).randn(1024).astype(np.float32))
cv1, cv2 = TMP / "export.gguf", TMP / "envelope.gguf"
out = {"export_step": run(f"export_control_vector {vp.as_posix()} out={cv1.as_posix()}")["status"],
       "envelope_step": run(f"layer_envelope layers=4..12 vector={vp.as_posix()} out={cv2.as_posix()} n_layers=24")["status"]}
for name, cv in (("export_control_vector", cv1), ("layer_envelope", cv2), ("no_control_vector", None)):
    cmd = [str(EXE), "-m", str(MODEL), "-p", "Hello", "-n", "4", "-ngl", "0", "--no-warmup", "-no-cnv"]
    if cv:
        cmd += ["--control-vector", str(cv)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240, errors="replace")
        log = (p.stdout + p.stderr).splitlines()
        hits = [ln for ln in log if "control" in ln.lower() or "direction" in ln.lower() or "error" in ln.lower()]
        out[name] = {"returncode": p.returncode, "relevant_log": hits[:8], "tail": log[-3:]}
    except Exception as e:  # noqa: BLE001
        out[name] = {"exception": repr(e)}
(Path(__file__).parent / "results_controlvector.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
