"""Shared harness: run one WSPLang step exactly as the CLI pipeline does (validate -> check_feasibility -> execute)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

import wsp  # noqa: E402
# never let the audit write centroid caches into data/
wsp.DIR_DATA = OUT

from engine.wsplang import (parse_args_lines, validate, check_feasibility, plan_run,  # noqa: E402
                            execute_plan_v4)

PY = r"C:\Users\Anto\.venvs\weights-cuda\Scripts\python.exe"


def run(op_line, model=None, model_name="nanoGPT", allow_write=False, check=True):
    code = f"""experiment audit:
model: {model_name}
budgets:
    allow_write = {str(allow_write).lower()}
steps:
    step {op_line}
"""
    prog = parse_args_lines(code)
    steps, errors = validate(prog)
    feas_err, feas_warn = check_feasibility(prog, steps, str(model) if model else None)
    out = {"validate_errors": errors, "feasibility_errors": feas_err, "feasibility_warnings": feas_warn}
    if check and (errors or feas_err):
        out["cli_would_run"] = False
    else:
        out["cli_would_run"] = not (errors or feas_err)
    res = execute_plan_v4(plan_run(prog, steps), prog, model_path=str(model) if model else None)
    out["step"] = res[0] if res else None
    return out


def dump(name, obj):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
