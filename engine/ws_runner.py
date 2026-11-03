"""ws_runner.py — adaptador weightspace (harness etapa A, modelos npz/safetensors F32).

Envuelve el pip weightspace (leofmarciano/weightspace, Apache-2.0) para el flujo del estudio:
/ventaja vs GGUF: parches sparse en memoria (checkpoint/restore/patch-apply sin requant).
Público: jueces/harness de retos etapa A (Qwen3.5-0.8B HF) y replay de transcripciones sin tokens.

CLI JSON-print:
  info <file> [--pattern] [--describe]
  edit <file> <program.py> -o out.npz      (ns: W, R, np, diff, checkpoint, restore, revert)
  apply-patch <file> <patch.npz> -o out.npz
  run <file> <task> (--llm spec | --replay transcript.jsonl)
      [--mode read|fix|write|improve] [--context @file|str] [--forward-budget N]
      [--max-steps N] [--text-mode] [--no-python] [--thinking off|low|medium|high]
      [--out dir] [--save-weights out.npz]

Requiere .venv: python -m venv .venv && .venv/bin/pip install weightspace safetensors
(sin torch: los GGUF siguen en wsp.py/llama.cpp — esto es para HF/f32 de etapa A).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(__file__).resolve().parents[1]
DIR_JRN = ROOT / "journal"
DIR_RIG = ROOT / "rig_runs"
DIR_JRN.mkdir(exist_ok=True)
DIR_RIG.mkdir(exist_ok=True)
JRN = DIR_JRN / "ws_runner.jsonl"


def _p(*a):
    print(*a, flush=True)


def journal(kind, detail):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "detail": detail}
    with JRN.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ws(path):
    from weightspace import WeightSpace

    return WeightSpace.from_file(str(path))


def cmd_info(a):
    ws = _ws(a.file)
    out = []
    for n in ws.names(a.pattern):
        d = {"name": n, "shape": list(ws.shape(n)), "dtype": ws.dtype(n), "numel": ws.numel(n)}
        if a.describe:
            from weightspace import describe

            d["describe"] = describe(n, ws[n], ws.dtype(n))
        out.append(d)
    _p(json.dumps({"tensors": out, "n": len(out), "numel": ws.numel()}, ensure_ascii=False, indent=2))
    journal("info", {"file": a.file, "n": len(out)})


def cmd_edit(a):
    ws = _ws(a.file)
    ns = {"np": np, "R": ws.R, "W": ws.W, "diff": ws.diff, "checkpoint": ws.checkpoint,
          "restore": ws.restore, "revert": ws.revert}
    code = Path(a.program).read_text(encoding="utf-8")
    exec(compile(code, a.program, "exec"), ns)
    changes = [{"name": t.name, "changed": t.changed, "l2": t.l2_delta, "rel": t.rel_delta} for t in ws.diff().tensors]
    if a.out:
        ws.save(str(a.out))
        journal("write", {"file": a.file, "out": str(a.out), "changed": sum(c["changed"] for c in changes)})
    _p(json.dumps({"edit": changes, "total_params": ws.numel(), "saved": str(a.out) if a.out else None}, ensure_ascii=False, indent=2))


def cmd_apply_patch(a):
    ws = _ws(a.file)
    ws.apply_patch(str(a.patch))
    changes = [{"name": t.name, "changed": t.changed, "l2": t.l2_delta} for t in ws.diff().tensors]
    if a.out:
        ws.save(str(a.out))
        journal("write", {"file": a.file, "out": str(a.out), "patch": str(a.patch), "changed": sum(c["changed"] for c in changes)})
    _p(json.dumps({"applied": str(a.patch), "tensors": changes, "saved": str(a.out) if a.out else None}, ensure_ascii=False, indent=2))


def _steps_from_transcript(path):
    from weightspace.llm import LLMResponse
    from weightspace.llm.base import ToolCall
    from weightspace.llm.mock import ScriptedLLM

    steps = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            m = json.loads(line)
        except Exception:
            continue
        if m.get("role") != "assistant":
            continue
        calls = [ToolCall(c["id"], c["name"], c.get("arguments") or {}) for c in m.get("tool_calls") or []]
        steps.append(LLMResponse(text=m.get("content"), tool_calls=calls))
    if not steps:
        raise SystemExit("no assistant steps in transcript")
    return ScriptedLLM(steps)


def cmd_run(a):
    from weightspace import Session

    ws = _ws(a.file)
    if a.mode == "write":
        ws.zero_()
        ws.commit()
    llm = _steps_from_transcript(a.replay) if a.replay else a.llm
    if not llm:
        raise SystemExit("necesita --llm SPEC o --replay TRANSCRIPT.jsonl")
    context = a.context
    if context and context.startswith("@"):
        context = Path(context[1:]).read_text(encoding="utf-8")
    s = Session(
        ws,
        llm,
        mode=a.mode,
        context=context or "",
        forward_budget=a.forward_budget,
        python=not a.no_python,
        max_steps=a.max_steps,
        thinking=a.thinking,
    )
    res = s.run_text(a.task) if a.text_mode else s.run(a.task)
    _p(str(res))
    out = {}
    if a.out:
        out["dir"] = str(a.out)
        res.save(str(a.out))
    if a.save_weights:
        ws.save(str(a.save_weights))
        out["weights"] = str(a.save_weights)
    journal("run", {"mode": a.mode, "llm": a.llm or f"replay:{Path(a.replay).name}", "changed": res.diff.changed_params,
                    "steps": res.steps, "out": out})
    _p(json.dumps({"result": {"steps": res.steps, "fwd_used": res.forward_used,
                              "forward_budget": res.forward_budget, "changed": res.diff.changed_params,
                              "of": res.diff.total_params, "out": out}}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(prog="ws_runner")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("info"); a.add_argument("file"); a.add_argument("--pattern", default=None); a.add_argument("--describe", action="store_true")
    a = sp.add_parser("edit"); a.add_argument("file"); a.add_argument("program"); a.add_argument("-o", "--out", default=None)
    a = sp.add_parser("apply-patch"); a.add_argument("file"); a.add_argument("patch"); a.add_argument("-o", "--out", default=None)
    a = sp.add_parser("run"); a.add_argument("file"); a.add_argument("task")
    a.add_argument("--llm", default=None); a.add_argument("--replay", default=None)
    a.add_argument("--mode", default="fix", choices=["read", "fix", "write", "improve"])
    a.add_argument("--context", default="")
    a.add_argument("--forward-budget", type=int, default=0)
    a.add_argument("--max-steps", type=int, default=60)
    a.add_argument("--text-mode", action="store_true"); a.add_argument("--no-python", action="store_true")
    a.add_argument("--thinking", choices=["off", "low", "medium", "high"], default=None)
    a.add_argument("--out", default=None); a.add_argument("--save-weights", default=None)
    args = ap.parse_args()
    try:
        {"info": cmd_info, "edit": cmd_edit, "apply-patch": cmd_apply_patch, "run": cmd_run}[args.cmd](args)
    except ImportError as e:
        _p(json.dumps({"error": f"instalar en .venv: venv/bin/pip install weightspace safetensors ({e})"}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
