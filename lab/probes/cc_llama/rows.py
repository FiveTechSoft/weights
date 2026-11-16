"""Calibration rows for the third family, DeepSeek-R1-Distill-Llama-8B (lab/probes/cc_llama/PLAN.md).

Its chat template ends the generation prompt with "<think>\\n" (checked on the HF tokenizer_config, 2026-11-17), as
Qwen's does and Phi's does not. So: pos / neg rows for contrast_vector come from `cc.py rows` unchanged (CC_HF / CC_RUNS),
and the calibration rows for compile_control_vector carry NO prefix: the op appends `reasoning` to the template, which
already wrote the tag. A stray leading tag in a trace (server fallback) is stripped, never doubled.

  python lab/probes/cc_llama/rows.py calib      (CC_RUNS, default lab/runs/cc_llama) -> runs/calib.jsonl
  python lab/probes/cc_llama/rows.py template   (CC_HF, default hf_cache/r1llama8b-hf) -> does the template write <think>?
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TAG = "<think>"


def finished(r):
    return r.get("finish") == "stop" and (r.get("reasoning") or "").strip() and (r.get("content") or "").strip()


def make_calib(traces):
    """-> [{"prompt", "reasoning"}] for finished traces; reasoning without a leading <think> tag."""
    out = []
    for r in traces:
        if not finished(r):
            continue
        text = r["reasoning"].lstrip("\n")
        if text.startswith(TAG):
            text = text[len(TAG):].lstrip("\n")
        out.append({"prompt": r["prompt"], "reasoning": text})
    return out


def template_writes_think(generation_prompt):
    """True when the rendered generation prompt ends with the think tag (plus whitespace): the model's reasoning then
    starts after it and no row may repeat the tag."""
    return generation_prompt.rstrip().endswith(TAG)


def main():
    runs = Path(os.environ.get("CC_RUNS") or ROOT / "lab" / "runs" / "cc_llama")
    hf = Path(os.environ.get("CC_HF") or ROOT / "hf_cache" / "r1llama8b-hf")
    if sys.argv[1:] == ["calib"]:
        tr = [json.loads(x) for x in (runs / "traces.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = make_calib(tr)
        (runs / "calib.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        print(json.dumps({"traces": len(tr), "calib": len(rows)}))
        return
    if sys.argv[1:] == ["template"]:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(hf))
        p = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
        print(json.dumps({"tail": p[-40:], "template_writes_think": template_writes_think(p)}))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
