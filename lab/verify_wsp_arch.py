"""End-to-end check of programs/arch/nanogpt.wsparch on real text; exits 1 on any difference.

Trains the compiled model with the protocol of the official control in lab/calibrate_llmness.py and
runs the LLM-ness gate. Same seed and bit-exact parity must reproduce that control's report exactly.
Requires a prior calibration run, whose configuration is reused from lab/runs/llmness_calibration.json.
"""
import argparse
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab")]

import llmness as lm  # noqa: E402
from calibrate_llmness import CFG, train  # noqa: E402
from engine import wsp_arch  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "shakespeare_char" / "input.txt"))
    ap.add_argument("--calibration", default=str(R / "lab" / "runs" / "llmness_calibration.json"))
    ap.add_argument("--eval-chars", type=int, default=8000)
    ap.add_argument("--out", default=str(R / "lab" / "runs" / "wsp_arch_verification.json"))
    a = ap.parse_args()

    calibration = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
    run = calibration["config"]
    text = Path(a.corpus).read_text(encoding="utf-8")
    chars = sorted(set(text))
    overrides = dict(vocab=len(chars), block=CFG["block_size"], layers=CFG["n_layer"], heads=CFG["n_head"],
                     d=CFG["n_embd"], dropout=CFG["dropout"], bias=CFG["bias"])

    model = wsp_arch.load_arch_file(ARCH, overrides=overrides, seed=run["seed"])
    train(model, text[:int(len(text) * 0.9)], chars, run["steps"])
    report = lm.evaluate(lm.NanoGPTCharModel(model, chars), *lm.split_corpus(text, 0.1, a.eval_chars))

    control = calibration["results"]["trained_nanogpt"]
    differing = [k for k in control if control[k] != report[k]]
    result = {"arch_sha256": model.source_sha256, "steps": run["steps"], "seed": run["seed"],
              "matches_official_control": not differing, "differing_sections": differing, "report": report}
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict={report['verdict']} nll={report['L1']['nll']} control_nll={control['L1']['nll']}")
    print("MATCHES official control" if not differing else f"DIFFERS from official control: {differing}")
    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
