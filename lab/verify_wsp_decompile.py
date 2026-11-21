"""Round trip: trained nanoGPT -> WSPLang n-gram program -> recompiled nanoGPT, and how much of the model survives.

For each gradient-trained teacher:
  1. structural decompile must refuse it (its weights do not follow the table layout);
  2. behavioral decompile at several orders writes a .wspngram program;
  3. that program is compiled back without gradients, at the exact 6L/384/1536 shape and with a wide MLP that holds
     every unit, and scored on full val plus the gate.
The Witten-Bell table from the data, compiled the same way, is the control: it says how much of the result comes
from the teacher rather than from the corpus. retained = (H_unigram - nll) / (H_unigram - nll_teacher).
Writes lab/runs/wsp_decompile_round_trip.json and the programs under lab/runs/decompiled/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab")]

import llmness as lm  # noqa: E402
from engine import wsp_arch, wsp_decompile, wsp_synth, wsp_train  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
TEACHERS = {
    "nanogpt_10.7M": (R / "models/nanogpt_shakespeare_char/model.safetensors",
                      dict(block=256, layers=6, heads=6, d=384, ff=1536, dropout=0.0, bias=False)),
    "adam_small_0.1M": (R / "lab/runs/als_adam_small.pt",
                        dict(block=64, layers=2, heads=4, d=64, ff=256, dropout=0.0, bias=False)),
}
EXACT = dict(block=256, layers=6, heads=6, d=384, ff=1536, dropout=0.0, bias=False)
WIDE = dict(block=256, layers=1, heads=6, d=384, dropout=0.0, bias=False)
L2_SEEDS = range(3)


class DeviceModel(lm.NanoGPTCharModel):
    def logits(self, x):
        device = next(self.module.parameters()).device
        return self.module(x.to(device), torch.zeros_like(x).to(device))[0].float().cpu()


def measure(model, data, text, device):
    model = model.eval().to(device)
    adapter = DeviceModel(model, data.chars)
    val = lm.score_nll(adapter, text[int(len(text) * 0.9):], seq=model.config.block_size, batch=16)
    baseline_train, test_text = lm.split_corpus(text, 0.1, 8000)
    gate = lm.evaluate(adapter, baseline_train, test_text, ladder=())
    seeds = [lm.check_long_context(adapter, test_text, seed=s) for s in L2_SEEDS]
    return {"val_nll_full": round(val, 4), "verdict": gate["verdict"],
            "l2_passed": sum(r["verdict"] == "PASSES" for r in seeds), "l2_total": len(seeds),
            "l2_gains": [r["mean_gain_nats"] for r in seeds],
            "parameters": sum(p.numel() for p in model.parameters())}


def compile_and_measure(table, shape, data, text, device):
    model = wsp_synth.compile_ngram(ARCH, table, overrides=shape, seed=1337)
    assert all(p.grad is None for p in model.parameters())
    out = measure(model, data, text, device)
    out["units_used"] = model.synthesis_info["units_used"]
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="3,4,5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
    data = wsp_train.char_dataset(text)
    floor = lm.unigram_entropy(data.val)
    outdir = R / "lab/runs/decompiled"
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"unigram_entropy_val": round(floor, 4), "teachers": {}, "data_tables": {}}

    for order in [int(o) for o in a.orders.split(",")]:
        table = wsp_synth.WittenBellTable(data.train, len(data.chars), order)
        report["data_tables"][order] = {"units": table.num_units(),
                                        "exact": compile_and_measure(table, EXACT, data, text, a.device),
                                        "wide": compile_and_measure(table, WIDE, data, text, a.device)}
        print("data table", order, report["data_tables"][order], flush=True)

    for name, (path, shape) in TEACHERS.items():
        teacher = wsp_arch.load_arch_file(ARCH, overrides=dict(shape, vocab=len(data.chars)), seed=1337)
        teacher.load(path)
        teacher.eval().to(a.device)
        entry = {"checkpoint": str(path.relative_to(R)), "teacher": measure(teacher, data, text, a.device)}
        try:
            wsp_decompile.decompile_ngram(teacher, data.chars)
            entry["structural"] = "unexpectedly decompiled"
        except wsp_decompile.DecompileError as error:
            entry["structural"] = f"refused: {error}"
        entry["behavioral"] = {}
        for order in [int(o) for o in a.orders.split(",")]:
            t0 = time.time()
            with wsp_synth.forbid_gradients():
                program = wsp_decompile.behavioral_table(teacher, data.train, data.chars, order=order)
            text_program = wsp_decompile.format_program(program, source=f"behavioral decompile of {path.name}")
            program_path = outdir / f"{name}_order{order}.wspngram"
            program_path.write_text(text_program, encoding="utf-8")
            table = wsp_decompile.parse_program(text_program)
            row = {"program": str(program_path.relative_to(R)), "units": table.num_units(),
                   "program_bytes": len(text_program.encode("utf-8")), "decompile_seconds": round(time.time() - t0, 1),
                   "exact": compile_and_measure(table, EXACT, data, text, a.device),
                   "wide": compile_and_measure(table, WIDE, data, text, a.device)}
            for size in ("exact", "wide"):
                row[size]["retained"] = round(wsp_decompile.retained_fraction(
                    row[size]["val_nll_full"], entry["teacher"]["val_nll_full"], floor), 4)
            entry["behavioral"][order] = row
            print(name, order, row, flush=True)
        report["teachers"][name] = entry
        del teacher
        torch.cuda.empty_cache()
    (R / "lab/runs/wsp_decompile_round_trip.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
