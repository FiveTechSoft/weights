"""Dry run (lab/probes/column_edit/PLAN.md): the MIT Exp 8 edit on Qwen3.5-0.8B rebuilt through engine.column_edit
from the 64 recovered columns per layer (data/exp8_edit_recovered.npz, mode add with one vector per column), written
as an ediciones program + npz, applied to the Q6_K GGUF by the existing wsp.py add-cols-vectors op, and compared byte
for byte with servidos/qwen35-08b__exp8-replica.gguf on the two edited tensors. Also writes the structural placebo
program (not applied) and the report. CPU, minutes. Writes servidos/qwen35-08b__exp8-via-edit-column.gguf (ignored by
git) and lab/runs/column_edit/exp8/*. Deletes nothing.

  python lab/probes/column_edit/dryrun_exp8.py
"""
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import column_edit as ce  # noqa: E402
from engine import wsplpatch as wp  # noqa: E402

HF = ROOT / "hf_cache" / "qwen35-08b-hf"
GGUF = ROOT / "hf_cache" / "qwen35-08b-gguf" / "Qwen_Qwen3.5-0.8B-Q6_K.gguf"
REPLICA = ROOT / "servidos" / "qwen35-08b__exp8-replica.gguf"
OUT_GGUF = ROOT / "servidos" / "qwen35-08b__exp8-via-edit-column.gguf"
RUN = ROOT / "lab" / "runs" / "column_edit" / "exp8"
NPZ = ROOT / "data" / "exp8_edit_recovered.npz"
LAYERS = (15, 19)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    edits, placebos, reports = [], [], {}
    with np.load(NPZ, allow_pickle=False) as z:
        for i, L in enumerate(LAYERS):
            name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
            W0 = wp.read_tensor(HF, name).float().numpy()
            cols = [int(c) for c in z[f"cols_{i}"]]
            V = np.asarray(z[f"vec_{i}"], dtype=np.float32)
            dW = ce.plan_edit(W0, cols, V, alpha=1.0, mode="add")          # 2-D direction: used as given
            ce.check_budget(dW, budget_params=200_000)
            P = ce.structural_placebo(dW, cols, V, seed=2026)
            rep = ce.report(W0, dW, V, cols, 1.0, "add", name, P, 2026, 200_000)
            reports[name] = rep
            g = ce.gguf_name(name)
            edits.append({"gguf_tensor": g, "cols": cols, "dW": dW})
            placebos.append({"gguf_tensor": g, "cols": cols, "dW": P})
            print(f"{name}: params_touched {rep['params_touched']} delta_norm {rep['delta_norm']:.4f} "
                  f"rel {rep['rel_delta_norm']:.4f} cos {rep['cosine_with_direction']:.4f} "
                  f"placebo params {rep['placebo_params_touched']} norm {rep['placebo_delta_norm']:.4f}", flush=True)
    prog = ce.write_program(RUN, "exp8_via_edit_column", edits, placebos, gguf_out=OUT_GGUF,
                            gguf_out_placebo=ROOT / "servidos" / "qwen35-08b__exp8-via-edit-column-placebo.gguf")
    (RUN / "report.json").write_text(json.dumps(reports, indent=1), encoding="utf-8")
    if OUT_GGUF.exists():
        print(f"{OUT_GGUF} exists: not rebuilt (wsp.py never overwrites)")
    else:
        cmd = [sys.executable, str(ROOT / "wsp.py"), "edit", str(GGUF), prog["program"]]
        print("running:", " ".join(cmd), flush=True)
        cp = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        (RUN / "wsp_edit.log").write_text(cp.stdout + "\n" + cp.stderr, encoding="utf-8")
        if cp.returncode != 0:
            print("wsp.py edit failed; see", RUN / "wsp_edit.log")
            print(cp.stderr[-800:])
            return 1
    # byte compare the two edited tensors (and count every differing tensor) against the replica
    from gguf import GGUFReader
    ra, rb = GGUFReader(str(OUT_GGUF)), GGUFReader(str(REPLICA))
    tb = {t.name: t for t in rb.tensors}
    differ, edited_equal = [], {}
    for t in ra.tensors:
        a = np.asarray(t.data).view(np.uint8).ravel()
        b = np.asarray(tb[t.name].data).view(np.uint8).ravel()
        same = a.shape == b.shape and np.array_equal(a, b)
        if t.name in (f"blk.{L}.ffn_down.weight" for L in LAYERS):
            edited_equal[t.name] = bool(same)
        if not same:
            differ.append(t.name)
    del ra, rb
    gc.collect()
    result = {"out_gguf": str(OUT_GGUF), "replica": str(REPLICA), "edited_tensors_byte_equal": edited_equal,
              "tensors_differing_anywhere": differ, "sha256_out": wp.sha256_file(OUT_GGUF),
              "sha256_replica": wp.sha256_file(REPLICA), "program": prog}
    (RUN / "byte_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("edited tensors byte-equal to the replica:", edited_equal)
    print("tensors differing anywhere:", differ)
    print("whole-file sha256 equal:", result["sha256_out"] == result["sha256_replica"])
    return 0 if all(edited_equal.values()) and not differ else 2


if __name__ == "__main__":
    sys.exit(main())
