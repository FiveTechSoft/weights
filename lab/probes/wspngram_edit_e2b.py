"""E2b: swap g/d in unit " kin" (word-initial) of the decompiled nanoGPT program. Criterion in
lab/probes/wspngram_edit_criterion.md (addendum). Adds "E2b" to lab/runs/wspngram_edit.json."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wspngram_edit as w  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = (w.R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
    chars = w.wsp_train.char_dataset(text).chars
    stoi, itos = {c: j for j, c in enumerate(chars)}, dict(enumerate(chars))
    val_text = text[int(len(text) * 0.9):]
    val_ids = [stoi[c] for c in val_text]
    d, g = stoi["d"], stoi["g"]
    table = w.wsp_decompile.parse_program(w.PROGRAM.read_text(encoding="utf-8"))
    capacity = w.EXACT["layers"] * w.EXACT["ff"]
    unit = tuple(stoi[c] for c in " kin")
    rank = [c for c, _ in table.units()].index(unit)
    assert rank < capacity, rank

    s = w.chain_sum(table, unit, capacity)
    swap = s[d] - s[g]

    def swap_kd(v):
        v[g] += swap
        v[d] -= swap
        return v
    edited = w.edited_table(table, {unit: swap_kd})
    (w.R / "lab/runs/decompiled/nanogpt_10.7M_order5_E2b_swap_space_kin.wspngram").write_text(
        w.wsp_decompile.format_program(edited, source='E2b: unit " kin" g and d exchanged'), encoding="utf-8")

    m0 = w.compile_model(table, device)
    base = w.val_logprobs(w.logits_fn(m0), val_ids, device)
    del m0
    m2 = w.compile_model(edited, device)
    f2 = w.logits_fn(m2)
    new = w.val_logprobs(f2, val_ids, device)
    targets = torch.tensor(val_ids[1:])

    ctx4 = [val_text[max(0, j - 4):j] for j in range(1, len(val_text))]
    word = torch.tensor([c == " kin" for c in ctx4])
    other_kin = torch.tensor([c.endswith("kin") and c != " kin" for c in ctx4])
    rest = ~(word | other_kin)
    pref_g_base = base[word][:, g] > base[word][:, d]
    flipped = (new[word][:, d] > new[word][:, g])[pref_g_base]
    true_b = base.gather(1, targets[:, None])[:, 0]
    true_n = new.gather(1, targets[:, None])[:, 0]
    result = {
        "unit_rank": rank, "swap_nats": float(swap),
        "space_kin_positions": int(word.sum()), "base_prefers_g": int(pref_g_base.sum()),
        "flipped_fraction": float(flipped.float().mean()),
        "other_kin_positions": int(other_kin.sum()),
        "other_kin_max_abs_dlogp": float((new[other_kin] - base[other_kin]).abs().max()),
        "rest_mean_abs_dnll": float((true_n - true_b)[rest].abs().mean()),
        "rest_max_abs_dlogp_row": float((new[rest] - base[rest]).abs().max()),
        "val_nll": float(-true_n.mean()), "val_nll_base": float(-true_b.mean()),
        "samples": w.sample(f2, stoi, itos, device),
    }
    result["passes"] = (result["flipped_fraction"] >= 0.95 and result["other_kin_max_abs_dlogp"] <= 1e-6
                        and result["rest_mean_abs_dnll"] <= 0.001)
    print(result)
    path = w.R / "lab/runs/wspngram_edit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["E2b"] = result
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
