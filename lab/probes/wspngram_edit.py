"""Edit a decompiled .wspngram program as text, recompile without gradients, and measure targeted effect and
collateral against controls. Criterion fixed beforehand in lab/probes/wspngram_edit_criterion.md.

Writes lab/runs/wspngram_edit.json and the two edited programs under lab/runs/decompiled/.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "lab")]

from engine import wsp_arch, wsp_decompile, wsp_synth, wsp_train  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
PROGRAM = R / "lab/runs/decompiled/nanogpt_10.7M_order5.wspngram"
TEACHER = R / "models/nanogpt_shakespeare_char/model.safetensors"
EXACT = dict(block=256, layers=6, heads=6, d=384, ff=1536, dropout=0.0, bias=False)
DELTA = 12.0
SEED = 1337
WORDS = ("lord", "Lord", "LORD", "king", "kind", "King", "KING")


def logits_fn(model):
    def fwd(x):
        return model(x, torch.zeros_like(x))[0].float()
    return fwd


@torch.no_grad()
def val_logprobs(fwd, ids, device, block=256, stride=128, batch=32):
    """Row j-1 = log p(ids[j] | ids[:j]) for j = 1..N-1, each scored with >= 128 characters of context."""
    n = len(ids)
    out = torch.empty(n, 65)
    starts = list(range(0, n - block - 1, stride)) + [n - block - 1]
    ids_t = torch.tensor(ids, dtype=torch.long)
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        x = torch.stack([ids_t[s:s + block] for s in chunk]).to(device)
        logp = torch.log_softmax(fwd(x), -1).cpu()
        for row, s in enumerate(chunk):
            lo = 0 if s == 0 else stride - 1
            out[s + 1 + lo:s + 1 + block] = logp[row, lo:]
    return out[1:]


@torch.no_grad()
def sample(fwd, stoi, itos, device, streams=16, length=1250, seed=0, bias=None):
    gen = torch.Generator().manual_seed(seed)
    x = torch.full((streams, 1), stoi["\n"], dtype=torch.long, device=device)
    for _ in range(length):
        logits = fwd(x[:, -256:])[:, -1].cpu()
        if bias is not None:
            logits = logits + bias
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).to(device)
        x = torch.cat([x, nxt], 1)
    texts = ["".join(itos[i] for i in row[1:].tolist()) for row in x]
    return {w: sum(t.count(w) for t in texts) for w in WORDS}


def metrics(base, new, targets, masks, chars):
    """base/new: [M, V] log-probs; targets: [M] next-char ids; masks: name -> bool [M]."""
    true_b = base.gather(1, targets[:, None])[:, 0]
    true_n = new.gather(1, targets[:, None])[:, 0]
    delta = true_n - true_b
    target_any = torch.zeros_like(masks[next(iter(masks))])
    for m in masks.values():
        target_any |= m
    out = {"val_nll": round(float(-true_n.mean()), 5), "val_nll_base": round(float(-true_b.mean()), 5),
           "outside_mean_abs_dnll": float(delta[~target_any].abs().mean()),
           "outside_mean_dnll": float(-delta[~target_any].mean()),
           "outside_max_abs_dlogp_row": float((new - base)[~target_any].abs().max()),
           "outside_positions": int((~target_any).sum())}
    d, g = chars.index("d"), chars.index("g")
    for name, m in masks.items():
        p = new[m].exp()
        pb = base[m].exp()
        nd = targets[m] == d
        ng = targets[m] == g
        out[name] = {"positions": int(m.sum()),
                     "next_d": int(nd.sum()), "next_g": int(ng.sum()),
                     "mean_p_d_when_next_d": float(p[nd, d].mean()) if nd.any() else None,
                     "mean_p_d_when_next_d_base": float(pb[nd, d].mean()) if nd.any() else None,
                     "frac_pd_gt_pg_when_next_g": float((p[ng, d] > p[ng, g]).float().mean()) if ng.any() else None,
                     "frac_pg_gt_pd_when_next_d": float((p[nd, g] > p[nd, d]).float().mean()) if nd.any() else None,
                     "frac_pd_gt_pg_when_next_g_base": float((pb[ng, d] > pb[ng, g]).float().mean()) if ng.any() else None,
                     "frac_pg_gt_pd_when_next_d_base": float((pb[nd, g] > pb[nd, d]).float().mean()) if nd.any() else None,
                     "mean_dnll": float(-delta[m].mean())}
    return out


def edited_table(table, edits):
    """edits: context tuple -> function(values copy) -> new values. Returns a new UnitTable."""
    units = [(c, edits[c](v.copy()) if c in edits else v) for c, v in table.units()]
    return wsp_decompile.UnitTable(table.chars, table.order, units, provenance=table.provenance)


def chain_sum(table, context, limit):
    index = {c: v for c, v in table.units(limit=limit)}
    total = np.zeros(table.vocab)
    for k in range(len(context) + 1):
        suffix = context[len(context) - k:]
        if suffix not in index:
            break
        total += index[suffix]
    return total


def compile_model(table, device):
    with wsp_synth.forbid_gradients():
        model = wsp_synth.compile_ngram(ARCH, table, overrides=EXACT, seed=SEED)
    return model.eval().to(device)


def fine_tune(teacher, train_ids, stoi, mode, device, steps=300, lr=1e-4, batch=32, block=256, seed=0):
    """Control C3: gradient fine-tuning with settings fixed in the criterion."""
    l, o, r, k, i, n = (stoi[c] for c in "lorkin")
    d, g = stoi["d"], stoi["g"]
    data = torch.tensor(train_ids, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=0.0)
    teacher.train()
    for _ in range(steps):
        ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[j:j + block] for j in ix]).to(device)
        y = torch.stack([data[j + 1:j + block + 1] for j in ix]).to(device)
        ctx = torch.zeros_like(x, dtype=torch.bool)
        a, b2, c = (l, o, r) if mode == "unlearn" else (k, i, n)
        ctx[:, 2:] = (x[:, :-2] == a) & (x[:, 1:-1] == b2) & (x[:, 2:] == c)
        logp = torch.log_softmax(teacher(x, y)[0].float(), -1)
        if mode == "unlearn":
            hit = ctx & (y == d)
            ce = -logp.gather(-1, y[..., None])[..., 0]
            ul = -torch.log1p(-logp[..., d].exp().clamp(max=1 - 1e-6))
            loss = torch.where(hit, ul, ce).mean()
        else:
            y2 = y.clone()
            y2[ctx & (y == g)] = d
            y2[ctx & (y == d)] = g
            loss = -logp.gather(-1, y2[..., None])[..., 0].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    teacher.eval()
    return teacher


def load_teacher(vocab, device):
    model = wsp_arch.load_arch_file(ARCH, overrides=dict(EXACT, vocab=vocab), seed=SEED)
    model.load(TEACHER)
    return model.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    text = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
    data = wsp_train.char_dataset(text)
    chars = data.chars
    stoi, itos = {c: j for j, c in enumerate(chars)}, dict(enumerate(chars))
    val_text = text[int(len(text) * 0.9):]
    val_ids = [stoi[c] for c in val_text]
    targets = torch.tensor(val_ids[1:])
    ctx3 = [val_text[max(0, j - 3):j] for j in range(1, len(val_text))]
    masks = {"lor": torch.tensor([c == "lor" for c in ctx3]), "kin": torch.tensor([c == "kin" for c in ctx3])}
    d, g = stoi["d"], stoi["g"]

    table = wsp_decompile.parse_program(PROGRAM.read_text(encoding="utf-8"))
    capacity = EXACT["layers"] * EXACT["ff"]
    order = [c for c, _ in table.units()]
    rank = {c: j for j, c in enumerate(order)}
    lor, kin = tuple(stoi[c] for c in "lor"), tuple(stoi[c] for c in "kin")
    for c in (lor, kin):
        assert c in rank and rank[c] < capacity, f"unit {c} not compiled (rank {rank.get(c)})"
    report = {"program": str(PROGRAM.relative_to(R)), "units": table.num_units(), "capacity": capacity,
              "rank_lor": rank[lor], "rank_kin": rank[kin], "delta_nats": DELTA}

    # C0
    m0 = compile_model(table, a.device)
    f0 = logits_fn(m0)
    base = val_logprobs(f0, val_ids, a.device)
    report["C0"] = {"val_nll": round(float(-base.gather(1, targets[:, None]).mean()), 5),
                    "samples": sample(f0, stoi, itos, a.device)}
    print("C0", report["C0"], flush=True)

    # E1
    def unlearn(v):
        v[d] -= DELTA
        return v
    t1 = edited_table(table, {lor: unlearn})
    (R / "lab/runs/decompiled/nanogpt_10.7M_order5_E1_unlearn_lord.wspngram").write_text(
        wsp_decompile.format_program(t1, source="E1: unit \"lor\" entry d -= 12"), encoding="utf-8")
    m1 = compile_model(t1, a.device)
    f1 = logits_fn(m1)
    e1 = val_logprobs(f1, val_ids, a.device)
    report["E1"] = metrics(base, e1, targets, masks, chars)
    report["E1"]["samples"] = sample(f1, stoi, itos, a.device)
    print("E1", report["E1"], flush=True)

    # C4: weights that differ between C0 and E1, and a hand-made direct edit of C0
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    changed = {k: int((sd0[k] != sd1[k]).sum()) for k in sd0 if not torch.equal(sd0[k], sd1[k])}
    layer, unit = divmod(rank[lor], EXACT["ff"])
    cols = {k: torch.nonzero((sd0[k] != sd1[k]).any(0)).flatten().tolist() for k in changed if sd0[k].dim() == 2}
    activation = float(torch.nn.functional.gelu(torch.tensor(wsp_synth.UNIT_GAIN / 2, dtype=torch.float64)))
    delta_vec = np.zeros(len(chars))
    delta_vec[d] = -DELTA
    delta_vec -= delta_vec.mean()
    logit_dirs = np.asarray(m0.synthesis_layout["logit_dirs"], dtype=np.float64)
    with torch.no_grad():
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        param = m0.get_parameter(name)
        param[:, unit] += torch.from_numpy(logit_dirs @ delta_vec / activation).to(param.dtype).to(param.device)
    direct = val_logprobs(f0, val_ids, a.device)
    report["C4"] = {"changed_tensors": changed, "changed_columns": cols, "expected": [name, unit],
                    "max_abs_logprob_diff_direct_vs_recompiled": float((direct - e1).abs().max())}
    print("C4", report["C4"], flush=True)
    del m0, m1, direct
    torch.cuda.empty_cache()

    # E2
    s = chain_sum(table, kin, capacity)
    swap = s[d] - s[g]

    def swap_kd(v):
        v[g] += swap
        v[d] -= swap
        return v
    t2 = edited_table(table, {kin: swap_kd})
    (R / "lab/runs/decompiled/nanogpt_10.7M_order5_E2_swap_king_kind.wspngram").write_text(
        wsp_decompile.format_program(t2, source="E2: unit \"kin\" g and d exchanged"), encoding="utf-8")
    m2 = compile_model(t2, a.device)
    f2 = logits_fn(m2)
    e2 = val_logprobs(f2, val_ids, a.device)
    report["E2"] = metrics(base, e2, targets, masks, chars)
    report["E2"]["samples"] = sample(f2, stoi, itos, a.device)
    report["E2"]["swap_nats"] = float(swap)
    print("E2", report["E2"], flush=True)
    del m2, e2
    torch.cuda.empty_cache()

    # C1: the E1 change in 5 random units of similar frequency
    lor_count = text.count("lor")
    pool = [c for c in order[:capacity] if len(c) == 3 and c != lor
            and 0.5 * lor_count <= text.count("".join(chars[t] for t in c)) <= 2 * lor_count]
    picks = random.Random(0).sample(pool, 5)
    report["C1"] = []
    for c in picks:
        word = "".join(chars[t] for t in c)
        m = compile_model(edited_table(table, {c: unlearn}), a.device)
        lp = val_logprobs(logits_fn(m), val_ids, a.device)
        own = torch.tensor([x == word for x in ctx3])
        row = metrics(base, lp, targets, {"lor": masks["lor"], "own": own}, chars)
        report["C1"].append({"context": word, "count": text.count(word),
                             "lor_mean_p_d_change": row["lor"]["mean_p_d_when_next_d"] - row["lor"]["mean_p_d_when_next_d_base"],
                             "own_mean_dnll": row["own"]["mean_dnll"], "own_positions": row["own"]["positions"],
                             "outside_mean_abs_dnll": row["outside_mean_abs_dnll"]})
        print("C1", report["C1"][-1], flush=True)
        del m
        torch.cuda.empty_cache()

    # C2: output-level bias on C0
    bias = torch.zeros(len(chars))
    bias[d] = -DELTA
    c2 = torch.log_softmax(base + bias, -1)
    report["C2"] = metrics(base, c2, targets, masks, chars)
    m0 = compile_model(table, a.device)
    report["C2"]["samples"] = sample(logits_fn(m0), stoi, itos, a.device, bias=bias)
    print("C2", {k: v for k, v in report["C2"].items() if k != "kin"}, flush=True)
    del m0
    torch.cuda.empty_cache()

    # C3: gradient fine-tuning of the trained teacher
    teacher = load_teacher(len(chars), a.device)
    ft = logits_fn(teacher)
    tbase = val_logprobs(ft, val_ids, a.device)
    report["C3_teacher_base"] = {"val_nll": round(float(-tbase.gather(1, targets[:, None]).mean()), 5),
                                 "samples": sample(ft, stoi, itos, a.device)}
    print("C3 base", report["C3_teacher_base"], flush=True)
    for mode in ("unlearn", "swap"):
        teacher = fine_tune(load_teacher(len(chars), a.device), data.train, stoi, mode, a.device)
        ft = logits_fn(teacher)
        lp = val_logprobs(ft, val_ids, a.device)
        report[f"C3_{mode}"] = metrics(tbase, lp, targets, masks, chars)
        report[f"C3_{mode}"]["samples"] = sample(ft, stoi, itos, a.device)
        print("C3", mode, report[f"C3_{mode}"], flush=True)
        del teacher
        torch.cuda.empty_cache()

    report["seconds"] = round(time.time() - t0, 1)
    (R / "lab/runs/wspngram_edit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
