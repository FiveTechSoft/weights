"""N25: shared-factor law on natural text (plan: PLAN_N25.md, written before the run).

Reuses factor_law's estimator, compiler, lags, holdout mask and tolerances unchanged;
the only new code is the natural class stream (PLAN_N25.md) and the acceptance-split
evaluation. No optimizer, no donor, no retuning.
"""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from natural_n16 import HASH, V, tokenize
from cycle_law import new, snapshot
from factor_law import OBS, PAIR, B, conditional, fit, compile_factors


def class_map(train_ids, counts_v):
    """4 macro-classes by train unigram mass quartiles (PLAN_N25). Train-only."""
    order = np.argsort(-counts_v, kind="stable")
    cum = np.cumsum(counts_v[order])
    total = cum[-1]
    cls = np.zeros(len(counts_v), dtype=np.int64)
    bounds = [total * f for f in (.25, .5, .75)]
    k = 0
    for rank, idx in enumerate(order):
        while k < 3 and cum[rank] > bounds[k]:
            k += 1
        cls[idx] = k
    return cls


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    raw = Path(a.corpus).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASH
    text = raw.decode()
    n = len(text)
    train_t, acc_t = (tokenize(text[s:e]) for s, e in [(0, 7 * n // 10), (8 * n // 10, 9 * n // 10)])
    words = [w for w, c in Counter(train_t).most_common(V - 1)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    train = np.array([vocab.get(w, 0) for w in train_t], dtype=np.int64)
    acc = np.array([vocab.get(w, 0) for w in acc_t], dtype=np.int64)

    counts_v = np.bincount(train, minlength=V).astype(np.float64)
    cls = class_map(train, counts_v)
    ctr = cls[train]
    cac = cls[acc]

    # Counts on the train class stream; held-out context cells zeroed before the fit.
    a_, b_ = PAIR
    t0 = max(a_, b_)
    src = 4 * ctr[t0 - a_:len(ctr) - a_] + ctr[t0 - b_:len(ctr) - b_]
    y = ctr[t0:]
    c = torch.bincount(torch.from_numpy(4 * src + y), minlength=64).reshape(4, 4, 4).to(torch.double)
    c[~OBS] = 0

    U, Vv, additive = fit(c)
    T = conditional(U, Vv)
    m = compile_factors(U, Vv).eval()

    # G1: compiler fidelity on arbitrary probe histories (fixed seed).
    g = torch.Generator().manual_seed(40000)
    probe = torch.randint(4, (128, 32), generator=g)
    pp = m(probe, probe)[0][:, PAIR[1]:].double().softmax(-1)
    pos = torch.arange(PAIR[1], 32)
    expected = T[probe[:, pos - PAIR[0]], probe[:, pos - PAIR[1]]]
    fidelity_err = float((pp - expected).abs().max())

    # Acceptance evaluation, windowed exactly like factor_law's own evaluation.
    nw = len(cac) // 64
    te = torch.from_numpy(cac[:nw * 64].reshape(nw, 64))
    x = te[:, :32].contiguous()
    pos = torch.arange(16, 32)
    aa = x[:, pos - PAIR[0]]
    bb = x[:, pos - PAIR[1]]
    yy = te[:, 17:33]
    missing = ~OBS[aa, bb]
    probs = m(x, x)[0][:, 16:].double().softmax(-1)

    def nll(p):
        return float(-p[missing].gather(-1, yy[missing][:, None]).log().mean())

    native = nll(probs)
    factored = nll(T[aa, bb])
    add = nll(additive[aa, bb])
    uniform = math.log(4)
    # Reported-only baselines: train class marginal, and unmasked full-train MLE.
    marg = torch.tensor(np.bincount(ctr, minlength=4) + .5, dtype=torch.double)
    marg /= marg.sum()
    marginal = float(-marg[yy[missing]].log().mean())
    full = torch.bincount(torch.from_numpy(4 * src + y), minlength=64).reshape(4, 4, 4).to(torch.double)
    fullmle = (full + .5) / (full + .5).sum(-1, keepdim=True)
    full_ref = nll(fullmle[aa, bb])

    # G4: strict reload and zero-product ablation.
    path = out / "case.safetensors"
    save_file(snapshot(m), str(path))
    restored = new(99)
    restored.load_state_dict(load_file(str(path)), strict=True)
    restored.eval()
    reload_exact = torch.equal(m(x[:8], x[:8])[0], restored(x[:8], x[:8])[0])
    m.transformer.h[1].mlp.c_proj.weight[:, 32:44].zero_()
    ablation = float((m(probe, probe)[0][:, PAIR[1]:].double().softmax(-1) - .25).abs().max())

    gates = dict(
        G1_fidelity=fidelity_err <= .002,
        G2_compilation_free=native <= factored + .06,
        G3_generalization=native <= add - .05 and native <= uniform - .05,
        G4_reload_and_ablation=bool(reload_exact) and ablation <= .002,
    )
    result = dict(
        plan="PLAN_N25.md", corpus_sha256=HASH, optimizer_steps=0, donor_weights=False,
        parameters=6800, heldout_cells="contexts {2,3}x{2,3}", heldout_targets=int(missing.sum()),
        acceptance_windows=nw,
        class_mass=[float(counts_v[cls == k].sum() / counts_v.sum()) for k in range(4)],
        train_masked_counts=c.tolist(),
        max_probability_error=fidelity_err,
        native_nll=native, factored_nll=factored, additive_nll=add, uniform_nll=uniform,
        marginal_nll=marginal, full_train_mle_nll=full_ref,
        excess_vs_factored=native - factored, gain_additive=add - native, gain_uniform=uniform - native,
        reload_exact=bool(reload_exact), zero_product_error=ablation,
        gates=gates, gate_pass=all(gates.values()),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print("SUMMARY", json.dumps(result), flush=True)
    print("GATE", result["gate_pass"], flush=True)


if __name__ == "__main__":
    main()
