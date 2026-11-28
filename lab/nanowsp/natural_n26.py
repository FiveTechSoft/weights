"""N26: factor-law calibration on a synthetic grammar + semantic classes on natural text.

Plan: PLAN_N26.md (written before the run). Reuses factor_law's estimator,
compiler, lags, holdout mask and tolerances unchanged. No optimizer, no donor,
no retuning.
"""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from natural_n16 import HASH, V, tokenize
from cycle_law import new, snapshot
from joint_law import streams
from factor_law import OBS, PAIR, B, conditional, fit, compile_factors, count_data

LAMBDAS = (1.0, 0.5, 0.25, 0.1)
SEEDS = (91000, 91010, 91020, 91030)
N25_FREQ_GAIN_UNIFORM = -0.0184  # recorded in RESEARCH_N25.md (c839051); recomputed in-run below


def true_table(seed, lam):
    g = torch.Generator().manual_seed(seed)
    def factors():
        return (.6 + .8 * torch.rand(4, 3, dtype=torch.double, generator=g)) * \
               (2 * torch.randint(2, (4, 3), generator=g) - 1)
    t = conditional(factors(), factors())
    return lam * t + (1 - lam) * .25


@torch.no_grad()
def synthetic_case(true, seed, out):
    tr = streams(PAIR, true.reshape(16, 4), 4096, seed + 1)
    te = streams(PAIR, true.reshape(16, 4), 512, seed + 2)
    c = count_data(tr)
    c[~OBS] = 0
    U, Vv, additive = fit(c)
    T = conditional(U, Vv)
    m = compile_factors(U, Vv).eval()
    a_, b_ = PAIR
    g = torch.Generator().manual_seed(seed + 3)
    probe = torch.randint(4, (128, 32), generator=g)
    pp = m(probe, probe)[0][:, b_:].double().softmax(-1)
    pos = torch.arange(b_, 32)
    expected = T[probe[:, pos - a_], probe[:, pos - b_]]
    err = float((pp - expected).abs().max())
    x = te[:, :32].contiguous()
    pos = torch.arange(16, 32)
    aa = x[:, pos - a_]
    bb = x[:, pos - b_]
    y = te[:, 17:33]
    missing = ~OBS[aa, bb]
    probs = m(x, x)[0][:, 16:].double().softmax(-1)

    def nll(p):
        return float(-p[missing].gather(-1, y[missing][:, None]).log().mean())

    path = out / f"syn{seed}.safetensors"
    save_file(snapshot(m), str(path))
    restored = new(99)
    restored.load_state_dict(load_file(str(path)), strict=True)
    restored.eval()
    reload_exact = torch.equal(m(x[:8], x[:8])[0], restored(x[:8], x[:8])[0])
    m.transformer.h[1].mlp.c_proj.weight[:, 32:44].zero_()
    ablation = float((m(probe, probe)[0][:, b_:].double().softmax(-1) - .25).abs().max())
    native = nll(probs)
    oracle = nll(true[aa, bb])
    add = nll(additive[aa, bb])
    uniform = math.log(4)
    path.unlink()
    return dict(seed=seed, heldout_targets=int(missing.sum()),
                max_probability_error=err, native_nll=native, oracle_nll=oracle,
                additive_nll=add, uniform_nll=uniform,
                excess=native - oracle, gain_additive=add - native,
                gain_uniform=uniform - native, oracle_gain_uniform=uniform - oracle,
                reload_exact=bool(reload_exact), zero_product_error=ablation)


def freq_class_map(counts_v):
    """N25's 4 macro-classes by train unigram mass quartiles (train-only)."""
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


def semantic_class_map(train, counts_v):
    """K=4 k-means over 50-dim SVD of train-only PPMI co-occurrence (window 5)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    C = np.zeros((V, V), dtype=np.float64)
    for o in range(1, 6):
        i = train[:-o]
        j = train[o:]
        np.add.at(C, (i, j), 1)
        np.add.at(C, (j, i), 1)
    tot = C.sum()
    pr = C.sum(1, keepdims=True) / tot
    pc = C.sum(0, keepdims=True) / tot
    p = C / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p / (pr * pc))
    ppmi = np.where(C > 0, np.maximum(pmi, 0), 0)
    vec = TruncatedSVD(n_components=50, random_state=0).fit_transform(ppmi)
    return KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(vec).astype(np.int64)


@torch.no_grad()
def natural_arm(cls, ctr, cac, counts_v, out, tag):
    """The N25 pipeline on a given class map: count, mask, fit, compile, evaluate."""
    a_, b_ = PAIR
    t0 = max(a_, b_)
    src = 4 * ctr[t0 - a_:len(ctr) - a_] + ctr[t0 - b_:len(ctr) - b_]
    y = ctr[t0:]
    c = torch.bincount(torch.from_numpy(4 * src + y), minlength=64).reshape(4, 4, 4).to(torch.double)
    c[~OBS] = 0
    U, Vv, additive = fit(c)
    T = conditional(U, Vv)
    m = compile_factors(U, Vv).eval()

    g = torch.Generator().manual_seed(40000)
    probe = torch.randint(4, (128, 32), generator=g)
    pp = m(probe, probe)[0][:, b_:].double().softmax(-1)
    pos = torch.arange(b_, 32)
    expected = T[probe[:, pos - a_], probe[:, pos - b_]]
    fidelity_err = float((pp - expected).abs().max())

    nw = len(cac) // 64
    te = torch.from_numpy(cac[:nw * 64].reshape(nw, 64))
    x = te[:, :32].contiguous()
    pos = torch.arange(16, 32)
    aa = x[:, pos - a_]
    bb = x[:, pos - b_]
    yy = te[:, 17:33]
    missing = ~OBS[aa, bb]
    probs = m(x, x)[0][:, 16:].double().softmax(-1)

    def nll(p):
        return float(-p[missing].gather(-1, yy[missing][:, None]).log().mean())

    native = nll(probs)
    add = nll(additive[aa, bb])
    uniform = math.log(4)
    marg = torch.tensor(np.bincount(ctr, minlength=4) + .5, dtype=torch.double)
    marg /= marg.sum()
    marginal = float(-marg[yy[missing]].log().mean())

    path = out / f"nat{tag}.safetensors"
    save_file(snapshot(m), str(path))
    restored = new(99)
    restored.load_state_dict(load_file(str(path)), strict=True)
    restored.eval()
    reload_exact = torch.equal(m(x[:8], x[:8])[0], restored(x[:8], x[:8])[0])
    m.transformer.h[1].mlp.c_proj.weight[:, 32:44].zero_()
    ablation = float((m(probe, probe)[0][:, b_:].double().softmax(-1) - .25).abs().max())
    path.unlink()
    return dict(heldout_targets=int(missing.sum()), acceptance_windows=nw,
                class_mass=[float(counts_v[cls == k].sum() / counts_v.sum()) for k in range(4)],
                max_probability_error=fidelity_err, native_nll=native,
                additive_nll=add, uniform_nll=uniform, marginal_nll=marginal,
                gain_additive=add - native, gain_uniform=uniform - native,
                reload_exact=bool(reload_exact), zero_product_error=ablation)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    # ---- N26a: synthetic calibration
    syn = {}
    for lam in LAMBDAS:
        rows = [synthetic_case(true_table(s, lam), s, out) for s in SEEDS]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        syn[str(lam)] = dict(
            cases=rows, mean_native_nll=mean("native_nll"), mean_oracle_nll=mean("oracle_nll"),
            mean_additive_nll=mean("additive_nll"), mean_excess=mean("excess"),
            mean_gain_additive=mean("gain_additive"), mean_gain_uniform=mean("gain_uniform"),
            mean_oracle_gain_uniform=mean("oracle_gain_uniform"),
            max_probability_error=max(r["max_probability_error"] for r in rows),
            reload_exact=all(r["reload_exact"] for r in rows),
            max_zero_product_error=max(r["zero_product_error"] for r in rows),
        )
        print("SYN", lam, json.dumps({k: v for k, v in syn[str(lam)].items() if k != "cases"}), flush=True)
    all_cases = [r for lam in syn.values() for r in lam["cases"]]
    G_A1 = (max(r["max_probability_error"] for r in all_cases) <= .002
            and all(r["reload_exact"] for r in all_cases)
            and max(r["zero_product_error"] for r in all_cases) <= .002)
    s1 = syn["1.0"]
    G_A2 = (s1["mean_excess"] <= .06
            and s1["mean_gain_uniform"] >= .5 * s1["mean_oracle_gain_uniform"])
    detection = min((float(l) for l, v in syn.items() if v["mean_gain_additive"] >= .05), default=None)

    # ---- N26b: semantic vs frequency classes on natural text
    raw = Path(a.corpus).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASH
    text = raw.decode()
    n = len(text)
    train_t, acc_t = (tokenize(text[s:e]) for s, e in [(0, 7 * n // 10), (8 * n // 10, 9 * n // 10)])
    words = [w for w, c_ in Counter(train_t).most_common(V - 1)]
    vocab = {w: i + 1 for i, w in enumerate(words)}
    train = np.array([vocab.get(w, 0) for w in train_t], dtype=np.int64)
    acc = np.array([vocab.get(w, 0) for w in acc_t], dtype=np.int64)
    counts_v = np.bincount(train, minlength=V).astype(np.float64)

    arms = {}
    for tag, cls in (("frequency", freq_class_map(counts_v)),
                     ("semantic", semantic_class_map(train, counts_v))):
        arms[tag] = natural_arm(cls, cls[train], cls[acc], counts_v, out, tag)
        print("NAT", tag, json.dumps(arms[tag]), flush=True)
    sem, frq = arms["semantic"], arms["frequency"]
    G_B1 = (sem["max_probability_error"] <= .002 and sem["reload_exact"]
            and sem["zero_product_error"] <= .002)
    G_B2 = (sem["native_nll"] <= sem["additive_nll"] - .05
            and sem["native_nll"] <= sem["uniform_nll"] - .05)
    G_B3 = sem["gain_uniform"] >= frq["gain_uniform"] + .02

    gates = dict(G_A1_exactness=bool(G_A1), G_A2_recovery=bool(G_A2),
                 G_B1_exactness=bool(G_B1), G_B2_generalization=bool(G_B2),
                 G_B3_semantic_beats_frequency=bool(G_B3))
    result = dict(
        plan="PLAN_N26.md", corpus_sha256=HASH, optimizer_steps=0, donor_weights=False,
        parameters=6800, lambdas=LAMBDAS, seeds=SEEDS,
        detection_lambda_gain_additive_05=detection,
        synthetic={l: {k: v for k, v in s.items() if k != "cases"} for l, s in syn.items()},
        natural=arms, n25_recorded_freq_gain_uniform=N25_FREQ_GAIN_UNIFORM,
        gates=gates, gate_pass=all(gates.values()),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print("SUMMARY", json.dumps(result), flush=True)
    print("GATE", result["gate_pass"], flush=True)


if __name__ == "__main__":
    main()
