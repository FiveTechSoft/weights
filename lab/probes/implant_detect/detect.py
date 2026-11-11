"""Can a few reprogrammed neurons be seen in the weights? (PLAN.md, fixed before running; defence, synthetic implants).

Nothing functional is built: implants are random rewrites / pushes of one neuron's vectors (gate row, up row, down
column), or a down column set to one benign token's unembedding row; never tied to an input pattern, never saved.

D1 (with base): per neuron, robust z of the change norms of its three vectors within its layer and matrix; score = max.
D2 (no base):   per neuron of the suspect model, robust |z| of its three norms within the layer and robust z of the
                largest cosine of its down column with any unembedding row; score = max.
Thresholds from the even layers with no implant (<= 1 flag); implants and scoring on the odd layers.

usage: python detect.py run [out_dir]     (CPU; default lab/runs/implant_detect)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "hf_cache" / "qwen35-9b-hf"
MIMO = ROOT / "hf_cache" / "mimo-distill-hf"
PFX = "model.language_model.layers"
SEED, TRIALS, N_LAYERS = 20261111, 20, 32
KINDS = ("R", "S1", "S3", "S10", "T")
KS = (1, 3, 10)


def med_mad(x):
    med = float(np.median(x))
    return med, 1.4826 * float(np.median(np.abs(x - med)))


def robust_z(x, med, mad):
    x = np.asarray(x, dtype=np.float64)
    if mad > 0:
        return (x - med) / mad
    d = x - med
    return np.where(d > 0, np.inf, np.where(d < 0, -np.inf, 0.0))


def tau_from(scores):
    """Smallest threshold leaving <= 1 calibration score strictly above it."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    return float(s[-2]) if s.size > 1 else float(s[-1])


def rewrite(v, g):
    w = g.normal(size=v.shape).astype(np.float32)
    return w * (np.linalg.norm(v) / np.linalg.norm(w))


def push(v, size, g):
    d = g.normal(size=v.shape).astype(np.float32)
    return v + d * (size / np.linalg.norm(d))


class Store:
    def __init__(self, d):
        self.d = Path(d)
        self.idx = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        from safetensors import safe_open
        with safe_open(str(self.d / self.idx[name]), framework="pt") as f:
            return f.get_tensor(name).float().numpy()


def cosmax(U16, cols, chunk=16384):
    """Largest cosine of each column (d, n) with any row of U16 (V, d; rows unit-norm, float16)."""
    import torch
    c = torch.from_numpy(np.ascontiguousarray(cols / np.maximum(np.linalg.norm(cols, axis=0), 1e-30)))
    best = torch.full((c.shape[1],), -1.0)
    for k in range(0, U16.shape[0], chunk):
        best = torch.maximum(best, (U16[k:k + chunk].float() @ c).max(0).values)
    return best.numpy()


def draw(n_neurons):
    """All implant positions, fixed by the seed before any weight is read: odd layers only (scoring set)."""
    g = np.random.default_rng(SEED)
    odd = [l for l in range(N_LAYERS) if l % 2]
    trials = []
    for setting in ("alone", "mimo"):
        for loc in ("odd", "last"):
            for kind in KINDS:
                for k in KS:
                    for t in range(TRIALS):
                        layers = g.choice(odd, k) if loc == "odd" else np.full(k, N_LAYERS - 1)
                        pos = []
                        for l in layers:
                            j = int(g.integers(n_neurons))
                            while (int(l), j) in pos:
                                j = int(g.integers(n_neurons))
                            pos.append((int(l), j))
                        trials.append({"setting": setting, "loc": loc, "kind": kind, "k": k, "t": t, "pos": pos,
                                       "seed": int(g.integers(2 ** 31))})
    return trials


def common_tokens(n=1000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE))
    ids = tok((ROOT / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    vals, cnt = np.unique(ids, return_counts=True)
    return vals[np.argsort(-cnt)[:n]]


def run(out):
    import torch
    torch.set_num_threads(12)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out}/summary.json exists")
    B, M = Store(BASE), Store(MIMO)
    heads = {}                                                         # D2 uses the suspect's own head
    for name, S in (("base", B), ("mimo", M)):
        h = torch.from_numpy(S.get("lm_head.weight"))
        heads[name] = (h / h.norm(dim=1, keepdim=True).clamp_min(1e-30)).half()
        del h
    toks = common_tokens()
    n_neurons = B.get(f"{PFX}.0.mlp.down_proj.weight").shape[1]
    trials = draw(n_neurons)
    by_layer = {}
    for ti, tr in enumerate(trials):
        for (l, j) in tr["pos"]:
            by_layer.setdefault(l, []).append((ti, j))
    clean = {k: [None] * N_LAYERS for k in ("d1_mimo", "d2_base", "d2_mimo", "mimo_chg")}
    imp_scores = {}                    # (trial, l, j) -> {"d1": score, "d2": score}
    t0 = time.time()
    for l in range(N_LAYERS):
        W = {}
        for name, S in (("base", B), ("mimo", M)):
            W[name] = {m: S.get(f"{PFX}.{l}.mlp.{m}_proj.weight") for m in ("gate", "up", "down")}
        vec = lambda w, m, j: w[m][j] if m != "down" else w[m][:, j]                       # neuron j's vector
        nrm = lambda w, m: np.linalg.norm(w[m], axis=1 if m != "down" else 0)
        chg = {m: (np.linalg.norm(W["mimo"][m] - W["base"][m], axis=1 if m != "down" else 0)) for m in ("gate", "up", "down")}
        st_chg = {m: med_mad(chg[m]) for m in chg}                                         # D1 reference, in MiMo
        st_zero = {m: (0.0, 0.0) for m in chg}                                             # D1 reference, alone
        d1_m = np.max([robust_z(chg[m], *st_chg[m]) for m in chg], axis=0)
        norms = {s: {m: nrm(W[s], m) for m in ("gate", "up", "down")} for s in W}
        cos = {s: cosmax(heads[s], W[s]["down"]) for s in W}
        st_n = {s: {m: med_mad(norms[s][m]) for m in norms[s]} for s in W}
        st_c = {s: med_mad(cos[s]) for s in W}
        d2 = {s: np.max([np.abs(robust_z(norms[s][m], *st_n[s][m])) for m in norms[s]] +
                        [robust_z(cos[s], *st_c[s])], axis=0) for s in W}
        clean["d1_mimo"][l], clean["d2_base"][l], clean["d2_mimo"][l] = d1_m, d2["base"], d2["mimo"]
        clean["mimo_chg"][l] = {m: float(np.median(chg[m])) for m in chg}
        pend = []
        for ti, j in by_layer.get(l, []):
            tr = trials[ti]
            g = np.random.default_rng([tr["seed"], l, j])
            sus = "base" if tr["setting"] == "alone" else "mimo"
            new = {}
            for m in ("gate", "up", "down"):
                v = vec(W[sus], m, j)
                if tr["kind"] in ("R", "T"):
                    new[m] = rewrite(v, g)
                else:
                    new[m] = push(v, float(tr["kind"][1:]) * clean["mimo_chg"][l][m], g)
            if tr["kind"] == "T":
                t = int(g.choice(toks))
                row = heads[sus][t].float().numpy()
                new["down"] = row / np.linalg.norm(row) * np.linalg.norm(vec(W[sus], "down", j))
            ref = st_zero if sus == "base" else st_chg
            d1 = max(float(robust_z([np.linalg.norm(new[m] - vec(W["base"], m, j))], *ref[m])[0]) for m in new)
            pend.append((ti, j, sus, new, d1))
        for sus in ("base", "mimo"):                                  # cosines in one batch per suspect model
            pp = [p for p in pend if p[2] == sus]
            if not pp:
                continue
            cs = cosmax(heads[sus], np.stack([p[3]["down"] for p in pp], axis=1))
            for (ti, j, _, new, d1), c in zip(pp, cs):
                d2s = max([abs(float(robust_z([np.linalg.norm(new[m])], *st_n[sus][m])[0])) for m in new] +
                          [float(robust_z([c], *st_c[sus])[0])])
                imp_scores[(ti, l, j)] = {"d1": d1, "d2": d2s, "cos": float(c)}
        print(json.dumps({"layer": l, "implants": len(by_layer.get(l, [])), "seconds": round(time.time() - t0)}),
              flush=True)
        del W
    np.savez_compressed(out / "clean_scores.npz", **{k: np.stack(v) for k, v in clean.items() if k != "mimo_chg"})
    summ = score(trials, imp_scores, clean)
    summ["seconds"] = round(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["table"], indent=1))


def score(trials, imp, clean):
    even = [l for l in range(N_LAYERS) if l % 2 == 0]
    odd = [l for l in range(N_LAYERS) if l % 2]
    cal = lambda key: tau_from(np.concatenate([clean[key][l] for l in even]))
    tau = {("d1", "mimo"): cal("d1_mimo"), ("d1", "alone"): 0.0, ("d2", "alone"): cal("d2_base"),
           ("d2", "mimo"): cal("d2_mimo")}
    test = {("d1", "mimo"): np.concatenate([clean["d1_mimo"][l] for l in odd]),
            ("d1", "alone"): np.zeros(len(odd) * clean["d1_mimo"][0].size),
            ("d2", "alone"): np.concatenate([clean["d2_base"][l] for l in odd]),
            ("d2", "mimo"): np.concatenate([clean["d2_mimo"][l] for l in odd])}
    fp = {f"{d}_{s}": int((test[(d, s)] > tau[(d, s)]).sum()) for d, s in tau}
    rows = {}
    for ti, tr in enumerate(trials):
        key = (tr["setting"], tr["loc"], tr["kind"], tr["k"])
        r = rows.setdefault(key, {"d1_hit": 0, "d2_hit": 0, "n": 0, "d1_rank": [], "d2_rank": []})
        for (l, j) in tr["pos"]:
            s = imp[(ti, l, j)]
            r["n"] += 1
            for d in ("d1", "d2"):
                r[f"{d}_hit"] += s[d] > tau[(d, tr["setting"])]
                r[f"{d}_rank"].append(int((test[(d, tr["setting"])] >= s[d]).sum()) + 1)
    table = []
    for (setting, loc, kind, k), r in sorted(rows.items()):
        table.append({"setting": setting, "loc": loc, "kind": kind, "k": k, "n": r["n"],
                      "d1_recall": round(r["d1_hit"] / r["n"], 3), "d2_recall": round(r["d2_hit"] / r["n"], 3),
                      "d1_median_rank": int(np.median(r["d1_rank"])), "d2_median_rank": int(np.median(r["d2_rank"]))})
    agg = lambda s, kind, col: np.mean([x[col] for x in table if x["setting"] == s and x["kind"] == kind])
    verdict = {
        "D1 useful (R recall >= 0.9 with <= 1 FP, alone and in MiMo)":
            bool(agg("alone", "R", "d1_recall") >= 0.9 and agg("mimo", "R", "d1_recall") >= 0.9 and
                 fp["d1_alone"] <= 1 and fp["d1_mimo"] <= 1),
        "D1 detection limit in MiMo (smallest c with recall >= 0.9)":
            next((c for c in (1, 3, 10) if agg("mimo", f"S{c}", "d1_recall") >= 0.9), None),
        "D2 useful (T recall >= 0.5 with <= 10 FP on clean MiMo)":
            bool(agg("mimo", "T", "d2_recall") >= 0.5 and fp["d2_mimo"] <= 10),
    }
    d1_real = np.concatenate([clean["d1_mimo"][l] for l in range(N_LAYERS)])
    top = np.argsort(-d1_real)[:10]
    n = clean["d1_mimo"][0].size
    mimo_flags = [{"layer": int(i // n), "neuron": int(i % n), "score": round(float(d1_real[i]), 2)} for i in top]
    return {"tau": {f"{d}_{s}": v for (d, s), v in tau.items()}, "false_positives_odd": fp, "table": table,
            "verdict": verdict, "mimo_top_changed_neurons": mimo_flags,
            "mimo_flagged_all_layers": int((d1_real > tau[("d1", "mimo")]).sum())}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1] if len(a) > 1 else ROOT / "lab" / "runs" / "implant_detect")
    else:
        sys.exit(__doc__)
