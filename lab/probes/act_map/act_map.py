"""Gate 1 of cross-model porting (PLAN.md): is there a linear map between the residual streams of two models?

Two models with the SAME tokenizer read the same texts. Per hidden state (0 = embeddings, k = output of layer k - 1;
the last one is after the final norm) the activations of every token but the first are stored. Then, offline:

  cka      linear CKA between every hidden state of A and every one of B (which layer corresponds to which);
  maps     ridge, affine, fitted on the fit texts and scored on held-out texts (R2 over all dimensions), for each
           target hidden state from (a) the source state with the highest CKA, (b) the depth-matched one, and
           (c) the source EMBEDDINGS: what the token's identity alone explains (the control);
  shuffle  one map fitted on permuted rows (must explain nothing);
  contrast a mean difference between two groups of held-out tokens ("late": last quarter of each text vs the rest,
           context only; "digit": digit tokens vs the rest, mostly token identity), carried through the map and
           compared with the same difference measured in the target model. This is what porting a steering direction
           needs: a mean difference D measured in A becomes N D in B (the intercepts cancel).

commands:
  capture --model <hf dir> --name <tag>      -> hf_cache/act_map/<tag>/h<k>.npy (float16) + meta.json
  fit --a <tag> --b <tag> [--save-maps]      -> lab/runs/act_map/<a>__<b>.json, verdict by the PLAN rule

usage:
  python lab/probes/act_map/act_map.py capture --model hf_cache/qwen35-08b-hf --name q08
  python lab/probes/act_map/act_map.py capture --model hf_cache/qwen35-9b-hf --name q9
  python lab/probes/act_map/act_map.py fit --a q9 --b q08 --save-maps
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "lab" / "runs" / "act_map"
STORE = ROOT / "hf_cache" / "act_map"

N_TEXTS, N_TEST, MAX_TOKENS = 240, 40, 128
GSM_FIRST = 1000               # GSM8K-train rows 1000.. : not the retention rows (0-299) nor cc's discovery (0-63)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1)   # ridge strength, relative to tr(XtX) / d
N_CKA = 2000
# PLAN rule (fixed before running)
R2_MEDIAN, R2_OVER_EMB, SHARE_OVER_EMB, SHUFFLE_MAX, COS_LATE, SHARE_COS = 0.5, 0.1, 2 / 3, 0.05, 0.7, 0.5


# ---------- pieces with tests ----------

def split_texts(text_of, n_test, n_val_share=0.1):
    """Token masks (fit, val, test) from each token's text index. Test = the last n_test texts, val = the last
    n_val_share of the remaining ones (for the ridge strength). Whole texts, never single tokens."""
    text_of = np.asarray(text_of)
    texts = np.unique(text_of)
    if n_test <= 0 or n_test >= len(texts) - 1:
        raise ValueError(f"n_test={n_test} with {len(texts)} texts")
    rest, test = texts[:-n_test], texts[-n_test:]
    n_val = max(1, int(round(len(rest) * n_val_share)))
    fit, val = rest[:-n_val], rest[-n_val:]
    return np.isin(text_of, fit), np.isin(text_of, val), np.isin(text_of, test)


def linear_cka(X, Y):
    """Linear CKA of two activation matrices with the same rows (tokens). 1 = same up to rotation and scale."""
    X = X.double() - X.double().mean(0)
    Y = Y.double() - Y.double().mean(0)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / ((K * K).sum().sqrt() * (L * L).sum().sqrt()))


def gram(X):
    X = X.double() - X.double().mean(0)
    return X @ X.T


def cka_from_grams(K, L):
    return float((K * L).sum() / ((K * K).sum().sqrt() * (L * L).sum().sqrt()))


class Source:
    """One source hidden state prepared for many ridge fits: centered, with the eigendecomposition of XtX."""

    def __init__(self, X):
        X = X.double()
        self.mean = X.mean(0)
        self.Xc = X - self.mean
        self.s, self.V = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.scale = float(self.s.clamp_min(0).sum()) / X.shape[1]          # tr(XtX) / d

    def solve(self, Y, lam_rel):
        """W (d_src x d_dst) and the target mean: Y ~ (X - mean_x) W + mean_y."""
        Y = Y.double()
        my = Y.mean(0)
        B = self.V.T @ (self.Xc.T @ (Y - my))
        return self.V @ (B / (self.s.clamp_min(0) + lam_rel * self.scale).unsqueeze(1)), my

    def predict(self, X, W, my):
        return (X.double() - self.mean) @ W + my


def r2(Y, P):
    """Share of the variance of Y (all dimensions together, around Y's own mean) explained by P."""
    Y, P = Y.double(), P.double()
    return float(1 - (Y - P).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())


def fit_ridge(src, Y_fit, X_val, Y_val, lams=LAMS):
    """Ridge strength chosen on the validation texts. -> (W, mean_y, lam_rel, validation R2)."""
    best = None
    for lam in lams:
        W, my = src.solve(Y_fit, lam)
        v = r2(Y_val, src.predict(X_val, W, my))
        if best is None or v > best[3]:
            best = (W, my, lam, v)
    return best


def depth_match(j, n_dst, n_src):
    """Hidden-state index of the source at the same relative depth as index j of the target."""
    return int(round(j * (n_src - 1) / (n_dst - 1)))


def mean_diff(X, mask):
    X = X.double()
    return X[mask].mean(0) - X[~mask].mean(0)


def cos(a, b):
    a, b = a.double(), b.double()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def contrast_masks(text_of, pos, is_digit):
    """Two groupings of tokens: 'late' = last quarter of its text (context only), 'digit' = digit token."""
    text_of, pos = np.asarray(text_of), np.asarray(pos)
    last = {t: pos[text_of == t].max() for t in np.unique(text_of)}
    late = np.array([p >= 0.75 * last[t] for t, p in zip(text_of, pos)])
    return {"late": late, "digit": np.asarray(is_digit, dtype=bool)}


def verdict(rows, shuffle_r2):
    """rows: per target hidden state (embeddings excluded) {'r2', 'r2_emb', 'cos_late'}. PLAN rule."""
    r = np.array([x["r2"] for x in rows])
    over = np.mean([x["r2"] - x["r2_emb"] >= R2_OVER_EMB for x in rows])
    late = np.mean([x["cos_late"] >= COS_LATE for x in rows])
    checks = {"median_r2": float(np.median(r)) >= R2_MEDIAN, "over_embeddings": float(over) >= SHARE_OVER_EMB,
              "shuffle": shuffle_r2 <= SHUFFLE_MAX, "late_contrast": float(late) >= SHARE_COS}
    return {"pass": all(checks.values()), "checks": checks, "median_r2": float(np.median(r)),
            "share_over_embeddings": float(over), "share_late_cos": float(late), "shuffle_r2": shuffle_r2}


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# ---------- capture ----------

def default_texts(n):
    rows = (ROOT / "data" / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines()[GSM_FIRST:GSM_FIRST + n]
    return [(lambda r: r["question"] + "\n" + r["answer"])(json.loads(x)) for x in rows]


@torch.no_grad()
def capture(a):
    from tokenizers import Tokenizer            # not AutoTokenizer: findings F37
    from transformers import AutoModelForCausalLM
    from engine import contrast_vector as cv
    mdir = Path(a.model) if Path(a.model).is_absolute() else ROOT / a.model
    texts = ([json.loads(x)["text"] for x in Path(a.texts).read_text(encoding="utf-8").splitlines() if x.strip()]
             if a.texts else default_texts(a.n_texts))
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    ids = [tok.encode(t, add_special_tokens=False).ids[:a.max_tokens] for t in texts]
    if any(len(x) < 2 for x in ids):
        raise ValueError("a text has fewer than 2 tokens")
    total = sum(len(x) - 1 for x in ids)       # the first token of each text is not stored (no context)
    out = STORE / a.name
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.bfloat16).eval()
    core = model.model
    core = core.language_model if hasattr(core, "language_model") else core
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hooks = cv._stream(core.layers, device) if device != "cpu" else []
    mm, row, overflow = None, 0, 0
    try:
        for k in range(0, len(ids), a.batch):
            chunk = ids[k:k + a.batch]
            n = max(len(c) for c in chunk)
            x = torch.tensor([c + [0] * (n - len(c)) for c in chunk])        # right pad: causal, pads unseen
            hs = cv.hidden_states_pre_norm(core, x)     # last row before the final norm (audit 2026-11-13)
            if mm is None:
                mm = [np.lib.format.open_memmap(out / f"h{L:02d}.npy", mode="w+", dtype=np.float16,
                                                shape=(total, hs[L].shape[-1])) for L in range(len(hs))]
            for j, c in enumerate(chunk):
                m = len(c) - 1
                for L, h in enumerate(hs):
                    v = h[j, 1:len(c)].float().cpu().numpy()
                    overflow += int((np.abs(v) > np.finfo(np.float16).max).sum())
                    mm[L][row:row + m] = v.astype(np.float16)
                row += m
            print(f"[capture {a.name}] {min(k + a.batch, len(ids))}/{len(ids)} texts", flush=True)
    finally:
        for h in hooks:
            h.remove()
    for m_ in mm:
        m_.flush()
    flat = [t for c in ids for t in c[1:]]
    meta = {"model": str(mdir), "n_states": len(mm), "d": int(mm[0].shape[1]), "tokens": total,
            "ids_sha256": ids_hash(flat), "token_ids": flat,
            "text_of": [i for i, c in enumerate(ids) for _ in c[1:]],
            "pos": [p for c in ids for p in range(1, len(c))],
            "is_digit": [(tok.id_to_token(t) or "").strip("Ġ▁ ").isdigit() for t in flat],
            "float16_overflow": overflow}
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[capture {a.name}] {total} tokens, {len(mm)} hidden states, d={meta['d']}, overflow={overflow}", flush=True)


# ---------- fit ----------

def load_state(name, k, rows=None):
    x = np.load(STORE / name / f"h{k:02d}.npy", mmap_mode="r")
    return torch.from_numpy(np.asarray(x if rows is None else x[rows], dtype=np.float32))


def one_direction(src_name, dst_name, ms, md, masks, cka, save):
    """Maps src -> dst for every target hidden state >= 1. cka[i][j]: source state i, target state j."""
    fit, val, test = masks
    con = contrast_masks(np.asarray(md["text_of"])[test], np.asarray(md["pos"])[test], np.asarray(md["is_digit"])[test])
    cache, rows, kept = {}, [], {}

    def source(i):
        if i not in cache:
            if len(cache) >= 3:                              # eigendecompositions are large: keep a few
                cache.pop(next(k for k in cache if k != 0))
            X = load_state(src_name, i)
            cache[i] = (Source(X[fit]), X[val], X[test])
        return cache[i]

    for j in range(1, md["n_states"]):
        Y = load_state(dst_name, j)
        Yf, Yv, Yt = Y[fit], Y[val], Y[test]
        i_best = int(np.argmax([cka[i][j] for i in range(ms["n_states"])]))
        i_depth = depth_match(j, md["n_states"], ms["n_states"])
        row = {"target": j, "source_best": i_best, "cka": cka[i_best][j], "source_depth": i_depth}
        for tag, i in (("", i_best), ("_depth", i_depth), ("_emb", 0)):
            S, Xv, Xt = source(i)
            W, my, lam, _ = fit_ridge(S, Yf, Xv, Yv)
            row["r2" + tag] = r2(Yt, S.predict(Xt, W, my))
            row["lam" + tag] = lam
            for name, m in con.items():
                row[f"cos_{name}{tag}"] = cos(mean_diff(Xt, m) @ W, mean_diff(Yt, m))
            if tag == "" and save:
                kept[f"W.{j}"], kept[f"mean_src.{j}"], kept[f"mean_dst.{j}"] = W.float(), S.mean.float(), my.float()
                kept[f"source.{j}"] = torch.tensor(i)
        rows.append(row)
        print(f"{src_name}->{dst_name} state {j}: from {i_best} (cka {row['cka']:.2f}) r2 {row['r2']:.3f} "
              f"depth {row['r2_depth']:.3f} emb {row['r2_emb']:.3f} cos late {row['cos_late']:.2f} "
              f"digit {row['cos_digit']:.2f}", flush=True)
    # shuffle control on the middle target state, from its best source
    j = md["n_states"] // 2
    i = rows[j - 1]["source_best"]
    S, Xv, Xt = source(i)
    Y = load_state(dst_name, j)
    perm = torch.randperm(int(fit.sum()), generator=torch.Generator().manual_seed(2026))
    W, my, _, _ = fit_ridge(S, Y[fit][perm], Xv, Y[val])
    shuffle = r2(Y[test], S.predict(Xt, W, my))
    if save:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in kept.items()}, str(STORE / f"maps_{src_name}__{dst_name}.safetensors"),
                  metadata={"provenance": "act_map_ridge", "source": src_name, "target": dst_name})
    return {"rows": rows, "shuffle_r2": shuffle, "verdict": verdict(rows, shuffle)}


def fit(a):
    ma = json.loads((STORE / a.a / "meta.json").read_text(encoding="utf-8"))
    mb = json.loads((STORE / a.b / "meta.json").read_text(encoding="utf-8"))
    if ma["ids_sha256"] != mb["ids_sha256"]:
        raise ValueError("the two captures do not hold the same token ids (different tokenizer or texts): "
                         "this version needs identical tokenization")
    masks = split_texts(ma["text_of"], a.n_test)
    fit_idx = np.flatnonzero(masks[0])
    sub = fit_idx[np.linspace(0, len(fit_idx) - 1, min(a.n_cka, len(fit_idx))).astype(int)]
    GA = [gram(load_state(a.a, i, sub)) for i in range(ma["n_states"])]
    GB = [gram(load_state(a.b, j, sub)) for j in range(mb["n_states"])]
    cka = [[cka_from_grams(K, L) for L in GB] for K in GA]
    del GA, GB
    rec = {"a": a.a, "b": a.b, "tokens": {"fit": int(masks[0].sum()), "val": int(masks[1].sum()),
                                           "test": int(masks[2].sum())}, "cka_a_rows_b_cols": cka}
    rec[f"{a.a}->{a.b}"] = one_direction(a.a, a.b, ma, mb, masks, cka, a.save_maps)
    rec[f"{a.b}->{a.a}"] = one_direction(a.b, a.a, mb, ma, masks, [list(c) for c in zip(*cka)], a.save_maps)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{a.a}__{a.b}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for k in (f"{a.a}->{a.b}", f"{a.b}->{a.a}"):
        print(k, json.dumps(rec[k]["verdict"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--texts", default=None, help='jsonl of {"text": ...}; default: GSM8K-train rows 1000..')
    p.add_argument("--n-texts", type=int, default=N_TEXTS)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p = sub.add_parser("fit")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--n-cka", type=int, default=N_CKA)
    p.add_argument("--save-maps", action="store_true")
    a = ap.parse_args()
    {"capture": capture, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()
