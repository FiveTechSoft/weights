"""Gate 2 of cross-model porting (PLAN.md, addendum "Gate 2"): can the target model USE the mapped states?

The source's hidden state i (from the gate 1 capture) goes through the gate 1 map into the target's residual stream
at hidden state j (the output of target layer j-1), at every position but the first; the target finishes the
forward. Scored by next-token NLL on the 40 gate 1 test texts, against controls (token identity only, shuffled
map, mean state) and a mechanics check (the target's own state through the same hook must give the native NLL).

usage:
  python lab/probes/act_map/stitch.py run --src q08 --dst q9        # 0.8B -> 9B (the direction PLAN named)
  python lab/probes/act_map/stitch.py run --src q9 --dst q08        # 9B -> 0.8B (the one gate 3 needs)
Outputs: lab/runs/act_map/stitch_<src>__<dst>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_map as am  # noqa: E402

DEPTHS = {"q9": (8, 16, 24), "q08": (6, 12, 18)}        # 1/4, 1/2, 3/4 of the hidden states (fixed in PLAN)
G_MIN, SELF_TOL, BOOT_REPS, BOOT_SEED = 0.5, 0.01, 2000, 2026


# ---------- pieces with tests ----------

def gap_recovered(nll_emb, nll_stitch, nll_native):
    return (nll_emb - nll_stitch) / (nll_emb - nll_native)


def boot_diff(a_sum, b_sum, n, reps=BOOT_REPS, seed=BOOT_SEED):
    """95 % interval of the per-token mean of (a - b), resampling whole texts."""
    a_sum, b_sum, n = (np.asarray(v, dtype=float) for v in (a_sum, b_sum, n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    stat = (a_sum[idx].sum(1) - b_sum[idx].sum(1)) / n[idx].sum(1)
    return float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))


def verdict(rows):
    ok = [r["G"] >= G_MIN and r["diff_lo"] > 0 and r["nll_stitch"] < r["nll_shuffle"] and r["nll_stitch"] < r["nll_mean"]
          for r in rows]
    return {"pass": sum(ok) >= 2, "depths_ok": ok}


def _core(model):
    core = model.model
    return core.language_model if hasattr(core, "language_model") else core


@torch.no_grad()
def run_arm(model, ids, j, states, device, batch=8, ref_top1=None, stream=False):
    """Next-token NLL of the model on texts `ids`, with hidden state j replaced at positions >= 1 by states[t]
    (shape len(ids[t]) - 1 x d) when j is given. Scored positions: 1 .. n-2 (predicting tokens 2 .. n-1).
    -> {"nll_sum", "n", "top1", "top1_same"} per text."""
    if j is not None and any(s.shape[0] != len(x) - 1 for s, x in zip(states, ids)):
        raise ValueError("each text needs one state per position 1 .. n-1")
    core = _core(model)
    head = model.get_output_embeddings()
    hooks = []
    if stream:
        from engine import contrast_vector as cv
        hooks += cv._stream(core.layers, device)
    cur = {}

    def put(mod, args, kwargs, out):
        h = (out[0] if isinstance(out, tuple) else out).clone()
        for b, (t, n) in enumerate(cur["rows"]):
            h[b, 1:n] = states[t].to(device=h.device, dtype=h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    if j is not None:
        hooks.append(core.layers[j - 1].register_forward_hook(put, with_kwargs=True))
    res = {"nll_sum": [], "n": [], "top1": [], "top1_same": []}
    try:
        for k in range(0, len(ids), batch):
            chunk = ids[k:k + batch]
            cur["rows"] = [(k + b, len(c)) for b, c in enumerate(chunk)]
            n = max(len(c) for c in chunk)
            x = torch.tensor([c + [0] * (n - len(c)) for c in chunk])     # right pad: causal, pads unseen
            if not stream:
                x = x.to(device)
            last = core(input_ids=x, use_cache=False).last_hidden_state
            for b, c in enumerate(chunk):
                hh = last[b, 1:len(c) - 1].to(head.weight.device, head.weight.dtype)
                lp = torch.log_softmax(head(hh).float(), -1)
                tgt = torch.tensor(c[2:], device=lp.device)
                res["nll_sum"].append(float(-lp.gather(1, tgt[:, None]).sum()))
                res["n"].append(len(c) - 2)
                top = lp.argmax(-1).cpu()
                res["top1"].append(top)
                res["top1_same"].append(int((top == ref_top1[k + b]).sum()) if ref_top1 is not None else len(c) - 2)
    finally:
        for h in hooks:
            h.remove()
    return res


# ---------- run ----------

def texts_ids(meta, mdir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(mdir) / "tokenizer.json"))
    n_texts = max(meta["text_of"]) + 1
    ids = [tok.encode(t, add_special_tokens=False).ids[:am.MAX_TOKENS] for t in am.default_texts(n_texts)]
    if am.ids_hash([t for c in ids for t in c[1:]]) != meta["ids_sha256"]:
        raise ValueError("re-tokenized texts differ from the gate 1 capture")
    return ids


def per_text(Y, test_texts, ids):
    out, r = [], 0
    for t in test_texts:
        m = len(ids[t]) - 1
        out.append(Y[r:r + m].float())
        r += m
    if r != Y.shape[0]:
        raise ValueError("rows do not match the test texts")
    return out


def load_model(mdir, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.bfloat16).eval()
    stream = device != "cpu" and sum(p.numel() for p in m.parameters()) > 3e9
    if not stream:
        m.to(device)
    return m, stream


def summ(r):
    return float(np.sum(r["nll_sum"]) / np.sum(r["n"]))


def run(a):
    from safetensors.torch import load_file
    ms = json.loads((am.STORE / a.src / "meta.json").read_text(encoding="utf-8"))
    md = json.loads((am.STORE / a.dst / "meta.json").read_text(encoding="utf-8"))
    if ms["ids_sha256"] != md["ids_sha256"]:
        raise ValueError("source and target captures hold different token ids")
    ids = texts_ids(md, md["model"])
    fit, val, test = am.split_texts(md["text_of"], am.N_TEST)
    text_of = np.asarray(md["text_of"])
    test_texts = [int(t) for t in np.unique(text_of[test])]
    tids = [ids[t] for t in test_texts]
    maps = load_file(str(am.STORE / f"maps_{a.src}__{a.dst}.safetensors"))
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # every arm's states first (CPU), then one model at a time on the GPU
    arms = {}
    E = am.load_state(a.src, 0)
    emb_src, Xe_val, Xe_test = am.Source(E[fit]), E[val], E[test]
    del E
    for j in DEPTHS[a.dst]:
        Y = am.load_state(a.dst, j)
        i = int(maps[f"source.{j}"])
        X = am.load_state(a.src, i)
        W, mx, my = maps[f"W.{j}"].double(), maps[f"mean_src.{j}"].double(), maps[f"mean_dst.{j}"].double()
        st = {"stitch": (X[test].double() - mx) @ W + my}
        We, mye, lam_e, _ = am.fit_ridge(emb_src, Y[fit], Xe_val, Y[val])
        st["emb"] = emb_src.predict(Xe_test, We, mye)
        S = am.Source(X[fit])
        perm = torch.randperm(int(fit.sum()), generator=torch.Generator().manual_seed(2026))
        Ws, mys, _, _ = am.fit_ridge(S, Y[fit][perm], X[val], Y[val])
        st["shuffle"] = S.predict(X[test], Ws, mys)
        st["mean"] = Y[fit].double().mean(0).expand(int(test.sum()), -1)
        st["self"] = Y[test]
        arms[j] = {"source": i, "lam_emb": lam_e, "states": {k: per_text(v, test_texts, ids) for k, v in st.items()}}
        del S, X, Y
        print(f"[stitch {a.src}->{a.dst}] states for j={j} (source {i}) ready {time.time() - t0:.0f}s", flush=True)
    del emb_src

    model, stream = load_model(md["model"], device)
    nat = run_arm(model, tids, None, None, device, a.batch, stream=stream)
    out = {"src": a.src, "dst": a.dst, "test_texts": len(tids), "scored_tokens": int(np.sum(nat["n"])),
           "nll_native": summ(nat)}
    print(f"[stitch] native {out['nll_native']:.4f} ({out['scored_tokens']} tokens)", flush=True)
    rows = []
    for j, info in arms.items():
        row = {"j": j, "source": info["source"], "lam_emb": info["lam_emb"]}
        res = {}
        for k, states in info["states"].items():
            r = run_arm(model, tids, j, states, device, a.batch, ref_top1=nat["top1"], stream=stream)
            res[k] = r
            row[f"nll_{k}"] = summ(r)
            row[f"top1_{k}"] = float(np.sum(r["top1_same"]) / np.sum(r["n"]))
            print(f"[stitch] j={j} {k}: nll {row[f'nll_{k}']:.4f} top1 {row[f'top1_{k}']:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        row["self_ok"] = abs(row["nll_self"] - out["nll_native"]) <= SELF_TOL
        row["G"] = gap_recovered(row["nll_emb"], row["nll_stitch"], out["nll_native"])
        row["diff_lo"], row["diff_hi"] = boot_diff(res["emb"]["nll_sum"], res["stitch"]["nll_sum"], nat["n"])
        rows.append(row)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sm, sstream = load_model(ms["model"], device)
    out["nll_source_alone"] = summ(run_arm(sm, tids, None, None, device, a.batch, stream=sstream))
    out["depths"] = rows
    mech = all(r["self_ok"] for r in rows)
    out["verdict"] = verdict(rows) if mech else {"pass": None, "mechanics_fail": True}
    out["seconds"] = round(time.time() - t0)
    am.RUNS.mkdir(parents=True, exist_ok=True)
    (am.RUNS / f"stitch_{a.src}__{a.dst}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "depths"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
