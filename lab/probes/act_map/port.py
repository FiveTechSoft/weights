"""Gate 3 of cross-model porting (PLAN.md, addendum "Gate 3"): does the 9B's "happy" direction, written into the 0.8B
through the gate 1 maps, steer the 0.8B about as well as the 0.8B's own direction?

usage:
  python lab/probes/act_map/port.py vectors     # D per hidden state in both models (same 70 pair texts of results
                                                # 43.13), the ported and embport differences, step 0 cosines, and the
                                                # four llama.cpp control vectors for the 0.8B (own, port, embport,
                                                # placebo of port) -> hf_cache/act_map/cvec08/, lab/runs/act_map/happy08/
  python lab/probes/act_map/port.py choose <arm>   # dev scale for one arm (rule of results 43.13) -> prints it or "none"
  python lab/probes/act_map/port.py verdict        # the PLAN.md gate 3 rule, mechanically
Behavior runs go through lab/probes/happy/happy.py with HAPPY_RUNS=lab/runs/act_map/happy08 (chain script).
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "happy"))
import act_map as am  # noqa: E402

OUT = am.RUNS / "happy08"
CVEC = am.STORE / "cvec08"
PAIRS = ROOT / "lab" / "runs" / "happy" / "pairs"
ARMS = ("own", "port", "embport")
PLACEBO_SEED = 2026


# ---------- pieces with tests ----------

def port_diff(D_src, W_by_j, src_by_j, n_dst):
    """A difference of source states, per target state j: D_src[i(j)] @ W_j (the affine maps' intercepts cancel in a
    difference). Row 0 (embeddings) stays zero: llama.cpp never steers it."""
    W0 = next(iter(W_by_j.values()))
    out = torch.zeros(n_dst, W0.shape[1], dtype=torch.float64)
    for j, W in W_by_j.items():
        out[j] = D_src[src_by_j[j]].double() @ W.double()
    return out


def residual_increments(D):
    """contrast_vector.increments without the last layer: HF's last hidden state is after the final norm, so
    D[-1] - D[-2] is not a residual increment (plan amendment, gate 3). -> {N: increment}, N = 1 .. n_layers - 2."""
    from engine import contrast_vector as cv
    inc = cv.increments(D)
    inc.pop(max(inc))
    return inc


def cos_by_state(A, B):
    return [am.cos(A[j], B[j]) for j in range(1, A.shape[0])]


def arm_stats(rows, base):
    import happy
    dt = [(a["tone"] or 0) - (b["tone"] or 0) for a, b in zip(rows, base)]
    dl = [a["lex"] - b["lex"] for a, b in zip(rows, base)]
    return {"tone_diff": round(sum(dt) / len(dt), 3), "tone_sign_p": happy.sign_p(dt),
            "lex_diff": round(sum(dl) / len(dl), 3), "lex_sign_p": happy.sign_p(dl),
            "coherent": sum(bool(r["coherent"]) for r in rows), "n": len(rows)}


DEV_SLACK, TEST_SLACK = 1, 4          # amendment 2: coherence may fall at most this much below the base's own


def choose_scale(coherent_by_scale, base_dev_coherent):
    """Largest scale whose dev answers the judge calls coherent in >= base - DEV_SLACK of 10; None if none."""
    ok = [sc for sc, c in coherent_by_scale.items() if c >= base_dev_coherent - DEV_SLACK]
    return max(ok) if ok else None


def _c14(s, coh_min):
    return (s is not None and s["tone_diff"] >= 1.0 and s["tone_sign_p"] < 0.01 and s["lex_diff"] >= 2.0
            and s["lex_sign_p"] < 0.01 and s["coherent"] >= coh_min)


def decide(st, base_test_coherent):
    """st: arm -> arm_stats on test (None: the arm had no coherent dev scale). PLAN.md gate 3 (amendment 2: coherence
    relative to the base, >= base - 4 of 40, instead of >= 36)."""
    coh_min = base_test_coherent - TEST_SLACK
    out = {"coherent_min": coh_min, "own C1-C4": _c14(st.get("own"), coh_min)}
    if not out["own C1-C4"]:
        out["verdict"] = "NOT DECIDED: own fails C1-C4"
        return out
    p, o, e, pl = st.get("port"), st["own"], st.get("embport"), st.get("placebo")
    c = {"P1 tone +1.0, p<0.01": p is not None and p["tone_diff"] >= 1.0 and p["tone_sign_p"] < 0.01,
         "P2 lexicon +2/100, p<0.01": p is not None and p["lex_diff"] >= 2.0 and p["lex_sign_p"] < 0.01,
         "P3 placebo tone < +0.5": pl is not None and pl["tone_diff"] < 0.5,
         "P4 coherent >= base - 4": p is not None and p["coherent"] >= coh_min,
         "P5 port gain >= 0.5 x own gain": p is not None and p["tone_diff"] >= 0.5 * o["tone_diff"]}
    out.update(c)
    out["verdict"] = "PASS" if all(c.values()) else "FAIL"
    out["P6 port beats embport by >= 0.5"] = (p is not None and
                                               p["tone_diff"] - (e["tone_diff"] if e is not None else 0.0) >= 0.5)
    return out


# ---------- run ----------

def _means(model_dir, device):
    """Mean response-token state, happy minus neutral, per hidden state: (n_layers + 1, d)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine import contrast_vector as cv
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    P = cv._pairs(tok, cv._rows(PAIRS / "pos.jsonl"), 384)
    N = cv._pairs(tok, cv._rows(PAIRS / "neg.jsonl"), 384)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval()
    core = model.model
    layers = (core.language_model if hasattr(core, "language_model") else core).layers
    big = sum(q.numel() for q in model.parameters()) > 3e9
    hooks = cv._stream(layers, device) if (big and device != "cpu") else []
    if not big:
        model.to(device)
    try:
        if not big:                                  # layer_means feeds CPU ids; move them with a pre-hook on the core
            hooks.append(core.register_forward_pre_hook(
                lambda m, a, k: (a, {**k, "input_ids": k["input_ids"].to(device)}), with_kwargs=True))
        D = cv.layer_means(model, P) - cv.layer_means(model, N)
    finally:
        for h in hooks:
            h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return D, len(P), len(N)


def vectors():
    from safetensors.torch import load_file
    from engine import contrast_vector as cv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m9 = json.loads((am.STORE / "q9" / "meta.json").read_text(encoding="utf-8"))
    m08 = json.loads((am.STORE / "q08" / "meta.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    CVEC.mkdir(parents=True, exist_ok=True)
    if (OUT / "D.pt").exists():                      # forwards are deterministic: reuse (the 9B takes minutes)
        d = torch.load(OUT / "D.pt")
        D9, D08, n_pos, n_neg = d["D9"], d["D08"], d["pairs"][0], d["pairs"][1]
    else:
        D9, n_pos, n_neg = _means(m9["model"], device)
        D08, _, _ = _means(m08["model"], device)
        torch.save({"D9": D9, "D08": D08, "pairs": [n_pos, n_neg]}, OUT / "D.pt")
    n08 = D08.shape[0]

    maps = load_file(str(am.STORE / "maps_q9__q08.safetensors"))
    js = range(1, n08)
    Dport = port_diff(D9, {j: maps[f"W.{j}"] for j in js}, {j: int(maps[f"source.{j}"]) for j in js}, n08)

    # embport: the 9B's embedding difference through an embeddings ridge per target state (gate 2's emb rule)
    fit, val, _ = am.split_texts(m08["text_of"], am.N_TEST)
    E = am.load_state("q9", 0)
    src = am.Source(E[fit])
    We = {}
    for j in js:
        Y = am.load_state("q08", j)
        We[j] = am.fit_ridge(src, Y[fit], E[val], Y[val])[0]
    del E, src
    Demb = port_diff(D9, We, {j: 0 for j in js}, n08)

    step0 = {"cos_port_own": cos_by_state(Dport, D08), "cos_embport_own": cos_by_state(Demb, D08),
             "norm_own": [float(D08[j].norm()) for j in js], "norm_port": [float(Dport[j].norm()) for j in js],
             "norm_embport": [float(Demb[j].norm()) for j in js], "pairs": [n_pos, n_neg]}
    files = {}
    for name, D in (("own", D08), ("port", Dport), ("embport", Demb)):
        vecs = {k: v.float().numpy() for k, v in residual_increments(D.float()).items()}
        files[name] = CVEC / f"happy_{name}.gguf"
        cv.write_gguf(vecs, files[name], "qwen35")
        if name == "port":
            files["placebo"] = CVEC / f"happy_port.placebo{PLACEBO_SEED}.gguf"
            cv.write_gguf(cv.placebo(vecs, PLACEBO_SEED), files["placebo"], "qwen35")
    step0["files"] = {k: str(v) for k, v in files.items()}
    (OUT / "step0.json").write_text(json.dumps(step0, indent=1), encoding="utf-8")
    import statistics as s
    print(json.dumps({"median_cos_port_own": s.median(step0["cos_port_own"]),
                      "median_cos_embport_own": s.median(step0["cos_embport_own"])}), flush=True)


def _coherent(name):
    return sum(bool(r["coherent"]) for r in _rows(name))


def choose(arm):
    import happy
    sc = choose_scale({s: _coherent(f"dev_{arm}_s{s}") for s in happy.SCALES}, _coherent("dev_base"))
    print("none" if sc is None else sc)


def _rows(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))["rows"] if p.exists() else None


def verdict():
    base_dev = _rows("dev_base")
    base = _rows("test_base")
    st = {}
    for arm in ARMS + ("placebo",):
        rows = _rows(f"test_{arm}")
        st[arm] = arm_stats(rows, base) if rows is not None else None
    out = {"base_dev_coherent": sum(bool(r["coherent"]) for r in base_dev),
           "base_test_coherent": sum(bool(r["coherent"]) for r in base), "stats": st,
           "scales": json.loads((OUT / "scales.json").read_text()) if (OUT / "scales.json").exists() else None}
    out.update(decide(st, out["base_test_coherent"]))
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"vectors": vectors, "choose": lambda: choose(a[1]), "verdict": verdict}[a[0]]()
