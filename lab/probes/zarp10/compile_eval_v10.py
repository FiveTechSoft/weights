"""V10: coprocessor trained over a wide numeral range; OOD test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from compile_eval import MODEL, SEED, batches, encode, prose_nll, file_hash
from compile_eval_v2 import zarp, arithmetic, evaluate, digit_ids
from compile_eval_v6 import (Circuit, Extended, collect, prose_features, fit_trigger,
                             digest, load_base, DOSES)
from probe_v5 import fit, apply, targets, score

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="lab/runs/zarp10_v10")
    p.add_argument("--revision")
    p.add_argument("--prose", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    rev = args.revision or HfApi().model_info(MODEL).sha
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = load_base(rev)
    original_hash = digest(base)
    text = Path(args.prose).read_text(encoding="utf-8")
    rng = np.random.default_rng(SEED + 100)
    idxs = np.random.default_rng(SEED).permutation(10000)
    train_pairs = [(int(i) // 100, int(i) % 100) for i in idxs[:600]]
    dev_pairs = [(int(i) // 100, int(i) % 100) for i in idxs[600:800]]
    in_dist_sealed = [(int(i) // 100, int(i) % 100) for i in idxs[800:1000]]
    ood_ids = np.random.default_rng(SEED + 8).choice(10000, 200, replace=False)
    ood_sealed = [(100 + int(i) // 100, 100 + int(i) % 100) for i in ood_ids]
    # Guard: ranges must not overlap OOD operands.
    assert all(a < 100 and b < 100 for a, b in train_pairs + dev_pairs + in_dist_sealed)
    assert all(a >= 100 for a, b in ood_sealed)
    train, dev = zarp(train_pairs), zarp(dev_pairs)
    # in-distribution sealed metrics from separate fresh pairs.
    X, N = collect(base, tok, train)
    _, A = collect(base, tok, arithmetic("Calculate: "), neg=True)
    N = torch.cat([N, A, prose_features(base, tok, text[:8192])])
    Y = targets(list(map(tuple, train_pairs)))
    wa, ba = fit(X, torch.nn.functional.one_hot(Y[:, 0], 10))
    wb, bb = fit(X, torch.nn.functional.one_hot(Y[:, 1], 2))
    wt, bt = fit_trigger(X, N)
    rule = torch.zeros(20, 10)
    for a in range(10):
        for b in range(2):
            rule[2 * a + b, (3 * a + 5 * b + 7) % 10] = 1
    state = dict(wa=wa.float(), ba=ba.float(), wb=wb.float(), bb=bb.float(), wt=wt, bt=bt, rule=rule)
    circuit = Circuit(state)
    digits = digit_ids(tok)
    wrapper = Extended(base, circuit, digits, 0).eval()
    probes = [p for p, _ in dev[:16]]
    enc = encode(tok, probes, base)
    parity = float((base(**enc, use_cache=False, logits_to_keep=1).logits - wrapper(**enc, use_cache=False, logits_to_keep=1).logits).abs().max())
    print("ZERO_PARITY", parity, flush=True)
    dev_results = []
    for dose in DOSES:
        wrapper.dose = dose
        row = dict(dose=dose, skill=evaluate(wrapper, tok, dev)["accuracy"],
                   arithmetic=evaluate(wrapper, tok, arithmetic("Compute: "))["accuracy"],
                   prose_nll=prose_nll(wrapper, tok, text[8192:12288]))
        baseline = dev_results[0] if dev_results else row
        row["feasible"] = row["arithmetic"] >= baseline["arithmetic"] - .02 and row["prose_nll"] <= baseline["prose_nll"] + .1
        dev_results.append(row); print("DEV", json.dumps(row), flush=True)
    dose = max((r for r in dev_results if r["feasible"]), key=lambda r: (r["skill"], -r["dose"]))["dose"]
    def metrics(m, sealed_items):
        return dict(sealed=evaluate(m, tok, sealed_items), arithmetic=evaluate(m, tok, arithmetic()),
                    prose_nll=prose_nll(m, tok, text[-4096:]))
    report = dict(model=MODEL, revision=rev, provenance="calibrated_interface_wide_range_neural_skip",
        backward_calls=0, base_sha256=original_hash, zero_parity=parity, dose=dose, dev=dev_results,
        arms={"base": dict(in_dist=metrics(base, zarp(in_dist_sealed)), ood=metrics(base, zarp(ood_sealed)))})
    def diagnostics(pairs):
        E, _ = collect(base, tok, zarp(pairs))
        pred = torch.stack([(E @ state["wa"] + state["ba"]).argmax(-1), (E @ state["wb"] + state["bb"]).argmax(-1)], -1)
        ye = targets(pairs)
        res = circuit(E)
        _, ordinary = collect(base, tok, arithmetic(), neg=True)
        look = [(f"OTHER10({a},{b}) = ", "") for a, b in pairs]
        look_h, _ = collect(base, tok, look)
        return dict(operands=score(pred, ye), internal_accuracy=float((res.argmax(-1) == (3 * ye[:, 0] + 5 * ye[:, 1] + 7) % 10).float().mean()),
            positive_trigger_mean=float(circuit.trigger(E).mean()),
            negative_trigger_mean=float(circuit.trigger(prose_features(base, tok, text[-4096:])).mean()),
            ordinary_trigger_mean=float(circuit.trigger(ordinary).mean()),
            lookalike_trigger_mean=float(circuit.trigger(look_h).mean()))
    report["diagnostics"] = {
        "in_dist_sealed": diagnostics(in_dist_sealed),
        "ood_sealed": diagnostics(ood_sealed)}
    print("DIAGNOSTICS", json.dumps(report["diagnostics"]), flush=True)
    intact = digest(base) == original_hash
    del wrapper, base
    for arm, r in (("candidate", rule), ("placebo", rule.roll(1, dims=1))):
        path = out / (arm + ".safetensors")
        save_file({**state, "rule": r.contiguous()}, str(path), metadata={"base": MODEL, "revision": rev, "dose": str(dose)})
        reloaded = load_base(rev)
        intact = intact and digest(reloaded) == original_hash
        mod = Extended(reloaded, Circuit(load_file(str(path))), digits, dose).eval()
        mi = dict(in_dist=metrics(mod, zarp(in_dist_sealed)), ood=metrics(mod, zarp(ood_sealed)))
        report["arms"][arm] = dict(**mi, sha256=file_hash(path))
        intact = intact and digest(reloaded) == original_hash
        del mod, reloaded
    b, c, p_ = (report["arms"][k] for k in ("base", "candidate", "placebo"))
    acc = lambda a, s, arm=None: a[s]["accuracy"] if arm is None else a[arm][s]["accuracy"]
    def arms_for(arm):
        # unwrap nested arm dict
        a = report["arms"][arm]
        return a
    def skill_of(a):
        return a["ood"]["sealed"]["accuracy"]
    def in_dist_skill(a):
        return a["in_dist"]["sealed"]["accuracy"]
    report["gates"] = dict(
        in_dist_sealed=in_dist_skill(c) >= .95,
        ood_sealed=__import__("numpy").float(skill_of(c)) >= .95,
        beats_base_in_dist=in_dist_skill(c) - in_dist_skill(b) >= .2,
        beats_placebo_in_dist=in_dist_skill(c) - in_dist_skill(p_) >= .2,
        beats_base_ood=skill_of(c) - skill_of(b) >= .2,
        beats_placebo_ood=skill_of(c) - skill_of(p_) >= .2,
        in_dist_arithmetic_retention=c["in_dist"]["arithmetic"]["accuracy"] >= b["in_dist"]["arithmetic"]["accuracy"] - .02,
        prose_retention=c["in_dist"]["prose_nll"] <= b["in_dist"]["prose_nll"] + .1,
        nonzero_edit=dose > 0,
        zero_parity=parity == 0,
        original_weights_intact=intact)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("SUMMARY", json.dumps(dict(dose=dose, gates=report["gates"],
        arms={k: {"in_dist": in_dist_skill(v), "ood": skill_of(v)} for k, v in report["arms"].items()},
        diagnostics=report["diagnostics"])), flush=True)

if __name__ == "__main__":
    main()
