"""Evaluate V10 candidate/placebo from saved checkpoints."""
import json
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from compile_eval import MODEL, SEED, prose_nll
from compile_eval_v2 import zarp, arithmetic, evaluate, digit_ids
from compile_eval_v6 import (Circuit, Extended, collect, prose_features, load_base, DOSES)
from probe_v5 import fit, targets, score

def main():
    rev = args_rev()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = load_base(rev)
    rng_idxs = np.random.default_rng(SEED).permutation(10000)
    train_pairs = [(int(i)//100, int(i)%100) for i in rng_idxs[:600]]
    dev_pairs = [(int(i)//100, int(i)%100) for i in rng_idxs[600:800]]
    in_dist_sealed = [(int(i)//100, int(i)%100) for i in rng_idxs[800:1000]]
    ood_ids = np.random.default_rng(SEED+8).choice(10000, 200, replace=False)
    ood_sealed = [(100+int(i)//100, 100+int(i)%100) for i in ood_ids]
    text = open("lab/data/shakespeare_char/input.txt").read()
    from safetensors import safe_open
    with safe_open("lab/runs/zarp10_v10/candidate.safetensors", "pt") as f:
        dose = float(f.metadata()["dose"])
    digits = digit_ids(tok)
    out = {}
    for arm in ("candidate","placebo"):
        layer_state = load_file(f"lab/runs/zarp10_v10/{arm}.safetensors")
        circuit = Circuit(layer_state)
        wrapper = Extended(base, circuit, digits, dose).eval()
        mi = dict(
            in_dist=dict(sealed=evaluate(wrapper, tok, zarp(in_dist_sealed)), arithmetic=evaluate(wrapper, tok, arithmetic()), prose_nll=prose_nll(wrapper, tok, text[-4096:])),
            ood=dict(sealed=evaluate(wrapper, tok, zarp(ood_sealed)), arithmetic=evaluate(wrapper, tok, arithmetic()), prose_nll=prose_nll(wrapper, tok, text[-4096:])),
            dose=dose)
        out[arm] = mi
        print(arm.upper(), json.dumps(mi, indent=1), flush=True)
        del wrapper
    # base
    base_ood = dict(sealed=evaluate(base, tok, zarp(ood_sealed)), arithmetic=evaluate(base, tok, arithmetic()), prose_nll=prose_nll(base, tok, text[-4096:]))
    base_in = dict(sealed=evaluate(base, tok, zarp(in_dist_sealed)), arithmetic=evaluate(base, tok, arithmetic()), prose_nll=prose_nll(base, tok, text[-4096:]))
    out["base"] = {"in_dist": base_in, "ood": base_ood}
    print("BASE", json.dumps(out["base"], indent=1), flush=True)
    print("SUMMARY", json.dumps({k: {s: (v["in_dist"] if isinstance(v, dict) else v) for s in ()} for k, v in out.items()}) if False else "")
    (out_path := "lab/runs/zarp10_v10/arms_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

def args_rev():
    # we don't need the revision to load; load_base with no revision
    return HfApi().model_info(MODEL).sha

if __name__ == "__main__":
    main()
