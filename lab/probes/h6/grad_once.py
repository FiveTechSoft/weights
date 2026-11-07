"""H6e, synthetic update (h6_plan.md stage 2 + addendum c): one summed gradient at M0, no optimizer run.

G = gradient of the training loss at M0 over the first 512 training examples in the seed's order (the examples its
first ~13 steps saw). The update is built from -G and rescaled globally to alpha * N, N = mean |ΔW| of the other two
seeds' trainings (not this seed's). alpha ∈ {0.25, 0.5, 1, 2, 4} chosen on dev, then the full evaluation.
Arms at r (= r* from stage 1):
  G_r      rank-r truncation of -G per matrix
  GX_r     -G projected on the lane of the other two seeds (top-r of [ΔW_t | ΔW_u])
  G_full   -G as is (reported, no threshold)
  SX_r     sign(-G) projected on the same lane (Adam-like geometry; reported, no threshold)
McNemar against M3_r from lab/runs/h6/transfer/seed<s>.json. backward_calls = 512 examples.
G itself is kept: hf_cache/h6/seed<s>/grad512.safetensors.

usage: python lab/probes/h6/grad_once.py --seed 1 --r 102
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "lab" / "runs" / "h6"
KEEP = ROOT / "hf_cache" / "h6"
sys.path.insert(0, str(HERE))

N_EXAMPLES, MICRO = 512, 10
ALPHAS = (0.25, 0.5, 1, 2, 4)


def global_rescale(d, norm):
    tot = sum(float(v.double().pow(2).sum()) for v in d.values()) ** 0.5
    return {n: v * (norm / tot) for n, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    import eval_arith as ea
    import make_data as md
    import synth
    import train
    import transfer as tr

    out_dir = RUNS / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = ea.load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.to(torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    named = train.trained_params(model)
    names = sorted(n for n, _ in named)
    params = dict(named)
    base = {n: params[n].detach().float().cpu().clone() for n in names}
    t0 = time.time()

    gpath = KEEP / f"seed{a.seed}" / "grad512.safetensors"
    if gpath.exists():
        G = load_file(str(gpath))
    else:
        for _, p in named:
            p.requires_grad_(True)
        rows = ea.load_split("train")
        order = list(range(len(rows)))
        random.Random(a.seed).shuffle(order)                 # same order as train.py
        idx = order[:N_EXAMPLES]
        model.train()
        ntok = 0
        for k0 in range(0, len(idx), MICRO):
            enc = [train.encode_pair(tok.encode(md.prompt(rows[i]["a"], rows[i]["b"]), add_special_tokens=False).ids,
                                     tok.encode(md.target(rows[i]["a"], rows[i]["b"]), add_special_tokens=False).ids)
                   for i in idx[k0:k0 + MICRO]]
            L = max(len(x) for x, _ in enc)
            ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in enc]).cuda()
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in enc]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids).last_hidden_state[:, :-1]
                mask = lab[:, 1:] != -100
                logits = model.lm_head(hidden[mask])
            loss = torch.nn.functional.cross_entropy(logits.float(), lab[:, 1:][mask], reduction="sum")
            loss.backward()
            ntok += int(mask.sum())
            del logits, hidden
        G = {n: (params[n].grad / ntok).detach().cpu().contiguous() for n in names}
        for _, p in named:
            p.grad = None
            p.requires_grad_(False)
        save_file(G, str(gpath), metadata={"provenance": "h6_grad_once", "seed": str(a.seed),
                                            "examples": str(N_EXAMPLES), "reduction": "mean over target tokens"})
        print(f"gradient {time.time() - t0:.0f}s", flush=True)
    model.eval()

    lane, N = tr.lane_bases(base, a.seed, a.r)
    r = a.r

    def proj(M, U):
        return M if min(M.shape) <= r else tr.project(M, U[:, :r])

    upd = {
        "G_full": {n: -G[n] for n in names},
        "G_r": {n: (-G[n] if min(G[n].shape) <= r else tr.project(-G[n], tr.top_basis(G[n], r, "cuda").cpu()))
                for n in names},
        "GX_r": {n: proj(-G[n], lane[n]) for n in names},
        "SX_r": {n: proj(torch.sign(-G[n]), lane[n]) for n in names},
    }
    m0 = json.loads((RUNS / f"seed{a.seed}" / "chunk0.json").read_text(encoding="utf-8"))["eval"]
    m1 = json.loads((RUNS / f"seed{a.seed}" / "chunk10.json").read_text(encoding="utf-8"))["eval"]
    a0, a1 = m0["test_id"]["acc"], m1["test_id"]["acc"]
    st1 = json.loads((RUNS / "transfer" / f"seed{a.seed}.json").read_text(encoding="utf-8"))
    m3 = st1["arms"][f"M3_{r}"]
    rec = {"seed": a.seed, "r": r, "N_other": N, "backward_examples": N_EXAMPLES, "optimizer_steps": 0,
           "M0": a0, "M1": a1, "M3": {k: v for k, v in m3.items() if k != "ok_mask"}, "arms": {}}
    outp = out_dir / f"seed{a.seed}.json"
    dev = ea.load_split("dev")

    @torch.no_grad()
    def set_upd(u, alpha):
        for n, t in global_rescale(u, alpha * N).items():
            params[n].copy_(base[n].cuda() + t.cuda())

    for arm, u in upd.items():
        dev_acc = {}
        for al in ALPHAS:
            set_upd(u, al)
            dev_acc[al] = ea.eval_split(model, tok, dev)["acc"]
        al = synth.pick_alpha(dev_acc)
        set_upd(u, al)
        ev = ea.evaluate(model, tok)
        rec["arms"][arm] = {"dev_acc": dev_acc, "alpha": al, "test_id": ev["test_id"]["acc"],
                            "test_id_independent": ev["test_id"]["acc_independent"],
                            "ood_easy": ev["ood_easy"]["acc"], "ood_hard": ev["ood_hard"]["acc"],
                            "retention_nll": ev["retention_nll"],
                            "recovery": synth.recovery(ev["test_id"]["acc"], a0, a1),
                            "mcnemar_vs_M3": synth.mcnemar(ev["test_id"]["ok_mask"], m3["ok_mask"]),
                            "ok_mask": ev["test_id"]["ok_mask"]}
        outp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        x = rec["arms"][arm]
        print(f"{arm} alpha={al} dev={dev_acc} acc={x['test_id']:.3f} R={x['recovery']:.2f} "
              f"p={x['mcnemar_vs_M3']:.2g} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
