"""H6f (h6_plan.md addendum d): the per-direction amounts inside the lane (X basis, r = 20, other two seeds).

part a, shaped one-shot update (no optimizer steps; the 512-example gradient of H6e, hf_cache/h6/seed<s>/grad512):
  per matrix d_n = -G_n (GXm) or sign(-G_n) (SXm), projected on the lane, rescaled to alpha * nu_n,
  nu_n = mean over the other two seeds of |ΔW_t,n projected on the lane|. alpha chosen on dev.
part b, train only the lane coefficients: ΔW_n = U_n C_n on the smaller side, C_n trainable (init 0), AdamW on C,
  the full-weight gradient projected (dL/dC = U^T dL/dW), same data order as the seed. --basis random: control with a
  random orthonormal 20-dim basis per matrix. --probe: lr probe (dev only, schedule over --steps).

usage:
  python lab/probes/h6/h6f.py a --seed 1
  python lab/probes/h6/h6f.py b --seed 1 --lr 1e-5 --steps 50 --probe
  python lab/probes/h6/h6f.py b --seed 1 --lr <chosen> --steps 200 [--basis random]
"""
import argparse
import json
import math
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
import fingerprint as fp  # noqa: E402
import transfer as tr  # noqa: E402

R = 20
ALPHAS_A = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
EVAL_AT = (50, 100, 200)


# ---------- pieces with tests ----------

def shaped_update(G, lane, nu, alpha, use_sign):
    out = {}
    for n, g in G.items():
        d = torch.sign(-g) if use_sign else -g
        out[n] = tr.rescale_to(tr.project(d, lane[n]), alpha * nu[n])
    return out


def lane_norm(Ds, U):
    return sum(float(tr.project(D, U).norm()) for D in Ds) / len(Ds)


def coef_shape(shape, r):
    return (r, shape[1]) if fp.small_side(shape) == 0 else (shape[0], r)


def coef_apply(U, C, shape):
    U = U.to(C.device, C.dtype)
    return U @ C if fp.small_side(shape) == 0 else C @ U.T


def coef_grad(gW, U, shape):
    U = U.to(gW.device, gW.dtype)
    return U.T @ gW if fp.small_side(shape) == 0 else gW @ U


def lr_at(step, total, lr):
    warm = max(1, total // 10)
    if step < warm:
        return lr * (step + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


# ---------- shared ----------

def lane_and_nu(base, seed, r, device="cuda"):
    from safetensors.torch import load_file
    t, u = tr.other_seeds(seed)
    wa = load_file(str(KEEP / f"seed{t}" / "chunk10.safetensors"))
    wb = load_file(str(KEEP / f"seed{u}" / "chunk10.safetensors"))
    lane, nu = {}, {}
    for n in sorted(base):
        b = base[n].to(device)
        Da, Db = wa[n].to(device) - b, wb[n].to(device) - b
        lane[n] = tr.lane_basis(Da, Db, min(r, min(Da.shape)), device)
        nu[n] = lane_norm([Da, Db], lane[n])
    del wa, wb
    return lane, nu


def load_model(train_mode=False):
    from transformers import AutoModelForCausalLM
    import eval_arith as ea
    import train
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.embed_tokens.to(torch.bfloat16)
    if train_mode:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    named = train.trained_params(model)
    return model, dict(named)


def ref_evals(seed):
    m0 = json.loads((RUNS / f"seed{seed}" / "chunk0.json").read_text(encoding="utf-8"))["eval"]
    m1 = json.loads((RUNS / f"seed{seed}" / "chunk10.json").read_text(encoding="utf-8"))["eval"]
    m3 = json.loads((RUNS / "transfer" / f"seed{seed}.json").read_text(encoding="utf-8"))["arms"][f"M3_{R}"]
    return m0["test_id"]["acc"], m1["test_id"]["acc"], m3


def summary(ev, a0, a1, m3):
    import synth
    return {"test_id": ev["test_id"]["acc"], "test_id_independent": ev["test_id"]["acc_independent"],
            "ood_easy": ev["ood_easy"]["acc"], "ood_hard": ev["ood_hard"]["acc"],
            "retention_nll": ev["retention_nll"], "recovery": synth.recovery(ev["test_id"]["acc"], a0, a1),
            "mcnemar_vs_M3_20": synth.mcnemar(ev["test_id"]["ok_mask"], m3["ok_mask"]),
            "ok_mask": ev["test_id"]["ok_mask"]}


# ---------- part a ----------

def part_a(seed):
    from safetensors.torch import load_file
    import eval_arith as ea
    import synth
    out = RUNS / "h6f"
    out.mkdir(parents=True, exist_ok=True)
    tok = ea.load_tokenizer()
    model, params = load_model()
    names = sorted(params)
    base = {n: params[n].detach().float().cpu().clone() for n in names}
    G = load_file(str(KEEP / f"seed{seed}" / "grad512.safetensors"))
    t0 = time.time()
    lane, nu = lane_and_nu(base, seed, R)
    a0, a1, m3 = ref_evals(seed)
    rec = {"seed": seed, "r": R, "part": "a", "optimizer_steps": 0, "backward_examples": 512, "arms": {}}
    dev = ea.load_split("dev")

    @torch.no_grad()
    def set_arm(alpha, use_sign):
        for n in names:
            d = shaped_update({n: G[n].cuda()}, {n: lane[n].cuda()}, nu, alpha, use_sign)[n]
            params[n].copy_(base[n].cuda() + d)

    for arm, use_sign in (("GXm", False), ("SXm", True)):
        dev_acc = {}
        for al in ALPHAS_A:
            set_arm(al, use_sign)
            dev_acc[al] = ea.eval_split(model, tok, dev)["acc"]
            print(f"{arm} alpha {al} dev {dev_acc[al]:.3f} {time.time() - t0:.0f}s", flush=True)
        al = synth.pick_alpha(dev_acc)
        set_arm(al, use_sign)
        rec["arms"][arm] = {"dev_acc": dev_acc, "alpha": al} | summary(ea.evaluate(model, tok), a0, a1, m3)
        (out / f"a_seed{seed}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        x = rec["arms"][arm]
        print(f"{arm} alpha={al} acc={x['test_id']:.3f} R={x['recovery']:.2f} p={x['mcnemar_vs_M3_20']:.2g}",
              flush=True)


# ---------- part b ----------

def part_b(seed, lr, steps, basis, probe):
    from safetensors.torch import save_file
    import eval_arith as ea
    import make_data as md
    import train
    out = RUNS / "h6f"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"b_{basis}_seed{seed}" + (f"_probe_lr{lr:g}" if probe else "")
    torch.manual_seed(seed)
    tok = ea.load_tokenizer()
    model, params = load_model(train_mode=True)
    names = sorted(params)
    W0 = {n: params[n].detach().clone() for n in names}                     # on GPU, fp32
    t0 = time.time()
    if basis == "lane":
        U, _ = lane_and_nu({n: W0[n].cpu() for n in names}, seed, R)
    else:
        U = {n: tr.random_basis(min(params[n].shape), min(R, min(params[n].shape)), seed=20_000 * seed + i)
             for i, n in enumerate(names)}
    U = {n: u.cuda().float() for n, u in U.items()}
    C = {n: torch.nn.Parameter(torch.zeros(coef_shape(tuple(params[n].shape), U[n].shape[1]), device="cuda"))
         for n in names}
    n_coef = sum(c.numel() for c in C.values())
    opt = train.AdamW([C[n] for n in names])
    for n in names:
        params[n].requires_grad_(True)
    a0, a1, m3 = ref_evals(seed)
    rows = ea.load_split("train")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    enc = {}
    rec = {"seed": seed, "part": "b", "basis": basis, "r": R, "lr": lr, "steps": steps, "probe": probe,
           "n_coef": n_coef, "n_trained_full": sum(p.numel() for p in params.values()), "evals": {}}
    log = (out / f"{tag}.steps.jsonl").open("w", encoding="utf-8")
    for step in range(steps):
        idx = order[(step * train.BATCH) % len(order):(step * train.BATCH) % len(order) + train.BATCH]
        for i in idx:
            if i not in enc:
                enc[i] = train.encode_pair(
                    tok.encode(md.prompt(rows[i]["a"], rows[i]["b"]), add_special_tokens=False).ids,
                    tok.encode(md.target(rows[i]["a"], rows[i]["b"]), add_special_tokens=False).ids)
        batch = [enc[i] for i in idx]
        L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in batch]).cuda()
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()
        ntgt = int((lab[:, 1:] != -100).sum())
        model.train()
        loss_sum = 0.0
        for k0 in range(0, len(idx), train.MICRO):
            ids_k, lab_k = ids[k0:k0 + train.MICRO], lab[k0:k0 + train.MICRO]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids_k).last_hidden_state[:, :-1]
                mask = lab_k[:, 1:] != -100
                logits = model.lm_head(hidden[mask])
            l_k = torch.nn.functional.cross_entropy(logits.float(), lab_k[:, 1:][mask], reduction="sum")
            (l_k / ntgt).backward()
            loss_sum += float(l_k.detach())
            del logits, hidden
        with torch.no_grad():
            for n in names:
                C[n].grad = coef_grad(params[n].grad, U[n], tuple(params[n].shape))
                params[n].grad = None
        opt.step(lr_at(step, steps, lr))
        with torch.no_grad():
            for n in names:
                params[n].copy_(W0[n] + coef_apply(U[n], C[n], tuple(params[n].shape)))
        log.write(json.dumps({"step": step + 1, "lr": lr_at(step, steps, lr), "loss": loss_sum / ntgt,
                              "t": round(time.time() - t0, 1)}) + "\n")
        log.flush()
        done = step + 1
        if probe and done == steps:
            rec["evals"][done] = {"dev": ea.eval_split(model, tok, ea.load_split("dev"))["acc"]}
        elif not probe and done in EVAL_AT:
            rec["evals"][done] = summary(ea.evaluate(model, tok), a0, a1, m3) | {"seconds": round(time.time() - t0)}
        if done in rec["evals"]:
            (out / f"{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            e = rec["evals"][done]
            print(f"{tag} step {done} " + " ".join(f"{k}={v:.3g}" for k, v in e.items()
                                                  if isinstance(v, float)), flush=True)
    log.close()
    if not probe:
        save_file({n: C[n].detach().cpu().contiguous() for n in names}, str(KEEP / f"seed{seed}" / f"h6f_{tag}.safetensors"),
                  metadata={"provenance": "h6f_lane_coefficients", "basis": basis, "seed": str(seed), "lr": str(lr),
                            "steps": str(steps)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("a", "b"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--basis", choices=("lane", "random"), default="lane")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.part == "a":
        part_a(a.seed)
    else:
        part_b(a.seed, a.lr, a.steps, a.basis, a.probe)


if __name__ == "__main__":
    main()
