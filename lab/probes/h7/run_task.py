"""H7 step dataset (h7_plan.md + addendum a): train one task's coefficients C in the shared random rank-20 basis.

ΔW_n = U_n C_n (U_n fixed, shared by all tasks), AdamW on C, lr 1e-5, 200 steps of batch 40, 10 % warmup then
cosine (h6f.lr_at). Same loop and numerics as H6f-b.

Logged:
  hf_cache/h7/<run>/C_step<t>.safetensors     C (fp16) at steps 1-10 and every 5 steps; C_final.safetensors (fp32)
  hf_cache/h7/<run>/moments_step<t>.safetensors   AdamW m and v (fp16) at steps 10, 25, 50
  lab/runs/h7/<run>/steps.jsonl   per step: loss, lr, batch indices, per matrix |g_C|, |u|, cos(g,u), cos(u_t,u_t-1)
  lab/runs/h7/<run>/evals.json    G1 (M0 test), dev every 25 steps, test at 0/25/50/100/200, retention NLL 0/200
Exit code 3 if G1 fails (M0 test accuracy outside [0.05, 0.5]); nothing is trained then.

usage: python lab/probes/h7/run_task.py --task mul3x3 [--limit 10 --tag g3]
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))
import tasks as tk  # noqa: E402

R, LR, STEPS = 20, 1e-5, 200
MOMENT_STEPS = (10, 25, 50)
TEST_STEPS = (0, 25, 50, 100, 200)


def snapshot_steps(total):
    return list(range(1, 11)) + list(range(15, total + 1, 5))


def shared_basis(shapes):
    import transfer as tr
    return {n: tr.random_basis(min(shapes[n]), min(R, min(shapes[n])), seed=70_000 + i)
            for i, n in enumerate(sorted(shapes))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tk.ORDER)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many steps (schedule still over 200)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from safetensors.torch import save_file
    import eval_arith as ea
    import h6f
    import train

    run = a.task + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "lab" / "runs" / "h7" / run
    keep = ROOT / "hf_cache" / "h7" / run
    out.mkdir(parents=True, exist_ok=True)
    keep.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tok = ea.load_tokenizer()
    model, params = h6f.load_model(train_mode=True)
    names = sorted(params)
    shapes = {n: tuple(params[n].shape) for n in names}
    W0 = {n: params[n].detach().clone() for n in names}
    U = {n: u.cuda() for n, u in shared_basis(shapes).items()}
    C = {n: torch.nn.Parameter(torch.zeros(h6f.coef_shape(shapes[n], U[n].shape[1]), device="cuda")) for n in names}
    opt = train.AdamW([C[n] for n in names])
    sp = tk.make_splits(a.task)
    ev = {"task": a.task, "run": run, "lr": LR, "steps": STEPS, "limit": a.limit,
          "n_coef": sum(c.numel() for c in C.values()), "dev": {}, "test": {}, "retention_nll": {}}
    evp = out / "evals.json"

    def save_ev():
        evp.write_text(json.dumps(ev, indent=1), encoding="utf-8")

    def evaluate(step, test):
        ev["dev"][step] = tk.eval_task(model, tok, a.task, sp["dev"])["acc"]
        if test:
            ev["test"][step] = tk.eval_task(model, tok, a.task, sp["test"])
        if step in (0, STEPS):
            ev["retention_nll"][step] = ea.retention_nll(model, tok)
        save_ev()
        print(f"{run} step {step} dev {ev['dev'][step]:.3f}" +
              (f" test {ev['test'][step]['acc']:.3f}" if test else ""), flush=True)

    t0 = time.time()
    if not a.limit:
        evaluate(0, True)
        m0 = ev["test"][0]["acc"]
        ev["G1"] = {"m0_test": m0, "pass": 0.05 <= m0 <= 0.5}
        save_ev()
        if not ev["G1"]["pass"]:
            print(f"{run} G1 FAIL m0={m0}", flush=True)
            sys.exit(3)
    for n in names:
        params[n].requires_grad_(True)
    order = list(range(len(sp["train"])))
    random.Random(0).shuffle(order)
    enc = {}
    snaps = set(snapshot_steps(STEPS))
    prev_u = {}
    log = (out / "steps.jsonl").open("w", encoding="utf-8")
    last = a.limit or STEPS
    for step in range(last):
        idx = order[(step * train.BATCH) % len(order):(step * train.BATCH) % len(order) + train.BATCH]
        for i in idx:
            if i not in enc:
                q, ans = sp["train"][i]
                enc[i] = train.encode_pair(tok.encode(tk.prompt(a.task, q), add_special_tokens=False).ids,
                                           tok.encode(tk.target(ans), add_special_tokens=False).ids)
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
                C[n].grad = h6f.coef_grad(params[n].grad, U[n], shapes[n])
                params[n].grad = None
        lr = h6f.lr_at(step, STEPS, LR)
        st = opt.step(lr, keep_update=True)
        rows = {}
        with torch.no_grad():
            for n, s in zip(names, st):
                u = s.pop("u")
                if n in prev_u:
                    q = prev_u[n]
                    s["cos_uu"] = float((u * q).sum() / (u.norm() * q.norm()).clamp_min(1e-30))
                prev_u[n] = u
                rows[n] = {k: round(v, 6) for k, v in s.items()}
                params[n].copy_(W0[n] + h6f.coef_apply(U[n], C[n], shapes[n]))
        done = step + 1
        log.write(json.dumps({"step": done, "lr": lr, "loss": loss_sum / ntgt, "t": round(time.time() - t0, 1),
                              "batch": idx, "tensors": rows}) + "\n")
        log.flush()
        if done in snaps or done == last:
            save_file({n: C[n].detach().half().cpu().contiguous() for n in names}, str(keep / f"C_step{done}.safetensors"))
        if done in MOMENT_STEPS:
            mv = {}
            for n, m, v in zip(names, opt.m, opt.v):
                mv[f"m.{n}"], mv[f"v.{n}"] = m.half().cpu().contiguous(), v.half().cpu().contiguous()
            save_file(mv, str(keep / f"moments_step{done}.safetensors"))
        if not a.limit and done % 25 == 0:
            evaluate(done, done in TEST_STEPS)
    log.close()
    save_file({n: C[n].detach().cpu().contiguous() for n in names}, str(keep / "C_final.safetensors"),
              metadata={"provenance": "h7_run_task", "task": a.task, "steps": str(last), "lr": str(LR)})
    print(f"{run} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
