"""H9 warm start (h9_plan.md): H7's trainer with a different starting point C_0.

--init zero  C_0 = 0 (H7's Cold run; used only for the G0 check against H7's G3 C_10)
--init sum   C_0 = lam * sum of the other G2 tasks' C_final (H7)
--init rand  C_0 = random, same shapes and global norm as the sum init (seed 90 000 + task index)
Everything else (basis, data order, seed, lr schedule, numerics) is H7's run_task.py. Optimizer state starts fresh.
Logged: lab/runs/h9/<run>/{steps.jsonl, evals.json}; hf_cache/h9/<run>/C_step{25,50,100,200}.safetensors (fp16)
and C_final (fp32).

usage: python lab/probes/h9/run_warm.py --task mul3x3 --init sum [--lam 0.5] [--limit 10 --tag g0]
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
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h7"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))
import tasks as tk  # noqa: E402

H7 = ROOT / "lab" / "runs" / "h7"
KEEP7 = ROOT / "hf_cache" / "h7"
STEPS, LR = 200, 1e-5
TEST_STEPS = (0, 25, 50, 100, 200)


# ---------- pieces with tests ----------

def warm_init(Cs, held, lam):
    others = [t for t in sorted(Cs) if t != held]
    return {n: lam * sum(Cs[t][n].float() for t in others) for n in Cs[others[0]]}


def rand_like_norm(C0, seed):
    g = torch.Generator().manual_seed(seed)
    R = {n: torch.randn(C0[n].shape, generator=g) for n in sorted(C0)}
    tgt = float(torch.cat([C0[n].double().flatten() for n in C0]).norm())
    cur = float(torch.cat([R[n].double().flatten() for n in R]).norm())
    return {n: R[n] * (tgt / cur) for n in R}


def snap_steps(last):
    return {s for s in (25, 50, 100, 200) if s <= last} | {last}


def g2_tasks():
    out = []
    for n in tk.ORDER:
        e = json.loads((H7 / n / "evals.json").read_text(encoding="utf-8"))
        if e["test"]["200"]["acc"] - e["test"]["0"]["acc"] >= 0.10:
            out.append(n)
    return out


# ---------- run ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=tk.ORDER)
    ap.add_argument("--init", required=True, choices=("zero", "sum", "rand"))
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    import eval_arith as ea
    import h6f
    import run_task as rt
    import train

    run = f"{a.task}_{a.init}" + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "lab" / "runs" / "h9" / run
    keep = ROOT / "hf_cache" / "h9" / run
    out.mkdir(parents=True, exist_ok=True)
    keep.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tok = ea.load_tokenizer()
    model, params = h6f.load_model(train_mode=True)
    names = sorted(params)
    shapes = {n: tuple(params[n].shape) for n in names}
    W0 = {n: params[n].detach().clone() for n in names}
    U = {n: u.cuda() for n, u in rt.shared_basis(shapes).items()}
    C = {n: torch.nn.Parameter(torch.zeros(h6f.coef_shape(shapes[n], U[n].shape[1]), device="cuda")) for n in names}
    info = {"init": a.init, "lam": a.lam}
    if a.init != "zero":
        pool = g2_tasks()
        Cs = {t: load_file(str(KEEP7 / t / "C_final.safetensors")) for t in pool}
        C0 = warm_init(Cs, a.task, a.lam)
        info["others"] = [t for t in pool if t != a.task]
        if a.init == "rand":
            C0 = rand_like_norm(C0, seed=90_000 + tk.ORDER.index(a.task))
        info["norm_C0"] = float(torch.cat([C0[n].double().flatten() for n in C0]).norm())
        del Cs
        with torch.no_grad():
            for n in names:
                C[n].copy_(C0[n].cuda())
                params[n].copy_(W0[n] + h6f.coef_apply(U[n], C[n], shapes[n]))
    opt = train.AdamW([C[n] for n in names])
    sp = tk.make_splits(a.task)
    ev = {"task": a.task, "run": run, "lr": LR, "steps": STEPS, "limit": a.limit, **info,
          "dev": {}, "test": {}, "retention_nll": {}}
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
    for n in names:
        params[n].requires_grad_(True)
    order = list(range(len(sp["train"])))
    random.Random(0).shuffle(order)
    enc = {}
    last = a.limit or STEPS
    snaps = snap_steps(last)
    log = (out / "steps.jsonl").open("w", encoding="utf-8")
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
        opt.step(lr, keep_update=True)
        with torch.no_grad():
            for n in names:
                params[n].copy_(W0[n] + h6f.coef_apply(U[n], C[n], shapes[n]))
        done = step + 1
        log.write(json.dumps({"step": done, "lr": lr, "loss": loss_sum / ntgt, "t": round(time.time() - t0, 1)}) + "\n")
        log.flush()
        if done in snaps:
            save_file({n: C[n].detach().half().cpu().contiguous() for n in names}, str(keep / f"C_step{done}.safetensors"))
        if not a.limit and done % 25 == 0:
            evaluate(done, done in TEST_STEPS)
    log.close()
    save_file({n: C[n].detach().cpu().contiguous() for n in names}, str(keep / "C_final.safetensors"),
              metadata={"provenance": "h9_run_warm", "task": a.task, "init": a.init, "steps": str(last)})
    print(f"{run} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
