"""H6g (h6_plan.md addendum e): learn the lane coefficients with forward passes only (no backward, no donor).

Same parametrization as H6f-b: ΔW_n = U_n C_n on the smaller side, U_n fixed, C_n init 0. What changes is how the
update of C is obtained. SPSA (two-point zeroth-order estimate, as MeZO, Malladi et al. 2023), per step, on one batch:

    z_i ~ N(0, I) with the shape of C,   s_i = (L(C + eps z_i) - L(C - eps z_i)) / (2 eps),   g^ = mean_i s_i z_i

then a normalized-SGD or Adam step on C with g^. The thesis arm uses a RANDOM basis (no trained seed involved);
`--basis lane` is a diagnostic (the lane comes from two trained seeds, so it is not donor-free).

Cost unit: example-forwards. One SPSA step = 2 * pairs * batch; one backprop step = 3 * batch (a backward counted as
two forwards). B0 = 24 000 = the Adam reference of H6f-b (200 steps x 40 examples x 3). Budgets are multiples of B0.

commands:
  gate     before any run: SPSA against the true coefficient gradient of one batch (cosine vs its expected value
           sqrt(N / (N + d + 1)), and a finite difference along the true gradient), for each eps. Picks eps.
  run      the search itself (--method spsa) or its backprop ceiling in the same basis (--method bp).
  placebo  a random C with the same per-matrix norm as a finished run's C, evaluated; exact McNemar against the run.

usage:
  python lab/probes/h6/h6g.py gate --seed 1 --basis random --r 1 --only mlp.down_proj [--dtype fp32]
  python lab/probes/h6/h6g.py run --seed 1 --basis random --r 1 --only mlp.down_proj --eps <gate> --lr 1e-5 \
         --opt nsgd --budget 10 --probe
  python lab/probes/h6/h6g.py run --seed 1 --basis random --r 1 --only mlp.down_proj --eps <gate> --lr <chosen> \
         --opt <chosen> --budget 100
  python lab/probes/h6/h6g.py run --seed 1 --basis random --r 1 --only mlp.down_proj --method bp --lr 1e-5 --budget 1
  python lab/probes/h6/h6g.py placebo --tag <tag printed by run>
"""
import argparse
import contextlib
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
import h6f  # noqa: E402
import transfer as tr  # noqa: E402

BATCH = 40
B0 = 24_000                    # example-forwards of the H6f-b Adam reference (200 steps x 40 x 3)
BUDGETS = (1, 10, 100)         # evaluation points, in multiples of B0
PAIRS = 8                      # SPSA pairs per step (fixed, not searched)
EPS_GRID = (3e-5, 1e-4, 3e-4)
GATE_N = (64, 512)
GATE_RATIO = (0.8, 1.25)       # finite difference along the true gradient / its norm
GATE_COS = 0.5                 # measured cosine at GATE_N[-1] must reach this share of the expected one


# ---------- pieces with tests ----------

def select(names, only):
    """Matrices searched: all, or those whose name contains `only` (e.g. 'mlp.down_proj')."""
    return [n for n in names if not only or only in n]


def n_coef(C):
    return sum(c.numel() for c in C.values())


def expected_cos(pairs, d):
    """E[cos(g^, g)] for the SPSA estimate with `pairs` Gaussian directions in d dimensions (small eps)."""
    return math.sqrt(pairs / (pairs + d + 1))


def step_cost(method, pairs=PAIRS, batch=BATCH):
    return 2 * pairs * batch if method == "spsa" else 3 * batch


def eval_steps(budget, cost, budgets=BUDGETS, b0=B0):
    """{step: budget multiple} for every evaluation point <= budget; the step count is rounded up."""
    return {math.ceil(b * b0 / cost): b for b in budgets if b <= budget}


def flat_cos(a, b):
    num = sum(float((a[n].double() * b[n].double()).sum()) for n in a)
    na = math.sqrt(sum(float(a[n].double().pow(2).sum()) for n in a))
    nb = math.sqrt(sum(float(b[n].double().pow(2).sum()) for n in a))
    return num / (na * nb) if na > 0 and nb > 0 else 0.0


def flat_norm(a):
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in a.values()))


@torch.no_grad()
def spsa(loss_at, C, eps, pairs, gen):
    """Two-point zeroth-order gradient estimate of loss_at at C. loss_at({name: tensor}) -> float.
    Returns (g^, stats); g^ has the shapes of C. Uses 2 * pairs calls of loss_at and no backward."""
    names = sorted(C)
    g = {n: torch.zeros_like(C[n]) for n in names}
    diff, level = 0.0, 0.0
    for _ in range(pairs):
        z = {n: torch.randn(C[n].shape, generator=gen, device=C[n].device, dtype=C[n].dtype) for n in names}
        lp = loss_at({n: C[n] + eps * z[n] for n in names})
        lm = loss_at({n: C[n] - eps * z[n] for n in names})
        s = (lp - lm) / (2 * eps)
        for n in names:
            g[n].add_(z[n], alpha=s / pairs)
        diff += abs(lp - lm) / pairs
        level += (lp + lm) / (2 * pairs)
    return g, {"loss": level, "abs_diff": diff}


@torch.no_grad()
def directional(loss_at, C, g, scale):
    """Central difference of loss_at along g / |g| with a displacement of norm `scale`; equals |g| if linear."""
    gn = flat_norm(g)
    lp = loss_at({n: C[n] + (scale / gn) * g[n] for n in C})
    lm = loss_at({n: C[n] - (scale / gn) * g[n] for n in C})
    return (lp - lm) / (2 * scale)


@torch.no_grad()
def nsgd_step(C, g, lr):
    """Normalized SGD: g^ rescaled to unit RMS over all coefficients, so lr is the RMS move per coefficient
    (the same scale as Adam's lr)."""
    rms = flat_norm(g) / math.sqrt(n_coef(C))
    if rms > 0:
        for n in C:
            C[n].add_(g[n], alpha=-lr / rms)


def placebo_like(C, seed):
    """Gaussian coefficients with the per-matrix Frobenius norm of C."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for n in sorted(C):
        r = torch.randn(C[n].shape, generator=gen)
        out[n] = (r * (float(C[n].norm()) / float(r.norm()))).to(C[n].dtype)
    return out


def gate_verdict(rows):
    """rows: per eps {'eps', 'ratio', 'cos': {N: x}, 'expected': {N: x}}. Largest eps that passes, or None."""
    ok = [r["eps"] for r in rows
          if GATE_RATIO[0] <= r["ratio"] <= GATE_RATIO[1]
          and r["cos"][GATE_N[-1]] >= GATE_COS * r["expected"][GATE_N[-1]]]
    return max(ok) if ok else None


def run_tag(method, opt, basis, r, only, seed):
    return f"g_{method}_{opt if method == 'spsa' else 'adam'}_{basis}_r{r}_{only or 'all'}_seed{seed}"


# ---------- model side ----------

def load_model(dtype, grad):
    from transformers import AutoModelForCausalLM
    import eval_arith as ea
    import train
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    if dtype == "bf16":
        model.model.embed_tokens.to(torch.bfloat16)      # as train.py / h6f.py
    if grad:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model, dict(train.trained_params(model))


def bases(params, all_names, names, basis, r, seed):
    """Per searched matrix, the fixed basis U_n. Random seeds are indexed on the full name list, as in h6f.py,
    so restricting the matrices does not change any basis."""
    if basis == "lane":
        U, _ = h6f.lane_and_nu({n: params[n].detach().float().cpu() for n in names}, seed, r)
    else:
        U = {n: tr.random_basis(min(params[n].shape), min(r, min(params[n].shape)), seed=20_000 * seed + i)
             for i, n in enumerate(all_names) if n in set(names)}
    return {n: u.cuda().float() for n, u in U.items()}


class Task:
    """The model with W_n = W0_n + U_n C_n written in place, and the training loss of the current batch."""

    def __init__(self, seed, basis, r, only, dtype, grad):
        import eval_arith as ea
        self.ea, self.dtype = ea, dtype
        self.tok = ea.load_tokenizer()
        self.model, self.params = load_model(dtype, grad)
        all_names = sorted(self.params)
        self.names = select(all_names, only)
        if not self.names:
            raise ValueError(f"--only {only!r} matches no trained matrix")
        self.W0 = {n: self.params[n].detach().clone() for n in self.names}
        self.U = bases(self.params, all_names, self.names, basis, r, seed)
        self.shape = {n: tuple(self.params[n].shape) for n in self.names}
        self.rows = ea.load_split("train")
        self.order = list(range(len(self.rows)))
        random.Random(seed).shuffle(self.order)
        self.enc, self.ids, self.lab = {}, None, None

    def zeros(self):
        return {n: torch.zeros(h6f.coef_shape(self.shape[n], self.U[n].shape[1]), device="cuda") for n in self.names}

    def set_batch(self, step):
        import make_data as md
        import train
        o = self.order
        idx = o[(step * BATCH) % len(o):(step * BATCH) % len(o) + BATCH]
        for i in idx:
            if i not in self.enc:
                r = self.rows[i]
                self.enc[i] = train.encode_pair(
                    self.tok.encode(md.prompt(r["a"], r["b"]), add_special_tokens=False).ids,
                    self.tok.encode(md.target(r["a"], r["b"]), add_special_tokens=False).ids)
        batch = [self.enc[i] for i in idx]
        L = max(len(x) for x, _ in batch)
        self.ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in batch]).cuda()
        self.lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()

    @torch.no_grad()
    def write(self, C):
        for n in self.names:
            self.params[n].copy_(self.W0[n] + h6f.coef_apply(self.U[n], C[n], self.shape[n]))

    def _ctx(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.dtype == "bf16" else contextlib.nullcontext()

    @torch.no_grad()
    def loss_at(self, C):
        """Mean target-token cross-entropy of the current batch with coefficients C (forward only)."""
        self.write(C)
        self.model.eval()
        with self._ctx():
            hidden = self.model.model(input_ids=self.ids).last_hidden_state[:, :-1]
            mask = self.lab[:, 1:] != -100
            logits = self.model.lm_head(hidden[mask])
        tgt = self.lab[:, 1:][mask]
        return float(torch.nn.functional.cross_entropy(logits.double(), tgt, reduction="sum")) / int(mask.sum())

    def grad_at(self, C):
        """True coefficient gradient of the same loss (backward; always under bf16 autocast, as h6f.py)."""
        import train
        self.write(C)
        for n in self.names:
            self.params[n].requires_grad_(True)
        self.model.train()
        ntgt = int((self.lab[:, 1:] != -100).sum())
        loss_sum = 0.0
        for k0 in range(0, self.ids.shape[0], train.MICRO):
            ids_k, lab_k = self.ids[k0:k0 + train.MICRO], self.lab[k0:k0 + train.MICRO]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.model.model(input_ids=ids_k).last_hidden_state[:, :-1]
                mask = lab_k[:, 1:] != -100
                logits = self.model.lm_head(hidden[mask])
            l_k = torch.nn.functional.cross_entropy(logits.float(), lab_k[:, 1:][mask], reduction="sum")
            (l_k / ntgt).backward()
            loss_sum += float(l_k.detach())
            del logits, hidden
        g = {}
        with torch.no_grad():
            for n in self.names:
                g[n] = h6f.coef_grad(self.params[n].grad, self.U[n], self.shape[n]).float()
                self.params[n].grad = None
                self.params[n].requires_grad_(False)
        return g, loss_sum / ntgt


def ref_accs(seed):
    m0 = json.loads((RUNS / f"seed{seed}" / "chunk0.json").read_text(encoding="utf-8"))["eval"]
    m1 = json.loads((RUNS / f"seed{seed}" / "chunk10.json").read_text(encoding="utf-8"))["eval"]
    return m0["test_id"]["acc"], m1["test_id"]["acc"]


def summary(ev, a0, a1):
    import synth
    return {"test_id": ev["test_id"]["acc"], "test_id_independent": ev["test_id"]["acc_independent"],
            "ood_easy": ev["ood_easy"]["acc"], "ood_hard": ev["ood_hard"]["acc"],
            "retention_nll": ev["retention_nll"], "recovery": synth.recovery(ev["test_id"]["acc"], a0, a1),
            "ok_mask": ev["test_id"]["ok_mask"]}


# ---------- gate ----------

def gate(a):
    out = RUNS / "h6g"
    out.mkdir(parents=True, exist_ok=True)
    task = Task(a.seed, a.basis, a.r, a.only, a.dtype, grad=True)
    task.set_batch(0)
    C = task.zeros()
    d = n_coef(C)
    t0 = time.time()
    g, loss0 = task.grad_at(C)
    gnorm = flat_norm(g)
    rec = {"seed": a.seed, "basis": a.basis, "r": a.r, "only": a.only, "dtype": a.dtype, "n_coef": d,
           "loss": loss0, "grad_norm": gnorm, "rows": []}
    print(f"gate d={d} loss={loss0:.4f} |g|={gnorm:.4g}", flush=True)
    for eps in EPS_GRID:
        ratio = directional(task.loss_at, C, g, eps * math.sqrt(d)) / gnorm
        gen = torch.Generator(device="cuda").manual_seed(777 + a.seed)
        acc = {n: torch.zeros_like(C[n]) for n in C}
        cos, done, diff = {}, 0, 0.0
        for N in GATE_N:
            part, st = spsa(task.loss_at, C, eps, N - done, gen)
            for n in acc:                                   # running mean over all pairs drawn so far
                acc[n].mul_(done / N).add_(part[n], alpha=(N - done) / N)
            done, diff = N, st["abs_diff"]
            cos[N] = flat_cos(acc, g)
        row = {"eps": eps, "ratio": ratio, "cos": cos, "expected": {N: expected_cos(N, d) for N in GATE_N},
               "abs_diff": diff, "seconds": round(time.time() - t0)}
        rec["rows"].append(row)
        print(f"eps {eps:g} ratio {ratio:.3f} " + " ".join(
            f"cos{N} {cos[N]:.4g} (exp {row['expected'][N]:.4g})" for N in GATE_N) + f" |L+-L-| {diff:.3g}",
            flush=True)
    rec["eps_chosen"] = gate_verdict(rec["rows"])
    name = f"gate_{a.basis}_r{a.r}_{a.only or 'all'}_{a.dtype}_seed{a.seed}.json"
    (out / name).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("eps chosen:", rec["eps_chosen"], "(None = gate fails; try --dtype fp32)", flush=True)


# ---------- run ----------

def run(a):
    from safetensors.torch import save_file
    import train
    out = RUNS / "h6g"
    out.mkdir(parents=True, exist_ok=True)
    if a.method == "bp" and a.dtype != "bf16":
        raise ValueError("--method bp runs under bf16 autocast only (memory, as h6f.py)")
    if a.method == "spsa" and a.eps is None:
        raise ValueError("--eps is required for spsa: take it from the gate")
    cost = step_cost(a.method)
    steps = math.ceil(a.budget * B0 / cost)
    at = eval_steps(a.budget, cost)
    tag = run_tag(a.method, a.opt, a.basis, a.r, a.only, a.seed)
    if a.probe:
        tag += f"_probe_lr{a.lr:g}_b{a.budget:g}"
    torch.manual_seed(a.seed)
    task = Task(a.seed, a.basis, a.r, a.only, a.dtype, grad=a.method == "bp")
    C = {n: torch.nn.Parameter(c, requires_grad=False) for n, c in task.zeros().items()}
    names = task.names
    adam = train.AdamW([C[n] for n in names]) if (a.method == "bp" or a.opt == "adam") else None
    gen = torch.Generator(device="cuda").manual_seed(1_000_003 * a.seed + 17)
    a0, a1 = ref_accs(a.seed)
    rec = {"seed": a.seed, "method": a.method, "opt": a.opt if a.method == "spsa" else "adam", "basis": a.basis,
           "r": a.r, "only": a.only, "dtype": a.dtype, "eps": a.eps, "pairs": PAIRS if a.method == "spsa" else None,
           "lr": a.lr, "budget": a.budget, "steps": steps, "step_cost": cost, "n_coef": n_coef(C),
           "n_matrices": len(names), "probe": a.probe, "evals": {}}
    log = (out / f"{tag}.steps.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    losses = []
    for step in range(steps):
        task.set_batch(step)
        lr = h6f.lr_at(step, steps, a.lr)
        if a.method == "spsa":
            g, st = spsa(task.loss_at, C, a.eps, PAIRS, gen)
        else:
            g, loss = task.grad_at(C)
            st = {"loss": loss}
        if adam is None:
            nsgd_step(C, g, lr)
        else:
            with torch.no_grad():
                for n in names:
                    C[n].grad = g[n]
            adam.step(lr)
        losses.append(st["loss"])
        done = step + 1
        log.write(json.dumps({"step": done, "lr": lr, "forwards": done * cost, "t": round(time.time() - t0, 1)}
                             | {k: round(v, 6) for k, v in st.items()}) + "\n")
        log.flush()
        if a.probe and done == steps:
            task.write(C)
            rec["evals"][a.budget] = {"dev": task.ea.eval_split(task.model, task.tok, task.ea.load_split("dev"))["acc"],
                                      "loss_last50": sum(losses[-50:]) / len(losses[-50:])}
        elif not a.probe and done in at:
            task.write(C)
            rec["evals"][at[done]] = summary(task.ea.evaluate(task.model, task.tok), a0, a1) | {
                "step": done, "forwards": done * cost, "seconds": round(time.time() - t0),
                "coef_norm": flat_norm(C), "loss_last50": sum(losses[-50:]) / len(losses[-50:])}
        else:
            continue
        (out / f"{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        e = rec["evals"][a.budget if a.probe else at[done]]
        print(f"{tag} step {done} " + " ".join(f"{k}={v:.3g}" for k, v in e.items() if isinstance(v, float)),
              flush=True)
    log.close()
    if not a.probe:
        (KEEP / f"seed{a.seed}").mkdir(parents=True, exist_ok=True)
        save_file({n: C[n].detach().cpu().contiguous() for n in names},
                  str(KEEP / f"seed{a.seed}" / f"h6g_{tag}.safetensors"),
                  metadata={"provenance": "h6g_coefficients", "method": a.method, "basis": a.basis, "r": str(a.r),
                            "only": a.only or "", "seed": str(a.seed), "lr": str(a.lr), "eps": str(a.eps),
                            "dtype": a.dtype})
        print("tag:", tag, flush=True)


# ---------- placebo ----------

def placebo(a):
    from safetensors.torch import load_file
    import synth
    out = RUNS / "h6g"
    rec = json.loads((out / f"{a.tag}.json").read_text(encoding="utf-8"))
    seed = rec["seed"]
    C = load_file(str(KEEP / f"seed{seed}" / f"h6g_{a.tag}.safetensors"))
    task = Task(seed, rec["basis"], rec["r"], rec["only"], rec["dtype"], grad=False)
    P = {n: v.cuda() for n, v in placebo_like(C, seed=2026 + seed).items()}
    task.write(P)
    a0, a1 = ref_accs(seed)
    s = summary(task.ea.evaluate(task.model, task.tok), a0, a1)
    last = rec["evals"][max(rec["evals"], key=float)]
    s["mcnemar_run_vs_placebo"] = synth.mcnemar(last["ok_mask"], s["ok_mask"])
    s["coef_norm"] = flat_norm(P)
    (out / f"{a.tag}.placebo.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"placebo acc={s['test_id']:.3f} R={s['recovery']:.2f} run acc={last['test_id']:.3f} "
          f"p={s['mcnemar_run_vs_placebo']:.2g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--basis", choices=("random", "lane"), default="random")
        p.add_argument("--r", type=int, default=20)
        p.add_argument("--only", default=None, help="substring of the matrix names to search, e.g. mlp.down_proj")
        p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="precision of the SPSA forwards")
    p = sub.choices["run"]
    p.add_argument("--method", choices=("spsa", "bp"), default="spsa")
    p.add_argument("--opt", choices=("nsgd", "adam"), default="nsgd")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--budget", type=float, default=100, help="multiples of B0 = 24 000 example-forwards")
    p.add_argument("--probe", action="store_true", help="dev accuracy only, at the end of the budget")
    sub.add_parser("placebo").add_argument("--tag", required=True)
    a = ap.parse_args()
    {"gate": gate, "run": run, "placebo": placebo}[a.cmd](a)


if __name__ == "__main__":
    main()
