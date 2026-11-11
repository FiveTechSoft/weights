"""H6h (h6_plan.md addendum f): train mul3x3 in a few preconditioned closed-form writes.

Per round: K fresh training examples, one forward + backward at the current weights. Per written matrix (y = W x):
G = sum x x^T, P = sum x delta^T with delta = -dL/dy (exact backward signal, collected by hooks). Direction:
ridge (cf_distill.solve: P^T (G + mu * mean diag G * I)^-1) or the raw gradient P^T (control). The step size is set
by a line search on dev target NLL over relative sizes rho = |alpha D|_F / |W|_F. A round that lowers nothing stops
the arm. No donor; a backward per round (this is a cost question about training, not T4b).

usage:
  python lab/probes/h6/h6h.py run --arm W_all --seed 1 [--rounds 10 --k 1024]
  python lab/probes/h6/h6h.py run --arm W_all --seed 1 --rounds 1 --k 20 --smoke     # model path + timing
  python lab/probes/h6/h6h.py verdict
Outputs: lab/runs/h6/h6h/<arm>_seed<s>.json. No weights are saved (the placebo is evaluated in the same process).
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
import cf_distill  # noqa: E402

RUNS = ROOT / "lab" / "runs" / "h6"
OUT = RUNS / "h6h"
ARMS = {"W_all": ("ridge", None), "G_all": ("grad", None), "W_down": ("ridge", "mlp.down_proj")}
MU = 1e-2
RHOS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
EXTEND = 2
EVAL_ROUNDS = (1, 3, 10)
K_DEFAULT, ROUNDS_DEFAULT = 1024, 10
MICRO, DEV_BATCH = 10, 40
NLL_MAX = 1.331                       # base retention NLL 1.3209 + 0.01
R_PASS, R_CONTINUE, P_PLACEBO = 0.5, 0.25, 0.01


# ---------- pure parts (tested) ----------

class Collector:
    """Forward hooks on Linear modules: G[n] += x^T x and, through a hook on the output, P[n] += x^T (-dL/dy).
    Only rows where the current mask is true are used (padding excluded). Inputs keep the last dim."""

    def __init__(self, mods):
        self.G, self.P, self.rows, self.mask = {}, {}, 0, None
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in mods.items()]

    def set_mask(self, mask):
        self.mask = mask.reshape(-1)
        self.rows += int(self.mask.sum())

    def _hook(self, n):
        def f(mod, inp, out):
            m = self.mask
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])[m].float()
            g = self.G.get(n)
            self.G[n] = x.T @ x if g is None else g.add_(x.T @ x)
            if out.requires_grad:
                def back(grad, x=x, m=m):
                    d = -grad.reshape(-1, grad.shape[-1])[m].float()
                    p = self.P.get(n)
                    self.P[n] = x.T @ d if p is None else p.add_(x.T @ d)
                out.register_hook(back)
        return f

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def direction(G, P, kind, mu=MU):
    """Out x in update direction from G (in x in) and P (in x out)."""
    if kind == "grad":
        return P.T.float().contiguous()
    dev = G.device if G.device.type == "cuda" else "cpu"
    return cf_distill.solve(G, P, mu, device=dev)


def alpha_for(D, W, rho):
    dn = math.sqrt(sum(float(D[k].double().pow(2).sum()) for k in D))
    wn = math.sqrt(sum(float(W[k].double().pow(2).sum()) for k in D))
    return rho * wn / dn if dn > 0 else 0.0


def line_search(f, rhos, current, extend=EXTEND):
    """f(rho) -> loss. Best rho of the grid, extended by a factor 3 past an edge at most `extend` times.
    None if no rho lowers the loss below `current`. Returns (rho or None, {rho: loss})."""
    table = {r: f(r) for r in rhos}
    for _ in range(extend):
        best = min(table, key=table.get)
        if best == max(table):
            r = best * 3
        elif best == min(table):
            r = best / 3
        else:
            break
        table[r] = f(r)
    best = min(table, key=table.get)
    return (best if table[best] < current else None), table


def round_rows(order, t, k):
    n = len(order)
    return [order[(t * k + j) % n] for j in range(k)]


def placebo_delta(D, seed):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in sorted(D):
        r = torch.randn(D[k].shape, generator=gen)
        out[k] = (r * (float(D[k].norm()) / float(r.norm()))).to(D[k].dtype)
    return out


def kind_of(name):
    """'model.layers.3.mlp.down_proj.weight' -> 'mlp.down_proj'."""
    parts = name.split(".")
    i = parts.index("layers")
    return ".".join(parts[i + 2:-1])


# ---------- model side ----------

class Task:
    def __init__(self, seed, only):
        import eval_arith as ea
        import h6f
        import make_data as md
        self.ea, self.md = ea, md
        torch.manual_seed(seed)
        self.seed = seed
        self.tok = ea.load_tokenizer()
        self.model, params = h6f.load_model(train_mode=False)
        self.model.eval()
        names = sorted(n for n in params if not only or only in n)
        self.params = {n: params[n] for n in names}
        self.mods = {}
        for n in names:
            m = self.model.get_submodule(n[:-len(".weight")])
            if not isinstance(m, torch.nn.Linear) or m.weight is not self.params[n]:
                raise TypeError(f"{n} is not the weight of an nn.Linear ({type(m).__name__})")
            self.mods[n] = m
        self.W0 = {n: p.detach().cpu().clone() for n, p in self.params.items()}
        self.train_rows = ea.load_split("train")
        self.dev_rows = ea.load_split("dev")
        self.order = list(range(len(self.train_rows)))
        random.Random(seed).shuffle(self.order)
        self.enc = {}

    def encode(self, rows, keys):
        import train
        out = []
        for key, r in zip(keys, rows):
            if key not in self.enc:
                self.enc[key] = train.encode_pair(
                    self.tok.encode(self.md.prompt(r["a"], r["b"]), add_special_tokens=False).ids,
                    self.tok.encode(self.md.target(r["a"], r["b"]), add_special_tokens=False).ids)
            out.append(self.enc[key])
        return out

    @staticmethod
    def tensors(pairs):
        L = max(len(x) for x, _ in pairs)
        ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in pairs]).cuda()
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in pairs]).cuda()
        valid = torch.tensor([[j < len(x) for j in range(L)] for x, _ in pairs]).cuda()
        return ids, lab, valid

    def _loss_sum(self, ids, lab):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(input_ids=ids).last_hidden_state[:, :-1]
            mask = lab[:, 1:] != -100
            logits = self.model.lm_head(hidden[mask])
        return torch.nn.functional.cross_entropy(logits.float(), lab[:, 1:][mask], reduction="sum")

    def collect(self, idx):
        """One forward + backward over the round's examples; returns the Collector and the mean train loss."""
        pairs = self.encode([self.train_rows[i] for i in idx], [("t", i) for i in idx])
        ntgt = sum(sum(v != -100 for v in y[1:]) for _, y in pairs)     # shifted labels, as in _loss_sum
        col = Collector(self.mods)
        emb = self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
        total, counted = 0.0, 0
        try:
            for k0 in range(0, len(pairs), MICRO):
                ids, lab, valid = self.tensors(pairs[k0:k0 + MICRO])
                col.set_mask(valid)
                ls = self._loss_sum(ids, lab)
                counted += int((lab[:, 1:] != -100).sum())
                (ls / max(ntgt, 1)).backward()
                total += float(ls.detach())
        finally:
            emb.remove()
            col.close()
        return col, total / counted

    @torch.no_grad()
    def dev_nll(self):
        pairs = self.encode(self.dev_rows, [("d", i) for i in range(len(self.dev_rows))])
        tot, n = 0.0, 0
        for k0 in range(0, len(pairs), DEV_BATCH):
            ids, lab, _ = self.tensors(pairs[k0:k0 + DEV_BATCH])
            tot += float(self._loss_sum(ids, lab))
            n += int((lab[:, 1:] != -100).sum())
        return tot / n

    def dev_acc(self):
        return self.ea.eval_split(self.model, self.tok, self.dev_rows)["acc"]

    @torch.no_grad()
    def add(self, D, a):
        for n in D:
            self.params[n].add_(D[n], alpha=a)

    @torch.no_grad()
    def set_delta(self, delta):
        for n, p in self.params.items():
            p.copy_(self.W0[n].to(p.device) + delta[n].to(p.device))

    @torch.no_grad()
    def delta(self):
        return {n: p.detach().cpu() - self.W0[n] for n, p in self.params.items()}

    def evaluate(self):
        import h6g
        a0, a1 = h6g.ref_accs(self.seed)
        return h6g.summary(self.ea.evaluate(self.model, self.tok), a0, a1)


def run(a):
    import synth
    kind, only = ARMS[a.arm]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}_seed{a.seed}" + ("_smoke" if a.smoke else "")
    t = Task(a.seed, only)
    rec = {"arm": a.arm, "direction": kind, "only": only, "seed": a.seed, "k": a.k, "rounds_max": a.rounds,
           "mu": MU, "rhos": RHOS, "n_matrices": len(t.params),
           "n_weights": sum(p.numel() for p in t.params.values()), "rounds": [], "evals": {}}
    path = OUT / f"{tag}.json"
    t0 = time.time()
    fwd = 0
    cur = t.dev_nll()
    rec["dev_nll_0"] = cur
    last = 0
    for r in range(a.rounds):
        idx = round_rows(t.order, r, a.k)
        col, train_loss = t.collect(idx)
        fwd += 3 * a.k
        D = {}
        for n in sorted(t.params):
            D[n] = direction(col.G.pop(n), col.P.pop(n), kind)
        rows = col.rows
        del col
        torch.cuda.empty_cache()
        W = {n: t.params[n] for n in D}
        unit = alpha_for(D, W, 1.0)
        state = {"applied": 0.0}

        def f(rho):
            al = rho * unit
            t.add(D, al - state["applied"])
            state["applied"] = al
            return t.dev_nll()
        best, table = line_search(f, RHOS, cur)
        fwd += len(table) * len(t.dev_rows)
        final = 0.0 if best is None else best * unit
        t.add(D, final - state["applied"])
        row = {"round": r + 1, "rows": rows, "train_loss": train_loss, "dev_nll_before": cur,
               "table": {f"{k:g}": v for k, v in table.items()}, "rho": best, "accepted": best is not None,
               "seconds": round(time.time() - t0), "example_forwards": fwd}
        if best is None:
            row["note"] = "no rho lowers dev NLL: arm stops"
            rec["rounds"].append(row)
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"{tag} round {r + 1} rejected, stop", flush=True)
            break
        cur = table[best]
        rel = {}
        for n in D:
            kk = kind_of(n)
            num, den = rel.get(kk, (0.0, 0.0))
            rel[kk] = (num + float((final * D[n]).double().pow(2).sum()), den + float(W[n].double().pow(2).sum()))
        row["rel_change_by_kind"] = {k: math.sqrt(v[0] / v[1]) for k, v in rel.items()}
        row["dev_nll_after"] = cur
        row["dev_acc_after"] = t.dev_acc()
        del D
        torch.cuda.empty_cache()
        rec["rounds"].append(row)
        last = r + 1
        print(f"{tag} round {last} rho={best:g} dev_nll {row['dev_nll_before']:.4f} -> {cur:.4f} "
              f"dev_acc {row['dev_acc_after']:.3f} {row['seconds']}s", flush=True)
        if not a.smoke and last in EVAL_ROUNDS:
            rec["evals"][str(last)] = t.evaluate() | {"example_forwards": fwd, "seconds": round(time.time() - t0)}
            e = rec["evals"][str(last)]
            print(f"{tag} eval round {last} test {e['test_id']:.3f} R {e['recovery']:.2f} nll {e['retention_nll']:.4f}",
                  flush=True)
        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if a.smoke or last == 0:
        rec["last_round"] = last
        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return
    if str(last) not in rec["evals"]:
        rec["evals"][str(last)] = t.evaluate() | {"example_forwards": fwd, "seconds": round(time.time() - t0)}
    rec["last_round"] = last
    final_eval = rec["evals"][str(last)]
    delta = t.delta()
    rec["delta_norm_by_matrix_total"] = math.sqrt(sum(float(v.double().pow(2).sum()) for v in delta.values()))
    t.set_delta(placebo_delta(delta, seed=2026 + a.seed))
    pl = t.evaluate()
    rec["placebo"] = pl
    rec["mcnemar_vs_placebo"] = synth.mcnemar(final_eval["ok_mask"], pl["ok_mask"])
    rec["wins_vs_placebo"] = sum(x and not y for x, y in zip(final_eval["ok_mask"], pl["ok_mask"]))
    rec["losses_vs_placebo"] = sum(y and not x for x, y in zip(final_eval["ok_mask"], pl["ok_mask"]))
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{tag} done: R {final_eval['recovery']:.2f} placebo R {pl['recovery']:.2f} "
          f"p {rec['mcnemar_vs_placebo']:.2g}", flush=True)


def verdict(_a=None):
    out = {}
    for arm in ARMS:
        recs = {s: json.loads((OUT / f"{arm}_seed{s}.json").read_text(encoding="utf-8"))
                for s in (1, 2, 3) if (OUT / f"{arm}_seed{s}.json").exists()}
        if not recs:
            continue
        last = {s: r["evals"][str(r["last_round"])] for s, r in recs.items() if r.get("last_round")}
        Rs = {s: e["recovery"] for s, e in last.items()}
        row = {"R": Rs, "rounds": {s: r.get("last_round") for s, r in recs.items()},
               "p_placebo": {s: r.get("mcnemar_vs_placebo") for s, r in recs.items()},
               "nll": {s: e["retention_nll"] for s, e in last.items()}}
        if len(recs) < 3:
            row["status"] = ("stopped at seed 1 (R < 0.25): fail" if Rs.get(1, 0) < R_CONTINUE
                             else "seeds 2-3 pending")
        else:
            mean_R = sum(Rs.values()) / 3
            ok = (mean_R >= R_PASS
                  and all(recs[s]["mcnemar_vs_placebo"] < P_PLACEBO
                          and recs[s]["wins_vs_placebo"] > recs[s]["losses_vs_placebo"] for s in recs)
                  and all(v <= NLL_MAX for v in row["nll"].values()))
            row["mean_R"] = mean_R
            row["status"] = ("inconclusive" if abs(mean_R - R_PASS) <= 0.1 and not ok else
                             "PASS" if ok else "FAIL")
        out[arm] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in out.items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--smoke", action="store_true", help="no test evaluation, no placebo; tag gets _smoke")
    sub.add_parser("verdict")
    a = ap.parse_args()
    {"run": run, "verdict": verdict}[a.cmd](a)


if __name__ == "__main__":
    main()
