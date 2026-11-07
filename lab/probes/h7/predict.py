"""H7 learned jump (h7_plan.md + addenda): from k steps of a NEW task to its step-200 coefficients.

Leave one task out. Fitted on the other tasks' trajectories (all that passed G1), applied to the held-out task
(those that passed G2) after k ∈ {10, 25, 50} of its own steps:
  P_scale  alpha * C_k, alpha = median over training tasks of <C_k, C_200> / |C_k|^2
  P_lin    per coordinate C_200,i ≈ w_kind · phi_i, phi = (C_k, C_khalf, m_k, m_k/(sqrt(v_k)+eps), 1), least squares
  P_mlp    phi (standardised) -> 2-layer MLP (hidden 32), shared by all coordinates
  +dev     one extra scale gamma ∈ {0.5, 0.75, 1, 1.5, 2} on the prediction, chosen on the held-out dev set
References: B_k (C_k as is), B_200 (the real run). R_traj = (acc - acc0) / (acc(C_200) - acc0).
Writes lab/runs/h7/predict.json (after every evaluation).

usage: python lab/probes/h7/predict.py
"""
import json
import statistics as stt
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))
import fingerprint as fp  # noqa: E402
import tasks as tk  # noqa: E402

RUNS = ROOT / "lab" / "runs" / "h7"
KEEP = ROOT / "hf_cache" / "h7"
KS = (10, 25, 50)
K_HALF = {10: 5, 25: 10, 50: 25}
GAMMAS = (0.5, 0.75, 1, 1.5, 2)
EPS = 1e-8
MLP_ROWS, MLP_STEPS, MLP_BATCH = 200_000, 2000, 4096


# ---------- pieces with tests ----------

def features(Ck, Ch, m, v):
    cols = [Ck.flatten(), Ch.flatten(), m.flatten(), (m / (v.sqrt() + EPS)).flatten()]
    return torch.stack(cols + [torch.ones_like(cols[0])], dim=1)


class LinAcc:
    """Least squares by accumulated normal equations."""

    def __init__(self, d=5):
        self.A = torch.zeros(d, d, dtype=torch.float64)
        self.b = torch.zeros(d, dtype=torch.float64)

    def add(self, X, y):
        X, y = X.double().cpu(), y.double().cpu().flatten()
        self.A += X.T @ X
        self.b += X.T @ y

    def solve(self):
        return torch.linalg.solve(self.A + 1e-12 * torch.eye(len(self.b), dtype=torch.float64), self.b)


def alpha_scale(Ck, CT):
    num = sum(float((Ck[n].double() * CT[n].double()).sum()) for n in Ck)
    den = sum(float(Ck[n].double().pow(2).sum()) for n in Ck)
    return num / den


def r_traj(acc, acc0, accT):
    return (acc - acc0) / (accT - acc0)


class MLP(torch.nn.Module):
    def __init__(self, d=5, h=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                       torch.nn.GELU(), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- data ----------

def load_task(name):
    from safetensors.torch import load_file
    d = KEEP / name
    out = {"C": {}, "m": {}, "v": {}}
    for t in sorted(set(KS) | set(K_HALF.values())):
        out["C"][t] = {n: x.float() for n, x in load_file(str(d / f"C_step{t}.safetensors")).items()}
    for k in KS:
        mv = load_file(str(d / f"moments_step{k}.safetensors"))
        out["m"][k] = {n[2:]: x.float() for n, x in mv.items() if n.startswith("m.")}
        out["v"][k] = {n[2:]: x.float() for n, x in mv.items() if n.startswith("v.")}
    out["CT"] = load_file(str(d / "C_final.safetensors"))
    return out


def phi(T, k, n):
    return features(T["C"][k][n], T["C"][K_HALF[k]][n], T["m"][k][n], T["v"][k][n])


def fit_lin(train, k):
    accs = {}
    for T in train:
        for n in T["CT"]:
            accs.setdefault(fp.kind(n), LinAcc()).add(phi(T, k, n), T["CT"][n])
    return {kd: a.solve() for kd, a in accs.items()}


def pred_lin(w, T, k):
    return {n: (phi(T, k, n).double() @ w[fp.kind(n)]).float().reshape(T["CT"][n].shape) for n in T["CT"]}


def fit_mlp(train, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for T in train:
        names = sorted(T["CT"])
        X = torch.cat([phi(T, k, n) for n in names])
        y = torch.cat([T["CT"][n].flatten() for n in names])
        idx = torch.randint(0, len(y), (MLP_ROWS,), generator=g)
        Xs.append(X[idx])
        ys.append(y[idx])
    X, y = torch.cat(Xs), torch.cat(ys)
    sx = X[:, :4].std(0).clamp_min(1e-12)
    sy = y.std().clamp_min(1e-12)
    norm = lambda Z: torch.cat([Z[:, :4] / sx, Z[:, 4:]], 1)
    Xn, yn = norm(X).cuda(), (y / sy).cuda()
    torch.manual_seed(seed)
    net = MLP().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(MLP_STEPS):
        b = torch.randint(0, len(yn), (MLP_BATCH,), device="cuda")
        loss = torch.nn.functional.mse_loss(net(Xn[b]), yn[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, norm, sy, float(loss)


@torch.no_grad()
def pred_mlp(model, T, k):
    net, norm, sy, _ = model
    return {n: (net(norm(phi(T, k, n)).cuda()) * sy).cpu().reshape(T["CT"][n].shape) for n in T["CT"]}


def cos_rel(P, CT):
    num = sum(float((P[n].double() * CT[n].double()).sum()) for n in CT)
    a = sum(float(P[n].double().pow(2).sum()) for n in CT) ** 0.5
    b = sum(float(CT[n].double().pow(2).sum()) for n in CT) ** 0.5
    err = sum(float((P[n].double() - CT[n].double()).pow(2).sum()) for n in CT) ** 0.5
    return num / (a * b), err / b


# ---------- run ----------

def main():
    import eval_arith as ea
    import h6f
    import run_task as rt
    import synth
    ev = {}
    for name in tk.ORDER:
        p = RUNS / name / "evals.json"
        if p.exists() and (KEEP / name / "C_final.safetensors").exists():
            e = json.loads(p.read_text(encoding="utf-8"))
            if e.get("G1", {}).get("pass") and "200" in e["test"]:
                ev[name] = e
    pool = list(ev)
    held = [n for n in pool if ev[n]["test"]["200"]["acc"] - ev[n]["test"]["0"]["acc"] >= 0.10]
    print("trajectories:", pool, "held-out (G2):", held, flush=True)
    data = {n: load_task(n) for n in pool}
    tok = ea.load_tokenizer()
    model, params = h6f.load_model()
    names = sorted(params)
    shapes = {n: tuple(params[n].shape) for n in names}
    W0 = {n: params[n].detach().clone() for n in names}
    U = {n: u.cuda() for n, u in rt.shared_basis(shapes).items()}

    @torch.no_grad()
    def set_C(Cd, gamma=1.0):
        for n in names:
            params[n].copy_(W0[n] + gamma * h6f.coef_apply(U[n], Cd[n].cuda(), shapes[n]))

    out = {"pool": pool, "held": held, "results": {}}
    outp = RUNS / "predict.json"
    t0 = time.time()
    for h in held:
        sp = tk.make_splits(h)
        a0, aT = ev[h]["test"]["0"]["acc"], ev[h]["test"]["200"]["acc"]
        train = [data[n] for n in pool if n != h]
        T = data[h]
        res = out["results"].setdefault(h, {"acc0": a0, "accT": aT})
        for k in KS:
            if str(k) in ev[h]["test"]:
                bk = ev[h]["test"][str(k)]
            else:
                set_C(T["C"][k])
                bk = tk.eval_task(model, tok, h, sp["test"])
            r = res.setdefault(str(k), {})
            r["B_k"] = {"acc": bk["acc"], "R": r_traj(bk["acc"], a0, aT)}
            alphas = [alpha_scale(D["C"][k], D["CT"]) for D in train]
            mlp = fit_mlp(train, k)
            preds = {"P_scale": {n: stt.median(alphas) * T["C"][k][n] for n in names},
                     "P_lin": pred_lin(fit_lin(train, k), T, k),
                     "P_mlp": pred_mlp(mlp, T, k)}
            r["alpha_scale"] = stt.median(alphas)
            r["mlp_train_loss"] = mlp[3]
            for pn, P in preds.items():
                c, rel = cos_rel(P, T["CT"])
                set_C(P)
                e1 = tk.eval_task(model, tok, h, sp["test"])
                dev = {}
                for gm in GAMMAS:
                    set_C(P, gm)
                    dev[gm] = tk.eval_task(model, tok, h, sp["dev"])["acc"]
                gm = synth.pick_alpha(dev)
                set_C(P, gm)
                e2 = tk.eval_task(model, tok, h, sp["test"])
                r[pn] = {"acc": e1["acc"], "R": r_traj(e1["acc"], a0, aT), "cos_to_CT": c, "rel_err": rel,
                         "mcnemar_vs_Bk": synth.mcnemar(e1["ok_mask"], bk["ok_mask"]) if "ok_mask" in bk else None,
                         "dev": {"gamma": gm, "dev_acc": dev, "acc": e2["acc"], "R": r_traj(e2["acc"], a0, aT),
                                 "mcnemar_vs_Bk": synth.mcnemar(e2["ok_mask"], bk["ok_mask"])
                                 if "ok_mask" in bk else None},
                         "ok_mask": e1["ok_mask"]}
                outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
                print(f"{h} k={k} {pn} acc={e1['acc']:.3f} R={r[pn]['R']:.2f} cos={c:.3f} rel={rel:.2f} "
                      f"| dev g={gm} R={r[pn]['dev']['R']:.2f} | B_k R={r['B_k']['R']:.2f} {time.time() - t0:.0f}s",
                      flush=True)
    # selection rule (pre-registered): best of P_lin / P_mlp by mean R over held-out tasks at k = 10
    mean = lambda pn, k: stt.mean(out["results"][h][str(k)][pn]["R"] for h in held)
    best = max(("P_lin", "P_mlp"), key=lambda pn: mean(pn, 10))
    out["selected"] = best
    out["summary"] = {pn: {k: mean(pn, k) for k in KS} for pn in ("P_scale", "P_lin", "P_mlp")}
    out["summary"]["B_k"] = {k: stt.mean(out["results"][h][str(k)]["B_k"]["R"] for h in held) for k in KS}
    ps = [out["results"][h]["25"][best]["mcnemar_vs_Bk"] for h in held]
    wins = sum(1 for h, p in zip(held, ps) if p is not None and p < 0.01
               and out["results"][h]["25"][best]["acc"] > out["results"][h]["25"]["B_k"]["acc"])
    out["H7"] = {"selected": best, "mean_R_k25": mean(best, 25), "wins_vs_B25": wins, "n_held": len(held),
                 "pass": mean(best, 25) >= 0.7 and wins >= 2 * len(held) / 3}
    outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("selected", "summary", "H7")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
