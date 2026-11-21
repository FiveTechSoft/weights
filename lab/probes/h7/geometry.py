"""H7 descriptive questions (h7_plan.md): shape in time, across tasks, and the 9 x 9 interference matrix.

1. per task: cos(C_t, C_200) against t; share of the trajectory's energy in its top 1-3 temporal modes
   (SVD of the stacked snapshots, from C_0 = 0).
2. across tasks: cosine between tasks' C_200, overall and per matrix kind; share of |C_200|^2 per kind and per
   depth quarter.
3. interference: every task's C_200 evaluated on every other task's test set (GPU; --no-gpu skips it).
Writes lab/runs/h7/geometry.json.

usage: python lab/probes/h7/geometry.py [--no-gpu]
"""
import json
import sys
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


def flat(Cd):
    return torch.cat([Cd[n].double().flatten() for n in sorted(Cd)])


def temporal_shares(traj):
    """Cumulative energy share of the top temporal modes of a trajectory (list of flat vectors)."""
    X = torch.stack([t.double() for t in traj])
    S2 = torch.linalg.eigvalsh(X @ X.T).flip(0).clamp_min(0)
    return [float(x) for x in (S2.cumsum(0) / S2.sum())]


def cos_matrix(Cs):
    v = {k: flat(c) for k, c in Cs.items()}
    return {a: {b: float(v[a] @ v[b] / (v[a].norm() * v[b].norm())) for b in v} for a in v}


def main():
    from safetensors.torch import load_file
    import run_task as rt
    pool = [n for n in tk.ORDER if (KEEP / n / "C_final.safetensors").exists()]
    out = {"pool": pool, "time": {}, "kind_share": {}, "quarter_share": {}}
    CT = {}
    for n in pool:
        CT[n] = load_file(str(KEEP / n / "C_final.safetensors"))
        fT = flat(CT[n])
        steps = rt.snapshot_steps(200)
        traj = [flat(load_file(str(KEEP / n / f"C_step{t}.safetensors"))) for t in steps]
        out["time"][n] = {"steps": steps, "cos_to_final": [float(x @ fT / (x.norm() * fT.norm())) for x in traj],
                          "norm": [float(x.norm()) for x in traj], "temporal_shares": temporal_shares(traj)[:5]}
        tot = float(fT.pow(2).sum())
        ks, qs = {}, {}
        for m, x in CT[n].items():
            e = float(x.double().pow(2).sum()) / tot
            ks[fp.kind(m)] = ks.get(fp.kind(m), 0) + e
            q = f"q{(fp.layer(m) or 0) * 4 // 24}"
            qs[q] = qs.get(q, 0) + e
        out["kind_share"][n], out["quarter_share"][n] = ks, qs
        print(n, "cos@25", round(out["time"][n]["cos_to_final"][steps.index(25)], 3),
              "modes", [round(x, 3) for x in out["time"][n]["temporal_shares"][:3]], flush=True)
    out["cos_final"] = cos_matrix(CT)
    kinds = sorted({fp.kind(m) for m in CT[pool[0]]})
    out["cos_final_by_kind"] = {kd: cos_matrix({n: {m: x for m, x in CT[n].items() if fp.kind(m) == kd}
                                                for n in pool}) for kd in kinds}
    (RUNS / "geometry.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if "--no-gpu" in sys.argv:
        return
    import eval_arith as ea
    import h6f
    tok = ea.load_tokenizer()
    model, params = h6f.load_model()
    names = sorted(params)
    shapes = {n: tuple(params[n].shape) for n in names}
    W0 = {n: params[n].detach().clone() for n in names}
    U = {n: u.cuda() for n, u in rt.shared_basis(shapes).items()}
    out["interference"] = {}
    for src in pool:
        with torch.no_grad():
            for n in names:
                params[n].copy_(W0[n] + h6f.coef_apply(U[n], CT[src][n].cuda(), shapes[n]))
        out["interference"][src] = {}
        for dst in pool:
            out["interference"][src][dst] = tk.eval_task(model, tok, dst, tk.make_splits(dst)["test"])["acc"]
        (RUNS / "geometry.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("interference", src, out["interference"][src], flush=True)


if __name__ == "__main__":
    main()
