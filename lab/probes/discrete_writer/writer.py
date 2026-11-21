"""Discrete per-row writer in the bf16 alphabet (lab/probes/discrete_writer/PLAN.md, roadmap gap 20).

Given a base matrix W0 (bf16 values held as float32) and a target change T (our compiled dW = outer(v, u), or any
edit), choose per entry one move m in {none, sign flip, x2, x1/2}; every move maps a bf16 value to a bf16 value
(Tree-Ising's admissible set: sign and exponent bits, mantissa untouched). Realised change d_j = move(w_j) - w_j.

Separable objective (exact):   sum_j (d_j - t_j)^2 + lam * [m_j != none]
  Per entry the loss is independent, so the optimum is a threshold: take the best move iff
  gain_j = t_j^2 - min_m (d_{j,m} - t_j)^2 > lam.  lam_for_fraction() picks lam as a quantile of the gains.

Coupled objective (activations): ||X (d - t)||^2 + lam * #changed, X = calibration down_proj inputs (positions x n_in),
  as cc_compile.fit_layers captures them. Entries interact through X; greedy coordinate descent per row with a sweep
  budget (discrete_row_anneal). The tests use a synthetic X; real X needs a forward pass (not CPU work here).

Readings reuse engine.ternary.overlap (alpha, noise, cos). No GGUF is written by this module.
"""
import numpy as np

MOVE_NAMES = ("none", "flip", "x2", "half")


def to_bf16(x):
    """float32 -> nearest-even bf16, returned as float32 (exactly representable)."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = x.view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def moves(w):
    """(3, ...) candidate new values: sign flip, x2, x1/2. Exact in bf16 (exponent +-1, sign bit) away from the
    exponent range ends, which these weights never reach."""
    w = np.asarray(w, dtype=np.float32)
    return np.stack([-w, 2.0 * w, 0.5 * w]).astype(np.float32)


def bit_flips(a, b):
    """Hamming distance between the bf16 codes of a and b (both bf16-representable float32 arrays)."""
    ca = (np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32) >> 16).astype(np.uint16)
    cb = (np.ascontiguousarray(np.asarray(b, dtype=np.float32)).view(np.uint32) >> 16).astype(np.uint16)
    x = (ca ^ cb).astype(np.uint16)
    return int(np.unpackbits(x.view(np.uint8)).sum())


def _costs(w, t):
    """-> (cost_none, cost_moves (3, ...), d_moves (3, ...)): squared errors per entry and candidate."""
    w = np.asarray(w, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    d = moves(w) - w
    return t.astype(np.float64) ** 2, (d.astype(np.float64) - t) ** 2, d


def gains(w, t):
    """t^2 - min over moves of (d - t)^2, per entry (> 0 means some move helps)."""
    c0, cm, _ = _costs(w, t)
    return c0 - cm.min(axis=0)


def discrete_threshold(w, t, lam):
    """Exact separable solve on an array of any shape. -> (d realised change float32, chosen int8 codes 0..3, info)."""
    c0, cm, d = _costs(w, t)
    best = cm.argmin(axis=0)
    gain = c0 - np.take_along_axis(cm, best[None], 0)[0]
    take = gain > lam
    chosen = np.where(take, best + 1, 0).astype(np.int8)
    dd = np.where(take, np.take_along_axis(d, best[None], 0)[0], 0.0).astype(np.float32)
    info = {"changed": int(take.sum()), "n": int(take.size), "lam": float(lam),
            "moves": {MOVE_NAMES[k]: int((chosen == k).sum()) for k in (1, 2, 3)}}
    return dd, chosen, info


def discrete_row_threshold(w_row, t_row, lam):
    """One row; same exact separable solve (kept as a named entry point beside discrete_row_anneal)."""
    return discrete_threshold(w_row, t_row, lam)


def lam_for_fraction(w, t, frac):
    """lam such that about `frac` of the entries change (quantile of the positive-gain distribution). If fewer than
    frac of the entries have a positive gain, returns 0.0 (everything beneficial changes)."""
    g = gains(w, t).ravel()
    pos = g[g > 0]
    if pos.size <= frac * g.size:
        return 0.0
    k = int(round(frac * g.size))
    return float(np.partition(pos, pos.size - k)[pos.size - k]) if k > 0 else float(pos.max())


def discrete_row_anneal(w_row, t_row, X, lam=0.0, sweeps=3, seed=0):
    """Coupled objective ||X (d - t)||^2 + lam * #changed on one row, greedy coordinate descent: at each visit of an
    entry take the move (or none) with the lowest objective. Deterministic visiting order from `seed`.
    -> (d float32, chosen int8, trace of objective values after each accepted change, starting with the initial one)."""
    w = np.asarray(w_row, dtype=np.float32)
    t = np.asarray(t_row, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = w.size
    d_all = (moves(w) - w).astype(np.float64)          # (3, n)
    d = np.zeros(n)
    chosen = np.zeros(n, dtype=np.int8)
    r = X @ (d - t)                                     # residual
    xx = (X ** 2).sum(axis=0)
    obj = float(r @ r)
    trace = [obj]
    rng = np.random.default_rng(seed)
    for _ in range(sweeps):
        improved = False
        for j in rng.permutation(n):
            cur = d[j]
            xj = X[:, j]
            # objective change for setting d_j := cand (removing cur): delta = (cand - cur) * (2 xj.r + (cand - cur) xx_j)
            cands = np.concatenate([[0.0], d_all[:, j]])
            step = cands - cur
            delta = step * (2.0 * (xj @ r) + step * xx[j])
            delta += lam * ((cands != 0.0).astype(float) - float(cur != 0.0))
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                r = r + step[k] * xj
                d[j] = cands[k]
                chosen[j] = k
                obj = float(r @ r) + lam * int((chosen != 0).sum())
                trace.append(obj)
                improved = True
        if not improved:
            break
    return d.astype(np.float32), chosen, trace
