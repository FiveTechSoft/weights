"""Discover and transfer a full-batch linear least-squares update law.

Rows of X are frozen activations; rows of Y are desired outputs. W follows
PyTorch Linear convention (out_features, in_features). The reference uses
loss ||X W.T - Y||_F^2 / (2 * len(X)), no bias or weight decay. This is
an algebraic primitive, not a law for AdamW or end-to-end Transformer training.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LearningLaw:
    rate: float
    relative_error: float


def _check(w, x, y):
    if w.ndim != 2 or x.ndim != 2 or y.ndim != 2 or not len(x):
        raise ValueError("expected nonempty 2D activations, targets and weight")
    if x.shape != (len(x), w.shape[1]) or y.shape != (len(x), w.shape[0]):
        raise ValueError("shape mismatch")
    if any(not t.is_floating_point() or not torch.isfinite(t).all() for t in (w, x, y)):
        raise ValueError("expected finite floating-point tensors")
    return tuple(t.detach().to(device=w.device, dtype=torch.float64) for t in (w, x, y))


@torch.no_grad()
def discover(base, trained, x, y, *, tolerance=1e-5):
    """Infer a scalar step rate from a single reference update; reject mismatch."""
    if trained.shape != base.shape or not torch.isfinite(trained).all():
        raise ValueError("invalid reference weight")
    w, x, y = _check(base, x, y)
    g = (x @ w.T - y).T @ x / len(x)
    delta = trained.detach().to(w) - w
    norm2 = g.square().sum()
    if norm2 <= 0 or delta.norm() <= 0:
        raise ValueError("unidentifiable update")
    rate = float(-(delta * g).sum() / norm2)
    error = float((delta + rate * g).norm() / delta.norm())
    if not 0 < rate or error > tolerance:
        raise ValueError("reference does not match the supported update law")
    return LearningLaw(rate, error)


@torch.no_grad()
def generate(base, law, x, y, *, steps):
    """Return a native weight delta via a spectral sum, without target updates."""
    if not isinstance(steps, int) or steps < 0 or not 0 < law.rate < float("inf"):
        raise ValueError("invalid step count or rate")
    w, x, y = _check(base, x, y)
    h = x.T @ x / len(x)
    eig, q = torch.linalg.eigh(h)
    eig = eig.clamp_min(0)
    if law.rate * float(eig.max()) >= 2:
        raise ValueError("unstable rate on target activations")
    r = (x @ w.T - y).T @ x / len(x)
    # Sum rate*(1-rate*eig)**j using binary doubling; works at zero eigenvalue.
    total = torch.zeros_like(eig)
    power = torch.ones_like(eig)
    block = torch.full_like(eig, law.rate)
    decay = 1 - law.rate * eig
    n = steps
    while n:
        if n & 1:
            total = total + power * block
            power = power * decay
        n >>= 1
        if n:
            block = block * (1 + decay)
            decay = decay.square()
    delta = -((r @ q) * total) @ q.T
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite generated delta")
    return delta.to(base.dtype)


@torch.no_grad()
def apply(base, delta):
    if base.shape != delta.shape or not torch.isfinite(delta).all():
        raise ValueError("invalid delta")
    result = base + delta.to(base)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite result")
    return result


def self_test():
    """Independent autograd reference, two different frozen nonlinear tasks."""
    import json
    torch.manual_seed(824)
    torch.set_num_threads(2)
    dtype = torch.float64

    def task(seed, n, width, outputs):
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(n, 12, generator=g, dtype=dtype)
        feature = torch.randn(12, width, generator=g, dtype=dtype) / 12**.5
        x = torch.nn.functional.gelu(raw @ feature)
        # Center columns and include a dependent feature (singular Gram).
        x = x - x.mean(0)
        x[:, -1] = x[:, 0]
        teacher = torch.randn(outputs, width, generator=g, dtype=dtype)
        y = x @ teacher.T
        base = torch.randn(outputs, width, generator=g, dtype=dtype) * .1
        return base, x, y

    def reference(base, x, y, rate, steps):
        w = torch.nn.Parameter(base.clone())
        opt = torch.optim.SGD([w], lr=rate)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (x @ w.T - y).square().sum() / (2 * len(x))
            loss.backward()
            opt.step()
        return w.detach()

    a, x, y = task(12, 256, 20, 4)
    law = discover(a, reference(a, x, y, .2, 1), x, y)
    b, u, v = task(19, 384, 32, 7)
    # Construct target before running its conventional reference.
    got = apply(b, generate(b, law, u, v, steps=128))
    expected = reference(b, u, v, .2, 128)
    error = float((got - expected).abs().max())
    assert error < 1e-9
    assert torch.equal(apply(b, generate(b, law, u, v, steps=0)), b)
    assert abs(law.rate - .2) < 1e-12
    try:
        discover(a, a + .1, x, y)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched donor was accepted")
    print(json.dumps(dict(rate=law.rate, discovery_error=law.relative_error,
        target_weight_max_error=error, target_steps_reproduced=128,
        target_optimizer_steps_for_construction=0, target_backward_calls_for_construction=0,
        initial_mse=float((u @ b.T - v).square().mean()),
        constructed_mse=float((u @ got.T - v).square().mean()),
        reference_mse=float((u @ expected.T - v).square().mean()))))


if __name__ == "__main__":
    self_test()
