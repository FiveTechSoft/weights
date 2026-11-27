"""Known optima and negative controls for the convex dual compiler."""
import pytest
import torch

from engine.wsp_native_qp import compile_qp
from engine.wsp_native_readout import ReadoutRejected

pytest.importorskip("scipy")


def fixture():
    w = torch.zeros(2, 3, dtype=torch.float64)
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=w.dtype)
    p = torch.tensor([[0., 0., 1.]], dtype=w.dtype)
    return w, x, p, x @ w.T, torch.tensor([0, 1]), torch.ones(2, dtype=w.dtype)


def test_known_minimum():
    w, x, p, logits, labels, outside = fixture()
    result, cert = compile_qp(w, x, p, logits, labels, outside)
    # Analytic optimum has both correct rows increased by 3, all else zero.
    assert cert["dual_lower_norm"] == pytest.approx(2**.5 * 2.999, abs=.005)
    assert cert["delta_frobenius"] == pytest.approx(2**.5 * 3., abs=.005)
    assert cert["margin_violation"] <= .001
    assert torch.equal(p @ (result-w).T, torch.zeros(1, 2, dtype=w.dtype))
    assert torch.equal(w, torch.zeros_like(w))


def test_budget_bound():
    with pytest.raises(ReadoutRejected) as exc:
        compile_qp(*fixture(), max_delta=1.)
    assert exc.value.certificate["dual_lower_norm"] > 1.


def test_protected_span():
    w, x, _, l, y, b = fixture()
    with pytest.raises(ReadoutRejected, match="zero projected"):
        compile_qp(w, x, torch.eye(3, dtype=w.dtype), l, y, b)


def test_already_correct():
    w, x, p, _, y, b = fixture()
    w[0, 0] = w[1, 1] = 5.
    result, cert = compile_qp(w, x, p, x @ w.T, y, b)
    assert torch.equal(result, w)
    assert cert["delta_frobenius"] == cert["dual_lower_norm"] == 0.
