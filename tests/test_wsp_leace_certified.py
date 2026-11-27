"""Tests of the real rank-one erasure implementation, not a duplicate formula."""
import pytest
import torch
from engine.wsp_leace import erase_direction


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_leakage_certificate(dtype):
    rng = torch.Generator().manual_seed(91)
    w = torch.randn(12, 8, generator=rng, dtype=dtype)
    u = torch.randn(12, generator=rng, dtype=dtype)
    a = torch.randn(12, 20, generator=rng, dtype=torch.float64)
    s = a @ a.T / 20
    erased, cert = erase_direction(w, u, s)
    assert erased.dtype == w.dtype
    unit = u.double() / u.double().norm()
    assert float((unit @ erased.double()).norm()) <= 1e-6 * max(1., cert["leak_before"])
    assert cert["leak_after"] < cert["leak_before"]
    assert cert["expected_displacement"] >= cert["orthogonal_displacement"] - 1e-10


def test_identity_covariance_matches_orthogonal_projection():
    w = torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64)
    u = torch.tensor([1., 0.], dtype=torch.float64)
    result, _ = erase_direction(w, u, torch.eye(2))
    assert torch.equal(result, torch.tensor([[0., 0.], [4., 5.]], dtype=w.dtype))


@pytest.mark.parametrize("cov", [
    [[1., 2.], [0., 1.]],
    [[1., 0.], [0., -1.]],
    [[0., 0.], [0., 1.]],
    [[float("nan"), 0.], [0., 1.]],
])
def test_invalid_covariance_rejected(cov):
    with pytest.raises(ValueError):
        erase_direction(torch.eye(2), torch.tensor([1., 0.]), torch.tensor(cov))


def test_zero_direction_rejected():
    with pytest.raises(ValueError, match="zero concept"):
        erase_direction(torch.eye(2), torch.zeros(2), torch.eye(2))


def test_rounding_is_certified_or_rejected():
    w = torch.tensor([[.1234, .5678], [.2345, .6789]], dtype=torch.bfloat16)
    u = torch.tensor([1., 2.])
    unit = u.double() / u.double().norm()
    rounded = (w.double() - unit[:, None] * (unit @ w.double())[None, :]).to(w.dtype)
    leak = float((unit @ rounded.double()).norm())
    limit = 1e-12 * max(1., float((unit @ w.double()).norm()))
    if leak > limit:
        with pytest.raises(ValueError, match="rounded weights"):
            erase_direction(w, u, torch.eye(2), atol=1e-12)
    else:
        result, cert = erase_direction(w, u, torch.eye(2), atol=1e-12)
        assert torch.equal(result, rounded)
        assert cert["leak_after"] <= limit
