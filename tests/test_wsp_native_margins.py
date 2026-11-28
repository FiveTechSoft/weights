"""Positive and negative controls for native margin feasibility."""
import pytest
import torch

from engine.wsp_native_margins import compile_margins
from engine.wsp_native_readout import ReadoutRejected


def fixture():
    w = torch.zeros(2, 3)
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    p = torch.tensor([[0., 0., 1.]])
    logits = x @ w.T
    labels = torch.tensor([0, 1])
    outside = torch.ones(2)
    return w, x, p, logits, labels, outside


def test_feasible_margin():
    w,x,p,l,y,b = fixture()
    result, cert = compile_margins(w,x,p,l,y,b,max_delta=10.)
    scores = l+x@(result-w).T
    assert torch.all(scores[torch.arange(2),y] >= b+2.-1e-3)
    assert torch.all(scores[torch.arange(2),y]-scores[torch.arange(2),1-y] >= 2.-1e-3)
    assert torch.equal(p@(result-w).T,torch.zeros(1,2))
    assert cert["delta_frobenius"] <= 10.
    assert cert["backward_calls"] == cert["optimizer_steps"] == 0
    assert torch.equal(w,torch.zeros_like(w))


def test_budget_rejection():
    with pytest.raises(ReadoutRejected) as exc:
        compile_margins(*fixture(),max_delta=.1,max_sweeps=5)
    assert exc.value.certificate["margin_violation"] > .001


def test_protected_span_rejection():
    w,x,_,l,y,b = fixture()
    with pytest.raises(ReadoutRejected):
        compile_margins(w,x,torch.eye(3),l,y,b,max_sweeps=5)


def test_conflicting_labels_rejected():
    w,x,p,l,y,b = fixture()
    x[1] = x[0]
    with pytest.raises(ReadoutRejected):
        compile_margins(w,x,p,l,y,b,max_sweeps=10)


def test_frozen_vocabulary_is_checked():
    w,x,p,l,y,b = fixture()
    b.fill_(100.)
    with pytest.raises(ReadoutRejected):
        compile_margins(w,x,p,l,y,b,max_delta=10.,max_sweeps=5)
