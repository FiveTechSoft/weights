import torch
from engine.delta_apply import apply_selected, layer_range


class D(dict):
    def get(self, k):
        return self[k]


def _pair():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
    base = D({"model.layers.0.w": r(8, 6), "model.layers.1.w": r(8, 6), "model.layers.2.w": r(8, 6), "model.norm.weight": r(6)})
    model = D({k: (v.float() + torch.randn(v.shape, generator=g) * 0.1).to(torch.bfloat16) for k, v in base.items()})
    model["model.layers.2.w"] = model["model.layers.2.w"].float()  # dtype differs on purpose
    return base, model


def test_selection_leaves_the_rest_as_base_and_never_modifies_inputs():
    base, model = _pair()
    before = {k: v.clone() for k, v in base.items()}
    out, rep = apply_selected(base, model, layer_range(0, 1))
    assert all(torch.equal(base[k], before[k]) for k in base)
    assert torch.equal(out["model.norm.weight"], base["model.norm.weight"])
    assert torch.equal(out["model.layers.2.w"], base["model.layers.2.w"])
    assert [r["status"] for r in rep] == ["changed", "changed"]


def test_full_delta_alpha_one_reproduces_model_and_alpha_zero_the_base():
    base, model = _pair()
    out, _ = apply_selected(base, model, layer_range(0, 1), alpha=1.0)
    assert torch.allclose(out["model.layers.0.w"].float(), model["model.layers.0.w"].float(), atol=1e-6)
    out0, _ = apply_selected(base, model, layer_range(0, 1), alpha=0.0)
    assert torch.equal(out0["model.layers.0.w"], base["model.layers.0.w"])


def test_dtype_mismatch_is_reported_not_cast():
    base, model = _pair()
    _, rep = apply_selected(base, model, layer_range(2, 2))
    assert rep[0]["status"] == "skipped" and "dtype" in rep[0]["reason"]


def test_rank_mode_is_lossy_and_reports_energy_kept():
    base, model = _pair()
    out, rep = apply_selected(base, model, layer_range(0, 0), rank=2)
    assert 0 < rep[0]["delta_energy_kept"] < 1
    assert not torch.allclose(out["model.layers.0.w"].float(), model["model.layers.0.w"].float(), atol=1e-3)
