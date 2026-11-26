import torch
from engine import wsp_arch, wsp_context_ridge
from pathlib import Path

ARCH = Path(__file__).resolve().parents[1] / "programs/arch/nanowsp_lm.wsparch"


def test_context_ridge_roundtrip(tmp_path):
    opts = dict(vocab=5, block=8, layers=1, heads=2, d=16, bias=False)
    model = wsp_arch.load_arch_file(ARCH, overrides=opts, seed=42)
    before = model.export_state_dict()
    ids = torch.arange(256) % 5
    info = wsp_context_ridge.fit(model, ids, heads=2, block=8, width=16,
                                 layers=1, samples=128, seed=42)
    assert info["backward_calls"] == 0
    after = model.export_state_dict()
    assert not torch.equal(before[info["tensor"]], after[info["tensor"]])
    for key in before:
        if key != info["tensor"]:
            assert torch.equal(before[key], after[key])
    path = tmp_path / "model.safetensors"
    model.save(path)
    loaded = wsp_arch.load_arch_file(ARCH, overrides=opts, seed=99)
    loaded.load(path)
    model.eval()
    loaded.eval()
    x = ids[:8].reshape(1, -1)
    assert torch.equal(model(x)[0], loaded(x)[0])
    assert torch.isfinite(loaded(x)[0]).all()
    assert loaded.generate(x, 3, temperature=0).shape == (1, 11)
