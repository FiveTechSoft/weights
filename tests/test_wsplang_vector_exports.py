"""Regression tests for vector artifact write guards."""
from types import SimpleNamespace
import numpy as np
import pytest
from engine import wsplang_ops as ops
from engine.wsplang import WriteGuardError


@pytest.fixture
def ctx(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    return SimpleNamespace(model_path=model, target_path=model,
        default_gguf=tmp_path/"base.gguf", written_paths=set(),
        allow_write=True, dry_run=False)


@pytest.mark.parametrize("name", ["existing.npz", "wrong.gguf", "model/new.npz"])
def test_guard_refuses_unsafe_paths(ctx,tmp_path,name):
    (tmp_path/"existing.npz").write_bytes(b"original")
    with pytest.raises(WriteGuardError):
        ops._guard_vector_export(ctx,tmp_path/name,".npz",[])
    assert (tmp_path/"existing.npz").read_bytes()==b"original"


def test_guard_refuses_reserved_and_input_paths(ctx,tmp_path):
    p=tmp_path/"new.npz"
    with pytest.raises(WriteGuardError):
        ops._guard_vector_export(ctx,p,".npz",[p])
    ctx.written_paths.add(p.resolve())
    with pytest.raises(WriteGuardError):
        ops._guard_vector_export(ctx,p,".npz",[])


@pytest.mark.parametrize("allow_write,dry_run", [(False,False),(True,True)])
def test_exports_are_suppressed(ctx,tmp_path,monkeypatch,allow_write,dry_run):
    ctx.allow_write=allow_write
    ctx.dry_run=dry_run
    def spectral(*args,**kwargs):
        assert kwargs["out_vector"] is None
        return {}
    monkeypatch.setattr(ops,"compute_spectral_delta",spectral)
    monkeypatch.setattr(ops,"load_vector_numpy",lambda _: np.ones(4))
    target=tmp_path/"absent"
    ops._op_spectral_delta(ctx,"spectral_delta",[],{"tensor":"w","out":str(target/"a.npz")},{})
    result={"warnings":[]}
    ops._op_layer_envelope_multi_layer_envelope(ctx,"layer_envelope",[],
        {"out":str(target/"a.gguf")},result)
    assert not result["result"]["gguf_exported"]
    assert not target.exists()


def test_spectral_export_registers_artifact(ctx,tmp_path,monkeypatch):
    p=tmp_path/"vector.npz"
    def spectral(*args,**kwargs):
        out=kwargs["out_vector"]
        np.savez_compressed(out,u=np.ones(4))
        return {"saved_dominant_vector":str(out)}
    monkeypatch.setattr(ops,"compute_spectral_delta",spectral)
    ops._op_spectral_delta(ctx,"spectral_delta",[],{"tensor":"w","out":str(p)},{})
    assert p.is_file()
    assert p.resolve() in ctx.written_paths
    with pytest.raises(WriteGuardError):
        ops._op_spectral_delta(ctx,"spectral_delta",[],{"tensor":"w","out":str(p)},{})
