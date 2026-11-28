"""Exercise the actual parser, runtime and NanoGPT discovery adapter."""
import hashlib
import json

import pytest
from safetensors.torch import save_file
from engine import wsplang as w
from engine.nanogpt_runtime import GPTConfig, NanoGPT


def program(model, data, report, ranked, placebo="true"):
    return f"""experiment discovery_integration:
    model: nanoGPT
    budgets:
        max_modified_weights = 64
        allow_write = false
    steps:
        step probe_columns model={model} tensor=transformer.h.*.mlp.c_proj.weight eval={data} placebo={placebo} n_probe=4 out={report}
        step rank_candidates from={report} out={ranked}
"""


@pytest.mark.parametrize("placebo", ["false", ""])
def test_validator_rejects_missing_placebo(placebo):
    p = w.parse_args_lines(program("model.safetensors", "data.json", "probe.json", "rank.json", placebo))
    _, errors = w.validate(p)
    assert any("placebo=true" in e for e in errors)


def test_real_discovery_pipeline(tmp_path):
    cfg = GPTConfig(vocab_size=8, n_embd=8, n_head=2, n_layer=1,
                    block_size=8, d_ff=16, bias=False)
    rt = NanoGPT(cfg)
    model = tmp_path / "nano.safetensors"
    save_file({k:v.detach().clone().contiguous() for k,v in rt.export_state_dict().items()},
              str(model), metadata={"architecture":json.dumps(
                  dict(vocab=8,d=8,heads=2,layers=1,block=8,ff=16,bias=False))})
    original = hashlib.sha256(model.read_bytes()).hexdigest()
    data = tmp_path / "data.json"
    data.write_text(json.dumps({s:[{"ids":[i,0],"target":i}]
        for i,s in enumerate(("discovery","heldout","preservation"),1)}))
    report, ranked = tmp_path/"probe.json", tmp_path/"ranked.json"
    p = w.parse_args_lines(program(model,data,report,ranked))
    steps, errors = w.validate(p)
    assert not errors, errors
    plan = w.plan_run(p,steps)
    result = w.execute_plan_v4(plan,p,model_path=str(model),dry_run=True)
    assert all(r["status"]=="ok" for r in result), result
    assert not report.exists() and not ranked.exists()
    result = w.execute_plan_v4(plan,p,model_path=str(model))
    assert all(r["status"]=="ok" for r in result), result
    r = json.loads(report.read_text())
    assert r["placebo"] is True and r["model_sha256"]==original
    assert r["split_sizes"]==dict(discovery=1,heldout=1,preservation=1)
    assert len(r["candidates"])==4
    q = json.loads(ranked.read_text())
    assert q["model_sha256"]==original
    assert q["verdict"] in ("no_locus","locus_candidates_ok")
    assert hashlib.sha256(model.read_bytes()).hexdigest()==original
    data.write_text(json.dumps({s:[{"ids":[1,0],"target":1}]
        for s in ("discovery","heldout","preservation")}))
    from engine.wsp_locus_nanogpt import probe
    with pytest.raises(ValueError,match="duplicate prompt"):
        probe(model,data,"transformer.h.*.mlp.c_proj.weight",n_probe=4)
