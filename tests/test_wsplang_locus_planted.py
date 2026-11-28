"""Positive-control localization inside a genuine NanoGPT forward pass.

The planted fault is deliberately artificial; this is a machinery test,
not evidence of localization of naturally learned language behavior.
"""
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from engine.nanogpt_runtime import GPTConfig, NanoGPT
from engine.wsp_locus import rank_candidates
from engine.wsp_locus_nanogpt import probe


@pytest.mark.parametrize("fault", [3, 11])
@torch.no_grad()
def test_blind_planted_column(tmp_path, fault):
    cfg = GPTConfig(vocab_size=8, n_embd=8, n_head=2, n_layer=1,
                    block_size=8, d_ff=16, bias=False)
    rt = NanoGPT(cfg)
    m = rt.model.eval()
    # Exact copy-last-token circuit, with inactive attention and MLP.
    for name, param in m.named_parameters():
        param.fill_(1 if "ln_" in name else 0)
    m.transformer.wte.weight.copy_(torch.eye(8))
    mlp = m.transformer.h[0].mlp
    # Distractor columns have nonzero weights but zero activations.
    mlp.c_proj.weight.fill_(.01)
    mlp.c_fc.weight[fault, 1:3] = 4
    mlp.c_proj.weight[:, fault].zero_()
    mlp.c_proj.weight[7, fault] = 100

    # Labels are the copy rule, not outputs obtained by selecting a lesion.
    splits = {
        "discovery": [{"ids":[i, 1], "target":1} for i in range(8)],
        "heldout": [{"ids":[i, 2], "target":2} for i in range(8)],
        "preservation": [{"ids":[i, 4], "target":4} for i in range(8)],
    }
    data = tmp_path/"eval.json"
    data.write_text(json.dumps(splits))
    checkpoint = tmp_path/"model.safetensors"
    save_file({k:v.detach().clone().contiguous() for k,v in rt.export_state_dict().items()},
              str(checkpoint), metadata={"architecture":json.dumps(
                  dict(vocab=8,d=8,heads=2,layers=1,block=8,ff=16,bias=False))})
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # All 16 columns are exposed; the fault index is never passed to probe.
    report = probe(checkpoint,data,"transformer.h.*.mlp.c_proj.weight",
                   n_probe=16,dose=1,seed=1337,top=5)
    assert report["baseline"] == dict(discovery=0.,heldout=0.,preservation=1.)
    ranked = rank_candidates(report)
    assert [r["column"] for r in ranked] == [fault], report
    assert ranked[0]["discovery_gain"] == 1.
    assert ranked[0]["heldout_gain"] == 1.
    assert ranked[0]["preservation_gain"] == 0.
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
