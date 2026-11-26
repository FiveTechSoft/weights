import torch
from engine.wsp_delta import factorize,compile_delta

def test_delta():
    g=torch.Generator().manual_seed(7)
    base={"transformer.wte.weight":torch.randn(5,8,generator=g),
          "transformer.h.0.attn.c_attn.weight":torch.randn(24,8,generator=g),
          "transformer.ln_f.weight":torch.ones(8)}
    donor={k:v+torch.randn(v.shape,generator=g)*.1 for k,v in base.items()}
    for state in (base,donor):
        state["lm_head.weight"]=state["transformer.wte.weight"].clone()
    f=factorize(base,donor)
    full,_=compile_delta(base,f)
    assert all(torch.allclose(full[k],donor[k],atol=1e-7) for k in donor)
    low,stats=compile_delta(base,f,rank=2)
    fake,fs=compile_delta(base,f,rank=2,placebo=True)
    key="transformer.h.0.attn.c_attn.weight"
    assert torch.equal(low["lm_head.weight"],low["transformer.wte.weight"])
    a,b=low[key]-base[key],fake[key]-base[key]
    assert torch.allclose(torch.linalg.svdvals(a),torch.linalg.svdvals(b),atol=1e-6)
    assert stats[key]["energy"]==fs[key]["energy"]
    assert torch.linalg.matrix_rank(a,atol=1e-6)==2
    selected,_=compile_delta(base,f,groups={"attention"})
    assert torch.equal(selected["transformer.wte.weight"],base["transformer.wte.weight"])
