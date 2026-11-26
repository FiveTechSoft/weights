"""Test discovery, lowering, native compilation and checkpoint reload."""
from pathlib import Path

import pytest
import torch
from engine import wsp_discover_register as d
from engine import wsp_register as r
from engine.nanogpt_runtime import GPTConfig, NanoGPT

ROOT = Path(__file__).resolve().parents[1]


def interpret(source,x,y):
    regs={"x":x,"y":y}
    ops,ret=r.parse_program(source)
    for dest,op,a,b in ops:
        regs[dest]=(regs[a]+(1 if op=="ADD" else -1)*regs[b])%16
    return regs[ret]


def test_lower_all_supported_laws():
    supported=0
    for a in range(16):
        for b in range(16):
            try:
                src=d.lower((a,b))
            except ValueError as exc:
                assert "budget" in str(exc)
                continue
            supported+=1
            for x in range(16):
                for y in range(16):
                    assert interpret(src,x,y)==(a*x+b*y)%16
    assert supported>=200


def test_reject_ambiguous_inconsistent():
    with pytest.raises(ValueError,match="unique"):
        d.discover([(0,0,0)])
    with pytest.raises(ValueError,match="unique"):
        d.discover([(0,0,1)])


def test_discover_compile_reload(tmp_path):
    # No hidden weights; discovery sees only 4 out of 256 input pairs.
    a,b=3,13
    observed=[(x,y,(a*x+b*y)%16) for x,y in [(1,0),(0,1),(3,7),(8,5)]]
    arch=ROOT/"programs/arch/nanowsp_lm.wsparch"
    law,source,model=d.compile_examples(arch,observed)
    assert law==(a,b)
    cfg=r.config(source)
    path=tmp_path/"discovered.safetensors"
    model.save(path,metadata={"provenance":"closed_form_symbolic_discovery"})
    rt=NanoGPT(GPTConfig(vocab_size=cfg["vocab"],n_embd=cfg["d"],
        n_head=cfg["heads"],n_layer=cfg["layers"],block_size=cfg["block"],
        d_ff=cfg["ff"],bias=False))
    rt.load_checkpoint(path)
    rt.model.eval()
    excluded={(x,y) for x,y,_ in observed}
    pairs=[(x,y) for x in range(16) for y in range(16) if (x,y) not in excluded]
    correct=0
    with torch.no_grad():
        for start in range(0,len(pairs),32):
            batch=pairs[start:start+32]
            ids=torch.tensor([[x,y,16] for x,y in batch])
            logits,_=rt.forward(ids)
            pred=logits[:,-1].argmax(-1).tolist()
            correct+=sum(p==(a*x+b*y)%16 for p,(x,y) in zip(pred,batch))
    assert correct==len(pairs), (correct,len(pairs))
    print(f"discovered native checkpoint: {correct}/{len(pairs)} unseen pairs")
