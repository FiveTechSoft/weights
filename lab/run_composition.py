"""One persistent checkpoint, three operations, novel associations, no inference solver."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from engine import wsp_composition as c
from engine.nanogpt_runtime import NanoGPT,GPTConfig

def cases(seed,n):
    rng=np.random.default_rng(seed)
    for _ in range(n):
        k=int(rng.integers(3,17))
        cycle=rng.choice(16,k,replace=False)
        mapping={int(a):int(b) for a,b in zip(cycle,np.roll(cycle,-1))}
        key=int(rng.choice(cycle))
        records=list(mapping.items())
        rng.shuffle(records)
        prefix=[int(t) for a,b in records for t in (a,b,17)]+[key]
        # Three different correct answers, same prompt except the opcode.
        yield [prefix+[16],prefix+[18],prefix+[19]], [key,mapping[key],mapping[mapping[key]]]

@torch.no_grad()
def evaluate(runtime,examples):
    hits=[0,0,0]
    joint=0
    for x,y in examples:
        pred=runtime.forward(torch.tensor(x))[0][:,-1].argmax(-1).tolist()
        ok=[a==b for a,b in zip(pred,y)]
        hits=[a+int(b) for a,b in zip(hits,ok)]
        joint+=all(ok)
    return dict(n=len(examples),copy=hits[0]/len(examples),lookup=hits[1]/len(examples),
                chain=hits[2]/len(examples),joint=joint/len(examples))

def main():
    out=Path("lab/runs");out.mkdir(parents=True,exist_ok=True)
    examples=list(cases(20261125,400))
    report={}
    for arm in ("compiled","first_lookup","second_lookup","routing"):
        path=Path(f"models/primitives/composition_{arm}.safetensors")
        if arm!="compiled":
            m=c.compile_composition(Path("programs/arch/nanowsp_lm.wsparch"),ablate=arm)
            m.save(path,metadata={"provenance":"closed_form_data_free","ablation":arm})
        rt=NanoGPT(GPTConfig(vocab_size=20,n_embd=192,n_head=3,n_layer=4,block_size=64,d_ff=256,bias=False))
        rt.load_checkpoint(path);rt.model.eval()
        report[arm]=dict(**evaluate(rt,examples),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    r=report["compiled"];first=report["first_lookup"];second=report["second_lookup"];route=report["routing"]
    gates=dict(joint_accuracy=r["joint"]>=.95,
        first_lookup_necessary=first["lookup"]<=.2 and first["chain"]<=.2 and first["copy"]>=.95,
        second_lookup_necessary=second["chain"]<=.2 and second["copy"]>=.95 and second["lookup"]>=.95,
        routing_necessary=route["joint"]<=.05)
    summary=dict(provenance="closed_form_data_free",arms=report,gates=gates)
    (out/"nanowsp_composition.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    assert all(gates.values()),gates

if __name__=="__main__":
    main()
