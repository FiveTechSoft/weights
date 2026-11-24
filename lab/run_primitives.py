import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from engine import wsp_primitives as p
from engine.nanogpt_runtime import NanoGPT,GPTConfig

def cases(skill,seed,n):
    rng=np.random.default_rng(seed)
    for i in range(n):
        if skill=="copy":
            x=rng.integers(0,16,size=int(rng.integers(2,63))).tolist()
            yield x+[16],x[-1]
        else:
            k=int(rng.integers(2,17))
            keys=rng.choice(16,k,replace=False)
            vals=rng.integers(0,16,size=k)
            j=int(rng.integers(k))
            x=[int(t) for a,b in zip(keys,vals) for t in (a,b,17)]
            yield x+[int(keys[j]),16],int(vals[j])

def score(runtime,examples):
    correct=0
    with torch.no_grad():
        for x,y in examples:
            pred=int(runtime.forward(torch.tensor([x]))[0][0,-1].argmax())
            correct+=pred==y
    return correct/len(examples)

def main():
    out=Path("lab/runs");out.mkdir(parents=True,exist_ok=True)
    report={}
    for skill in ("copy","associate"):
        examples=list(cases(skill,20261124,400))
        report[skill]={}
        for arm in ("compiled","ablation"):
            path=Path(f"models/primitives/{skill}_{arm}.safetensors")
            if arm=="ablation":
                model=p.compile_primitive(Path("programs/arch/nanowsp_lm.wsparch"),skill,ablate=True)
                model.save(path,metadata={"provenance":"closed_form_data_free","skill":skill,"arm":arm,"backward_calls":"0"})
            elif not path.is_file():
                raise FileNotFoundError(path)
            runtime=NanoGPT(GPTConfig(vocab_size=18,n_embd=128,n_head=2,n_layer=2,block_size=64,d_ff=256,bias=False))
            runtime.load_checkpoint(path);runtime.model.eval()
            report[skill][arm]={"accuracy":score(runtime,examples),"n":len(examples),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        report[skill]["pass"]=report[skill]["compiled"]["accuracy"]>=.95 and report[skill]["compiled"]["accuracy"]-report[skill]["ablation"]["accuracy"]>=.5
    (out/"nanowsp_primitives.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert all(r["pass"] for r in report.values()),report

if __name__=="__main__":
    main()
