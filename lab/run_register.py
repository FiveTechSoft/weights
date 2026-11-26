import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from engine import wsp_register as r
from engine.nanogpt_runtime import NanoGPT,GPTConfig

def main():
    rng=np.random.default_rng(20261126)
    report={}
    arch=Path("programs/arch/nanowsp_lm.wsparch")
    for name in ("arithmetic","mixed","copy_sum","lookup_sum"):
        source=Path(f"programs/register/{name}.wspreg").read_text()
        cfg=r.config(source)
        examples=[]
        for i in range(512):
            x,y=i//16%16,i%16
            values=rng.permutation(16).tolist()
            keys=rng.permutation(16).tolist()
            prefix=[t for k in keys for t in (k,values[k],17)]
            expected={"arithmetic":x,"mixed":(values[x]+values[values[x]]-y)%16,
                      "copy_sum":(x+y)%16,"lookup_sum":values[(x+y)%16]}[name]
            examples.append((prefix+[x,y,16],expected))
        report[name]={}
        for arm in ("compiled","ablation"):
            path=Path(f"models/primitives/register_{name}{'_ablation' if arm=='ablation' else ''}.safetensors")
            if arm=="ablation":
                ops,ret=r.parse_program(source)
                m=r.compile_program(arch,source,ablate=ret)
                m.save(path,metadata={"provenance":"closed_form_data_free","ablation":ret})
            rt=NanoGPT(GPTConfig(vocab_size=cfg["vocab"],n_embd=cfg["d"],n_head=cfg["heads"],
                                 n_layer=cfg["layers"],block_size=cfg["block"],d_ff=cfg["ff"],bias=False))
            rt.load_checkpoint(path);rt.model.eval()
            correct=0
            with torch.no_grad():
                for start in range(0,len(examples),32):
                    batch=examples[start:start+32]
                    pred=rt.forward(torch.tensor([x for x,y in batch]))[0][:,-1].argmax(-1).tolist()
                    correct+=sum(a==y for a,(_,y) in zip(pred,batch))
            report[name][arm]=dict(n=len(examples),correct=correct,accuracy=correct/len(examples),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report[name]["pass"]=report[name]["compiled"]["accuracy"]>=.95 and report[name]["ablation"]["accuracy"]<=.2
    Path("lab/runs").mkdir(parents=True,exist_ok=True)
    Path("lab/runs/nanowsp_register.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert all(v["pass"] for v in report.values()),report
if __name__=="__main__":
    main()
