"""Diagnostic only: oracle interventions are not a programmable model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from compile_eval import MODEL,SEED,batches,encode
from compile_eval_v2 import zarp,digit_ids
from probe_v5 import fit,apply,targets,score

LAYERS=(1,2)
DOSES=(0,.125,.25,.5,1,2,4,8)

@torch.no_grad()
def collect(base,tok,items,layer):
    rows=[]
    def capture(_m,args):
        rows.append(args[0][:,-1].detach().double().cpu())
    hook=base.model.layers[layer].mlp.register_forward_pre_hook(capture)
    try:
        for batch in batches(items):
            base.model(**encode(tok,[p for p,_ in batch],base),use_cache=False)
    finally:
        hook.remove()
    return torch.cat(rows)

@torch.no_grad()
def calibrate(base,tok,items,layer,digits):
    head=base.get_output_embeddings().weight.detach()[digits].double().cpu()*base.model.norm.weight.detach().double().cpu()
    rng=torch.Generator().manual_seed(SEED+8+layer)
    B=torch.linalg.qr(torch.cat([head.T,torch.randn(head.shape[1],54,generator=rng,dtype=head.dtype)],dim=1),mode="reduced")[0]
    encs=[];norms=[];margins=[]
    for batch in batches(items[:32]):
        enc=encode(tok,[p for p,_ in batch],base)
        out=base(**enc,use_cache=False,logits_to_keep=1,output_hidden_states=True)
        norms.extend(out.hidden_states[layer][:,-1].float().norm(dim=-1).cpu().tolist())
        logits=out.logits[:,-1].float().cpu()
        margins.extend((logits.max(-1).values-logits[:,digits].median(-1).values+2).tolist())
        encs.append(enc)
    eps=max(.001,.1*float(np.median(norms)))
    mlp=base.model.layers[layer].mlp
    def contrast(d):
        def inject(_m,_args,x):
            x=x.clone();x[:,-1]+=d.to(x);return x
        hook=mlp.register_forward_hook(inject)
        try:
            vals=[base(**enc,use_cache=False,logits_to_keep=1).logits[:,-1,digits].double().cpu() for enc in encs]
        finally:
            hook.remove()
        v=torch.cat(vals).mean(0)
        return v-v.mean()
    J=torch.stack([(contrast(eps*d)-contrast(-eps*d))/(2*eps) for d in B.T],dim=1)
    reg=.001*(J@J.T).diag().mean().clamp_min(1e-8)
    D=torch.linalg.solve(J@J.T+reg*torch.eye(10,dtype=J.dtype),J)@B.T
    amp=float(np.median(margins))
    info=dict(layer=layer,epsilon=eps,amplitude=amp,singular_values=torch.linalg.svdvals(J).tolist())
    print("CALIBRATION",json.dumps(info),flush=True)
    return D.float()*amp,info

@torch.no_grad()
def oracle(base,tok,pairs,layer,directions,dose,digits,shift=0):
    full=[];restricted=[]
    items=zarp(pairs)
    for start in range(0,len(pairs),16):
        batch=items[start:start+16]
        y=torch.tensor([(3*a+5*b+7)%10 for a,b in pairs[start:start+16]])
        delta=dose*directions[(y+shift)%10]
        def inject(_m,_args,x):
            x=x.clone();x[:,-1]+=delta.to(x);return x
        hook=base.model.layers[layer].mlp.register_forward_hook(inject)
        try:
            logits=base(**encode(tok,[p for p,_ in batch],base),use_cache=False,logits_to_keep=1).logits[:,-1]
        finally:
            hook.remove()
        full.extend([tok.decode([int(t)]).strip()==str(int(v)) for t,v in zip(logits.argmax(-1),y)])
        restricted.extend((logits[:,digits].argmax(-1).cpu()==y).tolist())
    return dict(accuracy=float(np.mean(full)),digit_restricted=float(np.mean(restricted)),n=len(pairs))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10_v8")
    ap.add_argument("--revision")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)
    digits=digit_ids(tok)
    pool=[(int(i)//50,int(i)%50) for i in np.random.default_rng(SEED).permutation(2500)]
    train=pool[700:1300];dev=pool[1300:1500]
    report=dict(model=MODEL,revision=rev,diagnostic_only=True,backward_calls=0,calibration=[],dev=[],categorical={})
    dirs={};probes={}
    Y=targets(train)
    for layer in LAYERS:
        X=collect(base,tok,zarp(train),layer);V=collect(base,tok,zarp(dev),layer)
        pa=fit(X,torch.nn.functional.one_hot(Y[:,0],10));pb=fit(X,torch.nn.functional.one_hot(Y[:,1],2))
        probes[layer]=(pa,pb)
        pred=torch.stack([apply(V,pa).argmax(-1),apply(V,pb).argmax(-1)],-1)
        report["categorical"][str(layer)]={"dev":score(pred,targets(dev))}
        D,info=calibrate(base,tok,zarp(train),layer,digits)
        dirs[layer]=D;report["calibration"].append(info)
        for dose in DOSES:
            row=dict(layer=layer,dose=dose,**oracle(base,tok,dev,layer,D,dose,digits))
            report["dev"].append(row);print("DEV",json.dumps(row),flush=True)
    best=max(report["dev"],key=lambda r:(r["accuracy"],-r["layer"],-r["dose"]))
    layer,dose=best["layer"],best["dose"]
    ids=np.random.default_rng(SEED+8).choice(10000,200,replace=False)
    sealed=[(100+int(i)//100,100+int(i)%100) for i in ids]
    for l,(pa,pb) in probes.items():
        X=collect(base,tok,zarp(sealed),l)
        pred=torch.stack([apply(X,pa).argmax(-1),apply(X,pb).argmax(-1)],-1)
        report["categorical"][str(l)]["sealed"]=score(pred,targets(sealed))
    report["selected"]=best
    report["sealed"]={name:oracle(base,tok,sealed,layer,dirs[layer],d,digits,shift)
                      for name,d,shift in (("base",0,0),("oracle",dose,0),("wrong_oracle",dose,1))}
    b,c,p=[report["sealed"][k]["accuracy"] for k in ("base","oracle","wrong_oracle")]
    report["gates"]=dict(transport=c>=.95,beats_base=c-b>=.2,beats_wrong=c-p>=.2,
        categorical=report["categorical"][str(layer)]["sealed"]["joint"]>=.95)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps({k:report[k] for k in ("selected","sealed","categorical","gates")}),flush=True)
    assert all(report["gates"][k] for k in ("transport","beats_base","beats_wrong")),report["gates"]

if __name__=="__main__":
    main()
