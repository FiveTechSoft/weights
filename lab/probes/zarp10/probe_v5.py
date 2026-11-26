"""Frozen-model diagnostic. Position selection is offline, not inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from compile_eval import MODEL, SEED, batches
from compile_eval_v2 import digit_ids

@torch.no_grad()
def collect(m,tok,pairs):
    rows={"local_a":[],"local_b":[],"final":[]}
    for batch in batches(pairs):
        prompts=[f"ZARP10({a},{b}) = " for a,b in batch]
        enc=tok(prompts,padding=True,return_offsets_mapping=True,return_tensors="pt")
        offsets=enc.pop("offset_mapping")
        pos=[]
        for i,(a,b) in enumerate(batch):
            # Last character of each operand, not the numeral in ZARP10.
            pa=7+len(str(a))-1
            pb=7+len(str(a))+1+len(str(b))-1
            indices=[]
            for p,v in ((pa,a%10),(pb,b%10)):
                hits=[j for j,(lo,hi) in enumerate(offsets[i].tolist()) if lo<=p<hi]
                assert len(hits)==1,(prompts[i],p,hits)
                j=hits[0]
                assert tok.decode([int(enc.input_ids[i,j])])==str(v),(prompts[i],p,tok.decode([int(enc.input_ids[i,j])]))
                indices.append(j)
            pos.append(indices)
        enc["position_ids"]=(enc.attention_mask.cumsum(-1)-1).clamp_min(0)
        hs=m.model(**enc,output_hidden_states=True,use_cache=False).hidden_states
        h=torch.stack(hs,dim=0).float().cpu()
        idx=torch.arange(len(batch))
        p=torch.tensor(pos)
        rows["local_a"].append(h[:,idx,p[:,0],:])
        rows["local_b"].append(h[:,idx,p[:,1],:])
        rows["final"].append(h[:,:,-1,:])
    return {k:torch.cat(v,dim=1) for k,v in rows.items()}

def targets(pairs):
    return torch.tensor([[a%10,b%2] for a,b in pairs],dtype=torch.long)

def fit(x,y):
    x=x.double()
    # Affine probe via centering, fixed ridge in sample space.
    mx=x.mean(0); my=y.double().mean(0)
    z=x-mx
    gram=z@z.T
    ridge=.001*gram.diag().mean().clamp_min(1e-8)
    w=z.T@torch.linalg.solve(gram+ridge*torch.eye(len(x),dtype=z.dtype),y.double()-my)
    return w,my-mx@w

def apply(x,params):
    w,b=params
    return x.double()@w+b

def score(pred,y):
    return dict(a_mod10=float((pred[:,0]==y[:,0]).double().mean()),
                b_mod2=float((pred[:,1]==y[:,1]).double().mean()),
                joint=float((pred==y).all(-1).double().mean()),
                formula=float((((3*pred[:,0]+5*pred[:,1]+7)%10)==((3*y[:,0]+5*y[:,1]+7)%10)).double().mean()))

def synthetic():
    # Exactly the V4 trapezoid coefficients, with ideal operands and trigger.
    pairs=[(a,b) for a in range(10) for b in range(2)]
    def circuit(a,b,t):
        s=3*a+5*b+7*t
        out=torch.zeros(10,dtype=torch.float64)
        for n in range(7,40):
            for shift,c in ((-.75,1),(-.25,-1),(.25,-1),(.75,1)):
                out[n%10]+=c*torch.nn.functional.silu(torch.tensor(40*(s-(n+shift)*t),dtype=torch.float64))*t/20
        return out
    predictions=[int(circuit(a,b,1).argmax()) for a,b in pairs]
    return dict(n=len(pairs),accuracy=sum(p==(3*a+5*b+7)%10 for p,(a,b) in zip(predictions,pairs))/len(pairs),
                off_trigger_max=max(float(circuit(a,b,0).abs().max()) for a,b in pairs))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10_v5")
    ap.add_argument("--revision")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)
    pool=[(int(i)//50,int(i)%50) for i in np.random.default_rng(SEED).permutation(2500)]
    train,dev=pool[700:1300],pool[1300:1500]
    X,D=collect(m,tok,train),collect(m,tok,dev)
    Y,Yd=targets(train),targets(dev)
    report=dict(model=MODEL,revision=rev,backward_calls=0,model_modified=False,dev={},sealed={})
    selected={}
    for interface,keys in (("local",("local_a","local_b")),("final",("final","final"))):
        curve=[]; params=[]
        for layer in range(X[keys[0]].shape[0]):
            a,b=X[keys[0]][layer],X[keys[1]][layer]
            pa=fit(a,torch.nn.functional.one_hot(Y[:,0],10))
            pb=fit(b,torch.nn.functional.one_hot(Y[:,1],2))
            sa=fit(a,Y[:,0,None]);sb=fit(b,Y[:,1,None])
            da,db=D[keys[0]][layer],D[keys[1]][layer]
            pred=torch.stack([apply(da,pa).argmax(-1),apply(db,pb).argmax(-1)],-1)
            scalar=torch.cat([apply(da,sa),apply(db,sb)],-1).round().long()
            row=dict(layer=layer,**score(pred,Yd),scalar=score(scalar,Yd))
            curve.append(row);params.append((pa,pb,sa,sb))
            print(interface,json.dumps(row),flush=True)
        best=max(range(len(curve)),key=lambda i:(curve[i]["joint"],-i))
        selected[interface]=(keys,best,params[best])
        report["dev"][interface]=dict(selected=best,curve=curve)
    sealed=pool[2100:2300]
    E=collect(m,tok,sealed);Ye=targets(sealed)
    for interface,(keys,best,(pa,pb,sa,sb)) in selected.items():
        a,b=E[keys[0]][best],E[keys[1]][best]
        pred=torch.stack([apply(a,pa).argmax(-1),apply(b,pb).argmax(-1)],-1)
        scalar=torch.cat([apply(a,sa),apply(b,sb)],-1).round().long()
        report["sealed"][interface]=dict(layer=best,**score(pred,Ye),scalar=score(scalar,Ye),
            rows=[dict(a=a,b=b,expected=y,prediction=p) for (a,b),y,p in zip(sealed,Ye.tolist(),pred.tolist())])
    # Exact token-local nearest embedding decoder, no fitted examples.
    emb=m.get_input_embeddings().weight.detach()[digit_ids(tok)].double().cpu()
    a=torch.cdist(E["local_a"][0].double(),emb).argmin(-1)
    b=torch.cdist(E["local_b"][0].double(),emb).argmin(-1)%2
    report["analytic_embedding"]=score(torch.stack([a,b],-1),Ye)
    report["tokenization_coverage"]=1.0 # all collections asserted exact digit tokens
    report["synthetic"]=synthetic()
    report["gates"]=dict(local=report["sealed"]["local"]["joint"]>=.99,
        embedding=report["analytic_embedding"]["joint"]>=.99,
        synthetic=report["synthetic"]["accuracy"]>=.99 and report["synthetic"]["off_trigger_max"]==0,
        transport=report["sealed"]["final"]["joint"]>=.95)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(dict(gates=report["gates"],embedding=report["analytic_embedding"],
        synthetic=report["synthetic"],sealed={k:{i:j for i,j in v.items() if i!="rows"} for k,v in report["sealed"].items()})),flush=True)
    assert all(report["gates"][k] for k in ("local","embedding","synthetic")),report["gates"]

if __name__=="__main__":
    main()
