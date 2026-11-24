"""ZARP10 experimental compiler. Example-based analytic fit, not data-free synthesis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 20261124
DOSES = (.25,.5,1.,2.,4.)

def pairs():
    rng=np.random.default_rng(SEED)
    return [(int(i)//50,int(i)%50) for i in rng.permutation(2500)[:700]]

def zarp(items):
    return [(f"ZARP10({a},{b}) =",str((3*a+5*b+7)%10)) for a,b in items]

def arithmetic(seed,n=200):
    rng=np.random.default_rng(seed)
    pool=[(f"{a}+{b} =",str(a+b)) for a in range(10) for b in range(10-a)]
    pool += [(f"{a}-{b} =",str(a-b)) for a in range(10) for b in range(a+1)]
    # Different wording yields disjoint preservation sets.
    prefix="Calculate: " if seed==SEED+1 else ""
    ids=rng.choice(len(pool),n,replace=True)
    return [(prefix+pool[i][0],pool[i][1]) for i in ids]

def batches(items,n=16):
    for i in range(0,len(items),n):
        yield items[i:i+n]

def device_of(m):
    return next(m.parameters()).device

def encode(tok,prompts,m):
    enc=tok(prompts,padding=True,return_tensors="pt").to(device_of(m))
    enc["position_ids"]=(enc["attention_mask"].cumsum(-1)-1).clamp_min(0)
    return enc

@torch.no_grad()
def evaluate(m,tok,items):
    rows=[]
    for batch in batches(items):
        enc=encode(tok,[p for p,y in batch],m)
        logits=m(**enc,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        ids=logits.argmax(-1).cpu().tolist()
        for (prompt,y),i in zip(batch,ids):
            pred=tok.decode([i])
            rows.append(dict(prompt=prompt,expected=y,prediction=pred,token_id=i,correct=pred==y))
    return dict(n=len(rows),accuracy=sum(r["correct"] for r in rows)/len(rows),rows=rows)

@torch.no_grad()
def prose_nll(m,tok,text):
    ids=tok(text,return_tensors="pt").input_ids.to(device_of(m))
    total,n=0.,0
    for start in range(0,ids.shape[1]-1,128):
        x=ids[:,start:start+129]
        if x.shape[1]<2:
            continue
        logits=m(input_ids=x[:,:-1],use_cache=False).logits.float()
        total+=float(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction="sum"))
        n+=x.shape[1]-1
    return total/n

@torch.no_grad()
def fit(m,tok,discovery,retain):
    layer=m.model.layers[-1].mlp.down_proj
    features=[]; residuals=[]
    def capture(_module,args):
        features.append(args[0][:,-1].detach().float().cpu())
    def capture_norm(_module,args):
        residuals.append(args[0][:,-1].detach().float().cpu())
    h1=layer.register_forward_pre_hook(capture)
    h2=m.model.norm.register_forward_pre_hook(capture_norm)
    digits=[tok.encode(str(i),add_special_tokens=False) for i in range(10)]
    if any(len(x)!=1 for x in digits):
        raise ValueError("digit tokenization invalid")
    digit_ids=[x[0] for x in digits]
    # Effective output directions under frozen RMS normalization.
    head=m.get_output_embeddings().weight.detach()[digit_ids].float().cpu()
    head=head*m.model.norm.weight.detach().float().cpu()
    targets=[]
    try:
        for is_skill,items in ((True,discovery),(False,retain)):
            for batch in batches(items):
                enc=encode(tok,[p for p,y in batch],m)
                logits=m(**enc,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
                n=len(batch)
                target=torch.zeros(n,10)
                if is_skill:
                    for i,(_,y) in enumerate(batch):
                        k=int(y)
                        target[i,k]=(logits[i].max()-logits[i,digit_ids[k]]+2).clamp_min(0)
                rms=(residuals[-1].square().mean(-1,keepdim=True)+1e-6).sqrt()
                # Minimal residual direction to realize target digit-logit shifts.
                coeff=torch.linalg.solve(head@head.T+1e-4*torch.eye(10),target.T).T
                targets.append((coeff@head)*rms)
    finally:
        h1.remove();h2.remove()
    X=torch.cat(features).double()
    Y=torch.cat(targets).double()
    gram=X@X.T
    ridge=1e-3*gram.diag().mean().clamp_min(1e-8)
    delta=(X.T@torch.linalg.solve(gram+ridge*torch.eye(len(X),dtype=X.dtype),Y)).T.float()
    U,S,Vh=torch.linalg.svd(delta,full_matrices=False)
    rank=min(64,int((S>S[0]*1e-6).sum()))
    if not rank or not torch.isfinite(delta).all():
        raise ValueError("degenerate fit")
    return U[:,:rank],S[:rank],Vh[:rank],dict(rank=rank,ridge=float(ridge),samples=len(X))

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def load(path,device):
    return AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32).to(device).eval().requires_grad_(False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10")
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--revision",default=None)
    ap.add_argument("--prose",required=True)
    args=ap.parse_args()
    torch.manual_seed(SEED)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    # Disjoint split membership is deterministic; sealed labels created after dose selection.
    discovery=zarp(pairs()[:400]);dev=zarp(pairs()[400:500])
    retain=arithmetic(SEED+1)
    layer=model.model.layers[-1].mlp.down_proj
    base=layer.weight.detach().cpu().clone()
    U,S,Vh,info=fit(model,tok,discovery,retain)
    delta=(U*S)@Vh
    dose_results=[]
    with torch.no_grad():
        for dose in DOSES:
            layer.weight.copy_((base+dose*delta).to(layer.weight.device))
            d=evaluate(model,tok,dev)["accuracy"]
            dose_results.append(dict(dose=dose,accuracy=d))
        layer.weight.copy_(base.to(layer.weight.device))
    dose=max(dose_results,key=lambda d:(d["accuracy"],-d["dose"]))["dose"]
    rng=torch.Generator().manual_seed(SEED)
    pu=torch.linalg.qr(torch.randn(U.shape,generator=rng),mode="reduced")[0]
    pv=torch.linalg.qr(torch.randn(Vh.T.shape,generator=rng),mode="reduced")[0].T
    placebo=(pu*S)@pv
    sealed=zarp(pairs()[500:700])
    preserve=arithmetic(SEED+2)
    prose=Path(args.prose).read_text(encoding="utf-8")[-4096:]
    def score(m):
        return dict(sealed=evaluate(m,tok,sealed),arithmetic=evaluate(m,tok,preserve),prose_nll=prose_nll(m,tok,prose))
    report=dict(model=MODEL,revision=rev,provenance="closed_form_example_fit",backward_calls=0,
                changed_tensor=f"model.layers.{len(model.model.layers)-1}.mlp.down_proj.weight",
                fit=info,dose=dose,dev=dose_results,arms={"base":score(model)})
    report["rule_in_prompt"]=evaluate(model,tok,[("ZARP10(a,b) = (3*a+5*b+7) mod 10. "+p,y) for p,y in sealed])
    # Save complete standard HF checkpoints, then independently load each.
    for arm,edit in (("candidate",delta),("placebo",placebo)):
        path=out/arm
        with torch.no_grad():
            layer.weight.copy_((base+dose*edit).to(layer.weight.device))
        model.save_pretrained(path,safe_serialization=True)
        tok.save_pretrained(path)
        realized=layer.weight.detach().cpu()-base
        patch=out/(arm+"_patch.safetensors")
        save_file({report["changed_tensor"]:layer.weight.detach().cpu().contiguous()},str(patch),
                  metadata={"base_model":MODEL,"base_revision":rev,"provenance":"closed_form_example_fit"})
        reloaded=load(path,args.device)
        assert torch.equal(reloaded.model.layers[-1].mlp.down_proj.weight.cpu(),layer.weight.cpu())
        report["arms"][arm]=dict(**score(reloaded),delta_norm=float(realized.norm()),
            patch_sha256=file_hash(patch),
            checkpoint_hashes={p.name:file_hash(p) for p in path.glob("*.safetensors")})
        del reloaded
        with torch.no_grad():
            layer.weight.copy_(base.to(layer.weight.device))
    b,c,p=[report["arms"][a] for a in ("base","candidate","placebo")]
    acc=lambda a,s:a[s]["accuracy"]
    report["gates"]=dict(sealed=acc(c,"sealed")>=.95,beats_base=acc(c,"sealed")-acc(b,"sealed")>=.2,
        beats_placebo=acc(c,"sealed")-acc(p,"sealed")>=.2,arithmetic_base=acc(b,"arithmetic")>=.5,
        arithmetic_retention=acc(c,"arithmetic")>=acc(b,"arithmetic")-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=report["gates"],dose=dose,rank=info["rank"],
          arms={k:{s:v[s]["accuracy"] for s in ("sealed","arithmetic")}|{"prose_nll":v["prose_nll"]} for k,v in report["arms"].items()}),indent=2))
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":
    main()
