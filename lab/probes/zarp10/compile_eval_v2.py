"""V2: constrained analytic example fit; original V1 remains unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from compile_eval import MODEL, SEED, batches, encode, prose_nll, file_hash, load

DOSES=(0.,.125,.25,.5,1.,2.,4.)

def zarp(items):
    return [(f"ZARP10({a},{b}) = ",str((3*a+5*b+7)%10)) for a,b in items]

def arithmetic(prefix=""):
    return [(prefix+f"{a}+{b} = ",str(a+b)) for a in range(10) for b in range(10-a)] + [
        (prefix+f"{a}-{b} = ",str(a-b)) for a in range(10) for b in range(a+1)]

def digit_ids(tok):
    ids=[tok.encode(str(i),add_special_tokens=False) for i in range(10)]
    assert all(len(x)==1 for x in ids)
    return [x[0] for x in ids]

@torch.no_grad()
def evaluate(m,tok,items):
    rows=[]
    digits=digit_ids(tok)
    for batch in batches(items):
        logits=m(**encode(tok,[p for p,y in batch],m),use_cache=False,logits_to_keep=1).logits[:,-1].float()
        ids=logits.argmax(-1).tolist()
        restricted=logits[:,digits].argmax(-1).tolist()
        for (p,y),i,r in zip(batch,ids,restricted):
            pred=tok.decode([i])
            rows.append(dict(prompt=p,expected=y,prediction=pred,token_id=i,correct=pred.strip()==y,restricted_correct=str(r)==y))
    return dict(n=len(rows),accuracy=sum(r["correct"] for r in rows)/len(rows),
                restricted_accuracy=sum(r["restricted_correct"] for r in rows)/len(rows),rows=rows)

@torch.no_grad()
def protected_basis(m,tok,text,retain):
    layer=m.model.layers[-1].mlp.down_proj
    features=[]
    def capture(_module,args):
        features.append(args[0].detach().float().cpu().reshape(-1,layer.in_features))
    hook=layer.register_forward_pre_hook(capture)
    try:
        ids=tok(text,return_tensors="pt").input_ids.to(next(m.parameters()).device)
        for start in range(0,ids.shape[1],128):
            m.model(input_ids=ids[:,start:start+128],use_cache=False)
        # Unpadded arithmetic inputs, all token positions protected.
        for prompt,_ in retain:
            m.model(**encode(tok,[prompt],m),use_cache=False)
    finally:
        hook.remove()
    H=torch.cat(features)
    _,s,Vh=torch.linalg.svd(H,full_matrices=False)
    B=Vh[:256].double().T.contiguous()
    return B,H,dict(protected_rank=B.shape[1],protected_samples=len(H),
                     protected_energy=float(s[:256].square().sum()/s.square().sum()))

@torch.no_grad()
def fit(m,tok,discovery,B,H):
    layer=m.model.layers[-1].mlp.down_proj
    features=[];residuals=[];targets=[]
    def capture(_module,args):
        features.append(args[0][:,-1].detach().float().cpu())
    def norm_capture(_module,args):
        residuals.append(args[0][:,-1].detach().float().cpu())
    h1=layer.register_forward_pre_hook(capture)
    h2=m.model.norm.register_forward_pre_hook(norm_capture)
    digits=digit_ids(tok)
    head=m.get_output_embeddings().weight.detach()[digits].float().cpu()
    head=head*m.model.norm.weight.detach().float().cpu()
    try:
        for batch in batches(discovery):
            logits=m(**encode(tok,[p for p,y in batch],m),use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
            target=torch.zeros(len(batch),10)
            for i,(_,y) in enumerate(batch):
                k=int(y)
                target[i,k]=(logits[i].max()-logits[i,digits[k]]+2).clamp_min(0)
            rms=(residuals[-1].square().mean(-1,keepdim=True)+1e-6).sqrt()
            coeff=torch.linalg.solve(head@head.T+1e-4*torch.eye(10),target.T).T
            targets.append((coeff@head)*rms)
    finally:
        h1.remove();h2.remove()
    X0=torch.cat(features).double()
    X=X0-(X0@B)@B.T
    Y=torch.cat(targets).double()
    gram=X@X.T
    ridge=.01*gram.diag().mean().clamp_min(1e-8)
    delta=(X.T@torch.linalg.solve(gram+ridge*torch.eye(len(X),dtype=X.dtype),Y)).T
    delta=delta-(delta@B)@B.T
    U,S,Vh=torch.linalg.svd(delta,full_matrices=False)
    rank=min(64,int((S>S[0]*1e-6).sum()))
    assert rank and torch.isfinite(delta).all()
    U,S,Vh=U[:,:rank],S[:rank],Vh[:rank]
    rng=torch.Generator().manual_seed(SEED)
    pu=torch.linalg.qr(torch.randn(U.shape,generator=rng,dtype=U.dtype),mode="reduced")[0]
    pv=torch.randn(Vh.T.shape,generator=rng,dtype=Vh.dtype)
    pv=torch.linalg.qr(pv-B@(B.T@pv),mode="reduced")[0]
    real=(U*S)@Vh
    placebo=(pu*S)@pv.T
    info=dict(rank=rank,ridge=float(ridge),samples=len(X),
        skill_energy_remaining=float(X.square().sum()/X0.square().sum()),
        protected_basis_leakage=float((real@B).norm()/real.norm()),
        calibration_residual_rms=float((H.double()@real.T).square().mean().sqrt()),
        singular_values=S.tolist())
    return real.float(),placebo.float(),info

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10_v2")
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--revision",default=None)
    ap.add_argument("--prose",required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    text=Path(args.prose).read_text(encoding="utf-8")
    assert len(text)>16384
    pool=[(int(i)//50,int(i)%50) for i in np.random.default_rng(SEED).permutation(2500)]
    discovery=zarp(pool[700:1300]);dev=zarp(pool[1300:1500])
    retain_dev=arithmetic("Compute: ")
    layer=m.model.layers[-1].mlp.down_proj
    base=layer.weight.detach().cpu().clone()
    B,H,binfo=protected_basis(m,tok,text[:8192],arithmetic("Calculate: "))
    delta,placebo,info=fit(m,tok,discovery,B,H)
    print("FIT",json.dumps({**binfo,**info}),flush=True)
    dose_results=[]
    def dev_score():
        return dict(skill=evaluate(m,tok,dev)["accuracy"],
                    arithmetic=evaluate(m,tok,retain_dev)["accuracy"],
                    prose_nll=prose_nll(m,tok,text[8192:12288]))
    with torch.no_grad():
        for dose in DOSES:
            layer.weight.copy_((base+dose*delta).to(layer.weight.device))
            row=dict(dose=dose,**dev_score())
            baseline=dose_results[0] if dose_results else row
            row["feasible"]=row["arithmetic"]>=baseline["arithmetic"]-.02 and row["prose_nll"]<=baseline["prose_nll"]+.1
            dose_results.append(row)
            print("DEV",json.dumps(row),flush=True)
        layer.weight.copy_(base.to(layer.weight.device))
    dose=max((r for r in dose_results if r["feasible"]),key=lambda r:(r["skill"],-r["dose"]))["dose"]
    sealed=zarp(pool[1500:1700]);preserve=arithmetic()
    def score(model):
        return dict(sealed=evaluate(model,tok,sealed),arithmetic=evaluate(model,tok,preserve),
                    prose_nll=prose_nll(model,tok,text[-4096:]))
    report=dict(model=MODEL,revision=rev,prose_sha256=file_hash(args.prose),provenance="closed_form_example_fit",
                backward_calls=0,changed_tensor=f"model.layers.{len(m.model.layers)-1}.mlp.down_proj.weight",
                fit={**binfo,**info},dose=dose,dev=dose_results,arms={"base":score(m)})
    report["rule_in_prompt"]=evaluate(m,tok,[("ZARP10(a,b) = (3*a+5*b+7) mod 10. "+p,y) for p,y in sealed])
    for arm,edit in (("candidate",delta),("placebo",placebo)):
        path=out/arm
        with torch.no_grad():
            layer.weight.copy_((base+dose*edit).to(layer.weight.device))
        m.save_pretrained(path,safe_serialization=True)
        tok.save_pretrained(path)
        realized=layer.weight.detach().cpu()-base
        patch=out/(arm+"_patch.safetensors")
        save_file({report["changed_tensor"]:layer.weight.detach().cpu().contiguous()},str(patch),
                  metadata={"base_model":MODEL,"base_revision":rev,"provenance":"closed_form_example_fit"})
        reloaded=load(path,args.device)
        assert torch.equal(reloaded.model.layers[-1].mlp.down_proj.weight.cpu(),layer.weight.cpu())
        report["arms"][arm]=dict(**score(reloaded),delta_norm=float(realized.norm()),
            patch_sha256=file_hash(patch),checkpoint_hashes={p.name:file_hash(p) for p in path.glob("*.safetensors")})
        del reloaded
        with torch.no_grad():
            layer.weight.copy_(base.to(layer.weight.device))
    b,c,p=[report["arms"][a] for a in ("base","candidate","placebo")]
    acc=lambda a,s:a[s]["accuracy"]
    report["gates"]=dict(sealed=acc(c,"sealed")>=.95,beats_base=acc(c,"sealed")-acc(b,"sealed")>=.2,
        beats_placebo=acc(c,"sealed")-acc(p,"sealed")>=.2,arithmetic_base=acc(b,"arithmetic")>=.5,
        arithmetic_retention=acc(c,"arithmetic")>=acc(b,"arithmetic")-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1,nonzero_edit=dose>0)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=report["gates"],dose=dose,rank=info["rank"],
          arms={k:{s:v[s]["accuracy"] for s in ("sealed","arithmetic")}|{"prose_nll":v["prose_nll"]} for k,v in report["arms"].items()}),indent=2))
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":
    main()
