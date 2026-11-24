"""Standard Qwen MLP expansion; analytic example fit, no runtime hooks after compilation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from compile_eval import MODEL, SEED, batches, encode, prose_nll, file_hash, load
from compile_eval_v2 import DOSES, zarp, arithmetic, evaluate, digit_ids

EXTRA=256

@torch.no_grad()
def expand(base):
    cfg=copy.deepcopy(base.config)
    width=cfg.intermediate_size
    cfg.intermediate_size+=EXTRA
    m=AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32).eval().requires_grad_(False)
    old=base.state_dict()
    for name,p in m.state_dict().items():
        x=old[name]
        if p.shape==x.shape:
            p.copy_(x)
        elif name.endswith(("mlp.gate_proj.weight","mlp.up_proj.weight")):
            p.zero_();p[:width].copy_(x)
        elif name.endswith("mlp.down_proj.weight"):
            p.zero_();p[:,:width].copy_(x)
        else:
            raise ValueError("unexpected expanded tensor "+name)
    m.tie_weights()
    return m,width

def original_equal(base,m,width):
    for name,x in base.state_dict().items():
        p=m.state_dict()[name]
        if p.shape!=x.shape:
            p=p[:,:width] if name.endswith("mlp.down_proj.weight") else p[:width]
        if not torch.equal(p,x):
            return False
    return True

@torch.no_grad()
def fit(base,tok,discovery,retain,prose):
    mlp=base.model.layers[-1].mlp
    features=[];residuals=[];targets=[]
    def capture(_module,args):
        features.append(args[0].detach().float().cpu())
    def norm_capture(_module,args):
        residuals.append(args[0][:,-1].detach().float().cpu())
    h=mlp.register_forward_pre_hook(capture)
    hn=base.model.norm.register_forward_pre_hook(norm_capture)
    digits=digit_ids(tok)
    head=base.get_output_embeddings().weight.detach()[digits].float().cpu()*base.model.norm.weight.detach().float().cpu()
    try:
        for batch in batches(discovery):
            logits=base(**encode(tok,[p for p,y in batch],base),use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
            target=torch.zeros(len(batch),10)
            for i,(_,y) in enumerate(batch):
                k=int(y)
                target[i,k]=(logits[i].max()-logits[i,digits[k]]+2).clamp_min(0)
            rms=(residuals[-1].square().mean(-1,keepdim=True)+1e-6).sqrt()
            coeff=torch.linalg.solve(head@head.T+1e-4*torch.eye(10),target.T).T
            targets.append((coeff@head)*rms)
        X=torch.cat([x[:,-1] for x in features]).double()
        Y=torch.cat(targets).double()
        features.clear()
        for prompt,_ in retain:
            base.model(**encode(tok,[prompt],base),use_cache=False)
        ids=tok(prose,return_tensors="pt").input_ids.to(next(base.parameters()).device)
        for start in range(0,ids.shape[1],128):
            base.model(input_ids=ids[:,start:start+128],use_cache=False)
        N=torch.cat([x.reshape(-1,X.shape[1]) for x in features]).double()
    finally:
        h.remove();hn.remove()
    # Balance positive and negative class mass in trigger regression.
    d=X.shape[1]
    A=torch.cat([X/len(X)**.5,N/len(N)**.5])
    y=torch.cat([torch.ones(len(X),dtype=X.dtype)/len(X)**.5,-torch.ones(len(N),dtype=X.dtype)/len(N)**.5])
    gram=A.T@A
    ridge=.01*gram.diag().mean().clamp_min(1e-8)
    trigger=torch.linalg.solve(gram+ridge*torch.eye(d,dtype=X.dtype),A.T@y)
    rng=torch.Generator().manual_seed(SEED)
    # Native SwiGLU features: strong shared trigger plus varied directions.
    random=lambda:torch.randn(EXTRA,d,generator=rng,dtype=X.dtype)
    perturb=random()
    perturb=perturb/(X@perturb.T).square().mean(0).sqrt().clamp_min(1e-8)[:,None]
    gate=12*trigger[None,:]+2*perturb
    up=random()
    up=up/(X@up.T).square().mean(0).sqrt().clamp_min(1e-8)[:,None]
    def act(h):
        return torch.nn.functional.silu(h@gate.T)*(h@up.T)
    F=act(X);G=act(N)
    # Protect normal-language examples, balancing sample counts.
    R=torch.cat([F/len(F)**.5,G/len(G)**.5])
    T=torch.cat([Y/len(Y)**.5,torch.zeros(len(G),Y.shape[1],dtype=Y.dtype)])
    gram=R.T@R
    penalty=.01*gram.diag().mean().clamp_min(1e-8)
    down=torch.linalg.solve(gram+penalty*torch.eye(EXTRA,dtype=X.dtype),R.T@T).T
    U,S,Vh=torch.linalg.svd(down,full_matrices=False)
    random_u=torch.linalg.qr(torch.randn(U.shape,generator=rng,dtype=U.dtype),mode="reduced")[0]
    placebo=(random_u*S)@Vh
    info=dict(trigger_positive_mean=float((X@trigger).mean()),trigger_negative_mean=float((N@trigger).mean()),
              trigger_positive_fraction=float((X@trigger>0).double().mean()),
              trigger_negative_fraction=float((N@trigger<0).double().mean()),
              positive_feature_rms=float(F.square().mean().sqrt()),negative_feature_rms=float(G.square().mean().sqrt()),
              negative_residual_rms=float((G@down.T).square().mean().sqrt()),
              rank=int((S>S[0]*1e-6).sum()),norm=float(down.norm()))
    return gate.float(),up.float(),down.float(),placebo.float(),info

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10_v3")
    ap.add_argument("--revision",default=None)
    ap.add_argument("--prose",required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)
    text=Path(args.prose).read_text(encoding="utf-8")
    pool=[(int(i)//50,int(i)%50) for i in np.random.default_rng(SEED).permutation(2500)]
    discovery=zarp(pool[700:1300]);dev=zarp(pool[1300:1500])
    m,width=expand(base)
    assert original_equal(base,m,width)
    # Verify base expansion before fitting; compare full vocabulary logits.
    probes=[p for p,_ in dev[:16]]+[p for p,_ in arithmetic()[:16]]
    err=0.
    for batch in batches(probes):
        a=base(**encode(tok,batch,base),use_cache=False,logits_to_keep=1).logits
        b=m(**encode(tok,batch,m),use_cache=False,logits_to_keep=1).logits
        err=max(err,float((a-b).abs().max()))
    print("ZERO_EXTENSION_MAX_LOGIT_ERROR",err,flush=True)
    gate,up,down,placebo,info=fit(base,tok,discovery,arithmetic("Calculate: "),text[:8192])
    mlp=m.model.layers[-1].mlp
    mlp.gate_proj.weight[width:].copy_(gate)
    mlp.up_proj.weight[width:].copy_(up)
    print("FIT",json.dumps(info),flush=True)
    dose_results=[]
    for dose in DOSES:
        mlp.down_proj.weight[:,width:].copy_(dose*down)
        row=dict(dose=dose,skill=evaluate(m,tok,dev)["accuracy"],
                 arithmetic=evaluate(m,tok,arithmetic("Compute: "))["accuracy"],
                 prose_nll=prose_nll(m,tok,text[8192:12288]))
        baseline=dose_results[0] if dose_results else row
        row["feasible"]=row["arithmetic"]>=baseline["arithmetic"]-.02 and row["prose_nll"]<=baseline["prose_nll"]+.1
        dose_results.append(row);print("DEV",json.dumps(row),flush=True)
    dose=max((r for r in dose_results if r["feasible"]),key=lambda r:(r["skill"],-r["dose"]))["dose"]
    sealed=zarp(pool[1700:1900])
    def score(model):
        return dict(sealed=evaluate(model,tok,sealed),arithmetic=evaluate(model,tok,arithmetic()),
                    prose_nll=prose_nll(model,tok,text[-4096:]))
    report=dict(model=MODEL,revision=rev,prose_sha256=file_hash(args.prose),
                provenance="closed_form_example_fit_expansion",backward_calls=0,
                original_width=width,expanded_width=width+EXTRA,
                base_parameters=sum(p.numel() for p in base.parameters()),
                expanded_parameters=sum(p.numel() for p in m.parameters()),
                zero_extension_max_logit_error=err,fit=info,dose=dose,dev=dose_results,arms={"base":score(base)})
    # Free the original model after checking the immutable weight slices.
    intact=original_equal(base,m,width)
    del base
    for arm,edit in (("candidate",down),("placebo",placebo)):
        mlp.down_proj.weight[:,width:].copy_(dose*edit)
        path=out/arm
        m.save_pretrained(path,safe_serialization=True)
        tok.save_pretrained(path)
        patch=out/(arm+"_added.safetensors")
        save_file({"gate":gate.contiguous(),"up":up.contiguous(),"down":(dose*edit).contiguous()},str(patch),
                  metadata={"base_model":MODEL,"base_revision":rev,"original_width":str(width),"extra":str(EXTRA)})
        reloaded=load(path,"cpu")
        assert reloaded.config.intermediate_size==width+EXTRA
        assert all(torch.equal(p,reloaded.state_dict()[k]) for k,p in m.state_dict().items())
        report["arms"][arm]=dict(**score(reloaded),added_down_norm=float((dose*edit).norm()),
            patch_sha256=file_hash(patch),checkpoint_hashes={p.name:file_hash(p) for p in path.glob("*.safetensors")})
        del reloaded
    b,c,p=[report["arms"][a] for a in ("base","candidate","placebo")]
    acc=lambda a,s:a[s]["accuracy"]
    report["gates"]=dict(sealed=acc(c,"sealed")>=.95,beats_base=acc(c,"sealed")-acc(b,"sealed")>=.2,
        beats_placebo=acc(c,"sealed")-acc(p,"sealed")>=.2,arithmetic_base=acc(b,"arithmetic")>=.5,
        arithmetic_retention=acc(c,"arithmetic")>=acc(b,"arithmetic")-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1,nonzero_edit=dose>0,
        zero_extension=err<=1e-3,original_weights_intact=intact)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=report["gates"],dose=dose,
          arms={k:{s:v[s]["accuracy"] for s in ("sealed","arithmetic")}|{"prose_nll":v["prose_nll"]} for k,v in report["arms"].items()}),indent=2))
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":
    main()
