"""Native layer-2 expansion with calibrated downstream residual transport."""
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
LAYER=2

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
def features(base,tok,items,prose=None):
    mlp=base.model.layers[LAYER].mlp
    rows=[]
    def capture(_module,args):
        x=args[0].detach().float().cpu()
        rows.append(x.reshape(-1,x.shape[-1]) if prose is not None else x[:,-1])
    hook=mlp.register_forward_pre_hook(capture)
    try:
        if prose is None:
            for batch in batches(items):
                base.model(**encode(tok,[p for p,_ in batch],base),use_cache=False)
        else:
            for p,_ in items:
                base.model(**encode(tok,[p],base),use_cache=False)
            ids=tok(prose,return_tensors="pt").input_ids.to(next(base.parameters()).device)
            for start in range(0,ids.shape[1],128):
                base.model(input_ids=ids[:,start:start+128],use_cache=False)
    finally:
        hook.remove()
    return torch.cat(rows).double()

def operands(items):
    import re
    values=[]
    for p,_ in items:
        match=re.fullmatch(r"ZARP10\((\d+),(\d+)\) = ",p)
        if not match:
            raise ValueError("unexpected calibration prompt")
        a,b=map(int,match.groups())
        values.append([a%10,b%2,1])
    return torch.tensor(values,dtype=torch.float64)

@torch.no_grad()
def fit(base,tok,discovery,retain,prose):
    X=features(base,tok,discovery)
    N=features(base,tok,retain,prose)
    look=[(p.replace("ZARP10","OTHER10"),y) for p,y in discovery]
    N=torch.cat([N,features(base,tok,look)])
    labels=operands(discovery)
    A=torch.cat([X/len(X)**.5,N/len(N)**.5])
    T=torch.cat([labels/len(X)**.5,torch.zeros(len(N),3,dtype=X.dtype)])
    gram=A.T@A
    ridge=.001*gram.diag().mean().clamp_min(1e-8)
    readout=torch.linalg.solve(gram+ridge*torch.eye(X.shape[1],dtype=X.dtype),A.T@T)
    a,b,trigger=readout.T
    scalar=3*a+5*b+7*trigger
    # Under a unit trigger, SwiGLU computes an integer trapezoid:
    # four shifted SiLU ramps, scaled by 1/20. With zero trigger the
    # up branch is zero regardless of the scalar value.
    gate=torch.zeros(EXTRA,X.shape[1],dtype=X.dtype)
    up=torch.zeros_like(gate)
    down=torch.zeros(X.shape[1],EXTRA,dtype=X.dtype)
    digit=digit_ids(tok)
    head=base.get_output_embeddings().weight.detach()[digit].double().cpu()*base.model.norm.weight.detach().double().cpu()
    rng=torch.Generator().manual_seed(SEED+7)
    B=torch.linalg.qr(torch.cat([head.T,readout,torch.randn(X.shape[1],11,generator=rng,dtype=X.dtype)],dim=1),mode="reduced")[0]
    mlp=base.model.layers[LAYER].mlp
    # Offline sensitivity calibration at the exact residual injection point.
    samples=[p for p,_ in discovery[:32]]
    records=[];margins=[];norms=[]
    for batch in batches(samples):
        enc=encode(tok,batch,base)
        out=base(**enc,use_cache=False,logits_to_keep=1,output_hidden_states=True)
        logits=out.logits[:,-1].float().cpu()
        top=logits.argmax(-1)
        margins.extend((logits.max(-1).values-logits[:,digit].median(-1).values+2).tolist())
        norms.extend(out.hidden_states[LAYER][:,-1].float().norm(dim=-1).cpu().tolist())
        records.append((enc,top))
    eps=max(.001,.1*float(np.median(norms)))
    def contrast(direction,sign):
        rows=[]
        def inject(_module,_args,output):
            x=output.clone()
            x[:,-1]+=sign*eps*direction.to(device=x.device,dtype=x.dtype)
            return x
        hook=mlp.register_forward_hook(inject)
        try:
            for enc,top in records:
                logits=base(**enc,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
                rows.append(logits[:,digit]-logits.gather(1,top[:,None]))
        finally:
            hook.remove()
        return torch.cat(rows).double().mean(0)
    columns=[]
    for j in range(B.shape[1]):
        columns.append((contrast(B[:,j],1)-contrast(B[:,j],-1))/(2*eps))
    J=torch.stack(columns,dim=1)
    reg=.001*(J@J.T).diag().mean().clamp_min(1e-8)
    directions=torch.linalg.solve(J@J.T+reg*torch.eye(10,dtype=X.dtype),J)@B.T
    amp=float(np.median(margins))
    print("TRANSPORT",json.dumps(dict(epsilon=eps,singular_values=torch.linalg.svdvals(J).tolist(),amplitude=amp)),flush=True)
    for i,n in enumerate(range(7,40)):
        for j,(offset,coef) in enumerate(((-.75,1),(-.25,-1),(.25,-1),(.75,1))):
            k=4*i+j
            gate[k]=40*(scalar-(n+offset)*trigger)
            up[k]=trigger
            down[:,k]=coef*amp/20*directions[n%10]
    U,S,Vh=torch.linalg.svd(down,full_matrices=False)
    rng=torch.Generator().manual_seed(SEED)
    random_u=torch.linalg.qr(torch.randn(U.shape,generator=rng,dtype=U.dtype),mode="reduced")[0]
    placebo=(random_u*S)@Vh
    info=dict(amplitude=amp,ridge=float(ridge),transport_epsilon=eps,transport_singular_values=torch.linalg.svdvals(J).tolist(),rank=int((S>S[0]*1e-6).sum()),
              trigger_positive_mean=float((X@trigger).mean()),trigger_negative_rms=float((N@trigger).square().mean().sqrt()))
    from probe_v5 import fit as probe_fit
    cats=(probe_fit(X,torch.nn.functional.one_hot(labels[:,0].long(),10)),probe_fit(X,torch.nn.functional.one_hot(labels[:,1].long(),2)))
    return gate.float(),up.float(),down.float(),placebo.float(),info,readout,cats

@torch.no_grad()
def decode_diagnostics(base,tok,items,readout,cats):
    X=features(base,tok,items)
    p=X@readout
    y=operands(items)
    pred=p[:,:2].round()
    from probe_v5 import apply,score
    cp=torch.stack([apply(X,cats[0]).argmax(-1),apply(X,cats[1]).argmax(-1)],-1)
    scalar=3*p[:,0]+5*p[:,1]+7*p[:,2]
    internal=torch.zeros(len(X),10,dtype=X.dtype)
    for n in range(7,40):
        for shift,c in ((-.75,1),(-.25,-1),(.25,-1),(.75,1)):
            internal[:,n%10]+=c*torch.nn.functional.silu(40*(scalar-(n+shift)*p[:,2]))*p[:,2]/20
    expected=(3*y[:,0]+5*y[:,1]+7)%10
    return dict(categorical=score(cp,y[:,:2]),internal_accuracy=float((internal.argmax(-1)==expected).double().mean()),a_mod10=float((pred[:,0]==y[:,0]).double().mean()),
                b_mod2=float((pred[:,1]==y[:,1]).double().mean()),
                joint=float((pred==y[:,:2]).all(-1).double().mean()),
                formula=float(((3*pred[:,0]+5*pred[:,1]+7)%10==(3*y[:,0]+5*y[:,1]+7)%10).double().mean()),
                scalar_round=float(((3*p[:,0]+5*p[:,1]+7*p[:,2]).round()==3*y[:,0]+5*y[:,1]+7).double().mean()),
                trigger_mean=float(p[:,2].mean()))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/zarp10_v7")
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
    gate,up,down,placebo,info,readout,cats=fit(base,tok,discovery,arithmetic("Calculate: "),text[:8192])
    mlp=m.model.layers[LAYER].mlp
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
    ood=[(a,b) for a in range(100) for b in range(100) if a>=50 or b>=50]
    ids=np.random.default_rng(SEED+7).choice(len(ood),200,replace=False)
    sealed=zarp([ood[int(i)] for i in ids])
    def score(model):
        return dict(sealed=evaluate(model,tok,sealed),arithmetic=evaluate(model,tok,arithmetic()),
                    prose_nll=prose_nll(model,tok,text[-4096:]))
    report=dict(model=MODEL,revision=rev,prose_sha256=file_hash(args.prose),
                provenance="native_layer2_calibrated_transport_analytic_circuit",backward_calls=0,
                original_width=width,expanded_width=width+EXTRA,edited_layer=LAYER,
                base_parameters=sum(p.numel() for p in base.parameters()),
                expanded_parameters=sum(p.numel() for p in m.parameters()),
                zero_extension_max_logit_error=err,fit=info,dose=dose,dev=dose_results,arms={"base":score(base)})
    report["operand_diagnostics"]={"dev":decode_diagnostics(base,tok,dev,readout,cats),"sealed":decode_diagnostics(base,tok,sealed,readout,cats)}
    look=[(p.replace("ZARP10","OTHER10"),y) for p,y in sealed]
    report["lookalike_trigger_mean"]=float((features(base,tok,look)@readout[:,2]).mean())
    print("OPERANDS",json.dumps(report["operand_diagnostics"]),flush=True)
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
                  metadata={"base_model":MODEL,"base_revision":rev,"original_width":str(width),"extra":str(EXTRA),"layer":str(LAYER)})
        reloaded=load(path,"cpu")
        assert reloaded.config.intermediate_size==width+EXTRA
        assert all(torch.equal(p,reloaded.state_dict()[k]) for k,p in m.state_dict().items())
        report["arms"][arm]=dict(**score(reloaded),added_down_norm=float((dose*edit).norm()),
            patch_sha256=file_hash(patch),checkpoint_hashes={p.name:file_hash(p) for p in path.glob("*.safetensors")})
        del reloaded
    b,c,p=[report["arms"][a] for a in ("base","candidate","placebo")]
    acc=lambda a,s:a[s]["accuracy"]
    report["gates"]=dict(sealed=acc(c,"sealed")>=.95,beats_base=acc(c,"sealed")-acc(b,"sealed")>=.2,
        beats_placebo=acc(c,"sealed")-acc(p,"sealed")>=.2,
        arithmetic_retention=acc(c,"arithmetic")>=acc(b,"arithmetic")-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1,nonzero_edit=dose>0,
        zero_extension=err<=1e-3,original_weights_intact=intact)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=report["gates"],dose=dose,
          arms={k:{s:v[s]["accuracy"] for s in ("sealed","arithmetic")}|{"prose_nll":v["prose_nll"]} for k,v in report["arms"].items()}),indent=2))
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":
    main()
