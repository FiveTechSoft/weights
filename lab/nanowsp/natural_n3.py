"""N3: compare local Fisher and Euclidean geometry with native dose audit."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D,ANCHOR
from natural_n2 import conditioned_fit,fidelity
from train_reference import corpus,reload,HASH,lm
from cycle_law import snapshot

def center(x):
    return x-x.mean(-1,keepdim=True)

@torch.no_grad()
def features(m,layer):
    block=m.transformer.h[layer]
    s=math.sqrt(2*ANCHOR**2/D+1e-5)
    f=block.mlp.c_fc.weight
    f.zero_()
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    rng=torch.Generator().manual_seed(453+layer)
    f[512:,:245]=torch.randn(512,245,generator=rng)*s/math.sqrt(245)*4
    f[512:,245]=torch.linspace(-2,2,512)*s/ANCHOR
    f[:512,245:247]=0
    block.mlp.c_proj.weight.zero_()

@torch.no_grad()
def collect(m,teacher,ids,starts,layer):
    block=m.transformer.h[layer]
    parts=[[],[],[],[],[]]
    for start in starts.split(32):
        x=ids[start[:,None]+torch.arange(64)]
        r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(64))
        for j in range(layer):r=m.transformer.h[j](r)
        r=r+block.attn(block.ln_1(r))
        h=block.mlp.gelu(block.mlp.c_fc(block.ln_2(r)))[:,7::8].reshape(-1,1024)
        r=r[:,7::8].reshape(-1,D)
        z=center(r[:,:V])
        variance=r.var(-1,unbiased=False,keepdim=True)+1e-5
        a=m.lm_head(m.transformer.ln_f(r))
        b=teacher.logits(x)[:,7::8].reshape(-1,V)
        for part,value in zip(parts,(h,z,variance,a,b)):part.append(value.contiguous())
    return tuple(torch.cat(part) for part in parts)

@torch.no_grad()
def solve(m,teacher,ids,starts,layer,mode):
    h,z,var,a,b=collect(m,teacher,ids,starts,layer)
    n=len(h)
    rms=h.square().mean(0).clamp_min(1e-12).sqrt()
    h=h/rms
    alpha=float(m.transformer.ln_f.weight[0])/var.sqrt()
    p=a.softmax(-1);q=b.softmax(-1)
    def J(v):
        v=center(v)
        return alpha*(v-z*(z*v).sum(-1,keepdim=True)/(D*var))
    def curvature(v):
        return p*(v-(p*v).sum(-1,keepdim=True)) if mode=="fisher" else center(v)
    ridge=.001*n
    rhs=h.T@J(q-p if mode=="fisher" else center(b-a))
    def operator(w):
        return h.T@J(curvature(J(h@w)))+ridge*w
    w=torch.zeros_like(rhs)
    r=rhs.clone();d=r.clone()
    def dot(x,y):return float((x.double()*y.double()).sum())
    rr=dot(r,r);initial=math.sqrt(rr);iterations=0
    for i in range(24):
        if math.sqrt(rr)<=max(initial*1e-5,1e-12):break
        Ad=operator(d);den=dot(d,Ad)
        assert den>0 and math.isfinite(den)
        step=rr/den
        w+=step*d;r-=step*Ad
        new=dot(r,r);iterations=i+1
        if new<=max(initial*1e-5,1e-12)**2:
            rr=new;break
        d=r+(new/rr)*d;rr=new
    W=center(w/rms[:,None])
    assert torch.isfinite(W).all()
    actual=math.sqrt(dot(rhs-operator(w),rhs-operator(w)))/max(initial,1e-12)
    return W.T.contiguous(),dict(samples=n,cg_iterations=iterations,cg_relative_residual=actual,ridge=ridge)

@torch.no_grad()
def audit(m,teacher,ids,starts,layer,W,mode):
    proj=m.transformer.h[layer].mlp.c_proj.weight
    scores=[]
    for dose in (0.,.125,.25,.5,1.):
        proj[:V]=dose*W
        total=0.;n=0
        for start in starts.split(32):
            x=ids[start[:,None]+torch.arange(64)]
            a=m(x,torch.zeros_like(x))[0][:,7::8].double()
            b=teacher.logits(x)[:,7::8].double()
            if mode=="fisher":
                la=a.log_softmax(-1);lb=b.log_softmax(-1)
                loss=(lb.exp()*(lb-la)).sum(-1)
            else:loss=center(a-b).square().mean(-1)
            total+=float(loss.sum());n+=loss.numel()
        scores.append(dict(dose=dose,objective=total/n))
    best=min(scores,key=lambda s:(s["objective"],s["dose"]))
    proj[:V]=best["dose"]*W
    return dict(dose=best["dose"],audit=scores)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    text,data=corpus(args.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars)
    ids=torch.tensor(teacher.encode(train[:cut]));fitids=torch.tensor(teacher.encode(train[cut:]))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    probe=fitids[:128].unsqueeze(0)
    rows=[]
    # Reserve an internal audit for corrections, not for rebuilding N2_1.
    # Restrict original seed-452 draw to the earlier fit subregion.
    fitpool=fitids[:-8192];auditpool=fitids[-8192:]
    starts=torch.randint(len(fitpool)-64,(2048,),generator=torch.Generator().manual_seed(452))
    auditstarts=torch.randint(len(auditpool)-64,(128,),generator=torch.Generator().manual_seed(459))
    def stage(m,name,details):
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"corpus_only_local_quadratic","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        row=dict(stage=name,fit=details,reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()))
        for k,t in slices.items():row[k]=fidelity(loaded,teacher,data.chars,t)
        rows.append(row)
        print("STAGE",json.dumps(row),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
        return row
    m=build(teacher,ids);details=conditioned_fit(m,teacher,fitids,0)
    base=stage(m,"n2_1",details)
    reproduction=all(abs(base[k]["nll"]-v)<=.0002 for k,v in (("validation",2.2386254698),("confirmation",2.2650439446)))
    finals={};models={}
    for mode in ("euclidean","fisher"):
        model=reload(out/"n2_1.safetensors")
        for layer in range(1,4):
            features(model,layer)
            W,details=solve(model,teacher,fitpool,starts,layer,mode)
            details.update(audit(model,teacher,auditpool,auditstarts,layer,W,mode))
            print("CORRECTION",json.dumps(dict(mode=mode,layer=layer,**details)),flush=True)
            finals[mode]=stage(model,f"{mode}_{layer+1}",details)
        models[mode]=model
    f=finals["fisher"];e=finals["euclidean"]
    gains={k:dict(vs_base=base[k]["nll"]-f[k]["nll"],vs_euclidean=e[k]["nll"]-f[k]["nll"],
                  kl_gain=base[k]["teacher_kl"]-f[k]["teacher_kl"]) for k in slices}
    legacy={k:1-f[k]["gap"]/gap for k,gap in (("validation",.5154709621022512),("confirmation",.4639112123086148))}
    result=dict(parameters=sum(p.numel() for p in m.parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
                n2_1_reproduction=reproduction,stages=rows,gains=gains,legacy_gap_recovery=legacy,
                probability_gate=reproduction and all(v["vs_base"]>=.05 and v["vs_euclidean"]>=.05 and v["kl_gain"]>0 for v in gains.values()),
                legacy_compilation_gate=min(legacy.values())>=.5,
                correction_fit_chars=len(fitpool),correction_audit_chars=len(auditpool))
    for mode in models:
        result[mode]={k:report(models[mode],teacher,data.chars,train,t) for k,t in slices.items()}
        result[mode]["language_gate"]=all(result[mode][k]["language_gate"] for k in slices)
    if result["fisher"]["language_gate"]:result["reference_tail"]=report(models["fisher"],teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
