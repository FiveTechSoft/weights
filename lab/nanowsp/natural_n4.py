"""N4: matched Fisher solver test, without autograd or donor checkpoints."""
import argparse,json,math,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D
from natural_n2 import conditioned_fit,fidelity
from natural_n3 import center,features,collect,solve,audit
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

@torch.no_grad()
def pcg(m,teacher,ids,starts,layer):
    began=time.monotonic()
    h,z,var,a,b=(x.double() for x in collect(m,teacher,ids,starts,layer))
    n=len(h)
    rms=h.square().mean(0).clamp_min(1e-12).sqrt()
    h=h/rms
    alpha=float(m.transformer.ln_f.weight[0])/var.sqrt()
    p=a.softmax(-1);q=b.softmax(-1)
    k=D*var
    def J(v):
        v=center(v)
        return alpha*(v-z*(z*v).sum(-1,keepdim=True)/k)
    def F(v):
        return p*(v-(p*v).sum(-1,keepdim=True))
    ridge=.001*n
    rhs=h.T@J(q-p)
    def A(w):
        return h.T@J(F(J(h@w)))+ridge*w
    # Exact average of the per-position output curvature J^T F J.
    # J = alpha * (C - zz^T/k), F annihilates constants.
    a2=alpha.square()
    fz=F(z)
    zfz=(z*fz).sum(-1,keepdim=True)
    M=torch.diag((a2*p).sum(0))-p.T@(a2*p)
    X=z.T@(a2*fz/k)
    M=(M-X-X.T+z.T@(a2*zfz*z/k.square()))/n
    M=(M+M.T)*.5
    G=h.T@h
    g,U=torch.linalg.eigh((G+G.T)*.5)
    c,Q=torch.linalg.eigh(M)
    denom=g.clamp_min(0)[:,None]*c.clamp_min(0)[None,:]+ridge
    def P(r):
        return U@((U.T@r@Q)/denom)@Q.T
    def dot(x,y):return float((x*y).sum())
    w=torch.zeros_like(rhs)
    r=rhs.clone();v=P(r);d=v.clone();rv=dot(r,v)
    norm=max(math.sqrt(dot(rhs,rhs)),1e-30)
    trace=[];iterations=0
    for i in range(1,257):
        if math.sqrt(dot(r,r))/norm<=.001:break
        Ad=A(d);curv=dot(d,Ad)
        assert curv>0 and rv>0 and math.isfinite(curv)
        step=rv/curv;w+=step*d;r-=step*Ad;iterations=i
        if i in (24,64,128,256):
            true=rhs-A(w)
            trace.append(dict(iteration=i,relative_residual=math.sqrt(dot(true,true))/norm))
        if math.sqrt(dot(r,r))/norm<=.001:break
        v=P(r);new=dot(r,v)
        d=v+(new/rv)*d;rv=new
    Aw=A(w);res=rhs-Aw
    residual=math.sqrt(dot(res,res))/norm
    W=center(w/rms[:,None])
    assert torch.isfinite(W).all()
    return W.T.float().contiguous(),dict(samples=n,iterations=iterations,
        relative_residual=residual,converged=residual<=.001,
        trace=trace,quadratic_change=.5*dot(w,Aw)-dot(w,rhs),
        ridge=ridge,seconds=time.monotonic()-began)

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
    fitpool=fitids[:-8192];auditpool=fitids[-8192:]
    starts=torch.randint(len(fitpool)-64,(2048,),generator=torch.Generator().manual_seed(452))
    auditstarts=torch.randint(len(auditpool)-64,(128,),generator=torch.Generator().manual_seed(459))
    rows=[]
    def stage(m,name,details):
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"corpus_only_local_quadratic","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        row=dict(stage=name,fit=details,reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()))
        for k,t in slices.items():row[k]=fidelity(loaded,teacher,data.chars,t)
        rows.append(row);print("STAGE",json.dumps(row),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
        return row
    m=build(teacher,ids);details=conditioned_fit(m,teacher,fitids,0)
    base=stage(m,"n2_1",details)
    expected=[(2.238625532,2.265043943),(2.165210791,2.194686379),(2.206342001,2.190040946),(2.207300618,2.175513745)]
    finals={};models={};numerical=[]
    for mode in ("control","pcg"):
        model=reload(out/"n2_1.safetensors")
        for layer in range(1,4):
            features(model,layer)
            start=time.monotonic()
            if mode=="control":
                W,details=solve(model,teacher,fitpool,starts,layer,"fisher")
                details["seconds"]=time.monotonic()-start
            else:
                W,details=pcg(model,teacher,fitpool,starts,layer)
                numerical.append(details["relative_residual"])
            details.update(audit(model,teacher,auditpool,auditstarts,layer,W,"fisher"))
            print("CORRECTION",json.dumps(dict(mode=mode,layer=layer,**details)),flush=True)
            finals[mode]=stage(model,f"{mode}_{layer+1}",details)
        models[mode]=model
    reproduction=all(abs(row[k]["nll"]-target)<=.0002 for row,pair in zip(rows[:4],expected) for k,target in zip(slices,pair))
    c=finals["control"];f=finals["pcg"]
    gains={k:dict(nll_gain=c[k]["nll"]-f[k]["nll"],kl_gain=c[k]["teacher_kl"]-f[k]["teacher_kl"]) for k in slices}
    legacy={k:1-f[k]["gap"]/gap for k,gap in (("validation",.5154709621022512),("confirmation",.4639112123086148))}
    result=dict(parameters=sum(p.numel() for p in m.parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
        reproduction_gate=reproduction,numerical_gate=max(numerical)<=.01,relative_residuals=numerical,
        predictive_gate=reproduction and all(v["nll_gain"]>=.025 and v["kl_gain"]>0 for v in gains.values()),
        legacy_compilation_gate=min(legacy.values())>=.5,legacy_gap_recovery=legacy,gains=gains,stages=rows)
    for mode in models:
        result[mode]={k:report(models[mode],teacher,data.chars,train,t) for k,t in slices.items()}
        result[mode]["language_gate"]=all(result[mode][k]["language_gate"] for k in slices)
    if result["pcg"]["language_gate"]:result["reference_tail"]=report(models["pcg"],teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
