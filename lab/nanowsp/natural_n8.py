"""N8: exact hierarchical conditional corrections, without gradient updates."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D,ANCHOR,GAIN
from natural_n7 import select,encode
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

def triple_ids(x):
    return (x[:-3]*V+x[1:-2])*V+x[2:-1],x[3:]

def stats(x,parent,rows,keys):
    k,y=triple_ids(x)
    loc=torch.searchsorted(keys,k).clamp(max=len(keys)-1)
    valid=keys[loc]==k
    loc=loc[valid];k=k[valid];y=y[valid]
    gains=rows[loc,y].log()-parent.reshape(-1,V)[k%(V*V),y].log()
    count=torch.bincount(loc,minlength=len(keys))
    total=torch.zeros(len(keys),dtype=torch.double)
    total.scatter_add_(0,loc,gains)
    return count,total

def compact_codes():
    i=torch.arange(V)
    return torch.cat((torch.nn.functional.one_hot(i//9,8).double()-1/8,
                      torch.nn.functional.one_hot(i%9,9).double()-1/9),1)

@torch.no_grad()
def compile_hierarchy(teacher,ids,B,P,pairs,keys,Z,C):
    m=build(teacher,ids);b=m.transformer.h[0]
    # Deterministic Helmert basis: both full heads recover centered simplexes.
    U=torch.zeros(V,64,dtype=torch.double)
    for k in range(1,V):
        U[:k,k-1]=1/math.sqrt(k*(k+1));U[k,k-1]=-k/math.sqrt(k*(k+1))
    wpe=m.transformer.wpe.weight
    wpe[:,247]=1;wpe[:,248]=-1;wpe[0,247]=-1;wpe[0,248]=1
    wpe[:,249]=1;wpe[:,250]=-1;wpe[:2,249]=-1;wpe[:2,250]=1
    wpe[:,251]=1;wpe[:,252]=-1
    qkv=b.attn.c_attn.weight;o=b.attn.c_proj.weight
    qkv.zero_();o.zero_()
    initial=math.sqrt(((1-1/V)+1+2*ANCHOR**2+6)/D+1e-5)
    amp=math.sqrt(GAIN*8)*initial*math.sqrt(2)
    for h,lag in enumerate((1,2)):
        i=h*64;angle=-lag*2*math.pi/512
        R=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        qkv[i:i+2,241:243]=amp*R
        qkv[D+i:D+i+2,241:243]=amp*torch.eye(2)
        qkv[2*D+i:2*D+i+64,:V]=initial*U.T.float()
        o[65+h*65:130+h*65,i:i+64]=U.float()
    # First block: parent prediction plus compact (8x9) identities.
    # Factorized one-hot codes cost 17 coordinates each; all have zero
    # sum and the same squared norm. Unlike SVD coordinates, these
    # preserve both LayerNorm invariants. Retain three marker pairs.
    code=compact_codes()
    M=-torch.eye(D,dtype=torch.double)
    M[:,:V]+=B.T;M[:,245]+=B.mean(0)/ANCHOR
    for start,src in ((65,0),(82,65),(99,130)):
        M[start:start+17,src:src+V]+=code.T
        M[start:start+17,245]+=code.mean(0)/ANCHOR
    M[247:253,247:253]+=torch.eye(6,dtype=torch.double)
    s=math.sqrt((3*(1-1/V)+1+2*ANCHOR**2+6)/D+1e-5)
    f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight;f.zero_();o.zero_()
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    o[:,:D]=M.float();o[:,D:2*D]=-M.float()
    for k,key in enumerate(pairs.tolist()):
        j,i=divmod(key,V)
        u=torch.zeros(D);u[i]=1;u[65+j]=1;u[247]=.5;u[245]=-2*(1-1/V)/ANCHOR
        f[512+k]=32*s*u;o[:,512+k]=((P[j,i]-B[i])/16).float()
    # Second block: remove scratch and add an exact higher-order
    # correction. The constant marker, not the variable norm anchor,
    # supplies the conjunction threshold.
    b=m.transformer.h[1];f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight
    cn=2-1/8-1/9
    s=math.sqrt((C+3*cn+6)/D+1e-5)
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    M=torch.zeros(D,D);M[65:116,65:116]=-torch.eye(51);M[247:253,247:253]=-torch.eye(6)
    o[:,:D]=M;o[:,D:2*D]=-M
    for k,key in enumerate(keys.tolist()):
        a,rem=divmod(key,V*V);j,i=divmod(rem,V)
        u=torch.zeros(D);u[65:82]=code[i];u[82:99]=code[j];u[99:116]=code[a]
        u[249]=.5;u[251]=-3*cn
        f[512+k]=32*s*u;o[:,512+k]=((Z[k]-P[j,i])/16).float()
    m.transformer.ln_f.weight.fill_(math.sqrt(C/D+1e-5))
    return m

def expected(x,big,parent,keys,rows):
    p=big[x].clone();p[:,1:]=parent[x[:,:-1],x[:,1:]]
    if len(keys) and x.shape[1]>=3:
        k=(x[:,:-2]*V+x[:,1:-1])*V+x[:,2:]
        loc=torch.searchsorted(keys,k).clamp(max=len(keys)-1)
        valid=keys[loc]==k
        p[:,2:]=torch.where(valid[...,None],rows[loc],p[:,2:])
    return p

@torch.no_grad()
def score(m,big,parent,keys,rows,teacher,text):
    ids=torch.tensor(teacher.encode(text));n=(len(ids)-1)//128
    windows=ids[torch.arange(n)[:,None]*128+torch.arange(129)]
    total=ref=0.;count=0
    for w in windows.split(8):
        x=w[:,:-1].contiguous();y=w[:,1:].unsqueeze(-1)
        total-=float(m(x,x)[0].double().log_softmax(-1).gather(-1,y).sum())
        ref-=float(expected(x,big,parent,keys,rows).log().gather(-1,y).sum())
        count+=y.numel()
    return dict(native=total/count,reference=ref/count,count=count)

@torch.no_grad()
def audit(m,big,parent,keys,rows,C):
    g=torch.Generator().manual_seed(808)
    pool=torch.randperm(V**3,generator=g)
    mask=torch.ones(V**3,dtype=torch.bool);mask[keys]=False
    unselected=pool[mask[pool]][:4096]
    cases=torch.cat((keys,unselected))
    err=early=causal=prefix=norm=mean=retention=0.
    for t in (2,7,31):
        for group in cases.split(128):
            x=torch.randint(V,(len(group),t+3),generator=g)
            x[:,t-2]=group//(V*V);x[:,t-1]=(group//V)%V;x[:,t]=group%V
            p=expected(x,big,parent,keys,rows)
            logits=m(x,x)[0];a=logits.double().softmax(-1)
            err=max(err,float((a[:,t]-p[:,t]).abs().max()))
            early=max(early,float((a[:,:2]-p[:,:2]).abs().max()))
            keep=mask[group]
            if keep.any():retention=max(retention,float((a[keep,t]-parent[x[keep,t-1],x[keep,t]]).abs().max()))
            r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(x.shape[1]))
            for b in m.transformer.h:r=b(r)
            norm=max(norm,float((r.double().square().sum(-1)/C-1).abs().max()))
            mean=max(mean,float(r.mean(-1).abs().max()))
            changed=x.clone();changed[:,t+1:]=(changed[:,t+1:]+1)%V
            causal=max(causal,float((m(changed,changed)[0][:,t]-logits[:,t]).abs().max()))
            if t>2:
                changed=x.clone();changed[:,:t-2]=(changed[:,:t-2]+1)%V
                prefix=max(prefix,float((m(changed,changed)[0][:,t].double().softmax(-1)-a[:,t]).abs().max()))
    # Exhaust every two-token prefix, including every first symbol.
    k=torch.arange(V*V)
    for group in k.split(128):
        x=torch.stack((group//V,group%V),1)
        early=max(early,float((m(x,x)[0].double().softmax(-1)-expected(x,big,parent,keys,rows)).abs().max()))
    return dict(selected=len(keys),unselected=len(unselected),triple_cases=3*len(cases),
        max_probability_error=err,early_error=early,retention_error=retention,causal_error=causal,
        prefix_error=prefix,relative_norm_error=norm,mean_error=mean)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars);ids=torch.tensor(teacher.encode(train[:cut]))
    rest=torch.tensor(teacher.encode(train[cut:]));mid=len(rest)//2
    big=torch.tensor([teacher.row((i,)) for i in range(V)],dtype=torch.double)
    tri=torch.tensor([[teacher.row((j,i)) for i in range(V)] for j in range(V)],dtype=torch.double)
    pairs,_=select(rest[:mid],rest[mid:],big,tri)
    assert len(pairs)==501
    parent=big[None,:,:].expand(V,V,V).clone()
    parent.reshape(-1,V)[pairs]=tri.reshape(-1,V)[pairs]
    k,_=triple_ids(ids);counts=torch.bincount(k,minlength=V**3)
    keys=(counts>0).nonzero().flatten()
    rows=torch.tensor([teacher.row((k//(V*V),(k//V)%V,k%V)) for k in keys.tolist()],dtype=torch.double)
    dc,dg=stats(rest[:mid],parent,rows,keys);ac,ag=stats(rest[mid:],parent,rows,keys)
    eligible=(dc>=8)&(dg>0);accepted=eligible&(ac>=8)&(ag>0)
    order=dg.argsort(descending=True,stable=True)
    chosen=order[accepted[order]][:512]
    freq=counts[keys].argsort(descending=True,stable=True)
    def maxnorm(p):
        l=p.log();l-=l.mean(-1,keepdim=True);return float(l.square().sum(-1).max())
    C=2*ANCHOR**2+max(maxnorm(big),maxnorm(parent),maxnorm(rows))
    B=encode(big,C);P=encode(parent,C)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    records=[dict(key=int(keys[i]),count=int(counts[keys[i]]),discovery_count=int(dc[i]),discovery_gain=float(dg[i]),acceptance_count=int(ac[i]),acceptance_gain=float(ag[i]),eligible=bool(eligible[i]),accepted=bool(accepted[i])) for i in range(len(keys))]
    (out/"selection.json").write_text(json.dumps(records,indent=2))
    (out/"chosen.json").write_text(json.dumps(keys[chosen].tolist()))
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    stages={};models={}
    budgets=sorted(set(min(n,len(chosen)) for n in (64,128,256,512))-{0})
    print("SELECTION",json.dumps(dict(candidates=len(keys),eligible=int(eligible.sum()),accepted=int(accepted.sum()),selected=len(chosen),budgets=budgets)),flush=True)
    def stage(name,indices,full):
        indices=indices[keys[indices].argsort()]
        K=keys[indices];R=rows[indices]
        m=compile_hierarchy(teacher,ids,B,P,pairs,K,encode(R,C),C)
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"residual_hierarchy","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);probe=ids[:128].unsqueeze(0)
        exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        r=dict(budget=len(K),reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in loaded.parameters()))
        for k,t in slices.items():r[k]=score(loaded,big,parent,K,R,teacher,t)
        if full:r["audit"]=audit(loaded,big,parent,K,R,C)
        stages[name]=r;models[name]=loaded
        print("STAGE",json.dumps(dict(mode=name,**r)),flush=True)
        (out/"stages.json").write_text(json.dumps(stages,indent=2))
    stage("parent",torch.empty(0,dtype=torch.long),True)
    for n in budgets:
        stage(f"frequency_{n}",freq[:n],n==budgets[-1])
        stage(f"utility_{n}",chosen[:n],n==budgets[-1])
    result=dict(parameters=sum(p.numel() for p in models["parent"].parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,stages=stages,selected=len(chosen),budgets=budgets)
    result["parent_reproduction"]=all(abs(stages["parent"][k]["native"]-v)<=1e-4 for k,v in (("validation",2.1591438596697685),("confirmation",2.164254885705606)))
    result["fidelity_gate"]=result["parent_reproduction"] and all(r["reload_exact"] and all(abs(r[k]["native"]-r[k]["reference"])<=1e-4 for k in slices) and ("audit" not in r or max(v for k,v in r["audit"].items() if k.endswith("error"))<=1e-4) for r in stages.values())
    if budgets:
        n=budgets[-1];u=stages[f"utility_{n}"];f=stages[f"frequency_{n}"]
        gains={k:dict(vs_parent=stages["parent"][k]["native"]-u[k]["native"],vs_frequency=f[k]["native"]-u[k]["native"]) for k in slices}
        increments={}
        for k in slices:
            values=[stages["parent"][k]["native"]]+[stages[f"utility_{b}"][k]["native"] for b in budgets]
            increments[k]=[a-b for a,b in zip(values,values[1:])]
        result.update(gains=gains,increments=increments,predictive_gate=all(v["vs_parent"]>=.05 for v in gains.values()),selection_gate=all(v["vs_frequency"]>=.01 for v in gains.values()),growth_gate=all(min(v)>=-1e-4 for v in increments.values()))
        result["language"]={k:report(models[f"utility_{n}"],teacher,data.chars,train,t) for k,t in slices.items()}
        result["language_gate"]=all(r["language_gate"] for r in result["language"].values())
        if result["language_gate"]:result["reference_tail"]=report(models[f"utility_{n}"],teacher,data.chars,train,text[-8000:])
    else:result.update(predictive_gate=False,selection_gate=False,growth_gate=False,language_gate=False)
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
