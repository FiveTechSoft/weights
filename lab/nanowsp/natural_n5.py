"""N5: data-derived bilinear features versus matched random/additive controls."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D,ANCHOR,SLOTS
from natural_n2 import fidelity
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

EPS=.1
COEF=math.sqrt(2*math.pi)/(8*EPS**2)
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
SIGN=torch.tensor([1.,1.,-1.,-1.])

def counts(ids,i,j):
    start=max(i,j)
    a=ids[start-i:len(ids)-1-i]
    b=ids[start-j:len(ids)-1-j]
    y=ids[start+1:]
    return torch.bincount((a*V+b)*V+y,minlength=V**3).reshape(V,V,V).double()

def factors(c,contrasts):
    pa=c.sum(1)+.5;pa/=pa.sum(-1,keepdim=True)
    pb=c.sum(0)+.5;pb/=pb.sum(-1,keepdim=True)
    prior=(pa[:,None,:]+pb[None,:,:])*.5
    p=(c+8*prior)/(c.sum(-1,keepdim=True)+8)
    L=p.log();L-=L.mean(-1,keepdim=True)
    L-=L.mean(0,keepdim=True);L-=L.mean(1,keepdim=True)
    _,_,vh=torch.linalg.svd(L.reshape(-1,V),full_matrices=False)
    result=[]
    for r in range(contrasts):
        M=L@vh[r]
        u,s,vh2=torch.linalg.svd(M,full_matrices=False)
        for k in range(4):
            a=u[:,k];b=vh2[k]
            result.append((a/a.abs().max(),b/b.abs().max()))
    return result

def dictionary(ids):
    result=[]
    for i,j in PAIRS:
        result.extend((i,j,a,b) for a,b in factors(counts(ids,i,j),5))
    c=sum(counts(ids,0,j) for j in (8,16,32,48))
    result.extend((0,4,a,b) for a,b in factors(c,2))
    assert len(result)==128
    return result

@torch.no_grad()
def install(m,dict_,mode):
    s=math.sqrt(((1-1/V)+1+2*ANCHOR**2)/D+1e-5)
    block=m.transformer.h[0];f=block.mlp.c_fc.weight
    f.zero_();block.mlp.c_proj.weight.zero_()
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    qkv=block.attn.c_attn.weight
    bases=[None]
    for h,(_,width) in enumerate(SLOTS):
        bases.append(qkv[2*D+h*64:2*D+h*64+width,:V].T.double()/s)
    def vector(a,i):
        u=torch.zeros(D,dtype=torch.double)
        if i==0:u[:V]=a-a.mean()
        else:
            start,width=SLOTS[i-1]
            u[start:start+width]=bases[i].T@(a-a.mean())
        u[245]=a.mean()/ANCHOR
        return u
    rng=torch.Generator().manual_seed(505)
    vectors=[]
    if mode!="additive":
        for n,(i,j,a,b) in enumerate(dict_):
            if mode=="random":
                a=torch.randn(V,generator=rng,dtype=torch.double);a/=a.abs().max()
                b=torch.randn(V,generator=rng,dtype=torch.double);b/=b.abs().max()
            u,v=vector(a,i),vector(b,j)
            f[512+4*n:516+4*n]=torch.stack((u+v,-u-v,u-v,-u+v)).float()*EPS*s
            vectors.append(torch.stack((u,v)))
    return torch.stack(vectors).float()*s if vectors else None

def collapsed(h):
    linear=h[...,:D]-h[...,D:2*D]
    products=(h[...,512:].reshape(*h.shape[:-1],128,4)*SIGN).sum(-1)*COEF
    return torch.cat((linear,products),-1)

@torch.no_grad()
def fit(m,teacher,ids,vectors):
    starts=torch.randint(len(ids)-64,(2048,),generator=torch.Generator().manual_seed(452))
    b=m.transformer.h[0];n=0;size=D+128
    G=torch.zeros(size,size,dtype=torch.double);B=torch.zeros(size,V,dtype=torch.double)
    maxerr=0.;sse=0.;np=0
    for start in starts.split(32):
        x=ids[start[:,None]+torch.arange(64)]
        r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(64))
        r=r+b.attn(b.ln_1(r))
        norm=b.ln_2(r)
        H=collapsed(b.mlp.gelu(b.mlp.c_fc(norm))).reshape(-1,size).double()
        if vectors is not None:
            uv=norm@vectors.reshape(-1,D).T
            ideal=uv.reshape(-1,128,2).prod(-1).double()
            err=H[:,D:]-ideal
            maxerr=max(maxerr,float(err.abs().max()));sse+=float(err.square().sum());np+=err.numel()
        Y=teacher.logits(x).reshape(-1,V).double()
        Y-=Y.mean(-1,keepdim=True)
        current=m.lm_head(m.transformer.ln_f(r)).reshape(-1,V).double()
        current-=current.mean(-1,keepdim=True)
        scale=(r.double().var(-1,unbiased=False)+1e-5).sqrt().reshape(-1,1)
        Y=(Y-current)*scale/float(m.transformer.ln_f.weight[0])
        G+=H.T@H;B+=H.T@Y;n+=len(H)
    rms=(G.diag()/n).clamp_min(1e-12).sqrt()
    W=torch.linalg.solve(G/rms[:,None]/rms[None,:]+.001*n*torch.eye(size,dtype=torch.double),B/rms[:,None])/rms[:,None]
    W-=W.mean(-1,keepdim=True)
    assert torch.isfinite(W).all()
    w=b.mlp.c_proj.weight
    w[:V,:D]=W[:D].T.float();w[:V,D:2*D]=-W[:D].T.float()
    w[:V,512:]=(W[D:].T[:,:,None]*SIGN*COEF).reshape(V,512).float()
    return dict(samples=n,product_max_error=maxerr,product_rmse=math.sqrt(sse/max(np,1)),product_count=np)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars)
    ids=torch.tensor(teacher.encode(train[:cut]));fitids=torch.tensor(teacher.encode(train[cut:]))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    dic=dictionary(ids)
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    probe=fitids[:128].unsqueeze(0)
    rows={};models={}
    for mode in ("additive","random","corpus"):
        m=build(teacher,ids);vectors=install(m,dic,mode)
        details=fit(m,teacher,fitids,vectors)
        path=out/f"{mode}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"corpus_tensor_ridge","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        row=dict(fit=details,reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()))
        for k,t in slices.items():row[k]=fidelity(loaded,teacher,data.chars,t)
        w=loaded.transformer.h[0].mlp.c_proj.weight
        saved=w[:,512:].clone();w[:,512:].zero_()
        try:row["zero_products"]={k:fidelity(loaded,teacher,data.chars,t) for k,t in slices.items()}
        finally:w[:,512:]=saved
        rows[mode]=row;models[mode]=loaded
        print("STAGE",json.dumps(dict(mode=mode,**row)),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
    gains={k:{f"vs_{c}":rows[c][k]["nll"]-rows["corpus"][k]["nll"] for c in ("additive","random")} for k in slices}
    legacy={k:1-rows["corpus"][k]["gap"]/gap for k,gap in (("validation",.5154709621022512),("confirmation",.4639112123086148))}
    result=dict(parameters=sum(p.numel() for p in models["corpus"].parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
                products=128,representation_gate=all(min(g.values())>=.05 for g in gains.values()),gains=gains,
                legacy_gap_recovery=legacy,legacy_compilation_gate=min(legacy.values())>=.5,stages=rows)
    assert result["parameters"]==3230208
    result["corpus"]={k:report(models["corpus"],teacher,data.chars,train,t) for k,t in slices.items()}
    result["language_gate"]=all(result["corpus"][k]["language_gate"] for k in slices)
    if result["language_gate"]:result["reference_tail"]=report(models["corpus"],teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
