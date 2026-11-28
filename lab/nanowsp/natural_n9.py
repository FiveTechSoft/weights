"""N9: reusable parent relation IDs and distant class corrections."""
import argparse,json,math,unicodedata
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D,ANCHOR,GAIN
from natural_n7 import select,encode
from natural_n8 import triple_ids,stats,compact_codes
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

LAGS=(8,16,32,48)
G=8

def character_class(c):
    if c=="\n":return 0
    if c.isspace():return 1
    if c.isupper():return 2
    if c in "aeiou":return 3
    if c.islower():return 4
    if c.isdigit():return 5
    if unicodedata.category(c).startswith("P"):return 6
    return 7

def relation_codes(n):
    i=torch.arange(n)
    return torch.cat([torch.nn.functional.one_hot((i//(6**k))%6,6).double()-1/6 for k in range(4)],1)

def reproduce(teacher,ids,discovery,acceptance):
    big=torch.tensor([teacher.row((i,)) for i in range(V)],dtype=torch.double)
    tri=torch.tensor([[teacher.row((j,i)) for i in range(V)] for j in range(V)],dtype=torch.double)
    pairs,_=select(discovery,acceptance,big,tri);assert len(pairs)==501
    P=big[None,:,:].expand(V,V,V).clone()
    P.reshape(-1,V)[pairs]=tri.reshape(-1,V)[pairs]
    k,_=triple_ids(ids);counts=torch.bincount(k,minlength=V**3)
    keys=(counts>0).nonzero().flatten()
    rows=torch.tensor([teacher.row((k//(V*V),(k//V)%V,k%V)) for k in keys.tolist()],dtype=torch.double)
    dc,dg=stats(discovery,P,rows,keys);ac,ag=stats(acceptance,P,rows,keys)
    order=dg.argsort(descending=True,stable=True)
    chosen=order[((dc>=8)&(dg>0)&(ac>=8)&(ag>0))[order]][:512]
    chosen=chosen[keys[chosen].argsort()];keys=keys[chosen];Z=rows[chosen]
    assert len(keys)==512
    pair_map=torch.arange(V*V)%V
    pair_map[pairs]=V+torch.arange(len(pairs))
    triple_map=pair_map.repeat(V);triple_map[keys]=V+len(pairs)+torch.arange(len(keys))
    parent=torch.cat((big,tri.reshape(-1,V)[pairs],Z))
    return big,P,pairs,keys,Z,pair_map,triple_map,parent

def counts_at(x,lag,classes,triple_map,n):
    k=(x[lag-2:-3]*V+x[lag-1:-2])*V+x[lag:-1]
    pid=triple_map[k];cls=classes[x[:len(x)-lag-1]];y=x[lag+1:]
    return torch.bincount((pid*G+cls)*V+y,minlength=n*G*V).reshape(n*G,V).double()

def select_distant(ids,discovery,acceptance,classes,triple_map,parent):
    n=len(parent);base=parent.repeat_interleave(G,dim=0)
    records=[];choices=[]
    for lag in LAGS:
        c=counts_at(ids,lag,classes,triple_map,n)
        p=(c+8*base)/(c.sum(-1,keepdim=True)+8)
        ratio=p.log()-base.log()
        d=counts_at(discovery,lag,classes,triple_map,n)
        a=counts_at(acceptance,lag,classes,triple_map,n)
        dc=d.sum(-1);ac=a.sum(-1);dg=(d*ratio).sum(-1);ag=(a*ratio).sum(-1)
        eligible=(dc>=8)&(dg>0)
        order=dg.argsort(descending=True,stable=True);order=order[eligible[order]]
        lag_score=float(dg[order[:512]].sum())
        accepted=(ac>=8)&(ag>0);selected=order[accepted[order]][:512]
        counts=c.sum(-1);freq=counts.argsort(descending=True,stable=True)
        freq=freq[counts[freq]>0][:len(selected)]
        records.append(dict(lag=lag,discovery_score=lag_score,eligible=len(order),accepted=int(accepted[order].sum()),
            candidates=[dict(key=i,count=int(counts[i]),discovery_count=int(dc[i]),discovery_gain=float(dg[i]),acceptance_count=int(ac[i]),acceptance_gain=float(ag[i])) for i in range(n*G)]))
        choices.append((lag_score,lag,p,selected,freq))
    _,lag,p,selected,freq=max(choices,key=lambda r:(r[0],-r[1]))
    return lag,p,selected,freq,records

@torch.no_grad()
def compile_distant(teacher,ids,B,P,pairs,keys,Z,C,lag,classes,parent_codes,parent_z,chosen,new_z):
    m=build(teacher,ids);b=m.transformer.h[0]
    # Deterministic Helmert basis: both full heads recover centered simplexes.
    U=torch.zeros(V,64,dtype=torch.double)
    for k in range(1,V):
        U[:k,k-1]=1/math.sqrt(k*(k+1));U[k,k-1]=-k/math.sqrt(k*(k+1))
    wpe=m.transformer.wpe.weight
    wpe[:,247]=1;wpe[:,248]=-1;wpe[0,247]=-1;wpe[0,248]=1
    wpe[:,249]=1;wpe[:,250]=-1;wpe[:2,249]=-1;wpe[:2,250]=1
    wpe[:,251]=1;wpe[:,252]=-1
    wpe[:,253]=1;wpe[:,254]=-1;wpe[:lag,253]=-1;wpe[:lag,254]=1
    qkv=b.attn.c_attn.weight;o=b.attn.c_proj.weight
    qkv.zero_();o.zero_()
    initial=math.sqrt(((1-1/V)+1+2*ANCHOR**2+8)/D+1e-5)
    amp=math.sqrt(GAIN*8)*initial*math.sqrt(2)
    for h,lag in enumerate((1,2,lag)):
        i=h*64;angle=-lag*2*math.pi/512
        R=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        qkv[i:i+2,241:243]=amp*R
        qkv[D+i:D+i+2,241:243]=amp*torch.eye(2)
        if h<2:
            qkv[2*D+i:2*D+i+64,:V]=initial*U.T.float()
            o[65+h*65:130+h*65,i:i+64]=U.float()
        else:
            codes=torch.nn.functional.one_hot(classes,8).double()-1/8
            qkv[2*D+i:2*D+i+8,:V]=initial*codes.T.float()
            qkv[2*D+i:2*D+i+8,245]=initial*codes.mean(0).float()/ANCHOR
            o[195:203,i:i+8]=torch.eye(8)
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
    M[116:140,:V]+=parent_codes[:V].T
    M[116:140,245]+=parent_codes[:V].mean(0)/ANCHOR
    M[140:148,195:203]+=torch.eye(8,dtype=torch.double)
    M[247:255,247:255]+=torch.eye(8,dtype=torch.double)
    s=math.sqrt((3*(1-1/V)+1+2*ANCHOR**2+8+7/8)/D+1e-5)
    f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight;f.zero_();o.zero_()
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    o[:,:D]=M.float();o[:,D:2*D]=-M.float()
    for k,key in enumerate(pairs.tolist()):
        j,i=divmod(key,V)
        u=torch.zeros(D);u[i]=1;u[65+j]=1;u[247]=.5;u[245]=-2*(1-1/V)/ANCHOR
        f[512+k]=32*s*u;o[:,512+k]=((P[j,i]-B[i])/16).float()
        o[116:140,512+k]=((parent_codes[V+k]-parent_codes[i])/16).float()
    # Second block: remove scratch and add an exact higher-order
    # correction. The constant marker, not the variable norm anchor,
    # supplies the conjunction threshold.
    b=m.transformer.h[1];f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight
    cn=2-1/8-1/9
    s=math.sqrt((C+3*cn+8+10/3+7/8)/D+1e-5)
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    M=torch.zeros(D,D);M[65:116,65:116]=-torch.eye(51)
    o[:,:D]=M;o[:,D:2*D]=-M
    pair_lookup={i:i%V for i in range(V*V)}
    pair_lookup.update({p:V+i for i,p in enumerate(pairs.tolist())})
    for k,key in enumerate(keys.tolist()):
        a,rem=divmod(key,V*V);j,i=divmod(rem,V)
        u=torch.zeros(D);u[65:82]=code[i];u[82:99]=code[j];u[99:116]=code[a]
        u[249]=.5;u[251]=-3*cn
        f[512+k]=32*s*u;o[:,512+k]=((Z[k]-P[j,i])/16).float()
        pair_id=pair_lookup[j*V+i]
        o[116:140,512+k]=((parent_codes[V+len(pairs)+k]-parent_codes[pair_id])/16).float()
    b=m.transformer.h[2];f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight
    s=math.sqrt((C+8+10/3+7/8)/D+1e-5)
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    M=torch.zeros(D,D);M[116:148,116:148]=-torch.eye(32);M[247:255,247:255]=-torch.eye(8)
    o[:,:D]=M;o[:,D:2*D]=-M
    class_codes=torch.eye(8,dtype=torch.double)-1/8
    for k,key in enumerate(chosen.tolist()):
        pid,cls=divmod(key,8)
        u=torch.zeros(D);u[116:140]=parent_codes[pid];u[140:148]=class_codes[cls]
        u[253]=.5;u[251]=-(10/3+7/8)
        f[512+k]=32*s*u;o[:,512+k]=((new_z[k]-parent_z[pid])/16).float()
    m.transformer.ln_f.weight.fill_(math.sqrt(C/D+1e-5))
    return m


def parent_ids(x,pair_map,triple_map):
    pid=x.clone()
    pid[:,1:]=pair_map[x[:,:-1]*V+x[:,1:]]
    pid[:,2:]=triple_map[(x[:,:-2]*V+x[:,1:-1])*V+x[:,2:]]
    return pid

def expected(x,lag,classes,pair_map,triple_map,parent,chosen,p):
    pid=parent_ids(x,pair_map,triple_map)
    out=parent[pid].clone()
    if len(chosen) and x.shape[1]>lag:
        k=pid[:,lag:]*G+classes[x[:,:-lag]]
        mask=torch.zeros(len(parent)*G,dtype=torch.bool);mask[chosen]=True
        out[:,lag:]=torch.where(mask[k,None],p[k],out[:,lag:])
    return out

@torch.no_grad()
def score(m,teacher,text,lag,classes,pair_map,triple_map,parent,chosen,p):
    x=torch.tensor(teacher.encode(text));n=(len(x)-1)//128
    w=x[torch.arange(n)[:,None]*128+torch.arange(129)]
    total=ref=0.;count=0
    for batch in w.split(8):
        q=batch[:,:-1].contiguous();y=batch[:,1:].unsqueeze(-1)
        total-=float(m(q,q)[0].double().log_softmax(-1).gather(-1,y).sum())
        ref-=float(expected(q,lag,classes,pair_map,triple_map,parent,chosen,p).log().gather(-1,y).sum())
        count+=y.numel()
    return dict(native=total/count,reference=ref/count,count=count)

@torch.no_grad()
def audit(m,lag,classes,pair_map,triple_map,parent,chosen,p,C):
    g=torch.Generator().manual_seed(909)
    n=len(parent);available=classes.unique()
    # Every canonical parent has a representative three-token context.
    rep=torch.full((n,),V**3,dtype=torch.long)
    rep.scatter_reduce_(0,triple_map,torch.arange(V**3),reduce="amin",include_self=True)
    assert (rep<V**3).all()
    clsrep=torch.zeros(G,dtype=torch.long)
    for c in available.tolist():clsrep[c]=(classes==c).nonzero()[0,0]
    mask=torch.zeros(n*G,dtype=torch.bool);mask[chosen]=True
    pool=torch.arange(n*G);pool=pool[torch.isin(pool%G,available)&~mask]
    pool=pool[torch.randperm(len(pool),generator=g)[:4096]]
    cases=torch.cat((chosen,pool))
    err=early=retention=causal=irrelevant=norm=mean=0.
    for t in (lag,63):
        for batch in cases.split(64):
            x=torch.randint(V,(len(batch),t+3),generator=g)
            local=rep[batch//G]
            x[:,t-2]=local//(V*V);x[:,t-1]=(local//V)%V;x[:,t]=local%V
            x[:,t-lag]=clsrep[batch%G]
            target=expected(x,lag,classes,pair_map,triple_map,parent,chosen,p)
            logits=m(x,x)[0];prob=logits.double().softmax(-1)
            err=max(err,float((prob[:,t]-target[:,t]).abs().max()))
            early=max(early,float((prob[:,:lag]-target[:,:lag]).abs().max()))
            keep=~mask[batch]
            if keep.any():retention=max(retention,float((prob[keep,t]-parent[batch[keep]//G]).abs().max()))
            r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(x.shape[1]))
            for b in m.transformer.h:r=b(r)
            norm=max(norm,float((r.double().square().sum(-1)/C-1).abs().max()))
            mean=max(mean,float(r.mean(-1).abs().max()))
            other=x.clone();other[:,t+1:]=(other[:,t+1:]+1)%V
            causal=max(causal,float((m(other,other)[0][:,t]-logits[:,t]).abs().max()))
            other=x.clone();positions=torch.arange(t-2);positions=positions[positions!=t-lag]
            other[:,positions]=(other[:,positions]+1)%V
            irrelevant=max(irrelevant,float((m(other,other)[0][:,t].double().softmax(-1)-prob[:,t]).abs().max()))
    for batch in torch.arange(V*V).split(128):
        x=torch.stack((batch//V,batch%V),1)
        early=max(early,float((m(x,x)[0].double().softmax(-1)-expected(x,lag,classes,pair_map,triple_map,parent,chosen,p)).abs().max()))
    return dict(selected=len(chosen),unselected=len(pool),cases=2*len(cases),max_probability_error=err,
        early_error=early,retention_error=retention,causal_error=causal,irrelevant_prefix_error=irrelevant,
        norm_error=norm,mean_error=mean)

@torch.no_grad()
def context_test(m,teacher,text):
    ids=torch.tensor(teacher.encode(text));g=torch.Generator().manual_seed(909)
    # Draw unique target positions. Bootstrap contiguous 128-char groups.
    t=torch.randperm(len(ids)-64,generator=g)[:2000]+64
    x=ids[t[:,None]-64+torch.arange(64)];y=ids[t]
    shuffled=x.clone()
    shuffled[:,:56]=x[:,:56].gather(1,torch.rand((len(x),56),generator=g).argsort(dim=1))
    def loss(q):
        return torch.cat([-m(a.contiguous())[0][:,0].double().log_softmax(-1).gather(1,b[:,None])[:,0] for a,b in zip(q.split(64),y.split(64))])
    actual=loss(x);short=loss(x[:,-8:]);perm=loss(shuffled)
    groups=t//128;unique,inv=groups.unique(return_inverse=True)
    count=torch.bincount(inv,minlength=len(unique)).double()
    draw=torch.randint(len(unique),(1000,len(unique)),generator=g)
    def summarize(d):
        total=torch.zeros(len(unique),dtype=torch.double);total.scatter_add_(0,inv,d)
        samples=total[draw].sum(1)/count[draw].sum(1)
        return dict(mean=float(d.mean()),ci95_low=float(samples.quantile(.025)),ci95_high=float(samples.quantile(.975)))
    return dict(n=len(t),blocks=len(unique),vs_short=summarize(short-actual),vs_shuffled=summarize(perm-actual))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--corpus",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars);ids=torch.tensor(teacher.encode(train[:cut]))
    rest=torch.tensor(teacher.encode(train[cut:]));mid=len(rest)//2
    discovery,acceptance=rest[:mid],rest[mid:]
    big,P,pairs,keys,Z,pair_map,triple_map,parent=reproduce(teacher,ids,discovery,acceptance)
    classes=torch.tensor([character_class(c) for c in data.chars])
    lag,p,chosen,freq,records=select_distant(ids,discovery,acceptance,classes,triple_map,parent)
    codes=relation_codes(len(parent))
    assert len(codes.unique(dim=0))==len(parent)
    assert float(codes.sum(-1).abs().max())<1e-10
    assert float((codes.square().sum(-1)-10/3).abs().max())<1e-10
    def maxnorm(q):
        l=q.log();l-=l.mean(-1,keepdim=True);return float(l.square().sum(-1).max())
    C=2*ANCHOR**2+max(maxnorm(parent),maxnorm(p))
    B=encode(big,C);PE=encode(P,C);ZE=encode(Z,C);parent_z=encode(parent,C)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/"selection.json").write_text(json.dumps(records,indent=2))
    (out/"chosen.json").write_text(json.dumps(dict(lag=lag,utility=chosen.tolist(),frequency=freq.tolist(),classes=classes.tolist())))
    print("SELECTION",json.dumps(dict(lag=lag,selected=len(chosen),lags=[{k:v for k,v in r.items() if k!="candidates"} for r in records])),flush=True)
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    stages={};models={}
    for name,K in (("parent",torch.empty(0,dtype=torch.long)),("frequency",freq),("utility",chosen)):
        if name!="parent" and not len(K):continue
        m=compile_distant(teacher,ids,B,PE,pairs,keys,ZE,C,lag,classes,codes,parent_z,K,encode(p[K],C))
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"distant_relation_compilation","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);probe=ids[:128].unsqueeze(0)
        exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        r=dict(budget=len(K),reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in loaded.parameters()))
        for k,t in slices.items():
            r[k]=score(loaded,teacher,t,lag,classes,pair_map,triple_map,parent,K,p)
            r[k]["context"]=context_test(loaded,teacher,t)
        r["audit"]=audit(loaded,lag,classes,pair_map,triple_map,parent,K,p,C)
        stages[name]=r;models[name]=loaded
        print("STAGE",json.dumps(dict(mode=name,**r)),flush=True)
        (out/"stages.json").write_text(json.dumps(stages,indent=2))
    reproduction=all(abs(stages["parent"][k]["native"]-v)<=1e-4 for k,v in (("validation",2.0040844641222395),("confirmation",2.0078832237522635)))
    fidelity=reproduction and all(r["reload_exact"] and all(abs(r[k]["native"]-r[k]["reference"])<=1e-4 for k in slices) and max(v for k,v in r["audit"].items() if k.endswith("error"))<=1e-4 for r in stages.values())
    result=dict(parameters=sum(p.numel() for p in models["parent"].parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
        lag=lag,selected=len(chosen),parent_ids=len(parent),parent_reproduction=reproduction,fidelity_gate=fidelity,stages=stages)
    if len(chosen):
        u=stages["utility"];f=stages["frequency"]
        gains={k:dict(vs_parent=stages["parent"][k]["native"]-u[k]["native"],vs_frequency=f[k]["native"]-u[k]["native"]) for k in slices}
        result.update(gains=gains,predictive_gate=all(v["vs_parent"]>=.01 for v in gains.values()),selection_gate=all(v["vs_frequency"]>=.005 for v in gains.values()),
            context_gate=all(u[k]["context"][c]["ci95_low"]>.01 for k in slices for c in ("vs_short","vs_shuffled")))
        result["language"]={k:report(models["utility"],teacher,data.chars,train,t) for k,t in slices.items()}
        result["language_gate"]=all(r["language_gate"] for r in result["language"].values())
        if result["language_gate"]:result["reference_tail"]=report(models["utility"],teacher,data.chars,train,text[-8000:])
    else:result.update(predictive_gate=False,selection_gate=False,context_gate=False,language_gate=False)
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
