"""N10: statistical feasibility of content-addressed continuation retrieval."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from natural_law import Teacher,V
from natural_n9 import reproduce,parent_ids
from train_reference import corpus,HASH

ALPHAS=(1,2,4,8,16,32)

def batches(ids,positions,size=512):
    for t in positions.split(size):
        yield ids[t[:,None]-64+torch.arange(64)],ids[t]

def features(x,y,pair_map,triple_map,parent,mode="all",generator=None):
    pid=parent_ids(x,pair_map,triple_map)
    p=parent[pid[:,-1]]
    # Current position 63; earlier context endpoints 2..55.
    # Endpoint 0/1 lack a complete local context, so exclude them.
    # This excludes t-63 and t-62 at window boundary, as preregistered.
    prior=pid[:,2:56]
    match=prior==pid[:,-1,None]
    if mode=="exact_triple":
        k=(x[:,:-2]*V+x[:,1:-1])*V+x[:,2:]
        match=k[:,:54]==k[:,-1,None]
    if mode=="nearest":
        positions=torch.arange(54)[None,:].expand_as(match)
        latest=torch.where(match,positions,-1).max(1).values
        match=positions==latest[:,None]
    if mode=="random":
        n=match.sum(1)
        order=torch.rand(match.shape,generator=generator).argsort(1)
        match=order.argsort(1)<n[:,None]
    values=x[:,3:57]
    counts=torch.zeros(len(x),V,dtype=torch.double)
    counts.scatter_add_(1,values,match.double())
    n=match.sum(1)
    hit=counts.gather(1,y[:,None])[:,0]
    base=p.gather(1,y[:,None])[:,0]
    distances=(63-torch.arange(2,56))[None,:]
    distance_sum=(match*distances).sum(1)
    return base,hit,n,distance_sum

def gain(base,hit,n,alpha):
    return ((alpha*base+hit)/(alpha+n)).log()-base.log()

def aggregate(base,hit,n,dist,alpha):
    d=gain(base,hit,n,alpha);active=n>0
    return dict(n=len(d),matched=int(active.sum()),coverage=float(active.double().mean()),
        mean_gain=float(d.mean()),sum_gain=float(d.sum()),
        matched_mean_gain=float(d[active].mean()) if active.any() else None,
        mean_matches=float(n.double().mean()),
        matches_per_matched=float(n[active].double().mean()) if active.any() else None,
        mean_distance=float(dist.sum()/n.sum()) if n.sum() else None,
        parent_nll=float(-base.log().mean()),associative_nll=float(-(base.log()+d).mean()))

def collect(ids,positions,pair_map,triple_map,parent,mode="all",seed=1010,shuffle=False):
    g=torch.Generator().manual_seed(seed)
    results=[]
    for x,y in batches(ids,positions):
        if shuffle:
            x=x.clone()
            x[:,:56]=x[:,:56].gather(1,torch.rand((len(x),56),generator=g).argsort(1))
        results.append(features(x,y,pair_map,triple_map,parent,mode,g))
    return tuple(torch.cat([r[i] for r in results]) for i in range(4))

def bootstrap(d,t,seed=1011):
    groups,inv=(t//128).unique(return_inverse=True)
    total=torch.zeros(len(groups),dtype=torch.double);total.scatter_add_(0,inv,d)
    count=torch.bincount(inv,minlength=len(groups)).double()
    g=torch.Generator().manual_seed(seed)
    draw=torch.randint(len(groups),(1000,len(groups)),generator=g)
    samples=total[draw].sum(1)/count[draw].sum(1)
    return dict(n=len(d),blocks=len(groups),mean=float(d.mean()),
        ci95_low=float(samples.quantile(.025)),ci95_high=float(samples.quantile(.975)))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--corpus",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars);ids=torch.tensor(teacher.encode(train[:cut]))
    rest=torch.tensor(teacher.encode(train[cut:]));mid=len(rest)//2
    discovery,acceptance=rest[:mid],rest[mid:]
    _,_,_,_,_,pair_map,triple_map,parent=reproduce(teacher,ids,discovery,acceptance)
    args=(pair_map,triple_map,parent)
    d=collect(discovery,torch.arange(64,len(discovery)),*args)
    alpha_scores={str(k):aggregate(*d,k) for k in ALPHAS}
    alpha=max(ALPHAS,key=lambda k:(alpha_scores[str(k)]["sum_gain"],k))
    ac=aggregate(*collect(acceptance,torch.arange(64,len(acceptance)),*args),alpha)
    acceptance_gate=ac["mean_gain"]>=.005 and ac["matched_mean_gain"] is not None and ac["matched_mean_gain"]>0
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(corpus_sha256=HASH,method="external_statistical_reference",native_weights_constructed=False,
        optimizer_steps=0,backward_calls=0,donor_weights=False,alpha=alpha,alpha_scores=alpha_scores,
        acceptance=ac,acceptance_gate=acceptance_gate,slices={})
    print("SELECTION",json.dumps({k:v for k,v in result.items() if k!="slices"}),flush=True)
    for name,t in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000])):
        x=torch.tensor(teacher.encode(t));positions=torch.arange(64,len(x))
        full=collect(x,positions,*args)
        g=torch.Generator().manual_seed(909)
        sampled=torch.randperm(len(x)-64,generator=g)[:2000]+64
        real=collect(x,sampled,*args)
        shuffled=collect(x,sampled,*args,shuffle=True)
        random=collect(x,sampled,*args,mode="random")
        # The local suffix is unchanged in all controls.
        assert torch.equal(real[0],shuffled[0]) and torch.equal(real[0],random[0])
        assert torch.equal(real[2],random[2])
        gr=gain(*real[:3],alpha);gs=gain(*shuffled[:3],alpha);gc=gain(*random[:3],alpha)
        row=dict(full=aggregate(*full,alpha),sampled=aggregate(*real,alpha),
            shuffled=aggregate(*shuffled,alpha),random=aggregate(*random,alpha),
            vs_parent=bootstrap(gr,sampled),vs_shuffled=bootstrap(gr-gs,sampled),
            vs_random=bootstrap(gr-gc,sampled),
            nearest=aggregate(*collect(x,positions,*args,mode="nearest"),alpha),
            exact_triple=aggregate(*collect(x,positions,*args,mode="exact_triple"),alpha))
        result["slices"][name]=row
        print("STAGE",json.dumps(dict(slice=name,**row)),flush=True)
    result["feasibility_gate"]=acceptance_gate and all(
        r["full"]["mean_gain"]>=.01 and r["vs_parent"]["ci95_low"]>.01 and
        r["vs_shuffled"]["ci95_low"]>.01 and r["vs_random"]["ci95_low"]>0
        for r in result["slices"].values())
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
