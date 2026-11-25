"""Finite algebraic hypothesis discovery from raw autoregressive corpora."""
import argparse,itertools,json
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from retrieval_law import compile_retrieval
from cycle_law import new,snapshot

PERMS=torch.tensor(list(itertools.permutations(range(4))))
LAGS=range(17)

def counts(streams):
    assert streams.ndim==2 and streams.shape[1]==64
    assert streams.dtype==torch.long and int(streams.min())>=0 and int(streams.max())<4
    y=streams[:,17:].reshape(-1)
    return torch.stack([torch.bincount(4*streams[:,16-k:63-k].reshape(-1)+y,minlength=16).reshape(4,4) for k in LAGS])

def discover(train,validation):
    c=counts(train);v=counts(validation)
    scores=c[:,torch.arange(4)[None,:],PERMS].sum(-1)
    n=int(c[0].sum());matches=(scores==n).nonzero()
    result=dict(counts=c.tolist(),candidate_scores=scores.tolist(),perfect_candidates=matches.tolist())
    if len(matches)!=1:
        return dict(result,status="ambiguous" if len(matches)>1 else "no_fit")
    k,i=matches[0].tolist();p=PERMS[i]
    if int(c[k].sum(-1).min())<32 or int(v[k].sum(-1).min())<32:
        return dict(result,status="insufficient_coverage")
    if int(v[k,torch.arange(4),p].sum())!=int(v[k].sum()):
        return dict(result,status="validation_rejected")
    return dict(result,status="accepted",lag=k,permutation=p.tolist())

def streams(k,p,n,seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.randint(4,(n,64),generator=g);p=torch.as_tensor(p)
    for t in range(k+1,64):x[:,t]=p[x[:,t-k-1]]
    return x

@torch.no_grad()
def check_model(model,k,p,test,seed):
    p=torch.tensor(p);correct=count=0
    for start in (0,31):
        x=test[:,start:start+32].contiguous();y=test[:,start+1:start+33].contiguous()
        pred=model(x,x)[0][:,k:].argmax(-1)
        correct+=int((pred==y[:,k:]).sum());count+=pred.numel()
    x=torch.randint(4,(32,32),generator=torch.Generator().manual_seed(seed))
    y=p[x[:,:32-k]] if k else p[x]
    probe=model(x,x)[0][:,k:].argmax(-1)
    generated=test[:16,:k+1].clone()
    while generated.shape[1]<32:
        nxt=model(generated)[0][:,-1].argmax(-1,keepdim=True)
        generated=torch.cat((generated,nxt),1)
    gen_correct=int((generated[:,k+1:]==test[:16,k+1:32]).sum())
    return dict(correct=correct,count=count,probe_correct=int((probe==y).sum()),probe_count=y.numel(),
                generation_correct=gen_correct,generation_count=16*(31-k))

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,(k,p) in enumerate(itertools.product(range(9,17),PERMS.tolist())):
        seed=10000+10*i
        train=streams(k,p,64,seed);val=streams(k,p,32,seed+1);test=streams(k,p,64,seed+2)
        r=discover(train,val)
        row=dict(case=i,oracle_lag=k,oracle_permutation=p,discovery=r)
        if r["status"]=="accepted":
            model=compile_retrieval(r["lag"],r["permutation"])
            row.update(check_model(model,k,p,test,seed+3))
            # Save representative models, but verify serialized reload on every case.
            path=out/(f"case{i}.safetensors" if i%24==0 else "temporary.safetensors")
            save_file(snapshot(model),str(path))
            restored=new(99);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
            x=test[:8,:32].contiguous()
            row["reload_exact"]=torch.equal(model(x,x)[0],restored(x,x)[0])
            if i%24:path.unlink()
            row["passed"]=r["lag"]==k and r["permutation"]==p and row["reload_exact"] and all(row[a]==row[b] for a,b in (("correct","count"),("probe_correct","probe_count"),("generation_correct","generation_count")))
        else:row["passed"]=False
        rows.append(row)
    g=torch.Generator().manual_seed(91)
    iid=lambda n:torch.randint(4,(n,64),generator=g)
    def constant(n):return torch.zeros(n,64,dtype=torch.long)
    def ambiguous(n):return torch.arange(n).remainder(4)[:,None].expand(n,64).clone()
    def shuffled(n,seed):
        x=streams(11,[1,2,3,0],n,seed)
        return x.reshape(-1)[torch.randperm(x.numel(),generator=g)].reshape_as(x)
    # Constant-output law is not a permutation: vary the initial context,
    # and keep a separate random prefix for each stream; constant suffix
    # makes the candidate set ambiguous or inconsistent, not accepted.
    def nonperm(n):
        x=iid(n);x[:,17:]=0;return x
    controls={}
    for name,a,b in (("iid",iid(64),iid(32)),("constant",constant(64),constant(32)),
                     ("nonpermutation",nonperm(64),nonperm(32)),("ambiguous",ambiguous(64),ambiguous(32)),
                     ("shuffled",shuffled(64,1),shuffled(32,2)),
                     ("validation_violation",streams(12,[1,2,3,0],64,3),iid(32))):
        controls[name]=discover(a,b)
    expected={"iid":"no_fit","ambiguous":"ambiguous","shuffled":"no_fit","validation_violation":"validation_rejected"}
    controls_pass=all(r["status"]!="accepted" and (name not in expected or r["status"]==expected[name]) for name,r in controls.items())
    summary=dict(cases=len(rows),passed=sum(r["passed"] for r in rows),optimizer_steps=0,donor_weights=False,
                 controls={k:v["status"] for k,v in controls.items()},controls_pass=controls_pass)
    for key in ("correct","count","probe_correct","probe_count","generation_correct","generation_count"):
        summary[key]=sum(r.get(key,0) for r in rows)
    (out/"report.json").write_text(json.dumps(dict(summary=summary,rows=rows,controls=controls),indent=2))
    print("SUMMARY",json.dumps(summary),flush=True)
    assert summary["passed"]==len(rows) and controls_pass

if __name__=="__main__":main()
