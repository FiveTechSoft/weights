"""Raw stochastic corpora -> lag and transition table -> native weights."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from discovery_law import counts
from retrieval_law import compile_retrieval
from cycle_law import new,snapshot

def discover(train,val):
    c=counts(train).double();v=counts(val).double()
    T=(c+.5)/(c+.5).sum(-1,keepdim=True)
    train_nll=-(c*T.log()).sum((1,2))/c.sum((1,2))
    val_nll=-(v*T.log()).sum((1,2))/v.sum((1,2))
    uni=c[0].sum(0)+.5;uni/=uni.sum()
    uv=-(v[0].sum(0)*uni.log()).sum()/v[0].sum()
    k=int(train_nll.argmin())
    other=torch.cat((val_nll[:k],val_nll[k+1:])).min()
    margin=float(other-val_nll[k]);gain=float(uv-val_nll[k])
    accepted=margin>=.02 and gain>=.03
    return dict(status="accepted" if accepted else "abstain",lag=k,
                table=T[k].tolist(),unigram=uni.tolist(),counts=c.tolist(),
                train_nll=train_nll.tolist(),val_nll=val_nll.tolist(),
                margin=margin,gain=gain)

@torch.no_grad()
def compile_probability(k,T):
    T=torch.as_tensor(T,dtype=torch.double)
    assert T.shape==(4,4) and bool((T>0).all())
    assert torch.allclose(T.sum(1),torch.ones(4,dtype=torch.double))
    m=compile_retrieval(k,(0,1,2,3))
    L=T.log();L-=L.mean(1,keepdim=True)
    C=float(L.square().sum(1).max())+1.
    h=torch.zeros(4,16,dtype=torch.double)
    h[:,:4]=L
    h[:,4]=((C-L.square().sum(1))/2).sqrt()
    h[:,5]=-h[:,4]
    E=torch.zeros(4,16,dtype=torch.double);E[:,:4]=torch.eye(4,dtype=torch.double)-.25
    H=torch.nn.functional.gelu(E[:,:4]/math.sqrt(.75/16+1e-5))
    # Four independent GELU responses give an exact finite interpolation.
    W=torch.linalg.solve(H,h-E).T
    mlp=m.transformer.h[1].mlp
    mlp.c_fc.weight.zero_();mlp.c_proj.weight.zero_()
    mlp.c_fc.weight[:4,:4]=torch.eye(4)
    mlp.c_proj.weight[:,:4]=W.float()
    m.transformer.ln_f.weight.fill_(math.sqrt(C/16+1e-5))
    return m.eval()

def make_table(seed):
    g=torch.Generator().manual_seed(seed)
    p=torch.randperm(4,generator=g)
    a=torch.rand(4,4,generator=g,dtype=torch.double)+.2
    a[torch.arange(4),p]=0
    a/=a.sum(1,keepdim=True)
    mass=.7+.15*torch.rand(4,generator=g,dtype=torch.double)
    a*=1-mass[:,None];a[torch.arange(4),p]=mass
    return a

def streams(k,T,n,seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.randint(4,(n,64),generator=g)
    for t in range(k+1,64):
        x[:,t]=torch.multinomial(T[x[:,t-k-1]],1,generator=g).squeeze(1)
    return x

@torch.no_grad()
def evaluate(m,k,T,true,uni,test,seed):
    native_nll=oracle_nll=true_nll=uni_nll=0.
    count=0;err=0.
    for start in (0,31):
        x=test[:,start:start+32].contiguous()
        y=test[:,start+1:start+33][:,k:].contiguous()
        src=x[:,:32-k]
        logits=m(x,x)[0][:,k:].double()
        prob=logits.softmax(-1)
        err=max(err,float((prob-T[src]).abs().max()))
        native_nll+=float(-logits.log_softmax(-1).gather(-1,y[...,None]).sum())
        oracle_nll+=float(-T[src].log().gather(-1,y[...,None]).sum())
        true_nll+=float(-true[src].log().gather(-1,y[...,None]).sum())
        uni_nll+=float(-uni[y].log().sum());count+=y.numel()
    g=torch.Generator().manual_seed(seed)
    x=torch.randint(4,(128,32),generator=g)
    err=max(err,float((m(x,x)[0][:,k:].double().softmax(-1)-T[x[:,:32-k]]).abs().max()))
    # Native sampling with no externally supplied transition during generation.
    # 128 independent histories; length32 limit is the native context.
    gen=torch.randint(4,(128,k+1),generator=g)
    while gen.shape[1]<32:
        p=m(gen)[0][:,-1].softmax(-1)
        nxt=torch.multinomial(p,1,generator=g)
        gen=torch.cat((gen,nxt),1)
    c=torch.bincount((4*gen[:,:31-k]+gen[:,k+1:]).reshape(-1),minlength=16).reshape(4,4).double()
    emp=c/c.sum(1,keepdim=True).clamp_min(1)
    tv=float((emp-T).abs().sum(1).mean()/2)
    kl=float((true*(true.log()-T.log())).sum(1).mean())
    return dict(count=count,max_probability_error=err,native_nll=native_nll/count,
                oracle_nll=oracle_nll/count,true_nll=true_nll/count,
                unigram_nll=uni_nll/count,excess_nll=(oracle_nll-true_nll)/count,
                conditional_kl=kl,sampling_tv=tv,sampling_counts=c.tolist())

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for k in range(9,17):
        for rep in range(3):
            seed=20000+100*k+10*rep
            true=make_table(seed)
            train=streams(k,true,256,seed+1);val=streams(k,true,128,seed+2);test=streams(k,true,128,seed+3)
            r=discover(train,val)
            row=dict(oracle_lag=k,true_table=true.tolist(),discovery=r)
            if r["status"]=="accepted":
                T=torch.tensor(r["table"],dtype=torch.double);uni=torch.tensor(r["unigram"],dtype=torch.double)
                m=compile_probability(r["lag"],T)
                row.update(evaluate(m,k,T,true,uni,test,seed+4))
                path=out/f"lag{k}_rep{rep}.safetensors";save_file(snapshot(m),str(path))
                restored=new(99);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
                x=test[:8,:32].contiguous()
                row["reload_exact"]=torch.equal(m(x,x)[0],restored(x,x)[0])
                row["passed"]=r["lag"]==k and row["reload_exact"] and row["max_probability_error"]<=1e-5 and abs(row["native_nll"]-row["oracle_nll"])<=1e-5 and row["unigram_nll"]-row["native_nll"]>=.03
            else:row["passed"]=False
            rows.append(row)
            print("ROW",json.dumps({key:v for key,v in row.items() if key not in ("true_table","discovery","sampling_counts")}),flush=True)
    controls={}
    for name,prob in (("uniform",[.25]*4),("skewed",[.7,.1,.1,.1])):
        g=torch.Generator().manual_seed(394)
        p=torch.tensor(prob)
        a=torch.multinomial(p,256*64,replacement=True,generator=g).reshape(256,64)
        b=torch.multinomial(p,128*64,replacement=True,generator=g).reshape(128,64)
        controls[name]=discover(a,b)
    good=[r for r in rows if r["passed"]]
    mean=lambda key:sum(r[key] for r in good)/len(good) if good else None
    summary=dict(cases=len(rows),passed=len(good),parameters=6800,optimizer_steps=0,donor_weights=False,
                 controls={k:v["status"] for k,v in controls.items()},
                 mean_excess_nll=mean("excess_nll"),mean_conditional_kl=mean("conditional_kl"),
                 mean_sampling_tv=mean("sampling_tv"),mean_native_nll=mean("native_nll"),
                 mean_unigram_nll=mean("unigram_nll"),count=sum(r.get("count",0) for r in rows),
                 max_probability_error=max((r["max_probability_error"] for r in good),default=None))
    summary["gate_pass"]=len(good)==len(rows) and all(r["status"]=="abstain" for r in controls.values()) and summary["mean_excess_nll"]<=.02 and summary["mean_sampling_tv"]<=.06
    (out/"report.json").write_text(json.dumps(dict(summary=summary,rows=rows,controls=controls),indent=2))
    print("SUMMARY",json.dumps(summary),flush=True)
    assert summary["gate_pass"]

if __name__=="__main__":main()
