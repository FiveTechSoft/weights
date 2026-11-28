"""Two-lag corpus discovery and native two-head compilation."""
import argparse,itertools,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from cycle_law import new,snapshot
from retrieval_law import linear_residual
from discovery_law import counts

PAIRS=list(itertools.combinations(range(17),2))
def joint_counts(x):
    y=x[:,17:].reshape(-1)
    return torch.stack([torch.bincount(16*x[:,16-a:63-a].reshape(-1)+4*x[:,16-b:63-b].reshape(-1)+y,minlength=64).reshape(16,4) for a,b in PAIRS]).double()
def estimate(c):
    return (c+.5)/(c+.5).sum(-1,keepdim=True)
def losses(c,T):
    return -(c*T.log()).sum((-2,-1))/c.sum((-2,-1))
def discover(x,v):
    c=joint_counts(x);cv=joint_counts(v);T=estimate(c)
    tr=losses(c,T);va=losses(cv,T);i=int(tr.argmin())
    single=estimate(counts(x).double())
    single_va=losses(counts(v).double(),single)
    uni=counts(x)[0].sum(0).double()+.5;uni/=uni.sum()
    vc=counts(v)[0].sum(0);uv=float(-(vc*uni.log()).sum()/vc.sum())
    margin=float(torch.cat((va[:i],va[i+1:])).min()-va[i])
    gain=min(float(single_va.min()),uv)-float(va[i])
    return dict(status="accepted" if gain>=.10 and margin>=.03 else "abstain",pair=PAIRS[i],table=T[i].tolist(),margin=margin,gain=gain,single_tables=single.tolist(),unigram=uni.tolist(),counts=c[i].tolist(),train_nll=tr.tolist(),val_nll=va.tolist())

@torch.no_grad()
def compile_joint(pair,T):
    a,b=pair;T=torch.as_tensor(T,dtype=torch.double)
    m=new(0)
    for w in m.parameters():w.zero_()
    for block in m.transformer.h:
        block.ln_1.weight.fill_(1);block.ln_2.weight.fill_(1)
    m.transformer.wte.weight[:,:4]=torch.eye(4)-.25
    theta=torch.arange(32,dtype=torch.double)*2*math.pi/64
    circle=torch.stack((theta.cos(),theta.sin()),1).float()/math.sqrt(2)
    m.transformer.wpe.weight[:,4:6]=circle
    m.transformer.wpe.weight[:,6:8]=-circle
    s=math.sqrt(1.75/16+1e-5)
    amp=math.sqrt(4096*math.sqrt(8))*s*math.sqrt(2)
    qkv=m.transformer.h[0].attn.c_attn.weight
    o=m.transformer.h[0].attn.c_proj.weight
    for h,k in enumerate(pair):
        i=h*8;angle=-k*2*math.pi/64
        R=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        qkv[i:i+2,4:6]=amp*R
        qkv[16+i:18+i,4:6]=amp*torch.eye(2)
        qkv[32+i:36+i,:4]=s*torch.eye(4)
        o[8+4*h:12+4*h,i:i+4]=torch.eye(4)
    D=-torch.eye(16);D[:4,8:12]=torch.eye(4);D[4:8,12:16]=torch.eye(4)
    linear_residual(m.transformer.h[0],D,math.sqrt(3.25/16+1e-5))
    E=torch.zeros(16,16,dtype=torch.double)
    eye=torch.eye(4,dtype=torch.double)-.25
    F=torch.zeros(16,16,dtype=torch.double)
    for i,j in itertools.product(range(4),repeat=2):
        n=4*i+j;E[n,:4]=eye[i];E[n,4:8]=eye[j]
        F[n,i]=1;F[n,4+j]=1
    H=torch.nn.functional.gelu(E@F.T/math.sqrt(1.5/16+1e-5))
    L=T.log();L-=L.mean(1,keepdim=True)
    C=float(L.square().sum(1).max())+1
    Z=torch.zeros_like(E);Z[:,:4]=L
    Z[:,8]=((C-L.square().sum(1))/2).sqrt();Z[:,9]=-Z[:,8]
    W=torch.linalg.solve(H,Z-E).T
    mlp=m.transformer.h[1].mlp
    mlp.c_fc.weight[:16]=F.float();mlp.c_proj.weight[:,:16]=W.float()
    m.transformer.ln_f.weight.fill_(math.sqrt(C/16+1e-5))
    return m.eval(),float(torch.linalg.cond(H))

def table(op):
    T=torch.full((16,4),.16/3,dtype=torch.double)
    for i,j in itertools.product(range(4),repeat=2):
        T[4*i+j,(i+j)%4 if op=="sum" else i^j]=.84
    return T
def streams(pair,T,n,seed):
    a,b=pair;g=torch.Generator().manual_seed(seed)
    x=torch.randint(4,(n,64),generator=g)
    for t in range(b+1,64):
        x[:,t]=torch.multinomial(T[4*x[:,t-a-1]+x[:,t-b-1]],1,generator=g).squeeze(1)
    return x
@torch.no_grad()
def evaluate(m,pair,T,true,r,test,seed):
    a,b=pair
    # common positions for all candidate baselines, avoid duplicate targets
    x=test[:,:32].contiguous();y=test[:,17:33]
    # target positions 17..32, query positions 16..31
    idx=torch.arange(16,32)
    src=4*x[:,idx-a]+x[:,idx-b]
    logits=m(x,x)[0][:,16:].double()
    err=float((logits.softmax(-1)-T[src]).abs().max())
    nll=lambda p:float(-p.gather(-1,y[...,None]).log().mean())
    native=nll(logits.softmax(-1));oracle=nll(T[src]);truth=nll(true[src])
    singles=torch.tensor(r["single_tables"],dtype=torch.double)
    single=min(nll(singles[k][x[:,idx-k]]) for k in range(17))
    uni=nll(torch.tensor(r["unigram"],dtype=torch.double).expand(*y.shape,4))
    g=torch.Generator().manual_seed(seed)
    z=torch.randint(4,(256,32),generator=g);t=torch.arange(b,32)
    err=max(err,float((m(z,z)[0][:,b:].double().softmax(-1)-T[4*z[:,t-a]+z[:,t-b]]).abs().max()))
    gen=torch.randint(4,(512,b+1),generator=g)
    while gen.shape[1]<32:
        gen=torch.cat((gen,torch.multinomial(m(gen)[0][:,-1].softmax(-1),1,generator=g)),1)
    t=torch.arange(b,31)
    c=torch.bincount((4*(4*gen[:,t-a]+gen[:,t-b])+gen[:,t+1]).reshape(-1),minlength=64).reshape(16,4).double()
    tv=float((c/c.sum(1,keepdim=True).clamp_min(1)-T).abs().sum(1).mean()/2)
    return dict(count=y.numel(),native_nll=native,oracle_nll=oracle,true_nll=truth,best_single_nll=min(single,uni),max_probability_error=err,sampling_tv=tv,excess_nll=oracle-truth)
@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i,(pair,op) in enumerate(itertools.product(((0,3),(2,7),(5,11),(9,16)),("sum","xor"))):
        true=table(op);seed=30000+i*10
        tr=streams(pair,true,512,seed);va=streams(pair,true,256,seed+1);te=streams(pair,true,256,seed+2)
        r=discover(tr,va);row=dict(pair=pair,op=op,discovery=r,passed=False)
        if r["status"]=="accepted":
            T=torch.tensor(r["table"],dtype=torch.double)
            m,cond=compile_joint(r["pair"],T);row["condition"]=cond
            if tuple(r["pair"])==pair:
                row.update(evaluate(m,pair,T,true,r,te,seed+3))
                path=out/f"case{i}.safetensors";save_file(snapshot(m),str(path))
                restored=new(99);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
                x=te[:8,:32].contiguous();row["reload_exact"]=torch.equal(m(x,x)[0],restored(x,x)[0])
                row["passed"]=row["reload_exact"] and row["max_probability_error"]<=1e-5 and abs(row["native_nll"]-row["oracle_nll"])<=1e-5 and row["best_single_nll"]-row["native_nll"]>=.10
        rows.append(row);print("ROW",json.dumps({k:v for k,v in row.items() if k!="discovery"}),flush=True)
    g=torch.Generator().manual_seed(99)
    iid=discover(torch.randint(4,(512,64),generator=g),torch.randint(4,(256,64),generator=g))
    from probability_law import streams as single_streams,make_table
    T=make_table(99)
    single=discover(single_streams(11,T,512,1),single_streams(11,T,256,2))
    good=[r for r in rows if r["passed"]]
    mean=lambda k:sum(r[k] for r in good)/len(good) if good else None
    summary=dict(cases=len(rows),passed=len(good),optimizer_steps=0,donor_weights=False,parameters=6800,
                 controls={"iid":iid["status"],"single":single["status"]},count=sum(r.get("count",0) for r in rows),
                 mean_native_nll=mean("native_nll"),mean_best_single_nll=mean("best_single_nll"),mean_excess_nll=mean("excess_nll"),mean_sampling_tv=mean("sampling_tv"))
    summary["gate_pass"]=len(good)==len(rows) and iid["status"]==single["status"]=="abstain" and summary["mean_excess_nll"]<=.03 and summary["mean_sampling_tv"]<=.08
    (out/"report.json").write_text(json.dumps(dict(summary=summary,rows=rows,controls={"iid":iid,"single":single}),indent=2))
    print("SUMMARY",json.dumps(summary),flush=True);assert summary["gate_pass"]
if __name__=="__main__":main()
