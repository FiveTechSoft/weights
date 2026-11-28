"""Analytic positional retrieval plus permutation, without learned weights."""
import argparse,math,json,itertools
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from cycle_law import new,snapshot

I=(0,1,2,3)
OPS=(I,(1,2,3,0),(1,0,2,3))
LAGS=(1,2,4,8,16)
GAIN=4096.

@torch.no_grad()
def linear_residual(block,D,s):
    f=block.mlp.c_fc.weight;o=block.mlp.c_proj.weight
    f.zero_();o.zero_()
    f[:16]=torch.eye(16);f[16:32]=-torch.eye(16)
    o[:,:16]=s*D;o[:,16:32]=-s*D

@torch.no_grad()
def compile_retrieval(k,p):
    assert 0<=k<32 and sorted(p)==list(I)
    m=new(0)
    for w in m.parameters():w.zero_()
    for b in m.transformer.h:
        b.ln_1.weight.fill_(1);b.ln_2.weight.fill_(1)
    m.transformer.ln_f.weight.fill_(4)
    m.transformer.wte.weight[:,:4]=torch.eye(4)-.25
    theta=torch.arange(32,dtype=torch.double)*(2*math.pi/64)
    circle=torch.stack((theta.cos(),theta.sin()),1).float()/math.sqrt(2)
    m.transformer.wpe.weight[:,4:6]=circle
    m.transformer.wpe.weight[:,6:8]=-circle
    s=math.sqrt(1.75/16+1e-5)
    angle=-k*2*math.pi/64
    R=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    qkv=m.transformer.h[0].attn.c_attn.weight
    # q^T k / sqrt(head_dim) = GAIN*cos(relative angle)
    amplitude=math.sqrt(GAIN*math.sqrt(8))*s*math.sqrt(2)
    qkv[:2,4:6]=amplitude*R
    qkv[16:18,4:6]=amplitude*torch.eye(2)
    qkv[32:36,:4]=s*torch.eye(4)
    m.transformer.h[0].attn.c_proj.weight[8:12,:4]=torch.eye(4)
    D=-torch.eye(16);D[:4,8:12]=torch.eye(4)
    # D acts on zero-mean residual; ideal norm squared is 2.5.
    linear_residual(m.transformer.h[0],D,math.sqrt(2.5/16+1e-5))
    D=torch.zeros(16,16)
    for i,j in enumerate(p):D[j,i]=1
    D[:4,:4]-=torch.eye(4)
    linear_residual(m.transformer.h[1],D,math.sqrt(.75/16+1e-5))
    return m.eval()

@torch.no_grad()
def audit(m,x,k,p):
    b=m.transformer.h[0]
    z=b.ln_1(m.transformer.wte(x)+m.transformer.wpe(torch.arange(x.shape[1])))
    q,keys,v=b.attn.c_attn(z).chunk(3,-1)
    scores=q[:,:,:8]@keys[:,:,:8].transpose(-1,-2)/math.sqrt(8)
    scores.masked_fill_(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool).triu(1),float("-inf"))
    a=scores.softmax(-1)
    t=torch.arange(k,x.shape[1])
    mass=float(a[:,t,t-k].min())
    seen=[]
    hook=b.register_forward_hook(lambda mod,inp,res:seen.append(res.clone()))
    try:logits=m(x,x)[0]
    finally:hook.remove()
    target=x[:,:x.shape[1]-k] if k else x
    expected=torch.zeros(*target.shape,16)
    expected[:,:,:4]=torch.nn.functional.one_hot(target,4).float()-.25
    err=float((seen[0][:,k:]-expected).abs().max())
    y=torch.tensor(p)[target]
    pred=logits[:,k:].argmax(-1)
    return dict(correct=int((pred==y).sum()),count=y.numel(),min_selected_mass=mass,
                max_residual_error=err,nll=float(torch.nn.functional.cross_entropy(logits[:,k:].reshape(-1,4),y.reshape(-1))))

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rng=torch.Generator().manual_seed(1729)
    x=torch.randint(4,(256,32),generator=rng)
    rows=[]
    for k,p in itertools.product(LAGS,OPS):
        m=compile_retrieval(k,p);row=dict(lag=k,permutation=p,**audit(m,x,k,p))
        # Cover every current/retrieved pair at every valid position.
        examples=[];positions=[]
        for t in range(k,32):
            for a,b in itertools.product(range(4),repeat=2):
                seq=torch.randint(4,(32,),generator=rng);seq[t-k]=a;seq[t]=b
                examples.append(seq);positions.append(t)
        ex=torch.stack(examples);ts=torch.tensor(positions);ii=torch.arange(len(ex))
        pred=m(ex,ex)[0][ii,ts].argmax(-1)
        expected=torch.tensor(p)[ex[ii,ts-k]]
        row["pair_correct"]=int((pred==expected).sum());row["pair_count"]=len(ex)
        changed=ex.clone();changed[ii,ts-k]=(changed[ii,ts-k]+1)%4
        pred2=m(changed,changed)[0][ii,ts].argmax(-1)
        row["counterfactual_correct"]=int((pred2==torch.tensor(p)[changed[ii,ts-k]]).sum())
        future=ex.clone();mask=torch.arange(32)[None,:]>ts[:,None];future[mask]=(future[mask]+1)%4
        row["causal_error"]=float((m(future,future)[0][ii,ts]-m(ex,ex)[0][ii,ts]).abs().max())
        sd=snapshot(m);path=out/f"lag{k}_perm{''.join(map(str,p))}.safetensors"
        save_file(sd,str(path));restored=new(99);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
        row["reload_exact"]=torch.equal(restored(x,x)[0],m(x,x)[0])
        m.transformer.h[0].attn.c_proj.weight.zero_()
        y=torch.tensor(p)[x[:,:32-k]]
        row["zero_attention_accuracy"]=float((m(x,x)[0][:,k:].argmax(-1)==y).float().mean())
        wrong=compile_retrieval(0,p)
        row["zero_lag_accuracy"]=float((wrong(x,x)[0][:,k:].argmax(-1)==y).float().mean())
        row["passed"]=row["correct"]==row["count"] and row["pair_correct"]==row["pair_count"] and row["counterfactual_correct"]==row["pair_count"] and row["min_selected_mass"]>=.99999 and row["max_residual_error"]<=1e-4 and row["causal_error"]<=1e-5 and row["reload_exact"] and max(row["zero_attention_accuracy"],row["zero_lag_accuracy"])<=.30
        rows.append(row);print("ROW",json.dumps(row),flush=True)
    report=dict(parameters=6800,optimizer_steps=0,donor_weights=False,models=len(rows),
        correct=sum(r["correct"] for r in rows),count=sum(r["count"] for r in rows),
        pair_count=sum(r["pair_count"] for r in rows),counterfactual_correct=sum(r["counterfactual_correct"] for r in rows),
        passed=sum(r["passed"] for r in rows),rows=rows)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
    assert report["passed"]==len(rows)

if __name__=="__main__":main()
