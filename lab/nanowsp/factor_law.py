"""Masked corpus counts -> CUR factors -> two-head native bilinear model."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from cycle_law import new,snapshot
from joint_law import streams,compile_joint
from retrieval_law import linear_residual

PAIR=(2,7)
B=torch.tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=torch.double)/2
OBS=torch.ones(4,4,dtype=torch.bool);OBS[2:,2:]=False
EPS=.02
ANCHOR=256.
def conditional(U,V):
    return ((U[:,None,:]*V[None,:,:])@B.T).softmax(-1)

def fit(c):
    # The four unseen context cells are never read by the estimator.
    assert c.shape==(4,4,4) and not c[~OBS].any()
    L=torch.zeros(4,4,3,dtype=torch.double)
    p=c[OBS].double()+.5;p/=p.sum(-1,keepdim=True)
    L[OBS]=p.log()@B
    U=[];V=[]
    for r in range(3):
        M=L[:,:,r]
        u,s,vh=torch.linalg.svd(M[:2,:2],full_matrices=False)
        a=M[:,:2]@vh[0]/s[0].sqrt()
        b=u[:,0]@M[:2,:]/s[0].sqrt()
        scale=(b.abs().max()/a.abs().max()).sqrt()
        U.append(a*scale);V.append(b/scale)
    # Additive logit baseline uses the same observed counts and output basis.
    eye=torch.eye(4,dtype=torch.double)
    F=torch.cat((eye[:,None,:].expand(4,4,4),eye[None,:,:].expand(4,4,4)),-1)
    X=F[OBS];W=torch.linalg.lstsq(X,L[OBS],driver="gelsd").solution
    additive=((F@W)@B.T).softmax(-1)
    return torch.stack(U,1),torch.stack(V,1),additive

@torch.no_grad()
def compile_factors(U,V):
    # Reuse only the already verified first-block positional selector.
    # Uniform dummy table has no knowledge of any fixture; its entire
    # second MLP is overwritten below.
    m,_=compile_joint(PAIR,torch.full((16,4),.25,dtype=torch.double))
    b0,b1=m.transformer.h
    s0=math.sqrt(3.25/16+1e-5)
    # Add constant +/-1 to the pair code using the identical sum of
    # four GELU responses for every centered one-hot current symbol.
    z=(torch.eye(4,dtype=torch.double)[0]-.25)/s0
    c=float(torch.nn.functional.gelu(z).sum())
    b0.mlp.c_fc.weight[32:36,:4]=torch.eye(4)
    b0.mlp.c_proj.weight[8,32:36]=1/c
    b0.mlp.c_proj.weight[9,32:36]=-1/c
    s1=math.sqrt(3.5/16+1e-5)
    D=torch.zeros(16,16)
    D[:8,:8]=-torch.eye(8);D[8,8]=ANCHOR-1;D[9,9]=ANCHOR-1
    linear_residual(b1,D,s1)
    # Each factor uses four shared GELU features, no context-specific units.
    coef=1/(8*(1/math.sqrt(2*math.pi))*EPS**2)
    for r in range(3):
        u=torch.zeros(16,dtype=torch.double);v=u.clone()
        u[:4]=U[:,r];u[8]=U[:,r].mean()
        v[4:8]=V[:,r];v[8]=V[:,r].mean()
        features=torch.stack((u+v,-u-v,u-v,-u+v))*EPS*s1
        start=32+4*r
        b1.mlp.c_fc.weight[start:start+4]=features.float()
        b1.mlp.c_proj.weight[:4,start:start+4]=(B[:,r,None]*torch.tensor([1.,1.,-1.,-1.])*coef).float()
    m.transformer.ln_f.weight.fill_(math.sqrt(2*ANCHOR**2/16+1e-5))
    return m.eval()

def count_data(x):
    a,b=PAIR
    src=4*x[:,16-a:63-a]+x[:,16-b:63-b]
    return torch.bincount((4*src+x[:,17:]).reshape(-1),minlength=64).reshape(4,4,4)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in range(8):
        seed=40000+i*10;g=torch.Generator().manual_seed(seed)
        def factors():
            return (.6+.8*torch.rand(4,3,dtype=torch.double,generator=g))*(2*torch.randint(2,(4,3),generator=g)-1)
        true=conditional(factors(),factors())
        tr=streams(PAIR,true.reshape(16,4),4096,seed+1)
        te=streams(PAIR,true.reshape(16,4),512,seed+2)
        c=count_data(tr);c[~OBS]=0
        U,V,additive=fit(c);T=conditional(U,V)
        m=compile_factors(U,V)
        a,b=PAIR
        # Independent arbitrary histories cover all context combinations.
        probe=torch.randint(4,(128,32),generator=g)
        pp=m(probe,probe)[0][:,b:].double().softmax(-1)
        pos=torch.arange(b,32)
        expected=T[probe[:,pos-a],probe[:,pos-b]]
        err=float((pp-expected).abs().max())
        x=te[:,:32].contiguous();pos=torch.arange(16,32)
        aa=x[:,pos-a];bb=x[:,pos-b];y=te[:,17:33]
        missing=~OBS[aa,bb]
        probs=m(x,x)[0][:,16:].double().softmax(-1)
        def nll(p):return float(-p[missing].gather(-1,y[missing][:,None]).log().mean())
        native=nll(probs);oracle=nll(true[aa,bb]);fit_nll=nll(T[aa,bb])
        add=nll(additive[aa,bb]);uniform=math.log(4)
        path=out/f"case{i}.safetensors";save_file(snapshot(m),str(path))
        restored=new(99);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
        reload=torch.equal(m(x[:8],x[:8])[0],restored(x[:8],x[:8])[0])
        m.transformer.h[1].mlp.c_proj.weight[:,32:44].zero_()
        ablation=float((m(probe,probe)[0][:,b:].double().softmax(-1)-.25).abs().max())
        # Counterexample: alternate law has identical observed conditional
        # probabilities but different unseen ones, so cannot be identified
        # from the retained conditional counts alone.
        alt=true.clone();alt[~OBS]=alt[~OBS].roll(1,-1)
        assert torch.equal(alt[OBS],true[OBS])
        row=dict(case=i,seed=seed,heldout_count=int(missing.sum()),min_observed_count=int(c[OBS].sum(-1).min()),
                 max_probability_error=err,native_nll=native,true_nll=oracle,factor_nll=fit_nll,
                 additive_nll=add,uniform_nll=uniform,excess_nll=native-oracle,
                 gain_additive=add-native,gain_uniform=uniform-native,reload_exact=reload,
                 zero_product_error=ablation,alternative_hidden_tv=float((alt[~OBS]-true[~OBS]).abs().sum(-1).mean()/2),
                 U=U.tolist(),V=V.tolist(),counts=c.tolist(),true_table=true.tolist(),
                 passed=err<=.002 and reload and ablation<=.002)
        rows.append(row);print("ROW",json.dumps({k:v for k,v in row.items() if k not in ("U","V","counts","true_table")}),flush=True)
    mean=lambda k:sum(r[k] for r in rows)/len(rows)
    summary=dict(cases=len(rows),passed=sum(r["passed"] for r in rows),optimizer_steps=0,donor_weights=False,
                 parameters=6800,factor_degrees_of_freedom=21,dense_centered_logits=48,
                 heldout_count=sum(r["heldout_count"] for r in rows),
                 mean_excess_nll=mean("excess_nll"),mean_gain_additive=mean("gain_additive"),
                 mean_gain_uniform=mean("gain_uniform"),mean_native_nll=mean("native_nll"),
                 max_probability_error=max(r["max_probability_error"] for r in rows))
    summary["gate_pass"]=summary["passed"]==8 and summary["mean_excess_nll"]<=.06 and min(summary["mean_gain_additive"],summary["mean_gain_uniform"])>=.05
    (out/"report.json").write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    print("SUMMARY",json.dumps(summary),flush=True);assert summary["gate_pass"]
if __name__=="__main__":main()
