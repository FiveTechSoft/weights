"""N1: fixed statistical features and algebraic readout, native inference."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from train_reference import corpus,config,reload,HASH,lm
from cycle_law import snapshot
from engine.nanogpt_runtime import KarpathyGPT

V=65;D=256;ANCHOR=64.;GAIN=200000.;SLOTS=((65,64),(129,48),(177,32),(209,32))
class Teacher(lm.CharCodec):
    def __init__(self,text,chars):
        super().__init__(chars)
        self.max_seq=256
        ids=np.array(self.encode(text),dtype=np.int64)
        self.tables={}
        for o in range(1,6):
            # Integer encoded contexts fit signed int64 for orders <=5.
            self.tables[o]=lm.NGramModel._count(self,ids,o)
        self.cache={}
    vocab=V
    def row(self,ctx):
        ctx=tuple(ctx[-4:])
        if ctx in self.cache:return self.cache[ctx]
        if not ctx:
            idx,c=self.tables[1][0]
            p=np.full(V,.5);p[idx]+=c;p/=p.sum()
        else:
            p=self.row(ctx[1:]).copy()*8.
            key=0
            for t in ctx:key=key*V+t
            e=self.tables[len(ctx)+1].get(key)
            if e is not None:p[e[0]]+=e[1]
            p/=p.sum()
        self.cache[ctx]=p
        return p
    def logits(self,x):
        return torch.from_numpy(np.log(np.stack([self.row(seq[max(0,t-3):t+1]) for seq in x.tolist() for t in range(x.shape[1])])).reshape(*x.shape,V)).float()

@torch.no_grad()
def build(teacher,ids):
    m=KarpathyGPT(config()).eval()
    for p in m.parameters():p.zero_()
    for b in m.transformer.h:
        b.ln_1.weight.fill_(1);b.ln_2.weight.fill_(1)
    E=torch.eye(V)-1/V
    m.transformer.wte.weight[:,:V]=E
    theta=torch.arange(256,dtype=torch.double)*2*math.pi/512
    circle=torch.stack((theta.cos(),theta.sin()),1).float()/math.sqrt(2)
    m.transformer.wpe.weight[:,241:243]=circle
    m.transformer.wpe.weight[:,243:245]=-circle
    m.transformer.wpe.weight[:,245]=ANCHOR;m.transformer.wpe.weight[:,246]=-ANCHOR
    s=math.sqrt(((1-1/V)+1+2*ANCHOR**2)/D+1e-5)
    m.transformer.ln_f.weight.fill_(math.sqrt(2*ANCHOR**2/D+1e-5))
    qkv=m.transformer.h[0].attn.c_attn.weight
    o=m.transformer.h[0].attn.c_proj.weight
    bases=[]
    for h,(slot,width) in enumerate(SLOTS):
        if h<3:
            k=h+1
            c=torch.bincount(V*ids[:-k-1]+ids[k+1:],minlength=V*V).reshape(V,V).double()
        else:
            c=torch.zeros(V,V,dtype=torch.double)
            for k in (8,16,32,48):
                c+=torch.bincount(V*ids[:-k-1]+ids[k+1:],minlength=V*V).reshape(V,V)
        T=(c+.5)/(c+.5).sum(1,keepdim=True)
        L=T.log();L-=L.mean(0,keepdim=True);L-=L.mean(1,keepdim=True)
        U,_,_=torch.linalg.svd(L,full_matrices=False)
        # Full head needs exactly the 64-dimensional zero-sum subspace.
        U=U[:,:width].float()
        bases.append(U)
        i=h*64
        if h<3:
            angle=-(h+1)*2*math.pi/512
            R=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            amp=math.sqrt(GAIN*8)*s*math.sqrt(2)
            qkv[i:i+2,241:243]=amp*R
            qkv[D+i:D+i+2,241:243]=amp*torch.eye(2)
        qkv[2*D+i:2*D+i+width,:V]=s*U.T
        o[slot:slot+width,i:i+width]=torch.eye(width)
    f=m.transformer.h[0].mlp.c_fc.weight
    # Linear signed pairs and fixed pair features, scaled to undo anchor LN.
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    pairs=torch.bincount(V*ids[1:]+ids[:-1],minlength=V*V).argsort(descending=True)[:256]
    for j,p in enumerate(pairs.tolist()):
        a,b=divmod(p,V)
        f[512+j,a]=8*s
        f[512+j,65:129]=8*s*bases[0][b]
        # Threshold near sum of 1.5 matches; constant anchor supplies bias.
        f[512+j,245]=-8*s*(1.5-2/V)/ANCHOR
    g=torch.Generator().manual_seed(451)
    f[768:,:241]=torch.randn(256,241,generator=g)*s
    f[768:,245]=torch.linspace(-1,1,256)*s/ANCHOR
    return m

@torch.no_grad()
def fit(m,teacher,ids):
    g=torch.Generator().manual_seed(452)
    starts=torch.randint(len(ids)-64,(2048,),generator=g)
    gram=torch.zeros(1024,1024,dtype=torch.double)
    cross=torch.zeros(1024,V,dtype=torch.double)
    b=m.transformer.h[0]
    for start in starts.split(32):
        x=ids[start[:,None]+torch.arange(64)]
        r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(64))
        r=r+b.attn(b.ln_1(r))
        h=b.mlp.gelu(b.mlp.c_fc(b.ln_2(r))).reshape(-1,1024).double()
        target=teacher.logits(x).reshape(-1,V).double()
        target-=target.mean(-1,keepdim=True)
        target-=r[:,:,:V].reshape(-1,V).double()
        gram+=h.T@h;cross+=h.T@target
    reg=.001*gram.diag().mean()
    W=torch.linalg.solve(gram+reg*torch.eye(1024,dtype=torch.double),cross)
    m.transformer.h[0].mlp.c_proj.weight[:V]=W.T.float()
    return float(reg)

@torch.no_grad()
def report(m,teacher,chars,train,test):
    model=lm.NanoGPTCharModel(m,chars)
    r=lm.evaluate(model,train,test)
    r["teacher_nll"]=lm.score_nll(teacher,test)
    x=torch.tensor(model.encode(test[:129])[:-1]).unsqueeze(0)
    a=model.logits(x).double();b=teacher.logits(x).double()
    a-=a.mean(-1,keepdim=True);b-=b.mean(-1,keepdim=True)
    r["teacher_logit_rmse"]=float((a-b).square().mean().sqrt())
    w=m.transformer.h[0].attn.c_proj.weight
    saved=w[:,192:256].clone();w[:,192:256].zero_()
    try:r["zero_prefix_nll"]=lm.score_nll(model,test)
    finally:w[:,192:256]=saved
    r["gates"]=dict(nll_margin=r["L1"]["nll"]<=min(r["L1"]["ngram_ladder_nll"].values())-.05,
        l1=r["L1"]["verdict"]=="PASSES",context=r["L2_long_context"]["verdict"]=="PASSES",
        generation=r["GEN"]["verdict"]=="PASSES",icl=r["L3_icl"]["verdict"]=="PASSES")
    r["language_gate"]=all(v for k,v in r["gates"].items() if k!="icl")
    return r

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)]
    cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars)
    ids=torch.tensor(teacher.encode(train[:cut]));fitids=torch.tensor(teacher.encode(train[cut:]))
    m=build(teacher,ids);reg=fit(m,teacher,fitids)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    path=out/"natural_n1.safetensors"
    save_file(snapshot(m),str(path),metadata={"provenance":"corpus_only_ridge","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
    restored=reload(path)
    probe=fitids[:128].unsqueeze(0)
    assert torch.equal(m(probe)[0],restored(probe)[0])
    result=dict(parameters=sum(p.numel() for p in m.parameters()),nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()),
        optimizer_steps=0,donor_weights=False,reload_exact=True,regularization=reg,teacher_chars=len(ids),fit_chars=len(fitids),fit_windows=2048)
    assert result["parameters"]==3230208
    for name,test in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000])):
        result[name]=report(restored,teacher,data.chars,train,test)
        print(name.upper(),json.dumps(result[name]),flush=True)
        (out/"report.json").write_text(json.dumps(result,indent=2))
    result["language_gate"]=all(result[n]["language_gate"] for n in ("validation","confirmation"))
    if result["language_gate"]:result["reference_tail"]=report(restored,teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
