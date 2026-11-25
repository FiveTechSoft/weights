"""N15: one causal Transformer, demonstration-conditioned guarded copy.
The symbolic functions are used only at compilation and test generation.
"""
import argparse,json,math,random
from itertools import combinations,permutations,product
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from natural_n11 import candidates,discover,apply,instantiate,partitions

V=18;Q=16;REJECT=17;T=12;D=240;H=10;A=64.
PAIRS=list(combinations(range(4),2))
# 10 retrieved token blocks, 20 equality flags as signed pairs.
EDGES=PAIRS+[(6+i,6+j) for i,j in PAIRS]+[(4+j,i) for j in range(2) for i in range(4)]
# token 0:18, position 18:30, anchors 30:32,
# retrieved 32:212, flags 212:252 => use D=280, head width 28.
D=280
FF=36+99*32
class Block(nn.Module):
    def __init__(self,width):
        super().__init__();self.ln1=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.o=nn.Linear(D,D);self.ln2=nn.LayerNorm(D);self.f=nn.Linear(D,width);self.g=nn.Linear(width,D)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=[z.view(b,t,H,D//H).transpose(1,2) for z in self.qkv(self.ln1(x)).chunk(3,-1)]
        x=x+self.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d))
        return x+self.g(F.gelu(self.f(self.ln2(x))))
class GPT(nn.Module):
    def __init__(self):
        super().__init__();self.wte=nn.Embedding(V,D);self.wpe=nn.Embedding(T,D);self.blocks=nn.ModuleList([Block(len(EDGES)*16),Block(FF)]);self.ln=nn.LayerNorm(D)
    def forward(self,x):
        z=self.wte(x)+self.wpe(torch.arange(x.shape[1],device=x.device))
        for b in self.blocks:z=b(z)
        return F.linear(self.ln(z),self.wte.weight)
def scale(extra=0):
    return math.sqrt(((1-1/V)+(1-1/T)+2*A*A+extra)/D+1e-5)
def discover_laws():
    rng=random.Random(1515);laws=[];train=range(8);guards=list(partitions(4))
    for h in candidates(4,2):
        g,o=h;examples=[]
        for _ in range(4):
            x=instantiate(g,rng.sample(train,max(g)+1));examples.append((x,apply(h,x)))
        for other in guards:
            if other!=g:examples.append((instantiate(other,rng.sample(train,max(other)+1)),None))
        learned,n=discover(examples,4,2);assert learned==h and n==1
        laws.append(learned)
    assert len(laws)==99 and len(set(laws))==99
    return laws
@torch.no_grad()
def compile_model(laws):
    m=GPT().eval()
    for p in m.parameters():p.zero_()
    for b in m.blocks:b.ln1.weight.fill_(1);b.ln2.weight.fill_(1)
    E=torch.eye(V)-1/V;P=torch.eye(T)-1/T
    m.wte.weight[:,:V]=E;m.wpe.weight[:,18:30]=P
    m.wpe.weight[:,30]=A;m.wpe.weight[:,31]=-A
    b=m.blocks[0];s=scale();width=D//H
    for i in range(10):
        start=i*width
        b.qkv.bias[start]=32*math.sqrt(width)
        b.qkv.weight[D+start,18+i]=32*s
        b.qkv.weight[2*D+start:2*D+start+V,:V]=s*torch.eye(V)
        b.o.weight[32+i*V:32+(i+1)*V,start:start+V]=torch.eye(V)
    s=scale(10*(1-1/V))
    for j,(i,k) in enumerate(EDGES):
        b.g.bias[212+2*j:214+2*j]=torch.tensor([-.5,.5])
        for v in range(16):
            r=j*16+v
            b.f.weight[r,32+i*V+v]=32*s;b.f.weight[r,32+k*V+v]=32*s
            b.f.bias[r]=-32*(1.5-2/V)
            b.g.weight[212+2*j:214+2*j,r]=torch.tensor([1/16,-1/16])
    s=scale(10*(1-1/V)+len(EDGES)*.5)
    b=m.blocks[1]
    b.f.weight[:V,:V]=s*torch.eye(V);b.f.weight[V:2*V,:V]=-s*torch.eye(V)
    b.g.weight[:V,:V]=-torch.eye(V);b.g.weight[:V,V:2*V]=torch.eye(V)
    b.g.bias[:V]=E[REJECT]
    for task,(guard,out) in enumerate(laws):
        # All twelve structural and eight demo-output equality checks
        # must match, even when only one output is being generated.
        flags=[guard[i]==guard[k] for i,k in PAIRS]*2
        flags += [out[j]==guard[i] for j in range(2) for i in range(4)]
        for j,var in enumerate(out):
            src=6+guard.index(var)
            for v in range(16):
                r=36+32*task+16*j+v
                for k,flag in enumerate(flags):
                    b.f.weight[r,212+2*k]=32*s*(1 if flag else -1)
                b.f.weight[r,32+src*V+v]=32*s
                b.f.weight[r,18+10+j]=32*s
                b.f.bias[r]=-32*(len(EDGES)*.5+1-1/V+1-1/T-.5)
                b.g.weight[:V,r]=(E[v]-E[REJECT])/16
    m.ln.weight.fill_(100*s)
    return m
@torch.no_grad()
def infer(m,x):
    p=torch.cat((x,torch.full((len(x),1),Q,dtype=torch.long)),1)
    a=m(p)[:,-1];y=a.argmax(-1);b=m(torch.cat((p,y[:,None]),1))[:,-1]
    return torch.stack((y,b.argmax(-1)),1),torch.stack((a,b),1)
@torch.no_grad()
def run(out):
    torch.set_num_threads(4);rng=random.Random(1516);laws=discover_laws();m=compile_model(laws)
    path=out/'model.pt';torch.save(m.state_dict(),path);loaded=GPT().eval();loaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
    before={k:v.clone() for k,v in loaded.state_dict().items()}
    counts={k:[0,0] for k in ('positive','invalid_query','invalid_demo','alternate_demo')}
    min_margin=float('inf');probe=[];alternate=[]
    def check(kind,rows,targets):
        nonlocal min_margin
        for start in range(0,len(rows),128):
            x=torch.tensor(rows[start:start+128]);y=torch.tensor(targets[start:start+128])
            actual,logits=infer(loaded,x);ok=(actual==y).all(1)
            counts[kind][0]+=int(ok.sum());counts[kind][1]+=len(x)
            selected=logits.gather(-1,y[...,None]).squeeze(-1)
            other=logits.scatter(-1,y[...,None],-float('inf')).amax(-1)
            min_margin=min(min_margin,float((selected-other).min()))
    for g,o in laws:
        demo=instantiate(g,rng.sample(range(8),max(g)+1));dy=apply((g,o),demo)
        pos=[instantiate(g,p) for p in permutations(range(8,16),max(g)+1)]
        neg=[x for x in product(range(8,16),repeat=4) if apply((g,o),x) is None]
        neg=rng.sample(neg,128)
        check('positive',[demo+dy+x for x in pos],[apply((g,o),x) for x in pos])
        check('invalid_query',[demo+dy+x for x in neg],[(REJECT,REJECT)]*len(neg))
        # At least one demonstration output is absent from demonstration input.
        bad=[]
        for _ in range(128):
            d=instantiate(g,rng.sample(range(8),max(g)+1));y=list(apply((g,o),d))
            y[rng.randrange(2)]=rng.choice([v for v in range(8) if v not in d])
            bad.append(d+tuple(y)+rng.choice(pos))
        check('invalid_demo',bad,[(REJECT,REJECT)]*len(bad))
        probe.append(demo+dy+pos[0])
        # Counterfactual changes to demonstrations, query held fixed.
        for oo in product(range(max(g)+1),repeat=2):
            if oo==o:continue
            alternate.append((demo+apply((g,oo),demo)+pos[0],apply((g,oo),pos[0])))
    check('alternate_demo',[x for x,y in alternate],[y for x,y in alternate])
    x=torch.tensor(probe);reload_exact=torch.equal(infer(m,x)[1],infer(loaded,x)[1])
    # Causality: changing last appended token cannot affect earlier logits.
    p=torch.cat((x,torch.full((len(x),2),Q,dtype=torch.long)),1)
    q=p.clone();q[:,-1]=REJECT
    causal=float((loaded(p)[:,:-1]-loaded(q)[:,:-1]).abs().max())
    unchanged=all(torch.equal(v,loaded.state_dict()[k]) for k,v in before.items())
    result=dict(counts=counts,parameters=sum(p.numel() for p in m.parameters()),nonzero=sum(int(p.count_nonzero()) for p in m.parameters()),laws=len(laws),optimizer_steps=0,backward_calls=0,donor_weights=False,task_ids=False,reload_exact=reload_exact,causal_error=causal,weights_unchanged=unchanged,min_logit_margin=min_margin)
    result['gate']=all(a==b for a,b in counts.values()) and counts['positive'][1]==46600 and counts['invalid_query'][1]==12672 and counts['invalid_demo'][1]==12672 and reload_exact and causal<=1e-5 and unchanged
    (out/'report.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps(result),flush=True);assert result['gate']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);run(out)
