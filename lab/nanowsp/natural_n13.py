"""N13: algebraic compilation of discovered guarded-copy laws to a causal GPT.
Standalone PyTorch implementation; no symbolic calls in inference.
"""
import argparse, json, math, random
from itertools import combinations, permutations, product
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from natural_n11 import discover, apply, candidates, instantiate, partitions
from natural_n12 import run as run_n12, evaluate
V=18; Q=16; REJECT=17; D=112; H=4; FF=128; T=6
PAIR=list(combinations(range(4),2)); A=64.
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.ln1=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.o=nn.Linear(D,D);self.ln2=nn.LayerNorm(D);self.f=nn.Linear(D,FF);self.g=nn.Linear(FF,D)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=[z.view(b,t,H,D//H).transpose(1,2) for z in self.qkv(self.ln1(x)).chunk(3,-1)]
        x=x+self.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d))
        return x+self.g(F.gelu(self.f(self.ln2(x))))
class GPT(nn.Module):
    def __init__(self):
        super().__init__();self.wte=nn.Embedding(V,D);self.wpe=nn.Embedding(T,D);self.blocks=nn.ModuleList([Block(),Block()]);self.ln=nn.LayerNorm(D)
    def forward(self,x):
        z=self.wte(x)+self.wpe(torch.arange(x.shape[1],device=x.device))
        for b in self.blocks:z=b(z)
        return F.linear(self.ln(z),self.wte.weight)
@torch.no_grad()
def compile_law(law):
    guard,out=law
    assert len(guard)==4 and len(out)==2
    m=GPT().eval()
    for p in m.parameters():p.zero_()
    for b in m.blocks:b.ln1.weight.fill_(1);b.ln2.weight.fill_(1)
    E=torch.eye(V)-1/V;P=torch.eye(T)-1/T
    m.wte.weight[:,:V]=E;m.wpe.weight[:,18:24]=P
    m.wpe.weight[:,24]=A;m.wpe.weight[:,25]=-A
    # scratch 26:98, equality bits 98:110. Fixed input norm.
    s0=math.sqrt(((1-1/V)+(1-1/T)+2*A*A)/D+1e-5)
    b=m.blocks[0];width=D//H
    for i in range(4):
        start=i*width
        b.qkv.bias[start]=32*math.sqrt(width)
        b.qkv.weight[D+start,18+i]=s0*32
        b.qkv.weight[2*D+start:2*D+start+V,:V]=s0*torch.eye(V)
        b.o.weight[26+i*V:26+(i+1)*V,start:start+V]=torch.eye(V)
    s1=math.sqrt((5*(1-1/V)+(1-1/T)+2*A*A)/D+1e-5)
    for j,(i,k) in enumerate(PAIR):
        b.g.bias[98+2*j:100+2*j]=torch.tensor([-.5,.5])
        for v in range(V-2):
            r=j*(V-2)+v
            b.f.weight[r,26+i*V+v]=32*s1;b.f.weight[r,26+k*V+v]=32*s1
            b.f.bias[r]=-32*(1.5-2/V)
            b.g.weight[98+2*j:100+2*j,r]=torch.tensor([1/16,-1/16])
    # Equalities are +/- .5 pairs; same norm for every structure.
    s2=math.sqrt((5*(1-1/V)+(1-1/T)+2*A*A+3)/D+1e-5)
    b=m.blocks[1]
    # Clear the residual token identity at output positions by signed GELU pairs.
    b.f.weight[:V,:V]=s2*torch.eye(V);b.f.weight[V:2*V,:V]=-s2*torch.eye(V)
    b.g.weight[:V,:V]=-torch.eye(V);b.g.weight[:V,V:2*V]=torch.eye(V)
    b.g.bias[:V]=E[REJECT]
    for j,var in enumerate(out):
        src=guard.index(var)
        for v in range(V-2):
            r=2*V+j*(V-2)+v
            for k,(i,h) in enumerate(PAIR):b.f.weight[r,98+2*k]=32*s2*(1 if guard[i]==guard[h] else -1)
            b.f.weight[r,26+src*V+v]=32*s2
            b.f.weight[r,18+4+j]=32*s2
            b.f.bias[r]=-32*(3+1-1/V+1-1/T-.5)
            b.g.weight[:V,r]=(E[v]-E[REJECT])/16
    m.ln.weight.fill_(100*s2)
    return m
@torch.no_grad()
def infer(m,x):
    q=torch.full((len(x),1),Q,dtype=torch.long)
    p=torch.cat((x,q),1);a=m(p)[:,-1];y=a.argmax(-1)
    b=m(torch.cat((p,y[:,None]),1))[:,-1];z=b.argmax(-1)
    return torch.stack((y,z),1),torch.stack((a,b),1)
def fusion(program):
    # Symbolic substitution only at compilation time. No lookup over identities.
    guard=program[0][0];state=guard
    for law in program:
        state=apply(law,state)
        if state is None:raise ValueError('incompatible composition')
    return guard,tuple(state[:2])
@torch.no_grad()
def run(out):
    torch.set_num_threads(4);rng=random.Random(1313);rows=[];train=range(8);test=range(8,16);guards=list(partitions(4))
    laws=[]
    for h in candidates(4,2):
        g,o=h;examples=[]
        for _ in range(4):
            x=instantiate(g,rng.sample(train,max(g)+1));examples.append((x,apply(h,x)))
        for other in guards:
            if other!=g:examples.append((instantiate(other,rng.sample(train,max(other)+1)),None))
        learned,n=discover(examples,4,2);assert n==1 and learned==h;laws.append(('N11',learned,None))
    for row in run_n12()['rows']:
        program=tuple((tuple(g),tuple(o)) for g,o in row['law']);laws.append(('N12',fusion(program),program))
    for i,(kind,law,program) in enumerate(laws):
        m=compile_law(law);path=out/f'{i:03d}.pt';torch.save(m.state_dict(),path)
        restored=GPT().eval();restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
        probe=torch.tensor([[8,9,8,9]]);assert torch.equal(infer(m,probe)[1],infer(restored,probe)[1])
        g,o=law;positive=[instantiate(g,p) for p in permutations(test,max(g)+1)]
        negative=[x for x in product(test,repeat=4) if apply(law,x) is None]
        if kind=='N11':negative=rng.sample(negative,min(256,len(negative)))
        else:negative=[x for x in negative if len(set(x))==2]
        xs=positive+negative;correct=pos=neg=0;margin=float('inf')
        for start in range(0,len(xs),256):
            batch=xs[start:start+256];x=torch.tensor(batch)
            targets=[]
            for item in batch:
                y=apply(law,item) if program is None else evaluate(program,item)
                targets.append((REJECT,REJECT) if y is None else tuple(y[:2]))
            target=torch.tensor(targets);actual,logits=infer(restored,x)
            ok=(actual==target).all(1);ispos=target[:,0]!=REJECT
            correct+=int(ok.sum());pos+=int((ok&ispos).sum());neg+=int((ok&~ispos).sum())
            chosen=logits.gather(-1,target[...,None]).squeeze(-1);other=logits.scatter(-1,target[...,None],-float('inf')).amax(-1)
            margin=min(margin,float((chosen-other).min()))
        row=dict(kind=kind,index=i,guard=g,out=o,positive=len(positive),correct_positive=pos,negative=len(negative),correct_negative=neg,min_logit_margin=margin,reload_exact=True)
        rows.append(row)
        if correct!=len(xs):print('FAIL',row,flush=True);break
        if i%20==0:print('PROGRESS',i,flush=True)
    result=dict(laws=len(rows),expected_laws=len(laws),parameters=sum(p.numel() for p in m.parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,positive=sum(r['positive'] for r in rows),correct_positive=sum(r['correct_positive'] for r in rows),negative=sum(r['negative'] for r in rows),correct_negative=sum(r['correct_negative'] for r in rows),min_logit_margin=min(r['min_logit_margin'] for r in rows),reload_exact=sum(r['reload_exact'] for r in rows),rows=rows)
    result['gate']=len(rows)==len(laws) and result['positive']==result['correct_positive'] and result['negative']==result['correct_negative']
    (out/'report.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);run(out)
