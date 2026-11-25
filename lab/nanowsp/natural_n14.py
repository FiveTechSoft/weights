"""N14: shared native inference, append-only law integration."""
import argparse,json,math,random
from itertools import combinations,permutations,product
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from natural_n11 import discover,apply,candidates,instantiate,partitions
from natural_n12 import run as run_n12,evaluate
from natural_n13 import fusion

V=18;Q=16;REJECT=17;N=163;D=160;H=5;T=7;A=64.
PAIRS=list(combinations(range(4),2))
FF=36+32*N
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
        super().__init__();self.wte=nn.Embedding(V+N,D);self.wpe=nn.Embedding(T,D);self.blocks=nn.ModuleList([Block(96),Block(FF)]);self.ln=nn.LayerNorm(D);self.head=nn.Linear(D,V,bias=False)
    def forward(self,x):
        z=self.wte(x)+self.wpe(torch.arange(x.shape[1],device=x.device))
        for b in self.blocks:z=b(z)
        return self.head(self.ln(z))
def codes():
    i=torch.arange(N+1)
    return torch.cat([F.one_hot((i//6**k)%6,6).float()-1/6 for k in range(3)],1)
def scale(extra=0):
    return math.sqrt(((1-1/V)+(1-1/T)+2*A*A+2.5+extra)/D+1e-5)
@torch.no_grad()
def initialize():
    m=GPT().eval()
    for p in m.parameters():p.zero_()
    for b in m.blocks:b.ln1.weight.fill_(1);b.ln2.weight.fill_(1)
    E=torch.eye(V)-1/V;P=torch.eye(T)-1/T;C=codes()
    m.wte.weight[:V,:V]=E;m.wte.weight[V:,:V]=E[Q]
    m.wte.weight[:V,18:36]=C[N];m.wte.weight[V:,18:36]=C[:N]
    m.wpe.weight[:,36:43]=P;m.wpe.weight[:,43]=A;m.wpe.weight[:,44]=-A
    b=m.blocks[0];width=D//H;s=scale()
    # Five heads retrieve four input symbols and the task identity.
    for h,pos in enumerate((1,2,3,4,0)):
        start=h*width;src=0 if h<4 else 18;dst=45+h*18
        b.qkv.bias[start]=32*math.sqrt(width)
        b.qkv.weight[D+start,36+pos]=32*s
        b.qkv.weight[2*D+start:2*D+start+18,src:src+18]=s*torch.eye(18)
        b.o.weight[dst:dst+18,start:start+18]=torch.eye(18)
    s=scale(4*(1-1/V)+2.5)
    for j,(i,k) in enumerate(PAIRS):
        b.g.bias[135+2*j:137+2*j]=torch.tensor([-.5,.5])
        for v in range(16):
            r=j*16+v
            b.f.weight[r,45+i*18+v]=32*s;b.f.weight[r,45+k*18+v]=32*s
            b.f.bias[r]=-32*(1.5-2/V)
            b.g.weight[135+2*j:137+2*j,r]=torch.tensor([1/16,-1/16])
    s=scale(4*(1-1/V)+2.5+3)
    b=m.blocks[1]
    b.f.weight[:V,:V]=s*torch.eye(V);b.f.weight[V:2*V,:V]=-s*torch.eye(V)
    b.g.weight[:V,:V]=-torch.eye(V);b.g.weight[:V,V:2*V]=torch.eye(V)
    b.g.bias[:V]=E[REJECT]
    m.ln.weight.fill_(100*s);m.head.weight[:,:V]=E
    return m
@torch.no_grad()
def insert(m,task,law):
    guard,out=law;C=codes();E=torch.eye(V)-1/V
    s=scale(4*(1-1/V)+2.5+3);b=m.blocks[1];start=36+32*task
    assert not b.f.weight[start:start+32].count_nonzero() and not b.f.bias[start:start+32].count_nonzero() and not b.g.weight[:,start:start+32].count_nonzero()
    for j,var in enumerate(out):
        src=guard.index(var)
        for v in range(16):
            r=start+j*16+v
            for k,(i,h) in enumerate(PAIRS):b.f.weight[r,135+2*k]=32*s*(1 if guard[i]==guard[h] else -1)
            b.f.weight[r,45+src*18+v]=32*s
            b.f.weight[r,36+5+j]=32*s
            b.f.weight[r,117:135]=32*s*C[task]
            b.f.bias[r]=-32*(3+1-1/V+1-1/T+2.5-.5)
            b.g.weight[:V,r]=(E[v]-E[REJECT])/16
@torch.no_grad()
def infer(m,x,task):
    if isinstance(task,int):task=torch.full((len(x),),task,dtype=torch.long)
    p=torch.cat((task[:,None]+V,x,torch.full((len(x),1),Q,dtype=torch.long)),1)
    a=m(p)[:,-1];y=a.argmax(-1);b=m(torch.cat((p,y[:,None]),1))[:,-1]
    return torch.stack((y,b.argmax(-1)),1),torch.stack((a,b),1)
def discover_laws():
    rng=random.Random(1414);train=range(8);guards=list(partitions(4));laws=[]
    for h in candidates(4,2):
        g,o=h;examples=[]
        for _ in range(4):
            x=instantiate(g,rng.sample(train,max(g)+1));examples.append((x,apply(h,x)))
        for other in guards:
            if other!=g:examples.append((instantiate(other,rng.sample(train,max(other)+1)),None))
        learned,n=discover(examples,4,2);assert n==1 and learned==h
        laws.append(learned)
    for row in run_n12()['rows']:
        program=tuple((tuple(g),tuple(o)) for g,o in row['law'])
        laws.append(fusion(program))
    assert len(laws)==N
    return laws
@torch.no_grad()
def run(out):
    torch.set_num_threads(4);rng=random.Random(1415);laws=discover_laws();test=range(8,16)
    m=initialize();data=[];probes=[]
    for g,o in laws:
        pos=[instantiate(g,p) for p in permutations(test,max(g)+1)]
        neg=[x for x in product(test,repeat=4) if apply((g,o),x) is None]
        neg=rng.sample(neg,min(128,len(neg)))
        xs=pos+neg;ys=[apply((g,o),x) or (REJECT,REJECT) for x in xs]
        data.append((torch.tensor(xs),torch.tensor(ys),len(pos)))
        probes.append(torch.tensor([pos[0],neg[0]]))
    stages=[];old=0;old_logits=None
    for installed in (0,1,16,64,99,163):
        before={k:v.clone() for k,v in m.state_dict().items()}
        for task in range(old,installed):insert(m,task,laws[task])
        # Verify all writes confined to the newly allocated slots.
        changed=0
        for k,v in m.state_dict().items():
            a=before[k];changed+=int((a!=v).sum())
            if k in ('blocks.1.f.weight','blocks.1.f.bias'):assert torch.equal(a[:36+32*old],v[:36+32*old]) and torch.equal(a[36+32*installed:],v[36+32*installed:])
            elif k=='blocks.1.g.weight':assert torch.equal(a[:,:36+32*old],v[:,:36+32*old]) and torch.equal(a[:,36+32*installed:],v[:,36+32*installed:])
            else:assert torch.equal(a,v)
        path=out/f'stage_{installed}.pt';torch.save(m.state_dict(),path);loaded=GPT().eval();loaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
        x=torch.cat(probes);tasks=torch.arange(N).repeat_interleave(2)
        def all_probes(model):
            return torch.cat([infer(model,a,b)[1] for a,b in zip(x.split(32),tasks.split(32))])
        logits=all_probes(loaded);exact=torch.equal(logits,all_probes(m));assert exact
        retention=0. if old_logits is None or old==0 else float((logits[:2*old]-old_logits[:2*old]).abs().max())
        assert retention<=1e-4
        p=n=pc=nc=0
        for task in range(installed):
            x,y,pos=data[task];offset=0
            for a,b in zip(x.split(256),y.split(256)):
                prediction,_=infer(loaded,a,task);ok=(prediction==b).all(1);mask=torch.arange(offset,offset+len(a))<pos
                pc+=int((ok&mask).sum());nc+=int((ok&~mask).sum());offset+=len(a)
            p+=pos;n+=len(x)-pos
        # Both valid and invalid probes for all absent task IDs must reject.
        absent=logits[2*installed:].argmax(-1)
        reject=int((absent==REJECT).all(1).sum())
        assert pc==p and nc==n and reject==2*(N-installed)
        row=dict(installed=installed,positive=p,correct_positive=pc,negative=n,correct_negative=nc,absent_reject=reject,changed_weights=changed,retention_logit_error=retention,reload_exact=exact,nonzero=sum(int(p.count_nonzero()) for p in m.parameters()))
        stages.append(row);print('STAGE',json.dumps(row),flush=True);old=installed;old_logits=logits
    result=dict(gate=True,parameters=sum(p.numel() for p in m.parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,stages=stages)
    (out/'report.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);run(out)
