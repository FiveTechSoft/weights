"""N17: exact dense probability compilation; external fitting only at build time."""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import csr_matrix,hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from natural_n16 import HASH,V,tokenize
D=2*V+4
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte=nn.Embedding(V,D);self.wpe=nn.Embedding(64,D)
        self.ln1=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.o=nn.Linear(D,D)
        self.ln2=nn.LayerNorm(D);self.f=nn.Linear(D,2*V);self.g=nn.Linear(2*V,D)
        self.ln=nn.LayerNorm(D)
    def forward(self,x):
        z=self.wte(x)+self.wpe(torch.arange(x.shape[1],device=x.device))
        q,k,v=self.qkv(self.ln1(z)).chunk(3,-1)
        z=z+self.o(F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],is_causal=True)[:,0])
        z=z+self.g(F.gelu(self.f(self.ln2(z))))
        return F.linear(self.ln(z),self.wte.weight)
@torch.no_grad()
def compile_model(P):
    # Disjoint embedding and logit spaces via a tied readout: the first
    # V coordinates are both input identity and final centered logit vector.
    # A second pair of coordinates supplies a fixed-norm output anchor.
    m=GPT().eval()
    for p in m.parameters():p.zero_()
    for ln in (m.ln1,m.ln2,m.ln):ln.weight.fill_(1)
    E=torch.eye(V)-1/V
    m.wte.weight[:,:V]=E
    L=torch.tensor(np.log(P),dtype=torch.float64)
    L-=L.mean(1,keepdim=True)
    C=float(L.square().sum(1).max())+1.
    Y=torch.zeros((V,D),dtype=torch.float64);Y[:,:V]=L
    Y[:,V]=torch.sqrt((C-L.square().sum(1))/2);Y[:,V+1]=-Y[:,V]
    # Signed GELU identity; rows represent the full centered simplex.
    # Use bias 1/V after undoing input normalization to recover one-hot.
    s=math.sqrt((1-1/V)/D+1e-5)
    m.f.weight[:V,:V]=s*torch.eye(V);m.f.bias[:V]=1/V
    m.f.weight[V:,:V]=-s*torch.eye(V);m.f.bias[V:]=-1/V
    M=(Y[:,:]-torch.nn.functional.pad(E.double(),(0,D-V))).T.float()
    m.g.weight[:,:V]=M;m.g.weight[:,V:]=-M
    m.ln.weight.fill_(math.sqrt(C/D+1e-5))
    return m

def distributions(path):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    t=raw.decode();n=len(t)
    seqs=[tokenize(t[a:b]) for a,b in [(0,7*n//10),(7*n//10,8*n//10),(8*n//10,9*n//10)]]
    words=[w for w,c in Counter(seqs[0]).most_common(V-1)];assert len(words)==V-1
    vocab={w:i+1 for i,w in enumerate(words)}
    tr,se,ac=[np.array([vocab.get(w,0) for w in seq],dtype=np.int64) for seq in seqs]
    C=np.bincount(tr[:-1]*V+tr[1:],minlength=V*V).reshape(V,V).astype(float)
    U=np.bincount(tr,minlength=V).astype(float)+.5;U/=U.sum()
    def lexical(a):return (C+a*U)/(C.sum(1)[:,None]+a)
    def nll(P,s):return float(-np.log(P[s[:-1],s[1:]]).mean())
    alpha=min((1,2,4,8,16,32,64),key=lambda a:nll(lexical(a),se))
    P=lexical(alpha)
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c]*M.sum()/(M.sum(1)[r]*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1616).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    best=None
    for k in (32,64,128):
        labels=KMeans(n_clusters=k,n_init=1,max_iter=20,random_state=1616).fit_predict(Z)
        B=np.zeros((k,V));np.add.at(B,labels,C)
        L=(B+alpha*U)/(B.sum(1)[:,None]+alpha)
        M=np.zeros((k,k))
        for j in range(V):M[:,labels[j]]+=B[:,j]
        mass=np.bincount(labels,weights=U,minlength=k)
        Fct=((M+alpha*mass)/(M.sum(1)[:,None]+alpha))[:,labels]*U[None,:]/mass[labels]
        for method,R in [('lexical',L),('factor',Fct)]:
            for w in (.05,.1,.2,.4,.6,.8,1.):
                # Selection only; avoid materializing each full mixture.
                x,y=se[:-1],se[1:]
                score=float(-np.log((1-w)*P[x,y]+w*R[labels[x],y]).mean())
                if best is None or score<best[0]:best=(score,k,method,w,labels.copy(),R.copy())
    score,k,method,w,labels,R=best;mix=(1-w)*P+w*R[labels]
    strong_alpha=min((2**i for i in range(11)),key=lambda a:nll(lexical(a),se))
    strong=lexical(strong_alpha)
    return mix,strong,ac,["<unk>"]+words,dict(alpha=alpha,k=k,method=method,weight=w,selection_nll=score,strong_alpha=strong_alpha,strong_selection_nll=nll(strong,se))
@torch.no_grad()
def run(corpus,out):
    torch.set_num_threads(2)
    P,strong,ac,words,selection=distributions(corpus)
    m=compile_model(P);path=out/'model.pt';torch.save(m.state_dict(),path)
    other=GPT().eval();other.load_state_dict(torch.load(path,weights_only=True),strict=True)
    rows=[]
    for ids in torch.arange(V).split(32):rows.append(other(ids[:,None])[:,0].softmax(-1).double().numpy())
    native=np.concatenate(rows)
    error=float(np.max(np.abs(native-P)))
    x,y=ac[:-1],ac[1:];nll=lambda p:float(-np.log(p[x,y]).mean())
    gain=np.log(P[x,y])-np.log(strong[x,y])
    rng=np.random.default_rng(1717);blocks=np.arange(len(gain))//128
    sums=np.bincount(blocks,weights=gain);sizes=np.bincount(blocks);draws=rng.integers(len(sizes),size=(1000,len(sizes)))
    ci=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist()
    probe=torch.tensor([[1,2,3,4],[8,7,3,4]])
    logits=other(probe);reload_exact=torch.equal(logits,m(probe))
    prefix=float((logits[0,-1].softmax(-1)-logits[1,-1].softmax(-1)).abs().max())
    changed=probe.clone();changed[:,-1]=9
    causal=float((logits[:,:-1]-other(changed)[:,:-1]).abs().max())
    samples={}
    for mode in ('greedy','sample'):
        tokens=[1];gen=torch.Generator().manual_seed(1718)
        for _ in range(64):
            # Native probabilities already evaluated for every possible last
            # token; use the actual model forward for generation.
            l=other(torch.tensor([[tokens[-1]]]))[0,-1]
            tokens.append(int(l.argmax() if mode=='greedy' else torch.multinomial(l.softmax(-1),1,generator=gen)))
        samples[mode]=' '.join(words[i] for i in tokens)
    result=dict(selection=selection,parameters=sum(p.numel() for p in m.parameters()),nonzero=sum(int(p.count_nonzero()) for p in m.parameters()),max_probability_error=error,native_nll=nll(native),reference_nll=nll(P),strong_nll=nll(strong),gain_vs_strong=float(gain.mean()),ci95=ci,prefix_error=prefix,causal_error=causal,reload_exact=reload_exact,optimizer_steps=0,backward_calls=0,donor_weights=False,samples=samples)
    result['fidelity_gate']=bool(error<=1e-5 and abs(nll(native)-nll(P))<=1e-5 and prefix<=1e-5 and causal<=1e-5 and reload_exact)
    result['predictive_gate']=bool(gain.mean()>=.02 and ci[0]>.01)
    (out/'report.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps(result),flush=True)
    assert result['fidelity_gate']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);run(a.corpus,out)
