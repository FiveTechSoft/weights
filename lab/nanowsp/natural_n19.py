"""N19: compact factorization of N17, not a Transformer."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix,hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from natural_n16 import HASH,V,tokenize
from natural_n17 import distributions

class CompactLM(nn.Module):
    def __init__(self,labels,u,a,t,mass,ptr,col,val):
        super().__init__()
        for name,value in [('u',u),('a',a),('transition',t),('mass',mass),('residual',val)]:
            self.register_parameter(name,nn.Parameter(torch.as_tensor(value,dtype=torch.float32),requires_grad=False))
        for name,value in [('labels',labels),('ptr',ptr),('col',col)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.long))
    def forward(self,x):
        flat=x.reshape(-1)
        p=self.u[None,:]*(self.a[flat,None]+self.transition[self.labels[flat]][:,self.labels]/self.mass[self.labels])
        starts=self.ptr[flat];lengths=self.ptr[flat+1]-starts
        batch=torch.arange(len(flat),device=x.device).repeat_interleave(lengths)
        begin=(lengths.cumsum(0)-lengths).repeat_interleave(lengths)
        idx=starts.repeat_interleave(lengths)+torch.arange(batch.numel(),device=x.device)-begin
        p.index_put_((batch,self.col[idx]),self.residual[idx],accumulate=True)
        return p.log().reshape(*x.shape,len(self.u))
def build(path):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    t=raw.decode();n=len(t)
    train=tokenize(t[:7*n//10])
    words=[w for w,c in Counter(train).most_common(V-1)]
    vocab={w:i+1 for i,w in enumerate(words)}
    tr=np.array([vocab.get(w,0) for w in train],dtype=np.int64)
    C=np.bincount(tr[:-1]*V+tr[1:],minlength=V*V).reshape(V,V).astype(float)
    U=np.bincount(tr,minlength=V).astype(float)+.5;U/=U.sum()
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c]*M.sum()/(M.sum(1)[r]*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1616).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    labels=KMeans(n_clusters=128,n_init=1,max_iter=20,random_state=1616).fit_predict(Z)
    B=np.zeros((128,V));np.add.at(B,labels,C)
    M=np.zeros((128,128))
    for j in range(V):M[:,labels[j]]+=B[:,j]
    mass=np.bincount(labels,weights=U,minlength=128)
    assert np.all(mass>0)
    alpha=64.;w=.4;den=C.sum(1)+alpha
    T=w*(M+alpha*mass)/(M.sum(1)[:,None]+alpha)
    sparse=csr_matrix((1-w)*C/den[:,None])
    return CompactLM(labels,U,(1-w)*alpha/den,T,mass,sparse.indptr,sparse.indices,sparse.data)
@torch.no_grad()
def run(corpus,out):
    torch.set_num_threads(2)
    P,strong,ac,words,selection=distributions(corpus)
    assert (selection['alpha'],selection['k'],selection['method'],selection['weight'])==(64,128,'factor',.4)
    m=build(corpus).eval()
    state=m.state_dict();torch.save(state,out/'model.pt')
    s=torch.load(out/'model.pt',weights_only=True)
    other=CompactLM(s['labels'],s['u'],s['a'],s['transition'],s['mass'],s['ptr'],s['col'],s['residual']).eval()
    other.load_state_dict(s,strict=True)
    rows=[];norm=0.
    for ids in torch.arange(V).split(128):
        p=other(ids).exp();norm=max(norm,float((p.sum(-1)-1).abs().max()))
        rows.append(p.double().numpy())
    native=np.concatenate(rows)
    error=float(np.abs(native-P).max())
    x,y=ac[:-1],ac[1:];nll=lambda p:float(-np.log(p[x,y]).mean())
    probe=torch.tensor([[1,2,3,4],[8,7,3,4]])
    logits=other(probe);changed=probe.clone();changed[:,-1]=9
    prefix=float((logits[0,-1]-logits[1,-1]).abs().max())
    causal=float((logits[:,:-1]-other(changed)[:,:-1]).abs().max())
    reload_exact=torch.equal(logits,m(probe))
    size=sum(p.numel()*p.element_size() for p in state.values())
    floats=sum(p.numel() for p in m.parameters());indices=sum(p.numel() for p in m.buffers())
    reduction=438142384/size
    result=dict(parameters=floats,indices=indices,state_bytes=size,checkpoint_bytes=(out/'model.pt').stat().st_size,reduction_vs_N17_fp32=reduction,lexical_nonzero=m.residual.numel(),max_probability_error=error,normalization_error=norm,nll=nll(native),reference_nll=nll(P),prefix_error=prefix,causal_error=causal,reload_exact=reload_exact,optimizer_steps=0,backward_calls=0,donor_weights=False,standard_transformer=False)
    result['gate']=bool(error<=1e-5 and norm<=1e-5 and abs(nll(native)-nll(P))<=1e-5 and prefix<=1e-5 and causal<=1e-5 and reload_exact and reduction>=100)
    (out/'report.json').write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps(result),flush=True);assert result['gate']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);run(a.corpus,out)
