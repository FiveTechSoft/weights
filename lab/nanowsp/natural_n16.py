"""N16 external statistical feasibility, no neural compilation."""
import argparse, hashlib,json,re
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
HASH="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
V=2048
def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\n|[^\w\s]",text.lower())
def run(path):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==HASH
    text=raw.decode();n=len(text)
    train,sel,acc=[tokenize(text[a:b]) for a,b in [(0,7*n//10),(7*n//10,8*n//10),(8*n//10,9*n//10)]]
    words=[w for w,c in Counter(train).most_common(V-1)]
    vocab={w:i+1 for i,w in enumerate(words)};v=len(words)+1
    train,sel,acc=[np.array([vocab.get(w,0) for w in seq],dtype=np.int64) for seq in (train,sel,acc)]
    C=np.bincount(train[:-1]*v+train[1:],minlength=v*v).reshape(v,v).astype(np.float64)
    U=np.bincount(train,minlength=v).astype(float)+.5;U/=U.sum()
    row=C.sum(1)
    def base(seq,alpha):
        x,y=seq[:-1],seq[1:];return (C[x,y]+alpha*U[y])/(row[x]+alpha)
    def nll(p):return float(-np.log(p).mean())
    alpha=min((1,2,4,8,16,32,64),key=lambda a:nll(base(sel,a)))
    ps=base(sel,alpha);pa=base(acc,alpha)
    def ppmi(mat):
        r,c=mat.nonzero();z=np.log(mat[r,c]*mat.sum()/(mat.sum(1)[r]*mat.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=mat.shape)
    X=hstack([ppmi(C),ppmi(C.T)]).tocsr()
    Z=TruncatedSVD(n_components=32,random_state=1616).fit_transform(X)
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    def class_probs(labels,seq,method):
        k=int(labels.max())+1
        # Count class -> lexical token and aggregate class transition matrix.
        B=np.zeros((k,v));np.add.at(B,labels,C)
        x,y=seq[:-1],seq[1:];a,b=labels[x],labels[y]
        if method=="lexical":
            return (B[a,y]+alpha*U[y])/(B.sum(1)[a]+alpha)
        M=np.zeros((k,k))
        for j in range(v):M[:,labels[j]]+=B[:,j]
        mass=np.bincount(labels,weights=U,minlength=k)
        return (M[a,b]+alpha*mass[b])/(M.sum(1)[a]+alpha)*U[y]/mass[b]
    results=[];classes={}
    for k in (32,64,128):
        labels=KMeans(n_clusters=k,n_init=1,max_iter=20,random_state=1616).fit_predict(Z)
        classes[k]=labels
        for method in ("lexical","factor"):
            p=class_probs(labels,sel,method)
            for w in (.05,.1,.2,.4,.6,.8,1.):
                score=nll((1-w)*ps+w*p)
                results.append(dict(k=k,method=method,weight=w,selection_nll=score,gain=nll(ps)-score))
    best=min(results,key=lambda r:r["selection_nll"]);k=best["k"];w=best["weight"];method=best["method"];labels=classes[k]
    actual=(1-w)*pa+w*class_probs(labels,acc,method)
    rng=np.random.default_rng(1617)
    shuffled=rng.permutation(labels)
    control=(1-w)*pa+w*class_probs(shuffled,acc,method)
    def interval(g):
        blocks=np.arange(len(g))//128;totals=np.bincount(blocks,weights=g);sizes=np.bincount(blocks)
        draws=rng.integers(len(sizes),size=(1000,len(sizes)))
        vals=totals[draws].sum(1)/sizes[draws].sum(1)
        return dict(mean=float(g.mean()),ci95=list(map(float,np.quantile(vals,[.025,.975]))))
    gain=np.log(actual)-np.log(pa);vs_random=np.log(actual)-np.log(control)
    count=C[acc[:-1],acc[1:]]
    subsets={}
    for name,mask in [('unseen',count==0),('rare',count<=3),('seen',count>0)]:
        subsets[name]=dict(n=int(mask.sum()),gain=float(gain[mask].mean()) if mask.any() else None)
    a=interval(gain);b=interval(vs_random)
    result=dict(corpus_sha256=HASH,tokens=[len(train),len(sel),len(acc)],vocab=v,alpha=alpha,selection_baseline=nll(ps),acceptance_baseline=nll(pa),acceptance_class=nll(actual),acceptance_random=nll(control),best=best,all_candidates=results,vs_baseline=a,vs_random=b,subsets=subsets,native_weights=False,optimizer_gradient_steps=0,donor_weights=False)
    result['gate']=bool(best['gain']>=.02 and a['mean']>=.02 and a['ci95'][0]>.01 and b['ci95'][0]>0)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
