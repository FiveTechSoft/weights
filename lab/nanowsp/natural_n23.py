"""N23: closed-form distributed contextual correction, no gradient."""
import argparse,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,hstack
from scipy.special import logsumexp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from natural_n16 import tokenize

def run(path):
    raw=Path(path).read_bytes();text=raw.decode('utf-8-sig')
    start=re.search(r'^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$',text,re.M)
    end=re.search(r'^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$',text,re.M)
    assert start and end and start.end()<end.start()
    text=text[start.end():end.start()];n=len(text)
    bounds=[0,6*n//10,7*n//10,85*n//100,n]
    seqs=[tokenize(text[a:b]) for a,b in zip(bounds,bounds[1:])]
    vocab={w:i+1 for i,(w,c) in enumerate(Counter(seqs[0]).most_common(2047))};V=len(vocab)+1
    tr,fit,se,ac=[np.array([vocab.get(w,0) for w in s],dtype=np.int64) for s in seqs]
    C=np.bincount(tr[:-1]*V+tr[1:],minlength=V*V).reshape(V,V)
    triples=Counter(zip(map(int,tr[:-2]),map(int,tr[1:-1]),map(int,tr[2:])))
    hist=defaultdict(list);cont=np.zeros((V,V),dtype=np.int64)
    for (a,b,c),count in triples.items():hist[a,b].append((c,count));cont[b,c]+=1
    U=(C>0).sum(0).astype(float)+.01;U/=U.sum()
    unigram=np.bincount(tr,minlength=V).astype(float)+.5;unigram/=unigram.sum()
    d=.9;rows=cont.sum(1);distinct=(cont>0).sum(1)
    back=(np.maximum(cont-d,0)+d*distinct[:,None]*U)/np.maximum(rows[:,None],1)
    back[rows==0]=U
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c].astype(float)*M.sum()/(M.sum(1)[r].astype(float)*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1818).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    labels=KMeans(n_clusters=128,n_init=1,max_iter=20,random_state=1818).fit_predict(Z)
    B=np.zeros((128,V));np.add.at(B,(labels[tr[:-1]],tr[1:]),1)
    cls=(B+32*unigram)/(B.sum(1)[:,None]+32)
    def batch(s,ix):
        p=back[s[ix-1]].copy()
        for i,t in enumerate(ix):
            h=hist.get((int(s[t-2]),int(s[t-1])))
            if h:
                c,v=np.array(h).T;total=v.sum();p[i]*=d*len(v)/total;p[i,c]+=(v-d)/total
        p=.8*p+.2*cls[labels[s[ix-1]]]
        assert np.max(abs(p.sum(1)-1))<1e-9
        return p
    def features(s,short=False):
        ix=np.arange(128,len(s))
        if short:return np.tile(Z[s[ix[:,None]-np.arange(1,4)]].mean(1),(1,3))
        prefix=np.vstack([np.zeros((1,32)),np.cumsum(Z[s],axis=0)])
        return np.concatenate([(prefix[ix-lo+1]-prefix[ix-hi])/(hi-lo+1) for lo,hi in ((9,32),(33,64),(65,128))],axis=1)
    def chunks(s):
        for i in range(128,len(s),256):yield np.arange(i,min(i+256,len(s)))
    def memories(s,ix):
        j=ix[:,None]-np.arange(128,8,-1)
        match=s[j]==s[ix-1,None];n=match.sum(1)
        cache=np.zeros((len(ix),V));exact=np.zeros_like(cache)
        row=np.broadcast_to(np.arange(len(ix))[:,None],j.shape)
        np.add.at(cache,(row,s[j+1]),1/j.shape[1])
        np.add.at(exact,(row[match],s[j+1][match]),1)
        return cache,exact,n
    def baseline(p,c,e,n,w,a):
        p=(1-w)*p+w*c
        return p if a is None else (a*p+e)/(a+n[:,None])
    def nll(p,y):return -np.log(p[np.arange(len(y)),y])
    configs=[(w,a) for w in (0,.02,.05,.1,.2) for a in (None,16,32,64,128,256)]
    sums=np.zeros(len(configs))
    for ix in chunks(se):
        p=batch(se,ix);c,e,n=memories(se,ix)
        for i,(w,a) in enumerate(configs):sums[i]+=nll(baseline(p,c,e,n,w,a),se[ix]).sum()
    scores=sums/(len(se)-128);w,a=configs[int(scores.argmin())]
    def fit_mapping(short):
        X=features(fit,short);mu=X.mean(0);X-=mu
        gram=X.T@X;rhs=np.zeros((96,V))
        for ix in chunks(fit):
            p=batch(fit,ix);p[np.arange(len(ix)),fit[ix]]-=1
            rhs-=X[ix-128].T@p
        candidates=[]
        for lam in (1,10,100):
            W=np.linalg.solve(gram+lam*np.eye(96),rhs)
            u,s,v=np.linalg.svd(W,full_matrices=False)
            for r in (8,16,32,64,96):candidates.append((lam,r,u[:,:r]*s[:r],v[:r]))
        return mu,candidates
    gains=(0,.25,1,4,16,64)
    def select(short):
        mu,candidates=fit_mapping(short);X=features(se,short)-mu
        sums=np.zeros((len(candidates),len(gains)))
        for ix in chunks(se):
            p=batch(se,ix);c,e,n=memories(se,ix);lp=np.log(baseline(p,c,e,n,w,a));y=se[ix]
            for j,(_,_,u,v) in enumerate(candidates):
                delta=(X[ix-128]@u)@v
                for k,g in enumerate(gains):
                    z=lp+g*delta;sums[j,k]+=(logsumexp(z,axis=1)-z[np.arange(len(ix)),y]).sum()
        scores=sums/(len(se)-128)
        j,k=np.unravel_index(scores.argmin(),scores.shape);lam,r,u,v=candidates[j]
        return (mu,u,v,gains[k]),dict(lam=lam,rank=r,gain=gains[k],nll=float(scores[j,k]),grid=[dict(lam=l,rank=r,scores=row.tolist()) for (l,r,_,_),row in zip(candidates,scores)])
    model,chosen=select(False);short_model,short_chosen=select(True)
    X=features(ac)-model[0];S=features(ac,True)-short_model[0]
    perm=np.roll(X,len(X)//2,axis=0)
    def correction(lp,x,m,y):
        _,u,v,g=m;z=lp+g*((x@u)@v)
        return logsumexp(z,axis=1)-z[np.arange(len(y)),y]
    losses={k:[] for k in ('base','context','permuted','short')}
    for ix in chunks(ac):
        p=batch(ac,ix);c,e,n=memories(ac,ix);lp=np.log(baseline(p,c,e,n,w,a));y=ac[ix]
        losses['base'].append(-lp[np.arange(len(ix)),y])
        for key,x,m in [('context',X,model),('permuted',perm,model),('short',S,short_model)]:
            losses[key].append(correction(lp,x[ix-128],m,y))
    losses={k:np.concatenate(v) for k,v in losses.items()}
    blocks=np.arange(len(losses['base']))//128;sizes=np.bincount(blocks)
    rng=np.random.default_rng(2323);draws=rng.integers(len(sizes),size=(1000,len(sizes)))
    def interval(p):
        g=p-losses['context'];sums=np.bincount(blocks,weights=g)
        return dict(mean=float(g.mean()),ci95=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist())
    effects={k:interval(v) for k,v in losses.items() if k!='context'}
    sg=float(scores.min()-chosen['nll'])
    gate=sg>=.02 and effects['base']['mean']>=.02 and effects['base']['ci95'][0]>.01 and all(effects[k]['ci95'][0]>0 for k in ('short','permuted'))
    return dict(corpus_sha256=hashlib.sha256(raw).hexdigest(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),tokens=[len(s) for s in (tr,fit,se,ac)],vocab=V,baseline=dict(cache_weight=w,alpha=a,grid=[dict(cache_weight=w,alpha=a,nll=float(s)) for (w,a),s in zip(configs,scores)]),context=chosen,short=short_chosen,selection_gain=sg,acceptance_nll={k:float(v.mean()) for k,v in losses.items()},effects=effects,gate=bool(gate),native_weights=False,optimizer_steps=0,backward_calls=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
