"""N20 external statistical feasibility; all retrieval strictly causal."""
import argparse,hashlib,json,re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import scipy,sklearn
from natural_n16 import tokenize
ALPHAS=(4,8,16,32,64,128,None)
def run(path):
    raw=Path(path).read_bytes()
    text=raw.decode('utf-8-sig')
    start=re.search(r'^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$',text,re.M)
    end=re.search(r'^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$',text,re.M)
    assert start and end and start.end()<end.start()
    text=text[start.end():end.start()];n=len(text)
    seqs=[tokenize(text[a:b]) for a,b in [(0,7*n//10),(7*n//10,85*n//100),(85*n//100,n)]]
    words=[w for w,c in Counter(seqs[0]).most_common(2047)]
    vocab={w:i+1 for i,w in enumerate(words)};V=len(words)+1
    tr,se,ac=[np.array([vocab.get(w,0) for w in s],dtype=np.int64) for s in seqs]
    C=np.bincount(tr[:-1]*V+tr[1:],minlength=V*V).reshape(V,V)
    triples=Counter(zip(map(int,tr[:-2]),map(int,tr[1:-1]),map(int,tr[2:])))
    ctx=Counter();types=Counter();cont=np.zeros((V,V),dtype=np.int64)
    for (a,b,c),count in triples.items():
        ctx[a,b]+=count;types[a,b]+=1;cont[b,c]+=1
    U=(C>0).sum(0).astype(float)+.01;U/=U.sum()
    unigram=np.bincount(tr,minlength=V).astype(float)+.5;unigram/=unigram.sum()
    rows=cont.sum(1);distinct=(cont>0).sum(1)
    def lexical(s,d):
        a,b,c=s[126:-2],s[127:-1],s[128:]
        back=np.where(rows[b]>0,(np.maximum(cont[b,c]-d,0)+d*distinct[b]*U[c])/np.maximum(rows[b],1),U[c])
        keys=list(zip(map(int,a),map(int,b)))
        counts=np.array([triples.get((x,y,int(z)),0) for (x,y),z in zip(keys,c)])
        totals=np.array([ctx[k] for k in keys]);nt=np.array([types[k] for k in keys])
        return np.where(totals>0,(np.maximum(counts-d,0)+d*nt*back)/np.maximum(totals,1),back)
    def nll(p):return float(-np.log(p).mean())
    ds=[.25,.5,.75,.9,.95,.99];dg=[nll(lexical(se,d)) for d in ds]
    d=ds[int(np.argmin(dg))]
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c].astype(float)*M.sum()/(M.sum(1)[r].astype(float)*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1818).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    labels=KMeans(n_clusters=128,n_init=1,max_iter=20,random_state=1818).fit_predict(Z)
    B=np.zeros((128,V),dtype=np.int64);np.add.at(B,(labels[tr[:-1]],tr[1:]),1)
    def last(s):
        b=labels[s[127:-1]];c=s[128:]
        return (B[b,c]+32*unigram[c])/(B.sum(1)[b]+32)
    ps=lexical(se,d);pa=lexical(ac,d)
    ws=[0,.05,.1,.2,.4,.6];wg=[nll((1-w)*ps+w*last(se)) for w in ws]
    w=ws[int(np.argmin(wg))]
    ps=(1-w)*ps+w*last(se);pa=(1-w)*pa+w*last(ac)
    rng=np.random.default_rng(2020);shuffled=rng.permutation(labels)
    def features(s):
        t=np.arange(128,len(s));j=t[:,None]-np.arange(128,8,-1)[None,:]
        prev=s[t-1,None];x=s[j];hit=s[j+1]==s[t,None]
        assert np.all(j+1<t[:,None])
        exact=x==prev
        cls=labels[x]==labels[prev]
        ext=cls&~exact
        randclass=(shuffled[x]==shuffled[prev])&~exact
        # Random positions matched in count, sampled without replacement,
        # restricted to non-exact identities just like the extension.
        ranking=np.argsort(np.where(exact,2.,rng.random(x.shape)),axis=1)
        random=np.zeros_like(ext)
        for i,k in enumerate(ext.sum(1)):random[i,ranking[i,:k]]=True
        masks=dict(cache=np.ones_like(exact),exact=exact,class_all=cls,extension=ext,random_class=randclass,random_positions=random)
        return {k:(m.sum(1),(m&hit).sum(1)) for k,m in masks.items()}
    fs=features(se);fa=features(ac)
    def update(p,f,a):
        if a is None:return p.copy()
        n,h=f;return (a*p+h)/(a+n)
    grids={}
    def select(name,p,f):
        grid=[dict(alpha=a,nll=nll(update(p,f,a))) for a in ALPHAS]
        grids[name]=grid
        return min(grid,key=lambda r:r['nll'])['alpha']
    opts={k:select(k,ps,fs[k]) for k in ('cache','exact','class_all')}
    es=update(ps,fs['exact'],opts['exact']);ea=update(pa,fa['exact'],opts['exact'])
    alpha=select('extension',es,fs['extension'])
    actual=update(ea,fa['extension'],alpha)
    controls=dict(base=pa,exact=ea,cache=update(pa,fa['cache'],opts['cache']),class_all=update(pa,fa['class_all'],opts['class_all']),random_class=update(ea,fa['random_class'],alpha),random_positions=update(ea,fa['random_positions'],alpha))
    def interval(p):
        g=np.log(actual)-np.log(p);blocks=np.arange(len(g))//128
        sums=np.bincount(blocks,weights=g);sizes=np.bincount(blocks)
        draws=rng.integers(len(sizes),size=(1000,len(sizes)))
        return dict(mean=float(g.mean()),ci95=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist())
    gains={k:interval(p) for k,p in controls.items()}
    sg=nll(es)-nll(update(es,fs['extension'],alpha))
    stats={k:dict(coverage=float((f[0]>0).mean()),mean_matches=float(f[0].mean()),hit_rate=float((f[1]>0).mean())) for k,f in fa.items()}
    matched=fa['extension'][0]>0
    result=dict(corpus_sha256=hashlib.sha256(raw).hexdigest(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),tokens=[len(tr),len(se),len(ac)],vocab=V,targets=[len(ps),len(pa)],discount=d,discount_grid=list(zip(ds,dg)),class_weight=w,class_weight_grid=list(zip(ws,wg)),retrieval_grids=grids,selected=dict(**opts,extension=alpha),selection_gain=sg,acceptance_nll=dict(extension=nll(actual),**{k:nll(p) for k,p in controls.items()}),gains=gains,stats=stats,matched_gain=float((np.log(actual[matched])-np.log(ea[matched])).mean()) if matched.any() else None,native_weights=False)
    result['gate']=bool(sg>=.02 and gains['exact']['mean']>=.02 and gains['exact']['ci95'][0]>.01 and all(gains[k]['ci95'][0]>0 for k in ('cache','random_class','random_positions')))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
