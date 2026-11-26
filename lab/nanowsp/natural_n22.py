"""N22 relevance-weighted retrieval; all retrieval strictly causal."""
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
        a,b,c=s[130:-2],s[131:-1],s[132:]
        back=np.where(rows[b]>0,(np.maximum(cont[b,c]-d,0)+d*distinct[b]*U[c])/np.maximum(rows[b],1),U[c])
        keys=list(zip(map(int,a),map(int,b)))
        counts=np.array([triples.get((x,y,int(z)),0) for (x,y),z in zip(keys,c)])
        totals=np.array([ctx[k] for k in keys]);nt=np.array([types[k] for k in keys])
        return np.where(totals>0,(np.maximum(counts-d,0)+d*nt*back)/np.maximum(totals,1),back)
    def nll(p):return float(-np.log(p).mean())
    d=.9
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c].astype(float)*M.sum()/(M.sum(1)[r].astype(float)*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1818).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    labels=KMeans(n_clusters=128,n_init=1,max_iter=20,random_state=1818).fit_predict(Z)
    B=np.zeros((128,V),dtype=np.int64);np.add.at(B,(labels[tr[:-1]],tr[1:]),1)
    def last(s):
        b=labels[s[131:-1]];c=s[132:]
        return (B[b,c]+32*unigram[c])/(B.sum(1)[b]+32)
    ds=[.75,.9,.95,.99]
    dg=[nll(lexical(se,d)) for d in ds];d=ds[int(np.argmin(dg))]
    ws=[0,.1,.2,.4];ps=lexical(se,d);pa=lexical(ac,d)
    wg=[nll((1-w)*ps+w*last(se)) for w in ws];w=ws[int(np.argmin(wg))]
    ps=(1-w)*ps+w*last(se);pa=(1-w)*pa+w*last(ac)
    four=Counter(zip(map(int,tr[:-3]),map(int,tr[1:-2]),map(int,tr[2:-1]),map(int,tr[3:])))
    totals=Counter();nt=Counter()
    for key,count in four.items():totals[key[:3]]+=count;nt[key[:3]]+=1
    def quad(s,p,d):
        if d==1:return p
        keys=list(zip(map(int,s[129:-3]),map(int,s[130:-2]),map(int,s[131:-1])))
        c=s[132:];h=np.array([four.get(k+(int(y),),0) for k,y in zip(keys,c)])
        n=np.array([totals[k] for k in keys]);u=np.array([nt[k] for k in keys])
        return np.where(n>0,(np.maximum(h-d,0)+d*u*p)/np.maximum(n,1),p)
    qs=[.5,.75,.9,.95,1];qg=[nll(quad(se,ps,q)) for q in qs];q=qs[int(np.argmin(qg))]
    ps=quad(se,ps,q);pa=quad(ac,pa,q)
    def features(s):
        t=np.arange(132,len(s));lags=np.arange(128,8,-1)
        j=t[:,None]-lags[None,:];match=s[j]==s[t-1,None]
        hit=s[j+1]==s[t,None]
        assert np.all(j+1<t[:,None]) and j.min()>=3
        ex=[];cl=[]
        for k in (1,2,3):
            ex.append(s[j-k]==s[t-1-k,None])
            cl.append(labels[s[j-k]]==labels[s[t-1-k,None]])
        return match,hit,np.sum(ex,axis=0),np.sum(cl,axis=0),lags
    fs=features(se);fa=features(ac)
    def weights(f,H,beta,mode):
        match,hit,ex,cl,lags=f
        rec=1 if H is None else np.exp2(-lags/H)
        return match*rec*np.power(1.+beta,ex if mode=='exact' else cl)
    def update(p,f,r,alpha):
        if alpha is None:return p
        return (alpha*p+(r*f[1]).sum(1))/(alpha+r.sum(1))
    grids={k:[] for k in ('uniform','recency','context')}
    for H in (None,32,128):
        for beta in (0,1,3):
            for mode in (('exact',) if beta==0 else ('exact','class')):
                kind='context' if beta else ('uniform' if H is None else 'recency')
                r=weights(fs,H,beta,mode)
                for alpha in (4,16,32,64,128,None):
                    grids[kind].append(dict(H=H,beta=beta,mode=mode,alpha=alpha,nll=nll(update(ps,fs,r,alpha))))
    best={k:min(v,key=lambda x:x['nll']) for k,v in grids.items()}
    def evaluate(p,f,c):return update(p,f,weights(f,c['H'],c['beta'],c['mode']),c['alpha'])
    chosen=best['context'];actual=evaluate(pa,fa,chosen)
    r=weights(fa,chosen['H'],chosen['beta'],chosen['mode'])
    rng=np.random.default_rng(2222);shuffled=r.copy()
    for i in range(len(r)):
        mask=fa[0][i];shuffled[i,mask]=rng.permutation(r[i,mask])
    controls=dict(base=pa,uniform=evaluate(pa,fa,best['uniform']),recency=evaluate(pa,fa,best['recency']),shuffled=update(pa,fa,shuffled,chosen['alpha']))
    blocks=np.arange(len(pa))//128;sizes=np.bincount(blocks);draws=rng.integers(len(sizes),size=(1000,len(sizes)))
    def interval(p):
        g=np.log(actual)-np.log(p);sums=np.bincount(blocks,weights=g)
        return dict(mean=float(g.mean()),ci95=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist())
    gains={k:interval(p) for k,p in controls.items()}
    non=min(('uniform','recency'),key=lambda k:best[k]['nll'])
    sg=nll(ps)-chosen['nll'];sn=best[non]['nll']-chosen['nll']
    gate=sg>=.02 and sn>=.01 and gains['base']['mean']>=.02 and gains[non]['mean']>=.01 and gains['base']['ci95'][0]>.01 and all(gains[k]['ci95'][0]>0 for k in ('uniform','recency','shuffled'))
    return dict(corpus_sha256=hashlib.sha256(raw).hexdigest(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),tokens=[len(tr),len(se),len(ac)],targets=[len(ps),len(pa)],vocab=V,discount=d,discount_grid=list(zip(ds,dg)),class_weight=w,class_grid=list(zip(ws,wg)),quad_discount=q,quad_grid=list(zip(qs,qg)),grids=grids,best=best,best_noncontext=non,selection_gain=sg,selection_gain_vs_noncontext=sn,acceptance_nll=dict(context=nll(actual),**{k:nll(p) for k,p in controls.items()}),gains=gains,coverage=float(fa[0].any(1).mean()),mean_weight=float(r.sum(1).mean()),gate=bool(gate),native_weights=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
