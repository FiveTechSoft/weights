"""N18: external natural-language predictive feasibility."""
import argparse,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from natural_n16 import tokenize

def run(path):
    raw=Path(path).read_bytes();text=raw.decode('utf-8-sig')
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
    ctx=Counter();types=Counter();continuation=np.zeros((V,V),dtype=np.int64)
    for (a,b,c),count in triples.items():
        ctx[a,b]+=count;types[a,b]+=1;continuation[b,c]+=1
    U=(C>0).sum(0).astype(float)+.01;U/=U.sum()
    unigram=np.bincount(tr,minlength=V).astype(float)+.5;unigram/=unigram.sum()
    rows=continuation.sum(1);distinct=(continuation>0).sum(1)
    def lexical(s,d,order=3):
        a,b,c=s[:-2],s[1:-1],s[2:]
        back=np.where(rows[b]>0,(np.maximum(continuation[b,c]-d,0)+d*distinct[b]*U[c])/np.maximum(rows[b],1),U[c])
        if order==2:return back
        keys=list(zip(map(int,a),map(int,b)))
        counts=np.array([triples.get((x,y,int(z)),0) for (x,y),z in zip(keys,c)])
        totals=np.array([ctx[k] for k in keys]);nt=np.array([types[k] for k in keys])
        return np.where(totals>0,(np.maximum(counts-d,0)+d*nt*back)/np.maximum(totals,1),back)
    def nll(p):return float(-np.log(p).mean())
    discounts={str(d):nll(lexical(se,d)) for d in (.25,.5,.75,.9)}
    d=float(min(discounts,key=discounts.get));ps=lexical(se,d);pa=lexical(ac,d)
    def ppmi(M):
        r,c=M.nonzero();z=np.log(M[r,c].astype(float)*M.sum()/(M.sum(1)[r].astype(float)*M.sum(0)[c]));keep=z>0
        return csr_matrix((z[keep],(r[keep],c[keep])),shape=M.shape)
    Z=TruncatedSVD(n_components=32,random_state=1818).fit_transform(hstack([ppmi(C),ppmi(C.T)]).tocsr())
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    labels=KMeans(n_clusters=128,n_init=1,max_iter=20,random_state=1818).fit_predict(Z)
    def class_model(lab):
        K=128;z=lab[tr]
        B=np.zeros((K,V),dtype=np.int64);np.add.at(B,(z[:-1],tr[1:]),1)
        br=B.sum(1)
        keys=z[:-2]*K+z[1:-1]
        counts=Counter(zip(map(int,keys),map(int,tr[2:])))
        totals=np.bincount(keys,minlength=K*K)
        def predict(s,history):
            a,b,c=lab[s[:-2]],lab[s[1:-1]],s[2:]
            p=(B[b,c]+32*unigram[c])/(br[b]+32)
            if not history:return p
            k=a*K+b;hits=np.array([counts.get((int(i),int(j)),0) for i,j in zip(k,c)])
            return (hits+32*p)/(totals[k]+32)
        return predict
    pred=class_model(labels)
    choices={}
    for name,history in [('history',True),('last',False)]:
        p=pred(se,history)
        grid=[dict(weight=w,nll=nll((1-w)*ps+w*p)) for w in (0,.05,.1,.2,.4,.6)]
        choices[name]=dict(grid=grid,best=min(grid,key=lambda x:x['nll']))
    w=choices['history']['best']['weight'];wl=choices['last']['best']['weight']
    actual=(1-w)*pa+w*pred(ac,True);last=(1-wl)*pa+wl*pred(ac,False)
    rng=np.random.default_rng(1819);control=(1-w)*pa+w*class_model(rng.permutation(labels))(ac,True)
    def interval(p):
        g=np.log(actual)-np.log(p);blocks=np.arange(len(g))//128
        totals=np.bincount(blocks,weights=g);sizes=np.bincount(blocks)
        draws=rng.integers(len(sizes),size=(1000,len(sizes)))
        return dict(mean=float(g.mean()),ci95=np.quantile(totals[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist())
    a,b,c=interval(pa),interval(last),interval(control)
    unseen=np.array([tuple(map(int,t)) not in triples for t in zip(ac[:-2],ac[1:-1],ac[2:])])
    gain=np.log(actual)-np.log(pa)
    selection_gain=nll(ps)-choices['history']['best']['nll']
    result=dict(corpus_sha256=hashlib.sha256(raw).hexdigest(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),tokens=[len(tr),len(se),len(ac)],vocab=V,distinct_triples=len(triples),discount=d,discount_grid=discounts,choices=choices,selection_gain=selection_gain,acceptance_nll=dict(lexical_bigram=nll(lexical(ac,d,2)),lexical_trigram=nll(pa),history=nll(actual),last=nll(last),random=nll(control)),vs_lexical=a,vs_last=b,vs_random=c,unseen_trigram=dict(n=int(unseen.sum()),gain=float(gain[unseen].mean()) if unseen.any() else None),native_weights=False)
    result['gate']=bool(selection_gain>=.02 and a['mean']>=.02 and a['ci95'][0]>.01 and b['ci95'][0]>0 and c['ci95'][0]>0)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
