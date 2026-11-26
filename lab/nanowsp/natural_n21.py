"""N21 prospective exact retrieval replication; all retrieval strictly causal."""
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
    d=.9
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
    w=.2
    rng=np.random.default_rng(2121)
    stages={}
    for name,seq in [('selection',se),('acceptance',ac)]:
        base=(1-w)*lexical(seq,d)+w*last(seq)
        t=np.arange(128,len(seq));j=t[:,None]-np.arange(128,8,-1)[None,:]
        assert np.all(j+1<t[:,None])
        match=seq[j]==seq[t-1,None]
        succ=seq[j+1];hit=succ==seq[t,None];counts=match.sum(1)
        random=np.zeros_like(match)
        for i,k in enumerate(counts):
            eligible=np.flatnonzero(~match[i])
            if len(eligible)<k:eligible=np.arange(j.shape[1])
            random[i,rng.choice(eligible,size=k,replace=False)]=True
        shuffled=np.take_along_axis(succ,np.argsort(rng.random(succ.shape),axis=1),axis=1)
        assert np.array_equal(random.sum(1),counts)
        def update(h):return (32*base+h)/(32+counts)
        actual=update((match&hit).sum(1))
        controls=dict(base=base,random=update((random&hit).sum(1)),shuffled=update((match&(shuffled==seq[t,None])).sum(1)))
        boot=np.random.default_rng(2122)
        blocks=np.arange(len(t))//128;sizes=np.bincount(blocks)
        draws=boot.integers(len(sizes),size=(1000,len(sizes)))
        def interval(p):
            g=np.log(actual)-np.log(p);sums=np.bincount(blocks,weights=g)
            return dict(mean=float(g.mean()),ci95=np.quantile(sums[draws].sum(1)/sizes[draws].sum(1),[.025,.975]).tolist())
        gains={k:interval(p) for k,p in controls.items()}
        matched=counts>0
        row=dict(targets=len(t),coverage=float(matched.mean()),mean_matches=float(counts.mean()),matched_gain=float((np.log(actual[matched])-np.log(base[matched])).mean()) if matched.any() else None,nll=dict(exact=nll(actual),**{k:nll(p) for k,p in controls.items()}),gains=gains)
        row['gate']=bool(gains['base']['mean']>=.02 and gains['base']['ci95'][0]>.01 and all(gains[k]['ci95'][0]>0 for k in ('random','shuffled')))
        stages[name]=row
    return dict(corpus_sha256=hashlib.sha256(raw).hexdigest(),versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),tokens=[len(tr),len(se),len(ac)],vocab=V,discount=d,class_weight=w,alpha=32,stages=stages,gate=all(r['gate'] for r in stages.values()),native_weights=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r=run(a.corpus)
    (out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps(r),flush=True)
