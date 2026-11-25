"""N11: infer guarded variable-copy laws from positive and negative examples."""
from itertools import product, permutations
import random, json, argparse
from pathlib import Path

def partitions(n):
    if n == 0:
        yield ()
    else:
        for p in partitions(n-1):
            for j in range(max(p, default=-1)+2):
                yield p+(j,)

def structure(x):
    names={}
    return tuple(names.setdefault(v,len(names)) for v in x)

def candidates(n,m):
    for guard in partitions(n):
        for out in product(range(max(guard)+1),repeat=m):
            yield guard,out

def apply(h,x):
    guard,out=h
    if structure(x)!=guard:return None
    vals=list(dict.fromkeys(x))
    return tuple(vals[i] for i in out)

def discover(examples,n,m):
    viable=[h for h in candidates(n,m) if all(apply(h,x)==y for x,y in examples)]
    return (viable[0] if len(viable)==1 else None),len(viable)

def instantiate(pattern,symbols):
    return tuple(symbols[i] for i in pattern)

def run():
    rng=random.Random(1125)
    train=tuple(range(8));test=tuple(range(8,16))
    guards=list(partitions(4));laws=list(candidates(4,2))
    rows=[];total=correct=0;negative=0;negcorrect=0;ambiguous=0
    for guard,out in laws:
        h=(guard,out);k=max(guard)+1
        examples=[]
        for _ in range(4):
            x=instantiate(guard,rng.sample(train,k));examples.append((x,apply(h,x)))
        for g in guards:
            if g!=guard:examples.append((instantiate(g,rng.sample(train,max(g)+1)),None))
        inferred,n=discover(examples,4,2)
        assert inferred==h and n==1
        count=0
        for symbols in permutations(test,k):
            x=instantiate(guard,symbols);expected=tuple(symbols[i] for i in out)
            correct+=int(apply(inferred,x)==expected);total+=1;count+=1
        for g in guards:
            if g==guard:continue
            for symbols in permutations(test,max(g)+1):
                x=instantiate(g,symbols);negative+=1;negcorrect+=int(apply(inferred,x) is None)
        rows.append(dict(guard=guard,out=out,heldout=count))
    for g in guards:
        other=next(p for p in guards if p!=g)
        h,n=discover([(instantiate(other,train),None)],4,2)
        ambiguous+=int(h is None and n>1)
    inconsistent=discover([((0,1,0,1),(0,1)),((0,1,0,1),(1,0))],4,2)
    outside=discover([((0,1,0,1),(2,3))],4,2)
    result=dict(laws=len(laws),discovered=len(rows),heldout_positive=total,correct_positive=correct,heldout_negative=negative,correct_negative=negcorrect,ambiguous_rejected=ambiguous,ambiguous_cases=len(guards),inconsistent_rejected=inconsistent[1]==0,outside_family_rejected=outside[1]==0,native_weights=False,rows=rows)
    result['gate']=correct==total and negative==negcorrect and ambiguous==len(guards) and inconsistent[1]==0 and outside[1]==0
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    r=run();(out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps({k:v for k,v in r.items() if k!='rows'}));assert r['gate']
