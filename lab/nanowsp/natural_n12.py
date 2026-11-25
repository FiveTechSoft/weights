"""N12: bounded symbolic search with discovered reusable macros."""
import argparse,json,random
from pathlib import Path
from itertools import product,permutations
from natural_n11 import apply,discover,instantiate

def evaluate(program,x):
    for h in program:
        x=apply(h,x)
        if x is None:return None
    return x

def search(library,examples,max_calls=2):
    viable=[];examined=0
    for length in range(1,max_calls+1):
        for seq in product(library,repeat=length):
            examined+=1
            program=tuple(h for macro in seq for h in macro)
            if all(evaluate(program,x)==y for x,y in examples):viable.append(program)
        if viable:break
    unique=list(dict.fromkeys(viable))
    return (unique[0] if len(unique)==1 else None),examined,len(unique)

def run():
    rng=random.Random(1212)
    patterns=[(0,0,0,1),(0,0,1,0),(0,1,0,0),(0,1,1,1),(0,1,0,1),(0,1,1,0),(0,0,1,1)]
    train=tuple(range(8));test=tuple(range(8,16));trials=[]
    for seed in range(64):
        guards=rng.sample(patterns,5);truth=[]
        for i in range(4):
            out=guards[i+1]
            if rng.randrange(2):out=tuple(1-v for v in out)
            truth.append((guards[i],out))
        library=[]
        for h in truth:
            examples=[]
            for _ in range(4):
                x=instantiate(h[0],rng.sample(train,2));examples.append((x,apply(h,x)))
            learned,n=discover(examples,4,4);assert n==1 and learned==h
            library.append((learned,))
        def examples_for(program):
            return [(x,evaluate(program,x)) for x in [instantiate(guards[0],rng.sample(train,2)) for _ in range(4)]]
        base=tuple(library);macros=[];counts=[]
        for depth in (2,3,4):
            inferred,examined,n=search(library,examples_for(truth[:depth]))
            assert inferred==tuple(truth[:depth]) and n==1
            macros.append(inferred);counts.append(examined);library.append(inferred)
        examples=examples_for(truth)
        cold,cold_examined,cold_n=search(base,examples)
        warm,warm_examined,warm_n=search(library[:-1],examples)
        budget=warm_examined;seen=0;budget_hits=0
        for length in range(1,5):
            for seq in product(base,repeat=length):
                if seen>=budget:break
                seen+=1;prog=tuple(h for macro in seq for h in macro)
                if all(evaluate(prog,x)==y for x,y in examples):budget_hits+=1
            if seen>=budget:break
        assert cold is None and cold_n==0 and warm==tuple(truth) and warm_n==1 and budget_hits==0
        pos=neg=0
        for g in patterns:
            for symbols in permutations(test,2):
                x=instantiate(g,symbols);y=evaluate(truth,x);assert evaluate(warm,x)==y
                if y is None:neg+=1
                else:pos+=1
        trials.append(dict(seed=seed,guards=guards,law=warm,macro_search_counts=counts,cold_examined=cold_examined,warm_examined=warm_examined,equal_budget_cold_examined=seen,equal_budget_cold_hits=budget_hits,positive=pos,negative=neg))
    return dict(trials=len(trials),learned_primitives=4*len(trials),learned_macros=3*len(trials),warm_success=len(trials),cold_success=0,equal_budget_cold_success=0,heldout_positive=sum(r['positive'] for r in trials),heldout_negative=sum(r['negative'] for r in trials),cold_examined=sum(r['cold_examined'] for r in trials),warm_examined=sum(r['warm_examined'] for r in trials),macro_total_examined=sum(sum(r['macro_search_counts']) for r in trials),native_weights=False,gate=True,rows=trials)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    r=run();(out/'report.json').write_text(json.dumps(r,indent=2));print('SUMMARY',json.dumps({k:v for k,v in r.items() if k!='rows'}))
