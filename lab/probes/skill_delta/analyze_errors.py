"""Error analysis of the stage-1 holdout generations (A vs B), from the saved eval files only; no model needed. Run from the repo root."""
import json,sys,re,collections
sys.path.insert(0,'lab/probes/closed_form_math')
from data import make_sets
S=make_sets('mul2x2'); H=S['holdout']
def load(p):
    g=[];v=[]
    for part in ('0-50','50-100'):
        d=json.load(open(f'lab/runs/skill_delta/eval{p}_holdout_{part}.json'))['holdout']; g+=d['gen']; v+=d['verdict']
    return g,v
gA,vA=load('A'); gB,vB=load('B')
def cat(g,it):
    t=g.split('\n')[0].strip()
    if re.fullmatch(r'\d+',t):
        n=int(t); y=it['a']*it['b']
        if n==y: return 'correct'
        if len(t)!=len(str(y)): return 'wrong_len'
        return 'wrong_same_len'
    if re.match(r'\d+\s*[*x]',t): return 'new_question'
    return 'other:'+t[:8]
print(H[0])
for nm,g in (('A',gA),('B',gB)):
    c=collections.Counter(cat(x,i) for x,i in zip(g,H)); print(nm,dict(c))
# by operands
for nm,g in (('A',gA),('B',gB)):
    r=collections.defaultdict(lambda:[0,0])
    for x,i in zip(g,H):
        k='ends0' if (i['a']*i['b'])%10==0 else 'nz'; r[k][0]+=cat(x,i)=='correct'; r[k][1]+=1
        k2='round_operand' if i['a']%10==0 or i['b']%10==0 else 'both_nonround'; r[k2][0]+=cat(x,i)=='correct'; r[k2][1]+=1
    print(nm,{k:tuple(v) for k,v in r.items()})
for x,y,i in list(zip(gA,gB,H))[:14]: print(i['a'],i['b'],i['a']*i['b'],repr(x),repr(y))
print(len(S['holdout']), {k:len(v) for k,v in S.items()})
print('-----')
cA=[cat(x,i) for x,i in zip(gA,H)]; cB=[cat(x,i) for x,i in zip(gB,H)]
nq=[k for k,c in enumerate(cA) if c=='new_question']; num=[k for k,c in enumerate(cA) if c!='new_question']
print('A new_question items',len(nq),'-> B correct',sum(cB[k]=='correct' for k in nq))
print('A numeric items',len(num),'A correct',sum(cA[k]=='correct' for k in num),'B correct',sum(cB[k]=='correct' for k in num))
b_only=sum(cA[k]!='correct' and cB[k]=='correct' for k in num); a_only=sum(cA[k]=='correct' and cB[k]!='correct' for k in num)
print('among numeric: B-only',b_only,'A-only',a_only)
from math import comb
def mc(b,c):
    n=b+c; k=min(b,c); return min(1,2*sum(comb(n,i) for i in range(k+1))/2**n)
print('McNemar numeric-only p',mc(b_only,a_only), ' format-only gain 13 items, 10 corrects')
# digit-level error profile
def dig(g,i):
    t=g.split('\n')[0].strip(); y=str(i['a']*i['b'])
    if not re.fullmatch(r'\d+',t) or len(t)!=len(y): return None
    return [k for k in range(len(y)) if t[k]!=y[k]]
for nm,g in (('A',gA),('B',gB)):
    c=collections.Counter()
    for x,i in zip(g,H):
        d=dig(x,i)
        if d: c[len(d)]+=1; 
    first=collections.Counter(min(dig(x,i)) for x,i in zip(g,H) if dig(x,i))
    print(nm,'wrong digits count',dict(c),'first wrong position',dict(first))
# closeness
import statistics
for nm,g in (('A',gA),('B',gB)):
    rel=[abs(int(x.split('\n')[0])-i['a']*i['b'])/(i['a']*i['b']) for x,i in zip(g,H) if re.fullmatch(r'\s*\d+\n.*',x,re.S) or re.fullmatch(r'\s*\d+',x.split('\n')[0])]
    print(nm,'median rel err (numeric answers)',round(statistics.median(rel),4))
print(open('lab/runs/skill_delta/train_report_B.json').read()[:600])
