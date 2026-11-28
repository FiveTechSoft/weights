"""First block V/O diagnostics; no gradient training or synthesized model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from train_reference import corpus, reload, lm, hashfile
from isolate_context import HASH100,HASH500,QKV,PROJ

@torch.no_grad()
def swap(base,donor,mask):
    out={k:v.clone() for k,v in base.items()}
    for h in range(4):
        if mask & (1<<h):
            sl=slice(h*64,(h+1)*64)
            vs=slice(512+h*64,512+(h+1)*64)
            out[QKV][vs]=donor[QKV][vs]
            out[PROJ][:,sl]=donor[PROJ][:,sl]
    return out

def windows(adapter,text,n,seed):
    ids=np.asarray(adapter.encode(text),dtype=np.int64)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(ids)-65,size=n)
    pairs=[]
    for i in range(0,n,32):
        w=torch.tensor(np.stack([ids[s:s+65] for s in starts[i:i+32]]))
        x=w[:,:-1]; bad=x.clone()
        bad[:,:56]=torch.tensor(ids[rng.integers(0,len(ids),size=(len(w),56))])
        pairs.append((x,bad,w[:,-1]))
    return pairs,rng

@torch.no_grad()
def paired(adapter,text):
    pairs,rng=windows(adapter,text,2000,0)
    good=[];bad=[]
    for x,z,y in pairs:
        for dst,inputs in ((good,x),(bad,z)):
            lp=torch.log_softmax(adapter.logits(inputs)[:,-1].double(),-1)
            dst.append(-lp.gather(1,y[:,None]).squeeze(1).numpy())
    g=np.concatenate(good);b=np.concatenate(bad)
    gains=b-g
    lo,hi=np.percentile(gains[rng.integers(0,len(g),size=(1000,len(g)))].mean(1),[2.5,97.5])
    return dict(mean_gain_nats=float(gains.mean()),ci95_low=float(lo),ci95_high=float(hi),
                intact_nll=float(g.mean()),corrupted_nll=float(b.mean()),n=len(g))

@torch.no_grad()
def head_trace(model,adapter,text):
    pairs,_=windows(adapter,text,128,71)
    block=model.transformer.h[0]
    attn=block.attn
    sums=torch.zeros(4,5,dtype=torch.float64)
    count=0
    def extract(ids):
        pos=torch.arange(ids.shape[1])
        x=block.ln_1(model.transformer.wte(ids)+model.transformer.wpe(pos))
        q,k,v=attn.c_attn(x).chunk(3,dim=-1)
        q,k,v=[t.reshape(len(ids),64,4,64).transpose(1,2) for t in (q,k,v)]
        scores=(q@k.transpose(-1,-2))/8.
        scores.masked_fill_(torch.ones(64,64,dtype=torch.bool).triu(1),float("-inf"))
        a=scores.softmax(-1)
        proj=attn.c_proj.weight.reshape(256,4,64).permute(1,0,2)
        contribution=torch.einsum("bhte,hoe->bhto",a@v,proj)
        torch.testing.assert_close(contribution.sum(1),attn(x),atol=2e-5,rtol=2e-4)
        old=torch.einsum("bhe,hoe->bho",(a[:,:,-1:,:56]@v[:,:,:56]).squeeze(2),proj)
        return a[:,:,-1],contribution[:,:,-1],old
    for x,z,_ in pairs:
        a,c,old=extract(x)
        _,cz,_=extract(z)
        metrics=torch.stack((a[:,:,-1],a[:,:,56:63].sum(-1),a[:,:,:56].sum(-1),
                             old.norm(dim=-1),(c-cz).norm(dim=-1)),dim=-1)
        sums+=metrics.double().sum(0);count+=len(x)
    return [dict(head=h,lag0=float(v[0]),lag1_7=float(v[1]),lag8_63=float(v[2]),
                 old_contribution_norm=float(v[3]),corruption_change_norm=float(v[4]))
            for h,v in enumerate(sums/count)]

def maps(sd,h):
    sl=slice(h*64,(h+1)*64);vs=slice(512+h*64,512+(h+1)*64)
    return sd[PROJ][:,sl].double()@sd[QKV][vs].double()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True);p.add_argument("--checkpoints",required=True)
    p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);d=Path(a.checkpoints);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    pa,pb=d/"step_00100.safetensors",d/"step_00500.safetensors"
    assert hashfile(pa)==HASH100 and hashfile(pb)==HASH500
    early,late=load_file(str(pa)),load_file(str(pb))
    model=reload(pa);adapter=lm.NanoGPTCharModel(model,data.chars)
    def evaluate(mask,reverse,subset):
        base,donor=(late,early) if reverse else (early,late)
        model.load_state_dict(swap(base,donor,mask),strict=True)
        row=dict(mask=mask,heads=[h for h in range(4) if mask&(1<<h)],
                 reverse=reverse,nll=lm.score_nll(adapter,subset),context=paired(adapter,subset))
        print("HEAD",json.dumps(row),flush=True)
        return row
    def score(forward,reverse):
        g=lambda r:r["context"]["mean_gain_nats"]
        lo,hi=g(forward[0]),g(reverse[0]);delta=hi-lo
        assert delta>0
        for mask,r in forward.items():
            r["recovery"]=(g(r)-lo)/delta
            r["reverse_residual"]=(g(reverse[mask])-lo)/delta
            r["context_pass"]=r["recovery"]>=.8 and r["context"]["ci95_low"]>.01
            r["joint_pass"]=r["context_pass"] and r["nll"]<=forward[0]["nll"]
            r["necessary"]=r["reverse_residual"]<=.2
        return forward
    val=text[-16000:-8000];confirm=text[-24000:-16000]
    f={};r={}
    for m in [0,15]+list(range(1,15)):
        f[m]=evaluate(m,False,val);r[m]=evaluate(m,True,val)
        (out/"partial.json").write_text(json.dumps(dict(forward=f,reverse=r),indent=2))
    for row,nll,gain in ((f[0],2.55985085,.0025),(r[0],2.35210155,.1039),
                         (f[15],2.71468448,.1093),(r[15],2.47353566,.0024)):
        assert abs(row["nll"]-nll)<1e-4 and abs(row["context"]["mean_gain_nats"]-gain)<1e-4,row
    score(f,r)
    eligible=[m for m,v in f.items() if v["joint_pass"]]
    minimum=[m for m in eligible if m.bit_count()==min(x.bit_count() for x in eligible)]
    context=[m for m,v in f.items() if v["context_pass"]]
    best_context=min(context,key=lambda m:(m.bit_count(),-f[m]["recovery"],m)) if context else None
    selected=sorted({0,15}|set(minimum)|({best_context} if best_context is not None else set()))
    cf={m:evaluate(m,False,confirm) for m in selected}
    cr={m:evaluate(m,True,confirm) for m in selected}
    score(cf,cr)
    spectra=[]
    for h in range(4):
        b0,b1=maps(early,h),maps(late,h);delta=b1-b0
        spectra.append(dict(head=h,early_norm=float(b0.norm()),late_norm=float(b1.norm()),
            delta_norm=float(delta.norm()),delta_singular_values=torch.linalg.svdvals(delta).tolist()))
    traces={}
    for step,sd in ((100,early),(500,late)):
        model.load_state_dict(sd,strict=True)
        traces[str(step)]=dict(train=head_trace(model,adapter,text[:int(len(text)*.9)]),
                                validation=head_trace(model,adapter,val))
    assert all(torch.equal(v,late[k]) for k,v in model.state_dict().items())
    report=dict(provenance="vo_head_diagnostics",training_updates=0,forward=f,reverse=r,
        minimum_joint=minimum,best_context_only=best_context,confirmation_forward=cf,
        confirmation_reverse=cr,confirmed_joint=[m for m in minimum if cf[m]["joint_pass"]],
        spectra=spectra,traces=traces,restored_exact=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps({k:v for k,v in report.items() if k!="spectra"}),flush=True)

if __name__=="__main__":main()
