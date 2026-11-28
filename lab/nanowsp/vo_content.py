"""Diagnostic only: native low-rank VO interventions and content probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from train_reference import corpus,reload,lm,hashfile
from isolate_context import HASH100,HASH500,QKV,PROJ
from trace_vo import maps,windows
from vo_geometry import ci

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
    model=reload(pb);adapter=lm.NanoGPTCharModel(model,data.chars)
    svds=[torch.linalg.svd(maps(late,h),full_matrices=False) for h in range(4)]
    perm=torch.randperm(256,generator=torch.Generator().manual_seed(20261124))
    def state(start,stop,permuted=False):
        sd={k:v.clone() for k,v in late.items()}
        for h,(u,s,vh) in enumerate(svds):
            sl=slice(h*64,(h+1)*64);vs=slice(512+h*64,512+(h+1)*64)
            left=u[:,start:stop]*s[start:stop].sqrt()
            right=s[start:stop].sqrt()[:,None]*vh[start:stop]
            if permuted:left=left[perm]
            sd[PROJ][:,sl].zero_();sd[QKV][vs].zero_()
            sd[PROJ][:,h*64:h*64+stop-start]=left.float()
            sd[QKV][512+h*64:512+h*64+stop-start]=right.float()
            torch.testing.assert_close(maps(sd,h),left@right,atol=1e-8,rtol=1e-5)
        return sd
    states={"early":early,"late":late,"rank64":state(0,64),
            "rank2":state(0,2),"rank4":state(0,4),"rank12":state(0,12),
            "without_top4":state(4,64),"rank4_permuted":state(0,4,True)}
    x=torch.tensor([adapter.encode(text[-16000:-15936])])
    model.load_state_dict(late,strict=True);ref=adapter.logits(x)
    model.load_state_dict(states["rank64"],strict=True)
    torch.testing.assert_close(adapter.logits(x),ref,atol=2e-5,rtol=2e-5)
    report=dict(provenance="diagnostic_vo_content",training_updates=0,splits={})
    for split,subset in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000])):
        pairs,rng=windows(adapter,subset,2000,0)
        indices=rng.integers(0,2000,size=(1000,2000))
        shuf_rng=np.random.default_rng(901)
        examples=[]
        for x,z,y in pairs:
            shuffled=x.clone()
            order=torch.tensor(np.argsort(shuf_rng.random((len(x),56)),axis=1))
            shuffled[:,:56]=x[:,:56].gather(1,order)
            assert torch.equal(shuffled[:,:56].sort(1).values,x[:,:56].sort(1).values)
            examples.append((x,z,shuffled,y))
        rows={}
        for name,sd in states.items():
            model.load_state_dict(sd,strict=True)
            losses=[[],[],[]]
            for x,z,sh,y in examples:
                for dst,inp in zip(losses,(x,z,sh)):
                    lp=adapter.logits(inp)[:,-1].double().log_softmax(-1)
                    dst.append(-lp.gather(1,y[:,None]).squeeze(1).numpy())
            g,b,s=[np.concatenate(v) for v in losses]
            row=dict(nll=lm.score_nll(adapter,subset),intact_nll=float(g.mean()),
                random_nll=float(b.mean()),shuffled_nll=float(s.mean()),
                gain=float((b-g).mean()),gain_ci=ci(b-g,indices),
                order_gain=float((s-g).mean()),order_ci=ci(s-g,indices),
                histogram_gain=float((b-s).mean()),histogram_ci=ci(b-s,indices))
            rows[name]=row
            print("ARM",json.dumps(dict(split=split,arm=name,**row)),flush=True)
        e,l=rows["early"],rows["late"]
        assert e["nll"]>l["nll"] and l["gain"]>e["gain"]
        for name,row in rows.items():
            row["nll_recovery"]=(e["nll"]-row["nll"])/(e["nll"]-l["nll"])
            row["context_recovery"]=(row["gain"]-e["gain"])/(l["gain"]-e["gain"])
            row["joint_pass"]=row["nll_recovery"]>=.8 and row["context_recovery"]>=.8 and row["gain_ci"][0]>.01
        assert abs(rows["rank64"]["nll"]-l["nll"])<1e-5
        report["splits"][split]=rows
        (out/"partial.json").write_text(json.dumps(report,indent=2))
    model.load_state_dict(late,strict=True)
    block=model.transformer.h[0]
    ids=torch.arange(65)[:,None].expand(-1,64)
    emb=block.ln_1(model.transformer.wte(ids)+model.transformer.wpe(torch.arange(64)))
    axes=[]
    for h,(u,s,vh) in enumerate(svds):
        scores=emb.double()@vh[:4].T
        means=scores.mean(1)
        axes.append(dict(head=h,character_means={c:means[i].tolist() for i,c in enumerate(data.chars)},
            positional_std=scores.std(1).mean(0).tolist(),
            between_character_std=means.std(0).tolist(),
            class_means={name:means[torch.tensor([i for i,c in enumerate(data.chars) if pred(c)])].mean(0).tolist()
                for name,pred in (("letter",str.isalpha),("space",str.isspace),("punctuation",lambda c:not c.isalpha() and not c.isspace()))}))
    def features(subset,n,seed):
        pairs,_=windows(adapter,subset,n,seed)
        fs=[];ys=[]
        for x,_,_ in pairs:
            emb=block.ln_1(model.transformer.wte(x)+model.transformer.wpe(torch.arange(64)))
            q,k,v=block.attn.c_attn(emb).chunk(3,-1)
            q,k,v=[t.reshape(len(x),64,4,64).transpose(1,2) for t in (q,k,v)]
            attention=((q[:,:,-1:]@k.transpose(-1,-2))/8).softmax(-1)
            proj=block.attn.c_proj.weight.reshape(256,4,64).permute(1,0,2)
            old=torch.einsum("bhe,hoe->bho",(attention[:,:,:,:56]@v[:,:,:56]).squeeze(2),proj)
            y=torch.cat([old[:,h].double()@svds[h][0][:,:4] for h in range(4)],1)
            hot=torch.nn.functional.one_hot(x,65).double()
            suffix=hot[:,56:].reshape(len(x),-1)
            counts=hot[:,:56].mean(1)
            fs.append(torch.cat((suffix,counts),1));ys.append(y)
        return torch.cat(fs),torch.cat(ys)
    fx,fy=features(text[:int(len(text)*.9)],2048,81)
    tests={name:features(sub,512,82) for name,sub in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000]))}
    probes={}
    for name,dim in (("suffix_only",520),("suffix_and_histogram",585)):
        x=fx[:,:dim];xm=x.mean(0);ym=fy.mean(0)
        z=x-xm;gram=z.T@z
        w=torch.linalg.solve(gram+torch.eye(dim,dtype=torch.float64)*(.001*gram.diag().mean()),z.T@(fy-ym))
        probes[name]={}
        for split,(tx,ty) in tests.items():
            pred=(tx[:,:dim]-xm)@w+ym
            r2=1-((ty-pred)**2).sum(0)/((ty-ym)**2).sum(0).clamp_min(1e-12)
            probes[name][split]=r2.reshape(4,4).tolist()
    report.update(axes=axes,probes=probes,passing_ranks=[k for k in (2,4,12,64)
        if all(rows[f"rank{k}"]["joint_pass"] for rows in report["splits"].values())],
        restored_exact=all(torch.equal(v,late[k]) for k,v in model.state_dict().items()))
    assert report["restored_exact"]
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps({k:v for k,v in report.items() if k!="axes"}),flush=True)

if __name__=="__main__":main()
