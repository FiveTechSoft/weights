"""Diagnostic: equal-norm VO maps on early and late backgrounds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from train_reference import corpus,reload,lm,hashfile
from isolate_context import HASH100,HASH500,PROJ
from trace_vo import maps,swap,windows

ARMS=("early","late","early_scaled","late_scaled","late_permuted")

@torch.no_grad()
def paired_losses(adapter,pairs):
    good=[];bad=[]
    for x,z,y in pairs:
        for dst,inputs in ((good,x),(bad,z)):
            lp=torch.log_softmax(adapter.logits(inputs)[:,-1].double(),-1)
            dst.append(-lp.gather(1,y[:,None]).squeeze(1).numpy())
    return np.concatenate(good),np.concatenate(bad)

def ci(values,indices):
    lo,hi=np.percentile(values[indices].mean(1),[2.5,97.5])
    return [float(lo),float(hi)]

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
    ratios=[];geometry=[]
    for h in range(4):
        e,l=maps(early,h),maps(late,h)
        en,ln=float(e.norm()),float(l.norm())
        assert en>0 and ln>0
        ratios.append(ln/en)
        def ranks(x):
            sv=torch.linalg.svdvals(x).square()
            energy=sv.cumsum(0)/sv.sum()
            return {str(q):int((energy<q).sum())+1 for q in (.5,.9,.99)}
        geometry.append(dict(head=h,early_norm=en,late_norm=ln,ratio=ln/en,
            cosine=float((e*l).sum()/(en*ln)),relative_delta=float((l-e).norm()/en),
            early_energy_ranks=ranks(e),late_energy_ranks=ranks(l),delta_energy_ranks=ranks(l-e)))
    perm=torch.randperm(256,generator=torch.Generator().manual_seed(20261124))
    def state(base,arm):
        src=early if arm.startswith("early") else late
        sd=swap(base,src,15)
        for h,ratio in enumerate(ratios):
            sl=slice(h*64,(h+1)*64)
            if arm=="early_scaled":sd[PROJ][:,sl]*=ratio
            elif arm=="late_scaled":sd[PROJ][:,sl]/=ratio
        if arm=="late_permuted":sd[PROJ]=sd[PROJ][perm].clone()
        for h in range(4):
            expected=geometry[h]["early_norm" if arm in ("early","late_scaled") else "late_norm"]
            assert abs(float(maps(sd,h).norm())/expected-1)<1e-6
        return sd
    report=dict(provenance="diagnostic_vo_geometry",training_updates=0,
                checkpoint_hashes=[HASH100,HASH500],geometry=geometry,splits={})
    for split,subset in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000])):
        pairs,rng=windows(adapter,subset,2000,0)
        indices=rng.integers(0,2000,size=(1000,2000))
        rows={}
        for background,base in (("early",early),("late",late)):
            native=state(base,background)
            assert all(torch.equal(v,base[k]) for k,v in native.items())
            model.load_state_dict(native,strict=True)
            native_good,_=paired_losses(adapter,pairs)
            rows[background]={}
            for arm in ARMS:
                sd=state(base,arm)
                model.load_state_dict(sd,strict=True)
                g,b=paired_losses(adapter,pairs);gain=b-g;improvement=native_good-g
                row=dict(nll=lm.score_nll(adapter,subset),intact_nll=float(g.mean()),
                         corrupted_nll=float(b.mean()),gain=float(gain.mean()),gain_ci=ci(gain,indices),
                         intact_improvement=float(improvement.mean()),intact_improvement_ci=ci(improvement,indices))
                rows[background][arm]=row
                print("ARM",json.dumps(dict(split=split,background=background,arm=arm,**row)),flush=True)
        baseline=rows["early"]["early"];target=rows["late"]["late"]
        delta=target["gain"]-baseline["gain"];assert delta>0
        for row in rows["early"].values():
            row["recovery"]=(row["gain"]-baseline["gain"])/delta
            row["joint_pass"]=row["recovery"]>=.8 and row["gain_ci"][0]>.01 and row["nll"]<=baseline["nll"]
        if split=="validation":
            for row,nll,gain in ((baseline,2.55985085,.0025),(target,2.35210155,.1039),
                (rows["early"]["late"],2.71468448,.1093),(rows["late"]["early"],2.47353566,.0024)):
                assert abs(row["nll"]-nll)<1e-4 and abs(row["gain"]-gain)<1e-4
        report["splits"][split]=rows
        (out/"partial.json").write_text(json.dumps(report,indent=2))
    report["magnitude_alone_supported"]=all(r["early"]["early_scaled"]["joint_pass"] for r in report["splits"].values())
    model.load_state_dict(late,strict=True)
    assert all(torch.equal(v,late[k]) for k,v in model.state_dict().items())
    report["restored_exact"]=True
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(report),flush=True)

if __name__=="__main__":main()
