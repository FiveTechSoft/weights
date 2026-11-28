"""Closed-form corpus correction, frozen trained step100 background."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from train_reference import corpus,reload,lm,hashfile
from isolate_context import HASH100,HASH500,QKV,PROJ
from trace_vo import maps,windows
from vo_geometry import paired_losses,ci

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ("corpus","checkpoints","out"):p.add_argument("--"+name,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    text,data=corpus(a.corpus);d=Path(a.checkpoints);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    pa,pb=d/"step_00100.safetensors",d/"step_00500.safetensors"
    assert hashfile(pa)==HASH100
    early=load_file(str(pa));model=reload(pa);adapter=lm.NanoGPTCharModel(model,data.chars)
    block=model.transformer.h[0];xs=[];ys=[]
    pairs,_=windows(adapter,text[:int(len(text)*.9)],8192,1729)
    for x,_,y in pairs:
        z=block.ln_1(model.transformer.wte(x)+model.transformer.wpe(torch.arange(64)))
        q,k,v=block.attn.c_attn(z).chunk(3,-1)
        q,k=[t.reshape(len(x),64,4,64).transpose(1,2) for t in (q,k)]
        att=((q[:,:,-1:]@k.transpose(-1,-2))/8).softmax(-1).squeeze(2)
        features=torch.einsum("bht,btd->bhd",att,z)
        probs=adapter.logits(x)[:,-1].double().softmax(-1)
        xs.append(features.reshape(len(x),1024).double())
        ys.append(torch.nn.functional.one_hot(y,65).double()-probs)
    X,Y=torch.cat(xs),torch.cat(ys)
    G=X.T@X;lam=.01*G.diag().mean()
    W=torch.linalg.solve(G+lam*torch.eye(1024,dtype=torch.double),X.T@Y)
    E=early["lm_head.weight"].double()
    H=E@E.T
    P=torch.linalg.solve(H+.01*H.diag().mean()*torch.eye(65,dtype=torch.double),E)
    correction=[];stats=[]
    for h in range(4):
        B=(W[h*256:(h+1)*256]@P).T
        u,s,v=torch.linalg.svd(B,full_matrices=False)
        C=(u[:,:4]*s[:4])@v[:4]
        correction.append(C)
        stats.append(dict(head=h,norm=float(C.norm()),energy_fraction=float(s[:4].square().sum()/s.square().sum())))
    perm=torch.randperm(256,generator=torch.Generator().manual_seed(20261124))
    def construct(dose,control):
        sd={k:v.clone() for k,v in early.items()};errors=[]
        for h,C in enumerate(correction):
            B=maps(early,h)+dose*(C[perm] if control else C)
            u,s,v=torch.linalg.svd(B,full_matrices=False)
            left=u[:,:64]*s[:64].sqrt();right=s[:64].sqrt()[:,None]*v[:64]
            sd[PROJ][:,h*64:(h+1)*64]=left.float()
            sd[QKV][512+h*64:512+(h+1)*64]=right.float()
            errors.append(float((maps(sd,h)-B).norm()/B.norm()))
        return sd,errors
    arms={"early":early};errors={}
    for dose in (.25,1.):
        for control in (False,True):
            name=("permuted" if control else "law")+str(dose)
            arms[name],errors[name]=construct(dose,control)
    # Late weights are loaded only after all candidate weights are fixed.
    assert hashfile(pb)==HASH500
    late=load_file(str(pb));arms["late"]=late
    report=dict(provenance="corpus_closed_form_on_trained_step100",training_updates=0,
                fit_windows=8192,rank=4,geometry=stats,refactor_relative_errors=errors,splits={})
    for split,subset in (("validation",text[-16000:-8000]),("confirmation",text[-24000:-16000])):
        pairs,rng=windows(adapter,subset,2000,0);idx=rng.integers(0,2000,size=(1000,2000))
        model.load_state_dict(early,strict=True);g0,_=paired_losses(adapter,pairs)
        rows={}
        for name,sd in arms.items():
            model.load_state_dict(sd,strict=True)
            g,b=paired_losses(adapter,pairs);gain=b-g
            rows[name]=dict(nll=lm.score_nll(adapter,subset),intact_nll=float(g.mean()),
                corrupted_nll=float(b.mean()),gain=float(gain.mean()),gain_ci=ci(gain,idx),
                intact_improvement=float((g0-g).mean()),improvement_ci=ci(g0-g,idx))
            print("ARM",json.dumps(dict(split=split,arm=name,**rows[name])),flush=True)
        for dose in (.25,1.):
            r=rows["law"+str(dose)];base=rows["early"];late_row=rows["late"]
            r["nll_recovery"]=(base["nll"]-r["nll"])/(base["nll"]-late_row["nll"])
            r["context_recovery"]=(r["gain"]-base["gain"])/(late_row["gain"]-base["gain"])
            r["pass"]=r["nll"]<base["nll"] and r["improvement_ci"][0]>0 and r["gain_ci"][0]>.01 and r["nll"]<rows["permuted"+str(dose)]["nll"]
        report["splits"][split]=rows
        (out/"partial.json").write_text(json.dumps(report,indent=2))
    # Portable real heatmaps, shared symmetric scale within each head.
    # PPM is lossless RGB; no plotting dependency.
    for h,C in enumerate(correction):
        e=maps(early,h);delta=maps(late,h)-e
        scale=max(float(t.abs().max()) for t in (e,C,delta))
        panels=[]
        for t in (e,C,delta):
            v=(t/scale).clamp(-1,1).numpy()
            rgb=np.stack((1-np.maximum(-v,0),1-np.abs(v),1-np.maximum(v,0)),axis=-1)
            panels.append((rgb*255).round().astype(np.uint8))
        im=np.concatenate(panels,axis=1)
        # PNG using standard library, no additional dependency.
        import struct,zlib
        def chunk(tag,payload):
            return struct.pack(">I",len(payload))+tag+payload+struct.pack(">I",zlib.crc32(tag+payload)&0xffffffff)
        raw=b"".join(b"\x00"+row.tobytes() for row in im)
        (out/f"head{h}.png").write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",768,256,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
        stats[h]["heatmap_scale"]=scale
    report["confirmed_doses"]=[dose for dose in (.25,1.) if all(s["law"+str(dose)]["pass"] for s in report["splits"].values())]
    model.load_state_dict(early,strict=True)
    assert all(torch.equal(v,early[k]) for k,v in model.state_dict().items())
    report["restored_exact"]=True
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(report),flush=True)

if __name__=="__main__":main()
