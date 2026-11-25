"""N2: reproduce N1, then condition and compile four residual blocks."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,fit,report,V,D,ANCHOR
from train_reference import corpus,reload,HASH,lm
from cycle_law import snapshot

@torch.no_grad()
def conditioned_fit(m,teacher,ids,layer):
    g=torch.Generator().manual_seed(452)
    starts=torch.randint(len(ids)-64,(2048,),generator=g)
    block=m.transformer.h[layer]
    block.mlp.c_proj.weight.zero_()
    if layer:
        s=math.sqrt(2*ANCHOR**2/D+1e-5)
        f=block.mlp.c_fc.weight
        f.zero_()
        f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
        rng=torch.Generator().manual_seed(453+layer)
        f[512:,:245]=torch.randn(512,245,generator=rng)*s/math.sqrt(245)*4
        f[512:,245]=torch.linspace(-2,2,512)*s/ANCHOR
        # Signed features exclude the large anchors. Bias-like constant
        # is supplied by the separate random-feature threshold above.
        f[:512,245:247]=0
    gram=torch.zeros(1024,1024,dtype=torch.double)
    cross=torch.zeros(1024,V,dtype=torch.double)
    count=0
    for start in starts.split(32):
        x=ids[start[:,None]+torch.arange(64)]
        r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(64))
        for j in range(layer):
            r=m.transformer.h[j](r)
        r=r+block.attn(block.ln_1(r))
        h=block.mlp.gelu(block.mlp.c_fc(block.ln_2(r))).reshape(-1,1024).double()
        current=m.lm_head(m.transformer.ln_f(r)).reshape(-1,V).double()
        target=teacher.logits(x).reshape(-1,V).double()
        target-=target.mean(-1,keepdim=True)
        current-=current.mean(-1,keepdim=True)
        target-=current
        # Convert desired logit increments to residual increments using
        # the actual current final-LN scale (the scale change induced by
        # the correction itself is neglected; measure native error).
        scale=(r.double().var(-1,unbiased=False)+1e-5).sqrt().reshape(-1,1)
        target*=scale/float(m.transformer.ln_f.weight[0])
        gram+=h.T@h;cross+=h.T@target;count+=h.shape[0]
    rms=(gram.diag()/count).clamp_min(1e-12).sqrt()
    G=gram/rms[:,None]/rms[None,:]
    W=torch.linalg.solve(G+.001*count*torch.eye(1024,dtype=torch.double),cross/rms[:,None])/rms[:,None]
    W-=W.mean(-1,keepdim=True)
    assert torch.isfinite(W).all()
    block.mlp.c_proj.weight[:V]=W.T.float()
    return dict(samples=count,lambda_normalized=.001*count,rms_min=float(rms.min()),rms_max=float(rms.max()))

@torch.no_grad()
def fidelity(m,teacher,chars,text):
    model=lm.NanoGPTCharModel(m,chars)
    ids=torch.tensor(model.encode(text))
    n=(len(ids)-1)//128
    windows=ids[torch.arange(n)[:,None]*128+torch.arange(129)]
    mse=kl=student=truth=0.;count=0
    for w in windows.split(8):
        x=w[:,:-1].contiguous();y=w[:,1:].unsqueeze(-1)
        a=model.logits(x).double();b=teacher.logits(x).double()
        ac=a-a.mean(-1,keepdim=True);bc=b-b.mean(-1,keepdim=True)
        la=a.log_softmax(-1);lb=b.log_softmax(-1)
        mse+=float((ac-bc).square().sum())/V
        kl+=float((lb.exp()*(lb-la)).sum())
        student-=float(la.gather(-1,y).sum());truth-=float(lb.gather(-1,y).sum())
        count+=y.numel()
    return dict(nll=student/count,teacher_nll=truth/count,gap=(student-truth)/count,
                logit_rmse=math.sqrt(mse/count),teacher_kl=kl/count,count=count)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars)
    ids=torch.tensor(teacher.encode(train[:cut]));fitids=torch.tensor(teacher.encode(train[cut:]))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    probe=fitids[:128].unsqueeze(0)
    rows=[]
    def stage(m,name,details):
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"corpus_only_ridge","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);exact=torch.equal(m(probe)[0],loaded(probe)[0])
        assert exact
        row=dict(stage=name,fit=details,reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()))
        for k,t in slices.items():row[k]=fidelity(loaded,teacher,data.chars,t)
        rows.append(row)
        print("STAGE",json.dumps(row),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
    m=build(teacher,ids);reg=fit(m,teacher,fitids)
    stage(m,"n1",dict(lambda_original=reg))
    reproduction=all(abs(rows[0][k]["nll"]-v)<=.0002 for k,v in (("validation",2.4099),("confirmation",2.3724)))
    for layer in range(4):
        details=conditioned_fit(m,teacher,fitids,layer)
        stage(m,f"n2_{layer+1}",details)
    gains={k:1-rows[-1][k]["gap"]/rows[0][k]["gap"] for k in slices}
    result=dict(parameters=sum(p.numel() for p in m.parameters()),optimizer_steps=0,donor_weights=False,
                n1_reproduction=reproduction,gap_recovery=gains,
                compilation_gate=reproduction and min(gains.values())>=.5,stages=rows)
    assert result["parameters"]==3230208
    for k,t in slices.items():
        result[k]=report(m,teacher,data.chars,train,t)
        print(k.upper(),json.dumps(result[k]),flush=True)
    result["language_gate"]=all(result[k]["language_gate"] for k in slices)
    if result["language_gate"]:result["reference_tail"]=report(m,teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
