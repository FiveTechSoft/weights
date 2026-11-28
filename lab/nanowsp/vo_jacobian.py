"""One closed-form local fit in a fixed input subspace; no optimizer."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from train_reference import corpus,reload,lm,hashfile
from isolate_context import HASH100,HASH500,QKV,PROJ
from trace_vo import maps,windows
from vo_geometry import paired_losses,ci

def main():
    p=argparse.ArgumentParser()
    for name in ("corpus","checkpoints","out"):p.add_argument("--"+name,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    text,data=corpus(a.corpus);d=Path(a.checkpoints);out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    pa,pb=d/"step_00100.safetensors",d/"step_00500.safetensors"
    assert hashfile(pa)==HASH100
    early=load_file(str(pa));model=reload(pa);adapter=lm.NanoGPTCharModel(model,data.chars)
    model.requires_grad_(False);block=model.transformer.h[0]
    train=text[:int(len(text)*.9)]
    # Disjoint corpus regions for direction selection, solve and local audit.
    cut1,cut2=int(len(train)*.8),int(len(train)*.9)
    xs=[];ys=[]
    with torch.no_grad():
        for x,_,y in windows(adapter,train[:cut1],8192,1729)[0]:
            z=block.ln_1(model.transformer.wte(x)+model.transformer.wpe(torch.arange(64)))
            q,k,_=block.attn.c_attn(z).chunk(3,-1)
            q,k=[t.reshape(len(x),64,4,64).transpose(1,2) for t in (q,k)]
            att=((q[:,:,-1:]@k.transpose(-1,-2))/8).softmax(-1).squeeze(2)
            xs.append(torch.einsum("bht,btd->bhd",att,z).reshape(len(x),1024).double())
            ys.append(torch.nn.functional.one_hot(y,65).double()-adapter.logits(x)[:,-1].double().softmax(-1))
        X,Y=torch.cat(xs),torch.cat(ys);G=X.T@X
        W=torch.linalg.solve(G+.01*G.diag().mean()*torch.eye(1024,dtype=torch.double),X.T@Y)
        E=early["lm_head.weight"].double();H=E@E.T
        P=torch.linalg.solve(H+.01*H.diag().mean()*torch.eye(65,dtype=torch.double),E)
        Ts=[];Rs=[]
        for h in range(4):
            B=(W[h*256:(h+1)*256]@P).T
            _,_,vh=torch.linalg.svd(B,full_matrices=False)
            V=early[QKV][512+h*64:512+(h+1)*64].double()
            T=vh[:4]@torch.linalg.pinv(V)
            R=T@V
            Q,_=torch.linalg.qr(R.T,mode="reduced");R=Q.T
            T=R@torch.linalg.pinv(V)
            torch.testing.assert_close(T@V,R,atol=1e-8,rtol=1e-7)
            Ts.append(T);Rs.append(R)
        T=torch.block_diag(*Ts).float() # 16 x 256
    # Exact functional forward: only O changes, shared over all positions.
    O=early[PROJ]
    def logits(x,D):
        def hook(module,inp,result):
            return result+(inp[0]@T.T)@D.T
        handle=block.attn.c_proj.register_forward_hook(hook)
        try:return model(x)[0][:,-1]
        finally:handle.remove()
    zero=torch.zeros(256,16)
    # Explicit Jacobian: 128 x 65 outputs, 4096 coordinates.
    # Batched reverse mode over output classes, with chunking for memory.
    # Each window is independent; summing same-class logits across batch
    # would mix parameter Jacobians, so use individual examples.
    def jac(x):
        D=zero.clone().requires_grad_(True);v=logits(x,D)
        rows=[]
        for start in range(0,65,8):
            basis=torch.eye(65)[start:start+8,None,:]
            grad=torch.autograd.grad(v,D,grad_outputs=basis,is_grads_batched=True,
                                      retain_graph=start+8<65)[0]
            rows.append(grad.detach().reshape(-1,4096).double())
        return v.detach()[0].double(),torch.cat(rows)
    fit=windows(adapter,train[cut1:cut2],128,1927)[0]
    Js=[];ps=[];targets=[]
    for x,_,y in fit:
        for j in range(len(x)):
            v,J=jac(x[j:j+1]);prob=v.softmax(-1)
            # Fisher square root: diag(sqrt(p)) (I - 1 p^T)
            A=prob.sqrt()[:,None]*(J-(prob@J)[None,:])
            # A^T b = J^T(onehot-p); b = (onehot-p)/sqrt(p)
            b=(torch.nn.functional.one_hot(y[j],65).double()-prob)/prob.sqrt()
            Js.append(A);targets.append(b);ps.append(prob)
    A=torch.cat(Js);b=torch.cat(targets)
    G=A.T@A;lam=.01*G.diag().mean()
    assert lam>0
    delta=torch.linalg.solve(G+lam*torch.eye(4096,dtype=torch.double),A.T@b).reshape(256,16).float()
    native_norm=sum(float(maps(early,h).square().sum()) for h in range(4))**.5
    correction_norm=sum(float((delta[:,h*4:(h+1)*4].double()@Rs[h]).square().sum()) for h in range(4))**.5
    scale=min(1.,.5*native_norm/max(correction_norm,1e-12))
    delta*=scale
    perm=torch.randperm(256,generator=torch.Generator().manual_seed(20261124))
    def state(dose,control=False):
        sd={k:v.clone() for k,v in early.items()}
        sd[PROJ]=O+dose*(delta[perm] if control else delta)@T
        return sd
    # Audit actual full-sequence logit change vs directional derivative.
    audit={}
    for dose in (.25,1.):
        num=den=pred_loss=base_loss=actual_loss=0.
        for x,_,y in windows(adapter,train[cut2:],128,1928)[0]:
            # Forward finite difference around zero, independent of exact dose.
            with torch.no_grad():
                base=logits(x,zero);eps=.001
                tangent=(logits(x,eps*delta)-logits(x,-eps*delta))/(2*eps)
                pred=base+dose*tangent;actual=logits(x,dose*delta)
                err=(actual-pred);change=actual-base
                err-=err.mean(-1,keepdim=True);change-=change.mean(-1,keepdim=True)
                num+=float(err.square().sum());den+=float(change.square().sum())
                base_loss+=float(torch.nn.functional.cross_entropy(base,y,reduction="sum"))
                pred_loss+=float(torch.nn.functional.cross_entropy(pred,y,reduction="sum"))
                actual_loss+=float(torch.nn.functional.cross_entropy(actual,y,reduction="sum"))
        audit[str(dose)]=dict(relative_error=(num/max(den,1e-20))**.5,
                              predicted_improvement=(base_loss-pred_loss)/128,
                              actual_improvement=(base_loss-actual_loss)/128)
    # Verify hook and persistent state agree.
    x=fit[0][0][:2]
    with torch.no_grad():expected=logits(x,delta)
    model.load_state_dict(state(1.),strict=True)
    with torch.no_grad():torch.testing.assert_close(model(x)[0][:,-1],expected,atol=3e-5,rtol=3e-5)
    model.load_state_dict(early,strict=True)
    arms={"early":early}
    for dose in (.25,1.):
        arms["law"+str(dose)]=state(dose);arms["permuted"+str(dose)]=state(dose,True)
    assert hashfile(pb)==HASH500
    arms["late"]=load_file(str(pb))
    report=dict(provenance="local_jacobian_on_trained_step100",optimizer_steps=0,
        sensitivity_method="autograd_full_jacobian",fit_windows=128,basis_windows=8192,
        audit_windows=128,lambda_value=float(lam),native_norm=native_norm,
        unscaled_correction_norm=correction_norm,scale=scale,audit=audit,splits={})
    print("AUDIT",json.dumps(audit),flush=True)
    with torch.no_grad():
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
                r=rows["law"+str(dose)];base=rows["early"];late=rows["late"]
                r["nll_recovery"]=(base["nll"]-r["nll"])/(base["nll"]-late["nll"])
                r["context_recovery"]=(r["gain"]-base["gain"])/(late["gain"]-base["gain"])
                r["pass"]=r["nll"]<base["nll"] and r["improvement_ci"][0]>0 and r["gain_ci"][0]>.01 and r["nll"]<rows["permuted"+str(dose)]["nll"]
            report["splits"][split]=rows
            (out/"partial.json").write_text(json.dumps(report,indent=2))
    report["audit_pass"]=audit["0.25"]["relative_error"]<=.25 and audit["0.25"]["predicted_improvement"]>0
    report["confirmed_doses"]=[d for d in (.25,1.) if report["audit_pass"] and all(s["law"+str(d)]["pass"] for s in report["splits"].values())]
    model.load_state_dict(early,strict=True)
    assert all(torch.equal(v,early[k]) for k,v in model.state_dict().items())
    report["restored_exact"]=True
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(report),flush=True)

if __name__=="__main__":main()
