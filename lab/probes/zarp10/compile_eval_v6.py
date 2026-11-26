"""V6: frozen Qwen plus persistent neural skip branch; custom runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file
from compile_eval import MODEL, SEED, batches, encode, prose_nll, file_hash
from compile_eval_v2 import zarp, arithmetic, evaluate, digit_ids
from probe_v5 import fit, apply, targets, score

DOSES=(0,4,8,16,32,64)

def ramp(x):
    return torch.relu(4*x-2)-torch.relu(4*x-3)

class Circuit(nn.Module):
    def __init__(self, state):
        super().__init__()
        for k,v in state.items():
            self.register_buffer(k,v)
    def forward(self,h):
        a=ramp(h@self.wa+self.ba)
        b=ramp(h@self.wb+self.bb)
        t=ramp(h@self.wt+self.bt)
        return ((a.unsqueeze(-1)*b.unsqueeze(-2)).flatten(-2)*t)@self.rule
    def trigger(self,h):
        return ramp(h@self.wt+self.bt)

class Extended(nn.Module):
    def __init__(self,base,circuit,digits,dose):
        super().__init__()
        self.base=base
        self.circuit=circuit
        self.register_buffer("digits",torch.tensor(digits,dtype=torch.long))
        self.dose=dose
    def forward(self,**kwargs):
        keep=kwargs.pop("logits_to_keep",0)
        # HF forward returns hidden states; no custom hooks.
        out=self.base(**kwargs,output_hidden_states=True,logits_to_keep=keep)
        h=out.hidden_states[2]
        if isinstance(keep,int) and keep>0:
            h=h[:,-keep:]
        residual=self.circuit(h.float())*self.dose
        logits=out.logits.clone()
        logits.index_add_(-1,self.digits,residual.to(logits.dtype))
        out.logits=logits
        return out

@torch.no_grad()
def collect(base,tok,items,neg=False):
    positive=[];negative=[]
    for batch in batches(items):
        enc=encode(tok,[p for p,_ in batch],base)
        h=base.model(**enc,output_hidden_states=True,use_cache=False).hidden_states[2].float().cpu()
        mask=enc.attention_mask.cpu().bool()
        if neg:
            negative.append(h[mask])
        else:
            positive.append(h[:,-1])
            mask[:,-1]=False
            negative.append(h[mask])
    return (torch.cat(positive) if positive else None,torch.cat(negative))

@torch.no_grad()
def prose_features(base,tok,text):
    ids=tok(text,return_tensors="pt").input_ids
    return torch.cat([base.model(input_ids=ids[:,i:i+128],output_hidden_states=True,use_cache=False).hidden_states[2][0].float().cpu()
                      for i in range(0,ids.shape[1],128)])

def fit_trigger(x,n):
    x=x.double();n=n.double()
    A=torch.cat([torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)/len(x)**.5,
                 torch.cat([n,torch.ones(len(n),1,dtype=n.dtype)],1)/len(n)**.5])
    y=torch.cat([torch.ones(len(x),1,dtype=x.dtype)/len(x)**.5,torch.zeros(len(n),1,dtype=n.dtype)])
    gram=A.T@A
    reg=.001*gram.diag().mean()
    penalty=torch.eye(gram.shape[0],dtype=x.dtype)*reg
    penalty[-1,-1]=0
    w=torch.linalg.solve(gram+penalty,A.T@y)
    return w[:-1].float(),w[-1].float()

def digest(m):
    h=hashlib.sha256()
    for name,p in sorted(m.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def load_base(rev):
    return AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="lab/runs/zarp10_v6")
    p.add_argument("--revision")
    p.add_argument("--prose",required=True)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    base=load_base(rev)
    original_hash=digest(base)
    text=Path(args.prose).read_text(encoding="utf-8")
    pool=[(int(i)//50,int(i)%50) for i in np.random.default_rng(SEED).permutation(2500)]
    train=pool[700:1300];dev=pool[1300:1500]
    X,N=collect(base,tok,zarp(train))
    _,A=collect(base,tok,arithmetic("Calculate: "),neg=True)
    N=torch.cat([N,A,prose_features(base,tok,text[:8192])])
    Y=targets(train)
    wa,ba=fit(X,torch.nn.functional.one_hot(Y[:,0],10))
    wb,bb=fit(X,torch.nn.functional.one_hot(Y[:,1],2))
    wt,bt=fit_trigger(X,N)
    rule=torch.zeros(20,10)
    for a in range(10):
        for b in range(2):
            rule[2*a+b,(3*a+5*b+7)%10]=1
    state=dict(wa=wa.float(),ba=ba.float(),wb=wb.float(),bb=bb.float(),wt=wt,bt=bt,rule=rule)
    circuit=Circuit(state)
    digits=digit_ids(tok)
    wrapper=Extended(base,circuit,digits,0).eval()
    probes=[p for p,_ in zarp(dev[:16])]
    enc=encode(tok,probes,base)
    parity=float((base(**enc,use_cache=False,logits_to_keep=1).logits-wrapper(**enc,use_cache=False,logits_to_keep=1).logits).abs().max())
    print("ZERO_PARITY",parity,flush=True)
    dev_results=[]
    for dose in DOSES:
        wrapper.dose=dose
        row=dict(dose=dose,skill=evaluate(wrapper,tok,zarp(dev))["accuracy"],
                 arithmetic=evaluate(wrapper,tok,arithmetic("Compute: "))["accuracy"],
                 prose_nll=prose_nll(wrapper,tok,text[8192:12288]))
        baseline=dev_results[0] if dev_results else row
        row["feasible"]=row["arithmetic"]>=baseline["arithmetic"]-.02 and row["prose_nll"]<=baseline["prose_nll"]+.1
        dev_results.append(row);print("DEV",json.dumps(row),flush=True)
    dose=max((r for r in dev_results if r["feasible"]),key=lambda r:(r["skill"],-r["dose"]))["dose"]
    sealed=pool[2300:2500]
    def metrics(m):
        return dict(sealed=evaluate(m,tok,zarp(sealed)),arithmetic=evaluate(m,tok,arithmetic()),prose_nll=prose_nll(m,tok,text[-4096:]))
    report=dict(model=MODEL,revision=rev,provenance="calibrated_interface_compiled_neural_skip",
        backward_calls=0,base_sha256=original_hash,zero_parity=parity,dose=dose,dev=dev_results,
        arms={"base":metrics(base)})
    E,_=collect(base,tok,zarp(sealed))
    pred=torch.stack([(E@state["wa"]+state["ba"]).argmax(-1),(E@state["wb"]+state["bb"]).argmax(-1)],-1)
    ye=targets(sealed)
    result=circuit(E)
    _,ordinary=collect(base,tok,arithmetic(),neg=True)
    look=[(f"OTHER10({a},{b}) = ","") for a,b in sealed]
    look_h,_=collect(base,tok,look)
    report["diagnostics"]=dict(operands=score(pred,ye),internal_accuracy=float((result.argmax(-1)==(3*ye[:,0]+5*ye[:,1]+7)%10).float().mean()),
        positive_trigger_mean=float(circuit.trigger(E).mean()),
        negative_trigger_mean=float(circuit.trigger(prose_features(base,tok,text[-4096:])).mean()),
        ordinary_trigger_mean=float(circuit.trigger(ordinary).mean()),
        lookalike_trigger_mean=float(circuit.trigger(look_h).mean()))
    print("DIAGNOSTICS",json.dumps(report["diagnostics"]),flush=True)
    intact=digest(base)==original_hash
    del wrapper,base
    for arm,r in (("candidate",rule),("placebo",rule.roll(1,dims=1))):
        path=out/(arm+".safetensors")
        save_file({**state,"rule":r.contiguous()},str(path),metadata={"base":MODEL,"revision":rev,"dose":str(dose)})
        reloaded=load_base(rev)
        intact=intact and digest(reloaded)==original_hash
        mod=Extended(reloaded,Circuit(load_file(str(path))),digits,dose).eval()
        report["arms"][arm]=dict(**metrics(mod),sha256=file_hash(path))
        intact=intact and digest(reloaded)==original_hash
        del mod,reloaded
    b,c,p=[report["arms"][a] for a in ("base","candidate","placebo")]
    acc=lambda a,s:a[s]["accuracy"]
    report["gates"]=dict(sealed=acc(c,"sealed")>=.95,beats_base=acc(c,"sealed")-acc(b,"sealed")>=.2,
        beats_placebo=acc(c,"sealed")-acc(p,"sealed")>=.2,arithmetic_base=acc(b,"arithmetic")>=.5,
        arithmetic_retention=acc(c,"arithmetic")>=acc(b,"arithmetic")-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1,nonzero_edit=dose>0,
        zero_parity=parity==0,original_weights_intact=intact)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(dict(gates=report["gates"],dose=dose,diagnostics=report["diagnostics"],
        arms={k:{s:v[s]["accuracy"] for s in ("sealed","arithmetic")}|{"prose_nll":v["prose_nll"]} for k,v in report["arms"].items()})),flush=True)
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":
    main()
