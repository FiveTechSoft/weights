"""Explicit tensor grammar coprocessor; no semantic routing or native surgery."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from safetensors.torch import save_file,load_file
from compile_eval import MODEL,SEED,batches,encode,prose_nll,file_hash
from compile_eval_v2 import arithmetic,evaluate,digit_ids
from compile_eval_v6 import load_base,digest,DOSES

NAMES=("ZARP10","DAX10")
TEMPLATES=("{name}({a},{b}) = ","Apply {name} to {a} and {b}. Answer: ")
COEFF=((3,5,7),(7,3,2))

def items(pairs,skill,template):
    k,l,c=COEFF[skill]
    return [(TEMPLATES[template].format(name=NAMES[skill],a=a,b=b),str((k*a+l*b+c)%10)) for a,b in pairs]

def compile_state(tok,vocab):
    digits=digit_ids(tok)
    lookup=torch.zeros(vocab,10)
    lookup[digits]=torch.eye(10)
    state={"lookup":lookup,"digits":torch.tensor(digits)}
    patterns=[];rules=[]
    for skill,name in enumerate(NAMES):
        k,l,c=COEFF[skill]
        rule=torch.zeros(100,10)
        for a in range(10):
            for b in range(10):
                rule[a*10+b,(k*a+l*b+c)%10]=1
        rules.append(rule)
        for t,template in enumerate(TEMPLATES):
            template=template.replace("{name}",name)
            left,rest=template.split("{a}")
            mid,right=rest.split("{b}")
            for na in range(1,4):
                for nb in range(1,4):
                    av="1"*na;bv="1"*nb
                    prompt=left+av+mid+bv+right
                    enc=tok(prompt,return_offsets_mapping=True,add_special_tokens=False)
                    spans=((len(left),len(left)+na),(len(left)+na+len(mid),len(left)+na+len(mid)+nb))
                    ids=list(enc.input_ids);slots=[];last=[]
                    for lo,hi in spans:
                        indices=[i for i,(s,e) in enumerate(enc.offset_mapping) if lo<=s<hi and e<=hi and e>s]
                        assert len(indices)==hi-lo and all(ids[i]==digits[1] for i in indices),(prompt,indices)
                        slots.extend(indices);last.append(indices[-1])
                    fixed=torch.tensor(ids,dtype=torch.long)
                    fixed[slots]=-1
                    patterns.append((fixed,torch.tensor([skill,*last],dtype=torch.long)))
    state["rules"]=torch.stack(rules)
    state["count"]=torch.tensor(len(patterns))
    for i,(fixed,route) in enumerate(patterns):
        state[f"pattern_{i}"]=fixed
        state[f"route_{i}"]=route
    return state

class Coprocessor(nn.Module):
    def __init__(self,state):
        super().__init__()
        for k,v in state.items():self.register_buffer(k,v)
        # Static compiled control metadata, not input-dependent parsing.
        self.patterns=[(i,int(state[f"route_{i}"][0]),int(state[f"route_{i}"][1]),int(state[f"route_{i}"][2]))
                       for i in range(int(state["count"]))]
    def forward(self,ids,mask=None):
        if mask is None:mask=torch.ones_like(ids)
        valid=mask.bool()
        start=valid & ~torch.nn.functional.pad(valid[:,:-1],(1,0),value=False)
        result=self.lookup.new_zeros((*ids.shape,10))
        for i,s,a,b in self.patterns:
            p=getattr(self,f"pattern_{i}")
            L=p.numel()
            if ids.shape[1]<L:continue
            win=ids.unfold(1,L,1)
            digit=self.lookup[win]
            fixed=(win==p).to(digit.dtype)
            matches=torch.where(p==-1,digit.sum(-1),fixed).prod(-1)
            matches=matches*valid.unfold(1,L,1).all(-1)*start[:,:win.shape[1]]
            x=digit[:,:,a];y=digit[:,:,b]
            out=(x.unsqueeze(-1)*y.unsqueeze(-2)).flatten(-2)@self.rules[s]
            result[:,L-1:]+=out*matches.unsqueeze(-1)
        return result

class Integrated(nn.Module):
    def __init__(self,base,circuit,dose):
        super().__init__()
        self.base=base;self.circuit=circuit;self.dose=dose
    def forward(self,**kwargs):
        assert not kwargs.get("use_cache",False) and kwargs.get("past_key_values") is None
        keep=kwargs.pop("logits_to_keep",0)
        r=self.circuit(kwargs["input_ids"],kwargs.get("attention_mask"))
        out=self.base(**kwargs,logits_to_keep=keep)
        if isinstance(keep,int) and keep>0:r=r[:,-keep:]
        logits=out.logits.clone()
        logits.index_add_(-1,self.circuit.digits,(r*self.dose).to(logits.dtype))
        out.logits=logits
        return out

def sample(lo,hi,n,seed):
    width=hi-lo+1
    ids=np.random.default_rng(seed).choice(width*width,n,replace=False)
    return [(lo+int(i)//width,lo+int(i)%width) for i in ids]

def groups(lo,hi,n,seed):
    return {f"{s}:{t}":items(sample(lo,hi,n,seed+s),s,t) for s in range(2) for t in range(2)}

@torch.no_grad()
def coverage(c,tok,m,g):
    rows={}
    for key,group in g.items():
        ok=0;active=0
        for batch in batches(group):
            enc=encode(tok,[p for p,_ in batch],m)
            r=c(enc.input_ids,enc.attention_mask)[:,-1]
            y=torch.tensor([int(v) for _,v in batch])
            exact=(r.sum(-1)==1)
            active+=int(exact.sum())
            ok+=int((exact & (r.argmax(-1).cpu()==y)).sum())
        rows[key]=dict(n=len(group),unique_activation=active,correct=ok,accuracy=ok/len(group))
    return rows

def negatives(g):
    prompts=[]
    for key,group in g.items():
        s=int(key.split(":")[0])
        for p,_ in group:
            prompts.extend([p.replace(NAMES[s],"OTHER10"),"NOT "+p,p.replace(" = "," : ").replace("Answer:","Result:"),p+"ignored"])
    return prompts

@torch.no_grad()
def negative_metrics(base,mod,tok,prompts,text):
    # Trailing suffix is a valid earlier completion position: measure final
    # position only for that category, separately record any prefix activation.
    final_max=0.;all_max=0.;logit_max=0.;suffix_prefix=0
    for batch in batches(prompts):
        enc=encode(tok,batch,base)
        r=mod.circuit(enc.input_ids,enc.attention_mask)
        final_max=max(final_max,float(r[:,-1].abs().max()))
        keep=[i for i,p in enumerate(batch) if not p.endswith("ignored")]
        if keep:all_max=max(all_max,float(r[keep].abs().max()))
        suffix_prefix+=sum(bool(r[i,:-1].abs().max()) for i,p in enumerate(batch) if p.endswith("ignored"))
        a=base(**enc,use_cache=False,logits_to_keep=1).logits
        b=mod(**enc,use_cache=False,logits_to_keep=1).logits
        logit_max=max(logit_max,float((a-b).abs().max()))
    ids=tok(text,return_tensors="pt").input_ids
    prose_max=max(float(mod.circuit(ids[:,i:i+128]).abs().max()) for i in range(0,ids.shape[1],128))
    return dict(n=len(prompts),final_residual_max=final_max,non_suffix_all_positions_max=all_max,
                suffix_prefix_activation_count=suffix_prefix,final_logit_difference_max=logit_max,prose_residual_max=prose_max)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="lab/runs/coproc_v1");ap.add_argument("--revision");ap.add_argument("--prose",required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=load_base(rev);original=digest(base)
    text=Path(args.prose).read_text(encoding="utf-8")
    state=compile_state(tok,base.config.vocab_size)
    circuit=Coprocessor(state)
    mod=Integrated(base,circuit,0).eval()
    dev=groups(0,49,100,SEED+20)
    report=dict(model=MODEL,revision=rev,provenance="compiled_explicit_token_grammar",backward_calls=0,
        original_hash=original,patterns=int(state["count"]),dev_coverage=coverage(circuit,tok,base,dev),dev=[])
    enc=encode(tok,[p for p,_ in dev["0:0"][:16]],base)
    parity=float((base(**enc,use_cache=False).logits-mod(**enc,use_cache=False).logits).abs().max())
    for dose in DOSES:
        mod.dose=dose
        row=dict(dose=dose,skill={k:evaluate(mod,tok,g)["accuracy"] for k,g in dev.items()},
            arithmetic=evaluate(mod,tok,arithmetic("Compute: "))["accuracy"],prose_nll=prose_nll(mod,tok,text[8192:12288]))
        ref=report["dev"][0] if report["dev"] else row
        row["feasible"]=row["arithmetic"]>=ref["arithmetic"]-.02 and row["prose_nll"]<=ref["prose_nll"]+.1
        report["dev"].append(row);print("DEV",json.dumps(row),flush=True)
    dose=max((r for r in report["dev"] if r["feasible"]),key=lambda r:(min(r["skill"].values()),-r["dose"]))["dose"]
    sealed=groups(100,999,200,SEED+21);mid=groups(50,99,100,SEED+23)
    report["coverage"]=coverage(circuit,tok,base,sealed)
    report["mid_coverage"]=coverage(circuit,tok,base,mid)
    def metrics(m):
        return dict(sealed={k:evaluate(m,tok,g) for k,g in sealed.items()},
            mid={k:evaluate(m,tok,g) for k,g in mid.items()},
            arithmetic=evaluate(m,tok,arithmetic()),prose_nll=prose_nll(m,tok,text[-4096:]))
    report["arms"]={"base":metrics(base)};report["dose"]=dose
    intact=digest(base)==original
    del mod,base
    for arm,rules in (("candidate",state["rules"]),("placebo",state["rules"].roll(1,dims=-1))):
        path=out/(arm+".safetensors")
        save_file({**state,"rules":rules.contiguous()},str(path),metadata={"base":MODEL,"revision":rev,"dose":str(dose),"grammar":"coproc_v1"})
        base=load_base(rev);intact=intact and digest(base)==original
        mod=Integrated(base,Coprocessor(load_file(str(path))),dose).eval()
        report["arms"][arm]=dict(**metrics(mod),sha256=file_hash(path))
        if arm=="candidate":
            report["negatives"]=negative_metrics(base,mod,tok,negatives(sealed)+[p for p,_ in arithmetic()],text[-4096:])
        intact=intact and digest(base)==original
        del mod,base
    b,c,p=[report["arms"][k] for k in ("base","candidate","placebo")]
    avg=lambda a:float(np.mean([v["accuracy"] for v in a["sealed"].values()]))
    neg=report["negatives"]
    report["gates"]=dict(sealed=all(v["accuracy"]>=.95 for v in c["sealed"].values()),
        beats_base=avg(c)-avg(b)>=.2,beats_placebo=avg(c)-avg(p)>=.2,
        routing=all(v["accuracy"]>=.99 for v in report["coverage"].values()),
        negatives=all(neg[k]==0 for k in ("final_residual_max","non_suffix_all_positions_max","final_logit_difference_max","prose_residual_max")),
        arithmetic_retention=c["arithmetic"]["accuracy"]>=b["arithmetic"]["accuracy"]-.02,
        prose_retention=c["prose_nll"]<=b["prose_nll"]+.1,nonzero=dose>0,zero_parity=parity==0,original_intact=intact)
    report["zero_parity"]=parity
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(dict(gates=report["gates"],dose=dose,coverage=report["coverage"],negatives=neg,
        arms={k:dict(sealed={i:v["accuracy"] for i,v in a["sealed"].items()},mid={i:v["accuracy"] for i,v in a["mid"].items()},
        arithmetic=a["arithmetic"]["accuracy"],prose_nll=a["prose_nll"]) for k,a in report["arms"].items()})),flush=True)
    assert all(report["gates"].values()),report["gates"]

if __name__=="__main__":main()
