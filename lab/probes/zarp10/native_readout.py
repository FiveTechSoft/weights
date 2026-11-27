"""Native output-only Qwen surgery; fail closed if protected constraints conflict.

No hooks during inference, no donor, no external arithmetic at inference.
The task is ZARP10; labels are used only for calibration and evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engine.wsp_native_readout import compile_rows
from engine.wsp_native_margins import compile_margins
from engine.wsp_native_qp import compile_qp
from compile_eval import MODEL, SEED, arithmetic, batches, encode, evaluate, prose_nll, file_hash


def items(lo, hi, n, seed):
    w = hi-lo+1
    ids = np.random.default_rng(seed).choice(w*w, n, replace=False)
    pairs = [(lo+int(i)//w, lo+int(i)%w) for i in ids]
    return [(f"ZARP10({a},{b}) =", str((3*a+5*b+7)%10)) for a,b in pairs]


@torch.no_grad()
def features(model, tok, prompts):
    xs = []
    for batch in batches(prompts):
        h = model.model(**encode(tok,batch,model),use_cache=False).last_hidden_state
        xs.append(h[:,-1].cpu())
    return torch.cat(xs)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="lab/runs/native_readout")
    p.add_argument("--prose", required=True)
    p.add_argument("--revision")
    p.add_argument("--solver", choices=("equality", "margin", "qp"), default="equality")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rev = a.revision or HfApi().model_info(MODEL).sha
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)
    text = Path(a.prose).read_text(encoding="utf-8")
    digits = [tok.encode(str(i),add_special_tokens=False) for i in range(10)]
    assert all(len(x)==1 for x in digits)
    idx = torch.tensor([x[0] for x in digits])
    assert len(idx.unique()) == 10
    discovery = items(0,49,100,SEED+101)
    sealed = items(100,999,200,SEED+102)
    retain = arithmetic(SEED+1, n=128)
    protected = [s for s,_ in retain] + [s.replace("ZARP10","OTHER10") for s,_ in discovery[:32]]
    x = features(model,tok,[s for s,_ in discovery])
    h = features(model,tok,protected)
    # Fixed additional protection from prose, across all positions.
    ids = tok(text[:4096],return_tensors="pt").input_ids[:,:128]
    hp = model.model(input_ids=ids,use_cache=False).last_hidden_state.reshape(-1,x.shape[-1])
    h = torch.cat([h,hp.cpu()])
    head = model.get_output_embeddings()
    original = head.weight[idx].clone()
    desired = torch.zeros(len(x),10)
    selected_logits = torch.empty_like(desired)
    outside_max = torch.empty(len(x))
    for start in range(0,len(x),16):
        logits = head(x[start:start+16])
        selected_logits[start:start+len(logits)] = logits[:,idx]
        frozen = logits.clone()
        frozen[:,idx] = -torch.inf
        outside_max[start:start+len(logits)] = frozen.max(dim=1).values
        for j,(_,label) in enumerate(discovery[start:start+16]):
            k = int(label)
            desired[start+j,k] = (logits[j].max()-logits[j,idx[k]]+2).clamp_min(0)
    report = dict(model=MODEL,revision=rev,solver=a.solver,prose_sha256=file_hash(a.prose),
                  calibration=len(x),protected=len(h),width=x.shape[-1],
                  protected_rank=int(torch.linalg.matrix_rank(h.double(),rtol=1e-8)),
                  backward_calls=0,optimizer_steps=0)
    def finish():
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print("SUMMARY",json.dumps(report),flush=True)
    try:
        if a.solver == "qp":
            labels = torch.tensor([int(label) for _,label in discovery],dtype=torch.long)
            candidate, cert = compile_qp(original,x,h,selected_logits,labels,outside_max,
                                          max_delta=10.,margin=2.,atol=1e-3,rtol=1e-8)
        elif a.solver == "margin":
            labels = torch.tensor([int(label) for _,label in discovery],dtype=torch.long)
            candidate, cert = compile_margins(original,x,h,selected_logits,labels,outside_max,
                                               max_delta=10.,margin=2.,atol=1e-3,rtol=1e-8)
        else:
            candidate, cert = compile_rows(original,x,h,desired,max_delta=10.,atol=1e-3,rtol=1e-8)
    except ValueError as exc:
        report.update(gate=False,status="rejected",reason=str(exc))
        if hasattr(exc, "certificate"):
            report["rejection_diagnostics"] = exc.certificate
        finish()
        return
    report["certificate"] = cert
    preservation = arithmetic(SEED+2)
    def score(m):
        return dict(sealed=evaluate(m,tok,sealed)["accuracy"],
                    arithmetic=evaluate(m,tok,preservation)["accuracy"],
                    prose_nll=prose_nll(m,tok,text[-4096:]))
    report["base"] = score(model)
    # Explicitly separate output from embedding: all input weights remain intact.
    embedding = model.get_input_embeddings().weight
    tied = head.weight.data_ptr()==embedding.data_ptr()
    report["original_tied"] = tied
    if tied:
        new = torch.nn.Linear(head.in_features,head.out_features,bias=False)
        new.weight.copy_(head.weight)
        model.config.tie_word_embeddings=False
        model.set_output_embeddings(new)
        head=new
    delta = candidate-original
    report["placebo_norm"] = float(delta.roll(1,0).norm())
    head.weight[idx] = original+delta.roll(1,0)
    report["placebo"] = score(model)
    head.weight[idx] = candidate
    path=out/"candidate"
    model.save_pretrained(path,safe_serialization=True)
    tok.save_pretrained(path)
    del head, embedding, model
    reloaded=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32).eval().requires_grad_(False)
    assert torch.equal(reloaded.get_output_embeddings().weight[idx],candidate)
    assert reloaded.get_output_embeddings().weight.data_ptr()!=reloaded.get_input_embeddings().weight.data_ptr()
    report["candidate"]=score(reloaded)
    b,c,q=report["base"],report["candidate"],report["placebo"]
    report["gates"]=dict(sealed=c["sealed"]>=.95,beats_base=c["sealed"]-b["sealed"]>=.2,
        beats_placebo=c["sealed"]-q["sealed"]>=.2,arithmetic=c["arithmetic"]>=b["arithmetic"]-.02,
        prose=c["prose_nll"]<=b["prose_nll"]+.1)
    report["gate"]=all(report["gates"].values())
    report["status"]="evaluated"
    finish()


if __name__=="__main__":
    main()
