"""Stage 1: diagnose emergence of context, not synthesize weights."""
import argparse
import itertools
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from train_reference import corpus, reload, lm, hashfile

HASH100="6142873d464aee4fd15031fe4e0e07b6e32a11d54c57761a692d3c483a327c94"
HASH500="53e4bcfd09c5e143f76fb92e414d2718f572f9b3de29ae9c704ab2a8d5610875"
GROUPS=("E","QK","VO","M","R")
QKV="transformer.h.0.attn.c_attn.weight"
PROJ="transformer.h.0.attn.c_proj.weight"
EMB={"transformer.wte.weight","lm_head.weight","transformer.wpe.weight"}

def group(k):
    if k in EMB:return "E"
    if k==PROJ:return "VO"
    if k.startswith("transformer.h.0.mlp."):return "M"
    return "R"

def hybrid(early,late,mask):
    selected={g for i,g in enumerate(GROUPS) if mask & (1<<i)}
    out={}
    for k,v in early.items():
        if k==QKV:
            d=v.shape[1]
            assert v.shape==(3*d,d)
            out[k]=torch.cat((late[k][:2*d] if "QK" in selected else v[:2*d],
                              late[k][2*d:] if "VO" in selected else v[2*d:]),dim=0)
        else:out[k]=(late[k] if group(k) in selected else v).clone()
    assert torch.equal(out["transformer.wte.weight"],out["lm_head.weight"])
    return out

def labels(mask):
    return [g for i,g in enumerate(GROUPS) if mask & (1<<i)]

def summarize(rows):
    lo,hi=rows[0],rows[31]
    delta=hi["context"]["mean_gain_nats"]-lo["context"]["mean_gain_nats"]
    assert delta>0, "no positive endpoint difference"
    for mask,r in rows.items():
        recovery=(r["context"]["mean_gain_nats"]-lo["context"]["mean_gain_nats"])/delta
        residual=(rows[31^mask]["context"]["mean_gain_nats"]-lo["context"]["mean_gain_nats"])/delta
        r.update(recovery=recovery,reverse_residual=residual,
                 sufficient=recovery>=.8 and r["context"]["ci95_low"]>.01 and r["nll"]<=lo["nll"],
                 necessary=residual<=.2)
    eligible=[m for m,r in rows.items() if r["sufficient"]]
    best=min((m.bit_count() for m in eligible),default=6)
    return sorted(m for m in eligible if m.bit_count()==best)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True)
    p.add_argument("--checkpoints",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus)
    directory=Path(a.checkpoints)
    p100=directory/"step_00100.safetensors"
    p500=directory/"step_00500.safetensors"
    assert hashfile(p100)==HASH100 and hashfile(p500)==HASH500
    early,late=load_file(str(p100)),load_file(str(p500))
    assert early.keys()==late.keys()
    model=reload(p100)
    adapter=lm.NanoGPTCharModel(model,data.chars)
    for mask,reference in ((0,early),(31,late)):
        assert all(torch.equal(v,reference[k]) for k,v in hybrid(early,late,mask).items())
    def evaluate(mask,subset):
        model.load_state_dict(hybrid(early,late,mask),strict=True)
        assert model.lm_head.weight is model.transformer.wte.weight
        row=dict(mask=mask,groups=labels(mask),nll=lm.score_nll(adapter,subset),
                 context=lm.check_long_context(adapter,subset))
        print("MASK",json.dumps(row),flush=True)
        return row
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    validation=text[-16000:-8000]
    rows={}
    for mask in [0,31]+list(range(1,31)):
        rows[mask]=evaluate(mask,validation)
        if mask==0:
            assert abs(rows[0]["nll"]-2.5598508372451683)<1e-4
            assert abs(rows[0]["context"]["mean_gain_nats"]-.0025)<1e-4
        if mask==31:
            assert abs(rows[31]["nll"]-2.3521015463136297)<1e-4
            assert abs(rows[31]["context"]["mean_gain_nats"]-.1039)<1e-4
        (out/"partial.json").write_text(json.dumps(rows,indent=2))
    minimal=summarize(rows)
    confirmation_text=text[-24000:-16000]
    confirm_masks=sorted({0,31}|set(minimal)|{31^m for m in minimal})
    confirm={m:evaluate(m,confirmation_text) for m in confirm_masks}
    # Confirmation only tests selected masks and their complements, not a second search.
    confirmed=[]
    delta=confirm[31]["context"]["mean_gain_nats"]-confirm[0]["context"]["mean_gain_nats"]
    if delta>0:
        for m in minimal:
            r=confirm[m]
            recovery=(r["context"]["mean_gain_nats"]-confirm[0]["context"]["mean_gain_nats"])/delta
            residual=(confirm[31^m]["context"]["mean_gain_nats"]-confirm[0]["context"]["mean_gain_nats"])/delta
            r.update(recovery=recovery,reverse_residual=residual,
                     sufficient=recovery>=.8 and r["context"]["ci95_low"]>.01 and r["nll"]<=confirm[0]["nll"],
                     necessary=residual<=.2)
            if r["sufficient"]:confirmed.append(m)
    model.load_state_dict(late,strict=True)
    assert all(torch.equal(v,late[k]) for k,v in model.state_dict().items())
    report=dict(provenance="diagnostic_checkpoint_swaps",training_updates=0,
        checkpoint_hashes=[HASH100,HASH500],rows=rows,minimum_masks=minimal,
        confirmation=confirm,confirmed_masks=confirmed,
        confirmed_localized=[m for m in confirmed if m!=31 and not m&16],
        confirmed_bidirectional=[m for m in confirmed if m!=31 and rows[m]["necessary"] and confirm[m]["necessary"]],
        restored_exact=True)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(report),flush=True)

if __name__=="__main__":main()
