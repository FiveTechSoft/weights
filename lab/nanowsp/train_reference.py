"""Gradient-trained reference only; reuse the repository's NanoGPT runtime."""
import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"lab")]
from engine.nanogpt_runtime import GPTConfig, NanoGPT
from engine.wsp_train import TrainConfig, char_dataset, build_optimizer
import llmness as lm

HASH="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
STEPS=(0,10,100,500,1000,2000,5000)
SEED=1337

def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def corpus(path):
    assert hashfile(path)==HASH, "unexpected corpus"
    text=Path(path).read_text(encoding="utf-8")
    data=char_dataset(text)
    assert len(data.chars)==65
    return text,data

def config():
    return GPTConfig(block_size=256,vocab_size=65,n_layer=4,n_head=4,n_embd=256,d_ff=1024,bias=False,dropout=0.)

def reload(path):
    rt=NanoGPT(config(),seed=SEED)
    sd=load_file(str(path))
    assert torch.equal(sd["lm_head.weight"],sd["transformer.wte.weight"])
    rt.load_checkpoint(path)
    assert rt.model.lm_head.weight is rt.model.transformer.wte.weight
    return rt.model.eval()

@torch.no_grad()
def snapshot(model,step,out,data,device):
    path=out/f"step_{step:05d}.safetensors"
    state={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    save_file(state,str(path),metadata={"provenance":"gradient_descent_reference",
        "backward_calls":str(step),"updates":str(step),"seed":str(SEED),
        "config":json.dumps(asdict(config())),"vocab_chars":data.chars,"corpus_sha256":HASH})
    loaded=reload(path)
    for k,v in loaded.state_dict().items():
        assert torch.equal(v,state[k]),k
    # Compare independent CPU copies, avoiding device-kernel numerical differences.
    check=NanoGPT(config(),seed=SEED).model.eval()
    check.load_state_dict(state,strict=True)
    x=torch.tensor(data.val[:128]).unsqueeze(0)
    assert torch.equal(check(x)[0],loaded(x)[0])
    validation=data.val[-16000:-8000]
    valtext="".join(data.chars[int(i)] for i in validation)
    nll=lm.score_nll(lm.NanoGPTCharModel(loaded,data.chars),valtext)
    row=dict(step=step,validation_nll=nll,sha256=hashfile(path),reload_exact=True)
    print("CHECKPOINT",json.dumps(row),flush=True)
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=NanoGPT(config(),seed=SEED).model.to(a.device)
    opt=build_optimizer(model,TrainConfig(device=a.device,learning_rate=6e-4,beta1=.9,beta2=.95,weight_decay=.1))
    generator=torch.Generator().manual_seed(SEED+1)
    train=torch.from_numpy(data.train)
    offsets=torch.arange(257)
    history=[snapshot(model,0,out,data,a.device)]
    for step in range(1,5001):
        model.train()
        starts=torch.randint(len(train)-256,(8,),generator=generator)
        window=train[starts[:,None]+offsets].to(a.device)
        x,y=window[:,:-1].contiguous(),window[:,1:].contiguous()
        lr=6e-4*step/100 if step<=100 else 6e-5+.5*(6e-4-6e-5)*(1+math.cos(math.pi*(step-100)/4900))
        for group in opt.param_groups:group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        loss=model(x,y)[1]
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in STEPS:
            history.append(snapshot(model,step,out,data,a.device))
            (out/"trajectory.json").write_text(json.dumps(history,indent=2))
    loaded=reload(out/"step_05000.safetensors")
    report=lm.evaluate(lm.NanoGPTCharModel(loaded,data.chars),*lm.split_corpus(text,.1,8000))
    report.update(provenance="gradient_descent_reference",updates=5000,backward_calls=5000,
        parameters=sum(p.numel() for p in loaded.parameters()),device=a.device,torch_version=torch.__version__,
        corpus_sha256=HASH,trajectory=history)
    l1=report["L1"]
    report["gates"]=dict(nll_margin=l1["nll"]<=min(l1["ngram_ladder_nll"].values())-.05,
        l1=l1["verdict"]=="PASSES",context=report["L2_long_context"]["verdict"]=="PASSES",
        generation=report["GEN"]["verdict"]=="PASSES",icl=report["L3_icl"]["verdict"]=="PASSES")
    report["language_gate"]=all(v for k,v in report["gates"].items() if k!="icl")
    report["full_gate"]=all(report["gates"].values())
    (out/"reference.json").write_text(json.dumps(report,indent=2))
    print("REFERENCE",json.dumps(report),flush=True)

if __name__=="__main__":main()
