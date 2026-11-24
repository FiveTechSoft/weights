"""Analyze reference on validation, never use the sealed test to select mechanisms."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from train_reference import corpus,reload,lm,STEPS

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus)
    out=Path(a.out)
    reference=json.loads((out/"reference.json").read_text())
    if not reference["language_gate"]:
        (out/"analysis.json").write_text(json.dumps({"status":"blocked_reference_failed"}))
        raise SystemExit("Reference language gate failed")
    val=text[-16000:-8000]
    model=reload(out/"step_05000.safetensors")
    initial=load_file(str(out/"step_00000.safetensors"))
    final={k:v.clone() for k,v in model.state_dict().items()}
    adapter=lm.NanoGPTCharModel(model,data.chars)
    def metrics():
        return dict(nll=lm.score_nll(adapter,val),
                    context=lm.check_long_context(adapter,val))
    baseline=metrics()
    rows=[]
    def measure(label,replace):
        state={k:v.clone() for k,v in final.items()}
        replace(state)
        model.load_state_dict(state,strict=True)
        row=dict(component=label,**metrics())
        row["nll_increase"]=row["nll"]-baseline["nll"]
        row["context_gain_lost"]=baseline["context"]["mean_gain_nats"]-row["context"]["mean_gain_nats"]
        rows.append(row)
        print("INTERVENTION",json.dumps(row),flush=True)
        model.load_state_dict(final,strict=True)
    for i in range(model.config.n_layer):
        prefix=f"transformer.h.{i}."
        qkv=prefix+"attn.c_attn.weight"
        proj=prefix+"attn.c_proj.weight"
        mlp=[k for k in final if k.startswith(prefix+"mlp.")]
        d=model.config.n_embd
        def qk(s):s[qkv][:2*d]=initial[qkv][:2*d]
        def vo(s):
            s[qkv][2*d:]=initial[qkv][2*d:]
            s[proj]=initial[proj].clone()
        def reset_mlp(s):
            for k in mlp:s[k]=initial[k].clone()
        measure(prefix+"QK_initial",qk)
        measure(prefix+"VO_initial",vo)
        measure(prefix+"MLP_initial",reset_mlp)
        measure(prefix+"attention_zero",lambda s:s[proj].zero_())
        measure(prefix+"MLP_zero",lambda s:s[prefix+"mlp.c_proj.weight"].zero_())
    def embeddings(s):
        for k in ("transformer.wte.weight","lm_head.weight"):s[k]=initial[k].clone()
    measure("tied_embeddings_initial",embeddings)
    measure("position_embeddings_initial",lambda s:s.update({"transformer.wpe.weight":initial["transformer.wpe.weight"].clone()}))
    assert all(torch.equal(v,final[k]) for k,v in model.state_dict().items())
    trajectory=[]
    for step in STEPS:
        m=reload(out/f"step_{step:05d}.safetensors")
        trajectory.append(dict(step=step,context=lm.check_long_context(lm.NanoGPTCharModel(m,data.chars),val)))
    report=dict(status="completed",baseline=baseline,
        interventions=sorted(rows,key=lambda r:r["context_gain_lost"],reverse=True),
        trajectory=trajectory,restored_exact=True,
        interpretation="necessity under intervention, not sufficiency or synthesis")
    (out/"analysis.json").write_text(json.dumps(report,indent=2))
    print("ANALYSIS",json.dumps(report),flush=True)
    if not reference["full_gate"]:
        raise SystemExit("Language reference passed but ICL gate failed")

if __name__=="__main__":main()
