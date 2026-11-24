"""Controlled cyclic languages: diagnostic training and analytic construction."""
import argparse,json,math,sys,struct,zlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from engine.nanogpt_runtime import GPTConfig,KarpathyGPT

STEPS=(0,1,5,10,20,50,100,500)
SEEDS=(1337,1338,1339)
def new(seed):
    torch.manual_seed(seed)
    m=KarpathyGPT(GPTConfig(block_size=32,vocab_size=4,n_layer=2,n_head=2,n_embd=16,d_ff=64,bias=False,dropout=0.))
    assert sum(p.numel() for p in m.parameters())==6800
    return m
def snapshot(m):
    return {k:v.detach().cpu().contiguous().clone() for k,v in m.state_dict().items()}
def batch(rng,rule,n=32,length=32):
    if rule==0:return torch.randint(4,(n,length+1),generator=rng)
    starts=torch.randint(4,(n,1),generator=rng)
    return (starts+rule*torch.arange(length+1))%4
@torch.no_grad()
def evaluate(m,rule):
    m.eval();nll=correct=count=0
    if rule:
        for length in (1,2,8,32):
            w=(torch.arange(4)[:,None]+rule*torch.arange(length+1))%4
            x,y=w[:,:-1].contiguous(),w[:,1:].contiguous()
            logits,loss=m(x,y);n=y.numel()
            nll+=float(loss)*n;correct+=int((logits.argmax(-1)==y).sum());count+=n
    else:
        rng=torch.Generator().manual_seed(98765)
        for _ in range(32):
            w=batch(rng,0);x,y=w[:,:-1].contiguous(),w[:,1:].contiguous()
            logits,loss=m(x,y);n=y.numel()
            nll+=float(loss)*n;correct+=int((logits.argmax(-1)==y).sum());count+=n
    return dict(nll=nll/count,accuracy=correct/count,n=count)
def effective(sd):
    result={}
    for layer in range(2):
        prefix=f"transformer.h.{layer}.attn."
        q,k,v=sd[prefix+"c_attn.weight"].chunk(3,0)
        o=sd[prefix+"c_proj.weight"]
        for h in range(2):
            s=slice(h*8,(h+1)*8)
            result[f"L{layer}H{h}_VO"]=o[:,s]@v[s]
            result[f"L{layer}H{h}_QK"]=q[s].T@k[s]/math.sqrt(8)
    return result
def png(path,im):
    def chunk(tag,payload):
        return struct.pack(">I",len(payload))+tag+payload+struct.pack(">I",zlib.crc32(tag+payload)&0xffffffff)
    h,w,_=im.shape;raw=b"".join(b"\x00"+row.tobytes() for row in im)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))
@torch.no_grad()
def compiled(rule):
    # One-hot embeddings in first four residual coordinates.
    # LN(e_i)=(e_i-1/16)/s, s=sqrt(15/256+eps).
    # MLP computes (P-I)e_i via GELU(z)-GELU(-z)=z.
    m=new(0)
    for p in m.parameters():p.zero_()
    for block in m.transformer.h:
        block.ln_1.weight.fill_(1);block.ln_2.weight.fill_(1)
    m.transformer.ln_f.weight.fill_(4)
    m.transformer.wte.weight[:,:4].copy_(torch.eye(4))
    P=torch.zeros(16,16)
    for i in range(4):P[(i+rule)%4,i]=1
    D=P.clone();D[:4,:4]-=torch.eye(4)
    s=math.sqrt(15/256+1e-5)
    mlp=m.transformer.h[0].mlp
    mlp.c_fc.weight[:16].copy_(torch.eye(16))
    mlp.c_fc.weight[16:32].copy_(-torch.eye(16))
    mlp.c_proj.weight[:,:16].copy_(s*D)
    mlp.c_proj.weight[:,16:32].copy_(-s*D)
    # D annihilates the all-ones vector; residual becomes e_{i+rule}.
    return m.eval()
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    report={"parameters":6800,"seeds":{},"compiled":{}}
    for seed in SEEDS:
        initial=snapshot(new(seed));base_maps=effective(initial);states={};rows={}
        for rule in (0,1,-1):
            m=new(seed);assert all(torch.equal(v,initial[k]) for k,v in m.state_dict().items())
            opt=torch.optim.AdamW(m.parameters(),lr=.003)
            rng=torch.Generator().manual_seed(seed+10000);trajectory={};states[rule]={}
            for step in range(501):
                if step in STEPS:
                    sd=snapshot(m);states[rule][step]=sd
                    save_file(sd,str(out/f"seed{seed}_rule{rule}_step{step}.safetensors"))
                    maps=effective(sd)
                    trajectory[str(step)]=dict(metrics=evaluate(m,rule),
                        delta_norms={k:float((v-initial[k]).norm()) for k,v in sd.items()},
                        effective_delta={k:float((v-base_maps[k]).norm()) for k,v in maps.items()})
                if step==500:break
                m.train();w=batch(rng,rule);x,y=w[:,:-1].contiguous(),w[:,1:].contiguous()
                opt.zero_grad(set_to_none=True);_,loss=m(x,y);loss.backward();opt.step()
            rows[str(rule)]=trajectory
        # Conjugation by r(x)=-x mod4. All non-embedding weights untouched.
        f=new(seed);f.load_state_dict(states[1][500]);f.eval()
        rev=new(seed);sd=snapshot(f);r=torch.tensor([0,3,2,1])
        sd["transformer.wte.weight"]=sd["transformer.wte.weight"][r].clone()
        sd["lm_head.weight"]=sd["transformer.wte.weight"].clone()
        rev.load_state_dict(sd);rev.eval()
        x=torch.randint(4,(64,32),generator=torch.Generator().manual_seed(99))
        with torch.no_grad():
            expected=f(r[x],x)[0][...,r];actual=rev(x,x)[0]
            error=float((actual-expected).abs().max())
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
        reverse=evaluate(rev,-1)
        distances={k:float((v-states[-1][500][k]).norm()) for k,v in sd.items()}
        # Fixed per-map scale, common to all arms and snapshots.
        scales={}
        for key in base_maps:
            deltas=[[effective(states[rule][step])[key]-base_maps[key] for step in STEPS] for rule in (0,1,-1)]
            scale=max(float(t.abs().max()) for row in deltas for t in row) or 1.
            scales[key]=scale
            panels=[]
            for row in deltas:
                imgs=[]
                for t in row:
                    v=(t/scale).numpy().clip(-1,1)
                    rgb=np.stack((1-np.maximum(-v,0),1-np.abs(v),1-np.maximum(v,0)),-1)
                    imgs.append((rgb*255).round().astype(np.uint8))
                panels.append(np.concatenate(imgs,1))
            png(out/f"seed{seed}_{key}.png",np.concatenate(panels,0).repeat(6,0).repeat(6,1))
        report["seeds"][str(seed)]=dict(trajectory=rows,conjugation_error=error,reverse=reverse,
            independent_reverse_distances=distances,heatmap_scales=scales,
            pass_reference=rows["1"]["500"]["metrics"]["accuracy"]>=.99 and rows["-1"]["500"]["metrics"]["accuracy"]>=.99 and abs(rows["0"]["500"]["metrics"]["nll"]-math.log(4))<=.15,
            pass_conjugation=reverse["accuracy"]>=.99)
        print("SEED",seed,json.dumps({k:v for k,v in report["seeds"][str(seed)].items() if k!="trajectory"}),flush=True)
    for rule in (1,-1):
        m=compiled(rule);metrics=evaluate(m,rule)
        # Arbitrary histories: rule depends only on current symbol.
        x=torch.randint(4,(128,32),generator=torch.Generator().manual_seed(123))
        with torch.no_grad():accuracy=float((m(x,x)[0].argmax(-1)==(x+rule)%4).float().mean())
        report["compiled"][str(rule)]=dict(metrics=metrics,arbitrary_history_accuracy=accuracy,optimizer_steps=0,
            passed=metrics["accuracy"]==1 and accuracy==1)
        save_file(snapshot(m),str(out/f"compiled_rule{rule}.safetensors"))
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(dict(compiled=report["compiled"],seeds={k:{name:v[name] for name in ("pass_reference","pass_conjugation","conjugation_error","reverse")} for k,v in report["seeds"].items()})),flush=True)
if __name__=="__main__":main()
