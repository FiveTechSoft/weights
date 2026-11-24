"""Native residual MLP composition, no donor or training."""
import argparse,itertools,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from cycle_law import new,snapshot

I=(0,1,2,3)
P=(1,2,3,0)
Q=(1,0,2,3)

def compose(a,b):
    return tuple(b[a[i]] for i in range(4))

@torch.no_grad()
def compile_block(block,p):
    assert sorted(p)==list(I)
    D=torch.zeros(16,16)
    for i,j in enumerate(p):D[j,i]=1
    D[:4,:4]-=torch.eye(4)
    s=math.sqrt(15/256+1e-5)
    block.mlp.c_fc.weight.zero_()
    block.mlp.c_proj.weight.zero_()
    block.mlp.c_fc.weight[:16]=torch.eye(16)
    block.mlp.c_fc.weight[16:32]=-torch.eye(16)
    block.mlp.c_proj.weight[:,:16]=s*D
    block.mlp.c_proj.weight[:,16:32]=-s*D

@torch.no_grad()
def compile_pair(a,b):
    m=new(0)
    for w in m.parameters():w.zero_()
    for block in m.transformer.h:
        block.ln_1.weight.fill_(1);block.ln_2.weight.fill_(1)
    m.transformer.ln_f.weight.fill_(4)
    m.transformer.wte.weight[:,:4]=torch.eye(4)
    compile_block(m.transformer.h[0],a)
    compile_block(m.transformer.h[1],b)
    return m.eval()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);args=p.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    permutations=list(itertools.permutations(I))
    histories=[torch.arange(4)[:,None],torch.randint(4,(16,32),generator=torch.Generator().manual_seed(1729))]
    direct={p:compile_pair(p,I) for p in permutations}
    count=correct=noncommuting=order_positions=order_correct=0
    max_residual=max_logits=0.
    rows=[]
    for a,b in itertools.product(permutations,repeat=2):
        m=compile_pair(a,b);ab=compose(a,b);ba=compose(b,a)
        if ab!=ba:noncommuting+=1
        for x in histories:
            seen=[]
            handle=m.transformer.h[0].register_forward_hook(lambda mod,inp,res:seen.append(res.clone()))
            try:logits=m(x,x)[0]
            finally:handle.remove()
            expected=torch.nn.functional.one_hot(torch.tensor(a)[x],16).float()
            err=float((seen[0]-expected).abs().max())
            max_residual=max(max_residual,err)
            torch.testing.assert_close(seen[0],expected,atol=1e-5,rtol=1e-5)
            ref=direct[ab](x,x)[0]
            max_logits=max(max_logits,float((logits-ref).abs().max()))
            torch.testing.assert_close(logits,ref,atol=1e-4,rtol=1e-5)
            y=torch.tensor(ab)[x];pred=logits.argmax(-1)
            correct+=int((pred==y).sum());count+=x.numel()
            differing=y!=torch.tensor(ba)[x]
            order_positions+=int(differing.sum());order_correct+=int(((pred==y)&differing).sum())
        rows.append(dict(first=a,second=b,composition=ab,reversed=ba,commutes=ab==ba))
    for name,a,b in (("P_then_Q",P,Q),("Q_then_P",Q,P),("identity",I,I),("inverse",P,tuple(P.index(i) for i in I))):
        m=compile_pair(a,b)
        path=out/(name+".safetensors");save_file(snapshot(m),str(path))
        restored=new(1);restored.load_state_dict(load_file(str(path)),strict=True);restored.eval()
        assert torch.equal(m(histories[1],histories[1])[0],restored(histories[1],histories[1])[0])
    report=dict(parameters=6800,optimizer_steps=0,donor_weights=False,pairs=len(rows),
        noncommuting_pairs=noncommuting,correct=correct,count=count,order_positions=order_positions,
        order_correct=order_correct,max_intermediate_residual_error=max_residual,
        max_direct_logit_error=max_logits,reload_exact=True,
        passed=correct==count and order_correct==order_positions,example={"P":P,"Q":Q,"P_then_Q":compose(P,Q),"Q_then_P":compose(Q,P)})
    (out/"report.json").write_text(json.dumps(dict(report,rows=rows),indent=2))
    print("SUMMARY",json.dumps(report),flush=True)
    assert report["passed"]
if __name__=="__main__":main()
