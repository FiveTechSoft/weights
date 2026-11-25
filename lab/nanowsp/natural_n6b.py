"""N6b: preserve zero mean by decoding the retrieved 64D basis to a 65D simplex."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,build,report,V,D,ANCHOR
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

@torch.no_grad()
def targets(teacher,ids):
    big=torch.tensor([teacher.row((i,)) for i in range(V)],dtype=torch.double)
    tri=torch.tensor([[teacher.row((j,i)) for i in range(V)] for j in range(V)],dtype=torch.double)
    freq=torch.bincount(ids[:-1]*V+ids[1:],minlength=V*V)
    selected=freq.argsort(descending=True,stable=True)[:512]
    sparse=big[None,:,:].expand(V,V,V).clone()
    sparse.reshape(-1,V)[selected]=tri.reshape(-1,V)[selected]
    def logits(p):
        l=p.log();return l-l.mean(-1,keepdim=True)
    B=logits(big);T=logits(sparse)
    C=2*ANCHOR**2+max(float(B.square().sum(-1).max()),float(T.square().sum(-1).max()))
    def encode(l):
        z=torch.zeros(*l.shape[:-1],D,dtype=torch.double)
        z[...,:V]=l
        a=((C-l.square().sum(-1))/2).sqrt()
        z[...,245]=a;z[...,246]=-a
        return z
    return big,tri,sparse,selected,encode(B),encode(T),C

@torch.no_grad()
def compile_model(teacher,ids,B,T,selected,C,compose):
    m=build(teacher,ids)
    b=m.transformer.h[0]
    # Remove all statistical attention except full lag-1 identity retrieval.
    qkv=b.attn.c_attn.weight
    qkv[64:D].zero_();qkv[D+64:2*D].zero_();qkv[2*D+64:].zero_()
    b.attn.c_proj.weight[:,64:].zero_()
    # The input LN now sees two extra unit coordinates at every position.
    # Rescale retrieval projections by ratio of new and original LN scales.
    old=math.sqrt(((1-1/V)+1+2*ANCHOR**2)/D+1e-5)
    initial=math.sqrt(((1-1/V)+3+2*ANCHOR**2)/D+1e-5)
    qkv.mul_(initial/old)
    m.transformer.wpe.weight[:,247]=1
    m.transformer.wpe.weight[:,248]=-1
    m.transformer.wpe.weight[0,247]=-1
    m.transformer.wpe.weight[0,248]=1
    # The full 64-dimensional SVD basis is the zero-sum character subspace.
    basis=qkv[2*D:2*D+64,:V].T.double()/initial
    if compose:
        b.attn.c_proj.weight.zero_()
        b.attn.c_proj.weight[65:130,:64]=basis.float()
    else:
        qkv.zero_();b.attn.c_proj.weight.zero_()
    s=math.sqrt(((1-1/V)*(2 if compose else 1)+3+2*ANCHOR**2)/D+1e-5)
    # For centered one-hot e_i - 1/V, map to desired B_i by adding mean
    # through constant anchor. Subtract identity to cancel the residual.
    M=-torch.eye(D,dtype=torch.double)
    M[:,:V]+=B.T
    M[:,245]+=B.mean(0)/ANCHOR
    f=b.mlp.c_fc.weight;o=b.mlp.c_proj.weight
    f.zero_();o.zero_()
    f[:D]=s*torch.eye(D);f[D:2*D]=-s*torch.eye(D)
    o[:,:D]=M.float();o[:,D:2*D]=-M.float()
    if compose:
        for k,idx in enumerate(selected.tolist()):
            j,i=divmod(idx,V)
            u=torch.zeros(D,dtype=torch.double)
            u[i]=1;u[65+j]=1
            u[247]=.5;u[245]=-2*(1-1/V)/ANCHOR
            f[512+k]=32*s*u.float()
            o[:,512+k]=((T[j,i]-B[i])/16).float()
    m.transformer.ln_f.weight.fill_(math.sqrt(C/D+1e-5))
    return m,basis

def expected(x,big,sparse,compose):
    p=big[x].clone()
    if compose:p[:,1:]=sparse[x[:,:-1],x[:,1:]]
    return p

@torch.no_grad()
def audit(m,basis,big,sparse,compose,C):
    g=torch.Generator().manual_seed(606)
    pairs=torch.arange(V*V)
    maxerr=normerr=causalerr=prefixerr=0.;mass=1.;retrievalerr=0.;count=0
    premean=prevariance=gateerr=0.;wrong_gates=0
    firsterr=0.
    # All 4225 pairs, at three positions, with independent random histories.
    for t in (1,7,31):
        for group in pairs.split(128):
            x=torch.randint(V,(len(group),t+3),generator=g)
            x[:,t-1]=group//V;x[:,t]=group%V
            target=expected(x,big,sparse,compose)
            r=m.transformer.wte(x)+m.transformer.wpe(torch.arange(x.shape[1]))
            b=m.transformer.h[0]
            if compose:
                q,k,v=b.attn.c_attn(b.ln_1(r)).chunk(3,-1)
                score=(q[:,t:t+1,:64]@k[:,:t+1,:64].transpose(-1,-2))/8
                mass=min(mass,float(score.softmax(-1)[:,:,t-1].min()))
                att=b.attn(b.ln_1(r))
                desired=torch.nn.functional.one_hot(x[:,t-1],V).double()-1/V
                retrievalerr=max(retrievalerr,float((att[:,t,65:130].double()-desired).abs().max()))
            pre=r+b.attn(b.ln_1(r))
            premean=max(premean,float(pre.mean(-1).abs().max()))
            prevariance=max(prevariance,float((pre.var(-1,unbiased=False)-((1-1/V)*(2 if compose else 1)+3+2*ANCHOR**2)/D).abs().max()))
            if compose:
                gates=b.mlp.gelu(b.mlp.c_fc(b.ln_2(pre)))[:,:,512:]
                # At most one selected pair may activate, and none at t=0.
                wrong_gates+=int((gates[:,0].abs()>1e-4).sum())
                wrong_gates+=int(((gates[:,1:]>1e-4).sum(-1)>1).sum())
                gateerr=max(gateerr,float(torch.minimum(gates.abs(),(gates-16).abs()).max()))
            for block in m.transformer.h:r=block(r)
            normerr=max(normerr,float((r[:,t].double().square().sum(-1)/C-1).abs().max()))
            a=m(x,x)[0].double().softmax(-1)
            maxerr=max(maxerr,float((a[:,t]-target[:,t]).abs().max()))
            firsterr=max(firsterr,float((a[:,0]-big[x[:,0]]).abs().max()))
            changed=x.clone();changed[:,t+1:]=(changed[:,t+1:]+1)%V
            causalerr=max(causalerr,float((m(changed,changed)[0][:,t]-m(x,x)[0][:,t]).abs().max()))
            if t>1:
                changed=x.clone();changed[:,:t-1]=(changed[:,:t-1]+1)%V
                prefixerr=max(prefixerr,float((m(changed,changed)[0][:,t].double().softmax(-1)-a[:,t]).abs().max()))
            count+=len(group)
    # Exhaustive single-character input with no preceding token.
    single=torch.arange(V)[:,None]
    firsterr=max(firsterr,float((m(single,single)[0][:,0].double().softmax(-1)-big).abs().max()))
    return dict(pair_cases=count,probability_comparisons=count*V,max_probability_error=maxerr,
        first_position_error=firsterr,relative_norm_error=normerr,causal_logit_error=causalerr,
        prefix_probability_error=prefixerr,min_attention_mass=mass,retrieval_error=retrievalerr,pre_layernorm_mean_max=premean,pre_layernorm_variance_error=prevariance,gate_amplitude_error=gateerr,wrong_gate_count=wrong_gates)

@torch.no_grad()
def score(m,big,tri,sparse,compose,chars,text):
    xids=torch.tensor([chars.index(c) for c in text])
    n=(len(xids)-1)//128
    windows=xids[torch.arange(n)[:,None]*128+torch.arange(129)]
    totals=dict(native=0.,reference=0.,bigram=0.,full_trigram=0.);count=0
    for w in windows.split(8):
        x=w[:,:-1].contiguous();y=w[:,1:].unsqueeze(-1)
        probs={"native":m(x,x)[0].double().log_softmax(-1),
               "reference":expected(x,big,sparse,compose).log(),
               "bigram":big[x].log(),"full_trigram":expected(x,big,tri,True).log()}
        for k,p in probs.items():totals[k]-=float(p.gather(-1,y).sum())
        count+=y.numel()
    return dict(count=count,**{k:v/count for k,v in totals.items()})

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars);ids=torch.tensor(teacher.encode(train[:cut]))
    big,tri,sparse,selected,B,T,C=targets(teacher,ids)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    rows={};models={}
    for compose in (False,True):
        name="sparse_trigram" if compose else "bigram"
        m,basis=compile_model(teacher,ids,B,T,selected,C,compose)
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"exact_count_compilation","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);probe=ids[:128].unsqueeze(0)
        exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        row=dict(audit=audit(loaded,basis,big,sparse,compose,C),reload_exact=exact,
                 nonzero_parameters=sum(int(p.count_nonzero()) for p in m.parameters()))
        for k,t in slices.items():row[k]=score(loaded,big,tri,sparse,compose,data.chars,t)
        rows[name]=row;models[name]=loaded
        print("STAGE",json.dumps(dict(mode=name,**row)),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
    fidelity_gate=all(max(r["audit"]["max_probability_error"],r["audit"]["first_position_error"],
                          r["audit"]["prefix_probability_error"],r["audit"]["causal_logit_error"],
                          r["audit"]["relative_norm_error"])<=1e-4 and
                      all(abs(r[k]["native"]-r[k]["reference"])<=1e-4 for k in slices)
                      and r["audit"]["pre_layernorm_mean_max"]<=1e-5 and r["audit"]["wrong_gate_count"]==0 and r["audit"]["gate_amplitude_error"]<=1e-3
                      and r["reload_exact"] for r in rows.values())
    retrieval_gate=rows["sparse_trigram"]["audit"]["min_attention_mass"]>=.99999 and rows["sparse_trigram"]["audit"]["retrieval_error"]<=1e-4
    gains={k:rows["bigram"][k]["native"]-rows["sparse_trigram"][k]["native"] for k in slices}
    result=dict(parameters=sum(p.numel() for p in models["bigram"].parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
                selected_pairs=512,total_pairs=V*V,squared_norm=C,
                selected_training_count_fraction=float(torch.bincount(ids[:-1]*V+ids[1:],minlength=V*V)[selected].sum())/(len(ids)-1),
                fidelity_gate=fidelity_gate,retrieval_gate=retrieval_gate,composition_gate=min(gains.values())>=.05,gains=gains,stages=rows)
    assert result["parameters"]==3230208
    result["language"]={k:report(models["sparse_trigram"],teacher,data.chars,train,t) for k,t in slices.items()}
    result["language_gate"]=all(result["language"][k]["language_gate"] for k in slices)
    if result["language_gate"]:result["reference_tail"]=report(models["sparse_trigram"],teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
