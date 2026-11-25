"""N7: autonomous selection inside a fixed conditional relation family."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import save_file
from natural_law import Teacher,report,V,D,ANCHOR
from natural_n6b import compile_model,audit,score
from train_reference import corpus,reload,HASH
from cycle_law import snapshot

def encode(p,C):
    l=p.log();l-=l.mean(-1,keepdim=True)
    z=torch.zeros(*l.shape[:-1],D,dtype=torch.double)
    z[...,:V]=l
    a=((C-l.square().sum(-1))/2).sqrt()
    z[...,245]=a;z[...,246]=-a
    return z

def statistics(ids,big,tri):
    pair=ids[:-2]*V+ids[1:-1]
    y=ids[2:]
    gain=(tri.log()-big.log()[None,:,:]).reshape(-1,V)[pair,y]
    count=torch.bincount(pair,minlength=V*V)
    sums=torch.zeros(V*V,dtype=torch.double)
    sums.scatter_add_(0,pair,gain)
    return count,sums

def select(discovery,acceptance,big,tri):
    dc,dg=statistics(discovery,big,tri)
    ac,ag=statistics(acceptance,big,tri)
    order=dg.argsort(descending=True,stable=True)
    chosen=[];records=[]
    for idx in order.tolist():
        eligible=int(dc[idx])>=8 and float(dg[idx])>0
        accepted=eligible and int(ac[idx])>=8 and float(ag[idx])>0
        take=accepted and len(chosen)<512
        if take:chosen.append(idx)
        records.append(dict(pair=idx,discovery_count=int(dc[idx]),discovery_gain=float(dg[idx]),
            acceptance_count=int(ac[idx]),acceptance_gain=float(ag[idx]),eligible=eligible,
            accepted_by_audit=accepted,selected=take))
    return torch.tensor(chosen,dtype=torch.long),records

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--corpus",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    text,data=corpus(a.corpus);train=text[:int(len(text)*.9)];cut=int(len(train)*.8)
    teacher=Teacher(train[:cut],data.chars);ids=torch.tensor(teacher.encode(train[:cut]))
    rest=torch.tensor(teacher.encode(train[cut:]));mid=len(rest)//2
    big=torch.stack([torch.from_numpy(teacher.row((i,))) for i in range(V)]).double()
    tri=torch.stack([torch.stack([torch.from_numpy(teacher.row((j,i))) for i in range(V)]) for j in range(V)]).double()
    chosen,records=select(rest[:mid],rest[mid:],big,tri)
    counts=torch.bincount(ids[:-1]*V+ids[1:],minlength=V*V)
    freq=counts.argsort(descending=True,stable=True)
    # Common norm independent of candidate selection.
    L=tri.log();L-=L.mean(-1,keepdim=True)
    l=big.log();l-=l.mean(-1,keepdim=True)
    C=2*ANCHOR**2+max(float(L.square().sum(-1).max()),float(l.square().sum(-1).max()))
    B=encode(big,C)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/"selection.json").write_text(json.dumps(records,indent=2))
    slices={"validation":text[-16000:-8000],"confirmation":text[-24000:-16000]}
    rows={};models={}
    budgets=sorted(set(min(n,len(chosen)) for n in (64,128,256,512))-{0})
    print("SELECTION",json.dumps(dict(candidates=len(records),eligible=sum(r["eligible"] for r in records),
        audit_accepted=sum(r["accepted_by_audit"] for r in records),selected=len(chosen),budgets=budgets)),flush=True)
    def stage(name,selected,full_audit):
        sparse=big[None,:,:].expand(V,V,V).clone()
        sparse.reshape(-1,V)[selected]=tri.reshape(-1,V)[selected]
        T=encode(sparse,C);compose=len(selected)>0
        m,basis=compile_model(teacher,ids,B,T,selected,C,compose)
        path=out/f"{name}.safetensors"
        save_file(snapshot(m),str(path),metadata={"provenance":"count_utility_compilation","optimizer_steps":"0","corpus_sha256":HASH,"vocab_chars":data.chars})
        loaded=reload(path);probe=ids[:128].unsqueeze(0)
        exact=torch.equal(m(probe)[0],loaded(probe)[0]);assert exact
        row=dict(budget=len(selected),reload_exact=exact,nonzero_parameters=sum(int(p.count_nonzero()) for p in loaded.parameters()),
            count_coverage=float(counts[selected].sum())/int(counts.sum()))
        for k,t in slices.items():row[k]=score(loaded,big,tri,sparse,compose,data.chars,t)
        if full_audit:row["audit"]=audit(loaded,basis,big,sparse,compose,C)
        rows[name]=row;models[name]=loaded
        print("STAGE",json.dumps(dict(mode=name,**row)),flush=True)
        (out/"stages.json").write_text(json.dumps(rows,indent=2))
    stage("bigram",torch.empty(0,dtype=torch.long),False)
    for n in budgets:
        stage(f"frequency_{n}",freq[:n],n==budgets[-1])
        stage(f"utility_{n}",chosen[:n],n==budgets[-1])
    if not budgets:
        result=dict(selection_gate=False,reason="no eligible accepted relations",stages=rows)
    else:
        n=budgets[-1];u=rows[f"utility_{n}"];f=rows[f"frequency_{n}"];base=rows["bigram"]
        gains={k:dict(vs_frequency=f[k]["native"]-u[k]["native"],vs_bigram=base[k]["native"]-u[k]["native"]) for k in slices}
        increments={}
        for k in slices:
            values=[base[k]["native"]]+[rows[f"utility_{b}"][k]["native"] for b in budgets]
            increments[k]=[a-b for a,b in zip(values,values[1:])]
        def audit_ok(row):
            a=row["audit"]
            return max(a["max_probability_error"],a["first_position_error"],a["relative_norm_error"],
                a["causal_logit_error"],a["prefix_probability_error"],a["retrieval_error"])<=1e-4 and a["min_attention_mass"]>=.99999 and a["wrong_gate_count"]==0 and a["pre_layernorm_mean_max"]<=1e-5 and a["gate_amplitude_error"]<=1e-3
        fidelity=all(audit_ok(r) for r in (u,f)) and all(r["reload_exact"] and all(abs(r[k]["native"]-r[k]["reference"])<=1e-4 for k in slices) for r in rows.values())
        result=dict(parameters=sum(p.numel() for p in models["bigram"].parameters()),optimizer_steps=0,backward_calls=0,donor_weights=False,
            selected=n,budgets=budgets,selection_gate=all(v["vs_frequency"]>=.01 and v["vs_bigram"]>=.05 for v in gains.values()),
            fidelity_gate=fidelity,growth_gate=all(min(v)>=-1e-4 for v in increments.values()),
            gains=gains,increments=increments,stages=rows,
            language={k:report(models[f"utility_{n}"],teacher,data.chars,train,t) for k,t in slices.items()})
        result["language_gate"]=all(result["language"][k]["language_gate"] for k in slices)
        if result["language_gate"]:result["reference_tail"]=report(models[f"utility_{n}"],teacher,data.chars,train,text[-8000:])
    (out/"report.json").write_text(json.dumps(result,indent=2))
    print("SUMMARY",json.dumps(result),flush=True)
if __name__=="__main__":main()
