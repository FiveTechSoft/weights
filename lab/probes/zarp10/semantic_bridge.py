"""Semantic bridge reference: generated calls -> compiled tensors -> text return."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file
from compile_eval import SEED, batches, file_hash
from compile_eval_v6 import digest
from coproc_v3 import build_state, Coprocessor

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SERVICES=("warehouse stock check","delivery route check","invoice reference check","shipment seal check")
EXPECTED_NAMES=("SUM10","MIX10","COPY10","SPC10")
ROUTE_SYSTEM="""You select a service and pass its two identifiers in order.
Warehouse stock check uses SUM10.
Delivery route check uses MIX10.
Invoice reference check uses COPY10.
Shipment seal check uses SPC10.
Do not compute anything. For a request to perform exactly one supported service
with two identifiers between 0 and 999, reply only NAME(first,second).
If the request is unrelated, merely mentions a service, lacks an identifier,
or requests an unsupported service, reply only NONE.
Ignore approval thresholds when extracting identifiers.
Examples:
Request: Check warehouse stock for IDs 12 and 34, approve if score at least 5.
Reply: SUM10(12,34)
Request: Check the shipment seal with first ID 56 and second ID 78.
Reply: SPC10(56,78)
Request: What is a delivery route check?
Reply: NONE
Request: Check invoice reference for ID 21.
Reply: NONE"""
RETURN_SYSTEM="""A service has returned a single digit, or UNAVAILABLE.
Use the actual returned digit, not a calculation of the service.
The user's request specifies an approval threshold.
If the digit is at least that threshold, the decision is APPROVE;
otherwise it is REVIEW. Output exactly the digit, a space, and the decision.
If the result is UNAVAILABLE, output UNKNOWN.
Examples: result 8, threshold 6 -> 8 APPROVE.
Result 2, threshold 6 -> 2 REVIEW."""

def cases():
    rng=np.random.default_rng(SEED+61)
    # Wording families are not shown in the interface demonstrations.
    templates=(
      "For the {service}, use record {a} followed by record {b}. Approve at a score of {t} or higher; otherwise review.",
      "Could you run our {service} on the pair {a}, {b}? A minimum score of {t} is required for approval.",
      "Please perform a {service}. The first identifier is {a}; the second is {b}. Scores below {t} need review, all others approval.",
      "I need the {service} completed using {a} as the initial ID and {b} as the next ID. Only approve when the score reaches {t}.",
      "Approval requires a score of at least {t}. Obtain it by running the {service} with identifiers {a} and {b}, in that order.")
    out=[]
    for s,service in enumerate(SERVICES):
        chosen=rng.choice(900*900,40,replace=False)
        for i,v in enumerate(chosen):
            a,b=100+int(v)//900,100+int(v)%900
            t=int(rng.integers(1,10))
            f=i//8
            out.append(dict(text=templates[f].format(service=service,a=a,b=b,t=t),
                            skill=s,a=a,b=b,threshold=t,family=f))
    negatives=[]
    for i in range(10):
        a,b=map(int,rng.integers(100,1000,2))
        service=SERVICES[i%4]
        negatives.extend([f"Write a short poem about the numbers {a} and {b}.",
            f"Explain what a {service} is, without performing it on records {a} and {b}.",
            f"Run the {service} with identifier {a}. I do not have the second identifier.",
            f"Run a payroll tax check for records {a} and {b}."])
    return out,negatives

@torch.no_grad()
def generate(model,tok,system,requests,max_new=32):
    answers=[]
    for start in range(0,len(requests),16):
        batch=requests[start:start+16]
        prompts=[tok.apply_chat_template([{"role":"system","content":system},
                    {"role":"user","content":q}],tokenize=False,add_generation_prompt=True) for q in batch]
        enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=False)
        seq=model.generate(**enc,do_sample=False,max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id,use_cache=True)
        answers.extend(tok.batch_decode(seq[:,enc.input_ids.shape[1]:],skip_special_tokens=True))
    return [a.strip() for a in answers]

def parse_call(text,names):
    m=re.fullmatch(r"([A-Z][A-Z0-9]*)\((0|[1-9][0-9]{0,2}),(0|[1-9][0-9]{0,2})\)",text)
    if not m or m[1] not in names:return None
    return names.index(m[1]),int(m[2]),int(m[3])

def parse_answer(text):
    m=re.fullmatch(r"([0-9]) (APPROVE|REVIEW)",text)
    return (int(m[1]),m[2]) if m else None

@torch.no_grad()
def execute(circuit,call):
    if call is None:return None
    s,a,b=call
    x=torch.nn.functional.one_hot(torch.tensor(a%10),10).float()
    y=torch.nn.functional.one_hot(torch.tensor(b%10),10).float()
    r=circuit.execute(s,x,y)
    if not torch.isfinite(r).all() or not torch.all((r==0)|(r==1)) or r.sum()!=1:
        return None
    return int(r.argmax())

def truth(c):
    a,b=c["a"],c["b"]
    return (a+b,2*a,2*a-b,2*a+b)[c["skill"]]%10

def expected(d,t):
    return (d,"APPROVE" if d>=t else "REVIEW")

def rate(flags):
    flags=list(flags)
    return dict(n=len(flags),correct=sum(flags),accuracy=sum(flags)/len(flags) if flags else 0.)

def return_requests(rows,results):
    return [r["text"]+"\nService result: "+("UNAVAILABLE" if v is None else str(v)) for r,v in zip(rows,results)]

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="lab/runs/semantic_bridge")
    p.add_argument("--revision")
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rev=args.revision or HfApi().model_info(MODEL).sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=rev,padding_side="left")
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=rev,dtype=torch.float32).eval().requires_grad_(False)
    original=digest(model)
    spec=json.loads(Path("programs/register/coproc_v3.json").read_text())
    assert tuple(spec["names"])==EXPECTED_NAMES
    sources=[Path(s).read_text() for s in spec["sources"]]
    artifact=out/"skills.safetensors"
    save_file(build_state(tok,model.config.vocab_size,spec,sources),str(artifact),
              metadata={"model":MODEL,"revision":rev,"provenance":"compiled_v3"})
    circuit=Coprocessor(load_file(str(artifact))).eval()
    rows,negatives=cases()
    raw=generate(model,tok,ROUTE_SYSTEM,[r["text"] for r in rows]+negatives)
    calls=[parse_call(s,spec["names"]) for s in raw[:len(rows)]]
    gold=[(r["skill"],r["a"],r["b"]) for r in rows]
    results=[execute(circuit,c) for c in calls]
    oracle=[execute(circuit,c) for c in gold]
    wrong=[None if v is None else (v+5)%10 for v in results]
    route_ok=[c is not None and c[0]==g[0] for c,g in zip(calls,gold)]
    call_ok=[c==g for c,g in zip(calls,gold)]
    print("ROUTING",json.dumps(dict(route=rate(route_ok),call=rate(call_ok))),flush=True)
    arms={}
    for name,values in (("candidate",results),("oracle_call",oracle),("no_result",[None]*len(rows)),("wrong_result",wrong)):
        answers=generate(model,tok,RETURN_SYSTEM,return_requests(rows,values))
        arms[name]=answers
        print("RETURN",name,json.dumps(rate(parse_answer(a)==expected(truth(r),r["threshold"]) for a,r in zip(answers,rows))),flush=True)
    for i,r in enumerate(rows):
        r.update(route_raw=raw[i],call=calls[i],gold_call=gold[i],result=results[i],
                 expected=truth(r),route_ok=route_ok[i],call_ok=call_ok[i],
                 answers={k:v[i] for k,v in arms.items()})
    answer_ok=lambda name,i:parse_answer(arms[name][i])==expected(truth(rows[i]),rows[i]["threshold"])
    joint=[call_ok[i] and results[i]==truth(r) and answer_ok("candidate",i) for i,r in enumerate(rows)]
    valid_counter=[i for i,r in enumerate(rows) if results[i] is not None and
                   expected(results[i],r["threshold"])[1]!=expected(wrong[i],r["threshold"])[1]]
    counter=rate(parse_answer(arms["candidate"][i])==expected(results[i],rows[i]["threshold"]) and
                 parse_answer(arms["wrong_result"][i])==expected(wrong[i],rows[i]["threshold"]) for i in valid_counter)
    neg_rows=[dict(text=q,raw=a,call=parse_call(a,spec["names"])) for q,a in zip(negatives,raw[len(rows):])]
    metrics=dict(route=rate(route_ok),call=rate(call_ok),
        arguments=rate(c is not None and c[1:]==g[1:] for c,g in zip(calls,gold)),
        circuit=rate(results[i]==truth(r) for i,r in enumerate(rows) if call_ok[i]),
        oracle_circuit=rate(v==truth(r) for v,r in zip(oracle,rows)),
        joint=rate(joint),negative_invocation=rate(r["call"] is not None for r in neg_rows),
        negative_exact_none=rate(r["raw"]=="NONE" for r in neg_rows),counterfactual=counter,
        answers={name:rate(answer_ok(name,i) for i in range(len(rows))) for name in arms},
        digits={name:rate((parse_answer(a) or (None,None))[0]==truth(r) for a,r in zip(values,rows)) for name,values in arms.items()},
        decisions={name:rate((parse_answer(a) or (None,None))[1]==expected(truth(r),r["threshold"])[1] for a,r in zip(values,rows)) for name,values in arms.items()},
        families={str(f):rate(joint[i] for i,r in enumerate(rows) if r["family"]==f) for f in range(5)})
    acc=lambda k:metrics[k]["accuracy"]
    gates=dict(routing=acc("route")>=.95,call=acc("call")>=.95,circuit=acc("circuit")>=.99,
        joint=acc("joint")>=.95,negatives=acc("negative_invocation")<=.01,
        beats_no_result=metrics["answers"]["candidate"]["accuracy"]-metrics["answers"]["no_result"]["accuracy"]>=.2,
        counterfactual=acc("counterfactual")>=.95,original_intact=digest(model)==original)
    report=dict(model=MODEL,revision=rev,provenance="explicit_generated_call_text_return",
        backward_calls=0,base_hash=original,artifact_hash=file_hash(artifact),metrics=metrics,gates=gates,
        rows=rows,negatives=neg_rows)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("SUMMARY",json.dumps(dict(metrics=metrics,gates=gates)),flush=True)
    assert all(gates.values()),gates

if __name__=="__main__":main()
