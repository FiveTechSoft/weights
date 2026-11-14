import time,json,torch,pyarrow.parquet as pq,re
from tokenizers import Tokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from engine import stream_score as ss
from engine.weight_diff import RangeSafetensors
from engine.stream_guard import ByteBudget, MemoryGuard
cfgd=json.load(open('/tmp/probe_cache/config.json'))['text_config']
cfgd['num_hidden_layers']=2; cfgd['layer_types']=cfgd['layer_types'][2:4]
cfg=Qwen3_5TextConfig(**cfgd); cfg._attn_implementation="eager"
tb=Tokenizer.from_file('/tmp/dsets/tok_base.json')
def pick(path,fn):
    sel=[];tot=0
    for r in pq.read_table(path).to_pylist()[::2]:
        x=tb.encode(fn(r)).ids
        if tot+len(x)>4000: break
        sel.append(x);tot+=len(x)
    return sel
sets={'he':pick('/tmp/dsets/he.parquet',lambda r:r['prompt']+r['canonical_solution']),
 'gsm':pick('/tmp/dsets/gsm.parquet',lambda r:r['question']+'\n'+r['answer']),
 'mbpp':pick('/tmp/dsets/mbpp.parquet',lambda r:r['text']+'\n'+r['code'])}
ids=[x for s in sets.values() for x in s]
print({k:sum(map(len,v)) for k,v in sets.items()},flush=True)
budget=ByteBudget(2*10**9,min_rate_bytes_s=None)
def fetch_local(f,s,e):
    with open(f,'rb') as fh: fh.seek(s); return fh.read(e-s+1)
fetch=budget.wrap(fetch_local)
rd={L:RangeSafetensors([f'/tmp/probe_cache/base_layer{L+2}.safetensors'],fetch) for L in range(2)}
mem=MemoryGuard(1750)
import sys
g=torch.Generator().manual_seed(0)
E=torch.randn(250000,64,generator=g);proj=torch.randn(64,4096,generator=g)*0.5
embed=lambda i:E[i]@proj
head=torch.randn(4096,4096,generator=g)*0.02
def load(L,layer):
    print("layer",L,"peak so far",round(mem.peak),flush=True)
    budget.check();mem.check()
    params=dict(layer.named_parameters());params.update(dict(layer.named_buffers()))
    r=rd[L]
    for k in r.keys():
        params[k[len(f'model.layers.{L+2}.'):]].data.copy_(r.get(k).float())
def head_chunks():
    print("head peak so far",round(mem.peak),flush=True)
    mem.check()
    for s in range(0,8192,4096): yield s,head[:min(4096,248320-s)]
print('anon before',round(mem._read()),flush=True)
t=time.time()
import tempfile;sp=tempfile.mkdtemp(dir="/tmp");out=ss.stream_nll(cfg,ids,embed,load,torch.ones(4096),head_chunks,[1]*len(ids),spill_dir=sp)
dt=time.time()-t
print('3 datasets',len(ids),'prompts, 2 layers + head: %ds'%dt,'peak anon MB',round(mem.peak),'bytes counted',budget.bytes,'calls',budget.calls)
import os;print('layer files bytes',sum(os.path.getsize(f'/tmp/probe_cache/base_layer{L}.safetensors') for L in (2,3)))
mem.close()
