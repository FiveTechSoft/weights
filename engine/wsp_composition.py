"""Data-free composition of three operations in one ordinary causal Transformer.

Tokens 0..15 are symbols; 16 COPY, 17 record separator, 18 LOOKUP, 19 CHAIN.
Input: (key,value,17)*, query, opcode, or arbitrary symbols,16 for COPY.
CHAIN computes f(f(query)) within a single forward pass, without extra tokens.
"""
import math
import numpy as np
import torch
from engine import wsp_arch,wsp_synth

ARCH=dict(vocab=20,d=192,heads=3,layers=4,ff=256,block=64,bias=False,dropout=0.0)

def compile_composition(arch,seed=1337,ablate=None):
    if ablate not in (None,"first_lookup","second_lookup","routing"):
        raise ValueError("invalid ablation")
    d,h,v=192,64,20
    b=wsp_synth._basis(d,seed)
    ballast=b[:,0]; out=b[:,1:21]; pos=b[:,21:53]
    prev,prev2,r1,r2=[b[:,53+j*v:53+(j+1)*v] for j in range(4)]
    freq=2*np.pi*np.arange(1,17)/64
    t=np.arange(64)[:,None]
    positions=np.concatenate([np.cos(t*freq),np.sin(t*freq)],axis=1)/4
    scale=math.sqrt(d)/10000
    with wsp_synth.forbid_gradients():
        m=wsp_arch.load_arch_file(arch,overrides=ARCH,seed=seed).eval()
        p=dict(m.named_parameters())
        def put(name,x):
            p[name].copy_(torch.as_tensor(x,dtype=p[name].dtype))
        for name,x in p.items():
            x.zero_()
            if '.ln_' in name and name.endswith('.weight'):
                x.fill_(1)
        put('transformer.wte.weight',out.T)
        put('transformer.wpe.weight',positions@pos.T+10000*ballast)
        put('transformer.ln_f.weight',np.full(d,1/scale))
        def attention(layer,qkv,proj):
            put(f'transformer.h.{layer}.attn.c_attn.weight',qkv)
            put(f'transformer.h.{layer}.attn.c_proj.weight',proj)
        qkv=np.zeros((3*d,d)); proj=np.zeros((d,d))
        for j,(lag,dest) in enumerate(((1,prev),(2,prev2))):
            lo=j*h
            qkv[lo:lo+32]=100*pos.T/scale
            qkv[d+lo:d+lo+32]=100*wsp_synth._shift_matrix(freq,lag)@pos.T/scale
            qkv[2*d+lo:2*d+lo+v]=out.T/scale
            proj[:,lo:lo+v]=dest
        attention(0,qkv,proj)
        # Independent residual registers preserve input keys and values.
        for layer,query,dest,label in ((1,prev,r1,"first_lookup"),(2,r1,r2,"second_lookup")):
            qkv.fill(0);proj.fill(0)
            if ablate==label:
                continue
            gain=math.sqrt(40*math.sqrt(h))/scale
            qkv[:v]=gain*query.T
            qkv[d:d+v]=gain*prev2.T
            qkv[v]=gain*ballast/10000
            qkv[d+v]=gain*.6*out[:,17]
            qkv[2*d:2*d+v]=prev.T/scale
            proj[:,:v]=dest
            attention(layer,qkv,proj)
        # GELU conjunction of opcode and register symbol. Both inputs must
        # equal one: gain*(1+1-1.5)=10, otherwise at most -10.
        fc=np.zeros((256,d));down=np.zeros((d,256))
        for j,(opcode,register) in enumerate(((16,prev),(18,r1),(19,r2))):
            for token in range(16):
                row=j*16+token
                fc[row]=20*(register[:,token]+out[:,opcode]-1.5*ballast/10000)/scale
                down[:,row]=1.2*out[:,token]
        if ablate!="routing":
            put('transformer.h.3.mlp.c_fc.weight',fc)
            put('transformer.h.3.mlp.c_proj.weight',down)
    return m
