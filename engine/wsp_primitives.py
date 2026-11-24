"""Donor-free, data-free synthesis of two algorithmic primitives.

Vocabulary: symbols 0..15; query marker 16; separator 17.
COPY: arbitrary symbols followed by marker -> previous symbol.
ASSOC: (key,value,separator)*, query_key, marker -> bound value.
No task-specific inference code; ordinary causal attention and tied readout.
"""
import math
import numpy as np
import torch
from engine import wsp_arch,wsp_synth

ARCH=dict(vocab=18,d=128,heads=2,layers=2,ff=256,block=64,bias=False,dropout=0.0)

def compile_primitive(arch,skill,seed=1337,ablate=False):
    if skill not in ("copy","associate"):
        raise ValueError("unknown primitive")
    d,h,v=128,64,18
    b=wsp_synth._basis(d,seed)
    ballast=b[:,0]
    out=b[:,1:1+v]
    pos=b[:,19:51]
    prev=b[:,51:69]
    prev2=b[:,69:87]
    # Use integer Fourier frequencies: exact shift and no repeated position
    # code inside a 64-token block.
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
        qkv=np.zeros((3*d,d)); proj=np.zeros((d,d))
        for j,(lag,dest) in enumerate(((1,prev),(2,prev2))):
            lo=j*h
            # High gain separates adjacent positions even at small distances.
            qkv[lo:lo+32]=100*pos.T/scale
            qkv[d+lo:d+lo+32]=100*wsp_synth._shift_matrix(freq,lag)@pos.T/scale
            qkv[2*d+lo:2*d+lo+v]=out.T/scale
            proj[:,lo:lo+v]=dest
        put('transformer.h.0.attn.c_attn.weight',qkv)
        put('transformer.h.0.attn.c_proj.weight',proj)
        qkv.fill(0);proj.fill(0)
        if skill=="copy":
            # Copy previous symbol through a pointwise linear GELU pair.
            fc=np.zeros((256,d)); down=np.zeros((d,256))
            fc[:v]=prev.T/scale;fc[v:2*v]=-prev.T/scale
            down[:,:v]=12*out;down[:,v:2*v]=-12*out
            if not ablate:
                put('transformer.h.1.mlp.c_fc.weight',fc)
                put('transformer.h.1.mlp.c_proj.weight',down)
        elif not ablate:
            # Query previous token (key). At separator positions, the key
            # lies two positions back and the value one position back.
            # A separator bonus suppresses non-record positions.
            g=math.sqrt(40*math.sqrt(h))/scale
            qkv[:v]=g*prev.T
            qkv[d:d+v]=g*prev2.T
            qkv[v]=g*ballast/10000
            qkv[d+v]=g*0.6*out[:,17]
            qkv[2*d:2*d+v]=prev.T/scale
            proj[:,:v]=12*out
            put('transformer.h.1.attn.c_attn.weight',qkv)
            put('transformer.h.1.attn.c_proj.weight',proj)
    return m
