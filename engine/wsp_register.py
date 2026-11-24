"""Compile straight-line register programs into ordinary nanoGPT weights.

Grammar: dest = COPY source | dest = LOOKUP source |
         dest = ADD left right | dest = SUB left right |
         RETURN source
Registers x,y are inputs; each assignment defines a fresh register.
Values are 0..15; ADD/SUB are modulo 16. Input format:
(key,value,17)*, x, y, 16. Unique record keys; queried keys must exist.
No examples, gradients or donor weights. Each operation gets a residual
register and one block. Arithmetic uses factored scalar encoding followed
by threshold differences, not a table of input pairs.
"""
import math
import re
import numpy as np
import torch
from engine import wsp_arch,wsp_synth

VOCAB=18
MAX_OPS=8
BLOCK=64
FF=256

def parse_program(source):
    defined={"x","y"}
    ops=[]
    lines=[line.split("#",1)[0].strip() for line in source.splitlines()]
    lines=[line for line in lines if line]
    if not lines or len(lines)>MAX_OPS+1:
        raise ValueError("expected 0..8 assignments and RETURN")
    for line in lines[:-1]:
        m=re.fullmatch(r"([a-z][a-z0-9_]*)\s*=\s*(COPY|LOOKUP|ADD|SUB)\s+([a-z][a-z0-9_]*)(?:\s+([a-z][a-z0-9_]*))?",line)
        if not m:
            raise ValueError("invalid assignment: "+line)
        dst,op,a,b=m.groups()
        if dst in defined or a not in defined or (op in ("ADD","SUB") and b not in defined) or (op in ("COPY","LOOKUP") and b is not None):
            raise ValueError("invalid register or arity: "+line)
        defined.add(dst)
        ops.append((dst,op,a,b))
    m=re.fullmatch(r"RETURN\s+([a-z][a-z0-9_]*)",lines[-1])
    if not m or m[1] not in defined:
        raise ValueError("invalid return")
    return ops,m[1]

def config(source):
    ops,_=parse_program(source)
    # Register banks are 18-dimensional, plus 32 positional dimensions.
    d=64*math.ceil((1+VOCAB+32+VOCAB*(2+len(ops))+1)/64)
    return dict(vocab=VOCAB,d=d,heads=d//64,layers=len(ops)+2,ff=FF,block=BLOCK,bias=False,dropout=0.0)

def compile_program(arch,source,seed=1337,ablate=None):
    ops,ret=parse_program(source)
    cfg=config(source)
    d,v,h=cfg["d"],VOCAB,64
    if ablate is not None and ablate not in [a[0] for a in ops]:
        raise ValueError("unknown ablation register")
    basis=wsp_synth._basis(d,seed)
    ballast=basis[:,0]; out=basis[:,1:19];pos=basis[:,19:51]
    regs={name:basis[:,51+j*v:51+(j+1)*v] for j,name in enumerate(["x","y"]+[a[0] for a in ops])}
    freq=2*np.pi*np.arange(1,17)/BLOCK
    t=np.arange(BLOCK)[:,None]
    positions=np.concatenate([np.cos(t*freq),np.sin(t*freq)],axis=1)/4
    scale=math.sqrt(d)/10000
    with wsp_synth.forbid_gradients():
        m=wsp_arch.load_arch_file(arch,overrides=cfg,seed=seed).eval()
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
        def mlp(layer,fc,down):
            put(f'transformer.h.{layer}.mlp.c_fc.weight',fc)
            put(f'transformer.h.{layer}.mlp.c_proj.weight',down)
        qkv=np.zeros((3*d,d));proj=np.zeros((d,d))
        for j,(lag,name) in enumerate(((1,"y"),(2,"x"))):
            lo=j*h
            qkv[lo:lo+32]=100*pos.T/scale
            qkv[d+lo:d+lo+32]=100*wsp_synth._shift_matrix(freq,lag)@pos.T/scale
            qkv[2*d+lo:2*d+lo+v]=out.T/scale
            proj[:,lo:lo+v]=regs[name]
        attention(0,qkv,proj)
        for layer,(dst,op,a,b) in enumerate(ops,1):
            if dst==ablate:
                continue
            if op=="LOOKUP":
                qkv.fill(0);proj.fill(0)
                gain=math.sqrt(40*math.sqrt(h))/scale
                qkv[:v]=gain*regs[a].T
                qkv[d:d+v]=gain*regs["x"].T
                qkv[v]=gain*ballast/10000
                qkv[d+v]=gain*.6*out[:,17]
                qkv[2*d:2*d+v]=regs["y"].T/scale
                proj[:,:v]=regs[dst]
                attention(layer,qkv,proj)
            else:
                fc=np.zeros((FF,d));down=np.zeros((d,FF))
                if op=="COPY":
                    fc[:v]=regs[a].T/scale
                    fc[v:2*v]=-regs[a].T/scale
                    down[:,:v]=regs[dst]
                    down[:,v:2*v]=-regs[dst]
                else:
                    # Integer sum or difference shifted into [0,30].
                    digits=np.arange(16,dtype=float)
                    direction=regs[a][:,:16]@digits
                    direction+= (1 if op=="ADD" else -1)*(regs[b][:,:16]@digits)
                    offset=0 if op=="ADD" else 15
                    # GELU approximates ReLU far from zero. A triangular
                    # hat around each integer s is 0 except at s where it
                    # is 1. Modular wrap is in the output projection.
                    for s in range(31):
                        result=(s-offset)%16
                        for j,(shift,coef) in enumerate(((-1,1),(0,-2),(1,1))):
                            # Half-integer kinks avoid evaluating GELU near zero:
                            # use two half-width ramps for a unit plateau.
                            row=4*s
                        for j,(shift,coef) in enumerate(((-1.5,1),(-.5,-1),(.5,-1),(1.5,1))):
                            # width 1.5 would overlap adjacent integers;
                            # instead use kinks at -0.75,-0.25,+0.25,+0.75.
                            shift*=.5
                            row=4*s+j
                            fc[row]=40*(direction+(offset-s-shift)*ballast/10000)/scale
                            down[:,row]=(coef/20)*regs[dst][:,result]
                mlp(layer,fc,down)
        fc=np.zeros((FF,d));down=np.zeros((d,FF))
        fc[:v]=regs[ret].T/scale;fc[v:2*v]=-regs[ret].T/scale
        down[:,:v]=12*out;down[:,v:2*v]=-12*out
        mlp(len(ops)+1,fc,down)
    return m
