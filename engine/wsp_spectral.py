"""Spectral distributional features and analytic layerwise residual prediction.

No table entries are compiled as individual neurons. All statistics and fits use
the supplied training split only. Attention uses fixed relative-position
retrieval and data-derived character similarity, then dense GELU features.
"""
import math
import numpy as np
import torch
from engine import wsp_arch, wsp_synth

ARCH = dict(d=256, heads=4, layers=4, ff=1024, block=128, bias=False, dropout=0.0)

def spectral_codes(ids, vocab, rank=32):
    counts = np.full((vocab, 8*vocab), 0.5)
    for j, lag in enumerate((-8,-4,-2,-1,1,2,4,8)):
        a, b = (ids[:-lag], ids[lag:]) if lag > 0 else (ids[-lag:], ids[:lag])
        np.add.at(counts, (a, j*vocab+b), 1)
    pmi = np.log(counts * counts.sum() / (counts.sum(0)[None,:]*counts.sum(1)[:,None]))
    u, s, _ = np.linalg.svd(np.maximum(pmi,0), full_matrices=False)
    z = u[:,:rank] * np.sqrt(s[:rank])
    return z / np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)

def compile_spectral(arch, ids, chars, seed=1337, spectral=True):
    ids = np.asarray(ids,dtype=np.int64)
    v = len(chars)
    d, h, block, ff = 256, 64, 128, 1024
    if v+1+32+4*32 > d-1:
        raise ValueError("alphabet too large for V6 layout")
    # Separate development slice, never use the external held-out corpus.
    cut = int(len(ids)*0.9)
    train, dev = ids[:cut], ids[cut:]
    rng = np.random.default_rng(seed)
    code = spectral_codes(train,v) if spectral else rng.normal(size=(v,32))
    code /= np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-8)
    basis = wsp_synth._basis(d,seed)
    ballast, out = basis[:,0], basis[:,1:1+v]
    pos = basis[:,1+v:1+v+32]
    slots = [basis[:,1+v+32+j*32:1+v+32+(j+1)*32] for j in range(4)]
    positions,freq = wsp_synth._position_codes(block,32)
    scale = math.sqrt(d)/wsp_synth.BALLAST
    with wsp_synth.forbid_gradients():
        model = wsp_arch.load_arch_file(arch, overrides=dict(ARCH,vocab=v),seed=seed).eval()
        params = dict(model.named_parameters())
        def put(name,x):
            params[name].copy_(torch.as_tensor(x,dtype=params[name].dtype))
        for name,p in params.items():
            p.zero_()
            if '.ln_' in name and name.endswith('.weight'):
                p.fill_(1)
        put('transformer.wte.weight',out.T)
        put('transformer.wpe.weight',positions@pos.T+wsp_synth.BALLAST*ballast)
        put('transformer.ln_f.weight',np.full(d,1/scale))
        qkv, proj = np.zeros((3*d,d)), np.zeros((d,d))
        for j,lag in enumerate((0,1,2,3)):
            lo=j*h
            qkv[lo:lo+32]=40*pos.T/scale
            qkv[d+lo:d+lo+32]=40*wsp_synth._shift_matrix(freq,lag)@pos.T/scale
            qkv[2*d+lo:2*d+lo+32]=code.T@out.T/scale
            proj[:,lo:lo+32]=slots[j]
        put('transformer.h.0.attn.c_attn.weight',qkv)
        put('transformer.h.0.attn.c_proj.weight',proj)
        # Layer 1 adds a distributed associative memory, in an existing slot.
        qkv.fill(0); proj.fill(0)
        gain=math.sqrt(8*math.sqrt(h))/scale
        qkv[:32]=gain*slots[0].T
        qkv[d:d+32]=gain*slots[1].T
        qkv[2*d:2*d+32]=slots[0].T/scale
        proj[:,:32]=0.3*slots[3]
        put('transformer.h.1.attn.c_attn.weight',qkv)
        put('transformer.h.1.attn.c_proj.weight',proj)
        # Fit dense nonlinear features layer by layer using forward-only
        # residuals and a regularized linear system. All future blocks start zero.
        def windows(data,n):
            starts=rng.integers(0,len(data)-block,size=n)
            return torch.tensor(np.stack([data[i:i+block+1] for i in starts]))
        train_w,dev_w=windows(train,512),windows(dev,64)
        eye=torch.eye(v,dtype=torch.float64)
        directions=np.concatenate(slots,axis=1)
        fits=[]
        for layer in range(4):
            prefix=f'transformer.h.{layer}.mlp'
            rows=rng.normal(0,1/math.sqrt(128),(ff,128))@directions.T/scale
            rows+=np.outer(rng.normal(0,0.5,ff),ballast)/(scale*wsp_synth.BALLAST)
            put(prefix+'.c_fc.weight',rows)
            fc=params[prefix+'.c_fc.weight']
            def features(w):
                x,y=w[:,:-1],w[:,1:]
                phi=torch.nn.functional.gelu(model.input_of(prefix+'.c_fc',x)@fc.T).flatten(0,1)
                logits=model(x,y.contiguous())[0].flatten(0,1)
                return phi,logits,y.flatten()
            gram=torch.zeros(ff,ff,dtype=torch.float64)
            cross=torch.zeros(ff,v,dtype=torch.float64)
            for w in train_w.split(16):
                phi,logits,y=features(w)
                phi=phi.double()
                p=logits.double().softmax(-1)
                curvature=(1-p.square().sum(-1)).clamp_min(.05)
                gram+=phi.T@(phi*curvature[:,None])
                cross+=phi.T@(eye[y]-p)
            sol=torch.linalg.solve(gram+10*torch.eye(ff,dtype=torch.float64),cross).float()
            sol-=sol.mean(1,keepdim=True)
            # Evaluate actual network logits for each candidate step, not an
            # approximation through subsequent nonlinear layers.
            def evaluate(step):
                put(prefix+'.c_proj.weight',out@(step*sol.numpy()).T)
                losses=[model(w[:,:-1],w[:,1:].contiguous())[1].item() for w in dev_w.split(16)]
                return float(np.mean(losses))
            trials={a:evaluate(a) for a in (0.,.25,.5,1.,2.,4.,8.,16.)}
            step=min(trials,key=trials.get)
            evaluate(step)
            fits.append(dict(layer=layer,step=step,dev_before=trials[0.],dev_after=trials[step]))
        model.synthesis_info=dict(provenance='closed_form_data_fit',method='spectral_residual',backward_calls=0,
                                  spectral=spectral,rank=32,fit_tokens=len(train),dev_tokens=len(dev),fits=fits)
    return model
