"""NanoGPT adapter for causal locus discovery on explicit token examples."""
import fnmatch
import hashlib
import json
from pathlib import Path
import random

import torch
from safetensors import safe_open
from engine.nanogpt_runtime import GPTConfig, NanoGPT
from engine.wsp_locus import Candidate, probe_columns


def probe(model_path, data_path, pattern, *, axis="column", n_probe=64,
          dose=.5, seed=1337, top=5, max_modified_weights=4096):
    model_path, data_path = Path(model_path), Path(data_path)
    with safe_open(str(model_path), framework="pt") as f:
        cfg = json.loads(f.metadata()["architecture"])
    rt = NanoGPT(GPTConfig(vocab_size=cfg["vocab"], n_embd=cfg["d"],
        n_head=cfg["heads"], n_layer=cfg["layers"], block_size=cfg["block"],
        d_ff=cfg["ff"], bias=cfg.get("bias",False)))
    rt.load_checkpoint(model_path)
    rt.model.eval()
    raw = data_path.read_bytes()
    data = json.loads(raw)
    seen = set()
    for split in ("discovery","heldout","preservation"):
        rows = data[split]
        if not isinstance(rows,list) or not rows:
            raise ValueError("empty evaluation split")
        for row in rows:
            ids, target = row["ids"], row["target"]
            if not isinstance(ids,list) or not 0<len(ids)<=cfg["block"]:
                raise ValueError("invalid sequence length")
            if any(type(i) is not int or not 0<=i<cfg["vocab"] for i in [*ids,target]):
                raise ValueError("invalid token id")
            key = tuple(ids)
            if key in seen:
                raise ValueError("duplicate prompt across evaluation data")
            seen.add(key)
    if axis not in ("column","row") or not 1<=n_probe<=256:
        raise ValueError("invalid axis or candidate count")
    tensors = {}
    for name,w in rt.model.named_parameters():
        # Restrict first backend to MLP projections; never tied embeddings.
        if w.ndim==2 and ".mlp." in name and fnmatch.fnmatchcase(name,pattern):
            view = w if axis=="column" else w.T
            if view.shape[0]<=max_modified_weights:
                tensors[name]=view
    pool = [Candidate(name,i) for name in sorted(tensors)
            for i in range(tensors[name].shape[1])]
    if not pool:
        raise ValueError("no eligible MLP tensors")
    candidates = random.Random(seed).sample(pool,min(n_probe,len(pool)))
    @torch.no_grad()
    def evaluate(split):
        correct=0
        for row in data[split]:
            logits,_=rt.forward(torch.tensor([row["ids"]],dtype=torch.long))
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite logits")
            correct+=int(logits[0,-1].argmax().item()==row["target"])
        return correct/len(data[split])
    report=probe_columns(tensors,candidates,evaluate,dose=dose,seed=seed,
                         top=top,max_modified_weights=max_modified_weights)
    h=hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    report.update(axis=axis,metric="task_acc",model_sha256=h.hexdigest(),
                  data_sha256=hashlib.sha256(raw).hexdigest(),
                  split_sizes={s:len(data[s]) for s in ("discovery","heldout","preservation")},
                  tensor_pattern=pattern,placebo=True)
    return report
