"""Behaviour distillation, step 2: train a small LoRA on the frozen base so it imitates a teacher's own answers
(distill_gen.py output: prompt + completion, loss on the completion and its end token only).

A 9B bf16 model (18 GB) does not fit a 12 GB GPU, and peft / bitsandbytes / accelerate are not installed, so:
  * the model sits on the CPU; each decoder layer's frozen weights go to the GPU only for its own forward (hooks);
  * the forward keeps only each layer's input (on the CPU); the backward is done by hand, last layer first: the layer
    is streamed in again, recomputed with grad from its input, backpropagated, then sent back;
  * the LoRA parameters (fp32) never leave the GPU;
  * the loss is computed in checkpointed chunks of positions (full logits would be B x T x 248k).
tests/test_distill_train.py checks on the real 0.8B that this gradient equals the plain one.

Export: PEFT layout of delta_to_lora (lora_alpha = r), converted by llama.cpp's convert_lora_to_gguf.py.

usage: python lab/probes/close_probe/distill_train.py <base_hf> <data.jsonl> <out_dir> [--rank 16] [--kinds ...]
       [--epochs 2] [--lr 2e-4] [--seqs_per_step 8] [--tokens_per_micro 6144] [--max_len 2600]
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint as ckpt

PREFIX = "model.language_model.layers"      # stored names in Qwen3.5 checkpoints (what the converter maps against)


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank, device, gen):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter((torch.randn(rank, base.in_features, generator=gen) / base.in_features ** 0.5)
                                    .to(device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=device))

    def forward(self, x):
        d = (x.to(self.A.dtype) @ self.A.T) @ self.B.T
        return self.base(x) + d.to(x.dtype)


def _layers(model):
    core = model.model
    return core.layers if hasattr(core, "layers") else core.language_model.layers


def _to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, dev) for k, v in x.items()}
    return x


_MASTER = {}   # id(tensor slot) -> the frozen CPU tensor; weights never change, so "back to the CPU" is a pointer swap


def _move_frozen(layer, dev, keep):
    """Frozen weights and buffers of layer onto dev, or back to their CPU master (no device-to-host copy)."""
    for p in layer.parameters():
        if id(p) in keep:
            continue
        cpu = _MASTER.setdefault(id(p), p.data if p.data.device.type == "cpu" else p.data.cpu())
        p.data = cpu if str(dev) == "cpu" else cpu.to(dev, non_blocking=True)
    for m in layer.modules():
        for k, b in m._buffers.items():
            if b is None:
                continue
            key = (id(m), k)
            cpu = _MASTER.setdefault(key, b if b.device.type == "cpu" else b.cpu())
            m._buffers[k] = cpu if str(dev) == "cpu" else cpu.to(dev, non_blocking=True)


def attach(model, rank, kinds=("gate_proj", "up_proj", "down_proj"), layers=None, device="cuda", seed=0):
    """Wrap the chosen MLP projections with LoRALinear (parameters on `device`). Returns {stored PEFT stem:
    LoRALinear}. The base stays frozen (and on the CPU)."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    out, keep = {}, set()
    L = _layers(model)
    for i, layer in enumerate(L):
        if layers is not None and i not in layers:
            continue
        for k in kinds:
            lin = getattr(layer.mlp, k)
            w = LoRALinear(lin, rank, device, gen)
            setattr(layer.mlp, k, w)
            out[f"{PREFIX}.{i}.mlp.{k}"] = w
            keep |= {id(w.A), id(w.B)}
    model._distill = {"keep": keep, "index": {id(l): i for i, l in enumerate(L)}}
    model.eval()
    return out


def detach(model):
    for layer in _layers(model):
        for k in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, k)
            if isinstance(m, LoRALinear):
                setattr(layer.mlp, k, m.base)
    model._distill = {}


def _ce_sum(h, W, t):
    return torch.nn.functional.cross_entropy((h @ W.T).float(), t, reduction="sum")


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def batch_backward(model, seqs, device="cuda", chunk=512, weight=1.0, pad_id=0):
    """Accumulate into the LoRA .grad the gradient of weight x (mean cross-entropy over the completion tokens of
    seqs = [(prompt_ids, completion_ids), ...]); returns that mean (unweighted). Right padding: causal, so real
    positions never see the pads.

    Forward without grad, each layer streamed through `device`, keeping on the CPU each layer's input and call
    arguments. Backward by hand, last layer first: stream the layer in, recompute it with grad from its saved input,
    backpropagate, and only then send it back. (Moving weights with p.data inside autograd's lifetime moved the
    tensors autograd had saved: tests caught it.)"""
    st = model._distill
    keep, index = st["keep"], st["index"]
    core, L = model.model, _layers(model)
    last = len(L) - 1
    Lmax = max(len(p) + len(c) for p, c in seqs)
    ids = torch.tensor([p + c + [pad_id] * (Lmax - len(p) - len(c)) for p, c in seqs])
    rec = {}

    def pre(mod, args, kwargs):
        rec[index[id(mod)]] = (_to(args, "cpu"), _to(kwargs, "cpu"))
        _move_frozen(mod, device, keep)
        return _to(args, device), _to(kwargs, device)

    def post(mod, args, kwargs, o):
        _move_frozen(mod, "cpu", keep)
        if index[id(mod)] == last:
            rec["out"] = _to(_first(o), "cpu")
        return _to(o, "cpu")
    hs_ = [l.register_forward_pre_hook(pre, with_kwargs=True) for l in L]
    hs_ += [l.register_forward_hook(post, with_kwargs=True) for l in L]
    try:
        with torch.no_grad():
            core(input_ids=ids, use_cache=False)
    finally:
        for h in hs_:
            h.remove()
    # head: final norm + chunked, checkpointed cross-entropy against a GPU copy of lm_head
    norm = core.norm
    _move_frozen(norm, device, set())
    hl = rec.pop("out").to(device).requires_grad_(True)
    h = norm(hl)
    rows, tg = [], []
    for b, (p, c) in enumerate(seqs):
        a, z = len(p) - 1, len(p) + len(c) - 1
        rows.append(h[b, a:z])
        tg.append(ids[b, a + 1:z + 1])
    hs, tgt = torch.cat(rows), torch.cat(tg).to(device)
    W = model.lm_head.weight.detach().to(device=device, dtype=hs.dtype)
    tot = 0.0
    for s in range(0, len(tgt), chunk):
        tot = tot + ckpt(_ce_sum, hs[s:s + chunk], W, tgt[s:s + chunk], use_reentrant=False)
    loss = tot / len(tgt)
    (loss * weight).backward()
    g = hl.grad
    del W, hs, h, hl
    _move_frozen(norm, "cpu", set())
    for i in range(last, -1, -1):
        layer = L[i]
        args, kwargs = rec.pop(i)
        _move_frozen(layer, device, keep)
        if args:
            x = args[0].to(device).detach().requires_grad_(True)
            a2, kw = (x,) + tuple(_to(args[1:], device)), _to(kwargs, device)
        else:
            x = kwargs["hidden_states"].to(device).detach().requires_grad_(True)
            a2, kw = (), dict(_to(kwargs, device), hidden_states=x)
        y = _first(layer(*a2, **kw))
        y.backward(g)
        g = x.grad
        del y, x, a2, kw
        _move_frozen(layer, "cpu", keep)
    return float(loss)


def export(loras, out, rank):
    from safetensors.torch import save_file
    out = Path(out)
    out.mkdir(parents=True)
    T = {}
    for stem, w in loras.items():
        T[f"base_model.model.{stem}.lora_A.weight"] = w.A.detach().cpu().to(torch.bfloat16).contiguous()
        T[f"base_model.model.{stem}.lora_B.weight"] = w.B.detach().cpu().to(torch.bfloat16).contiguous()
    save_file(T, str(out / "adapter_model.safetensors"))
    kinds = sorted({s.rsplit(".", 1)[1] for s in loras})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
           "bias": "none", "target_modules": kinds, "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--kinds", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seqs_per_step", type=int, default=8)
    ap.add_argument("--tokens_per_micro", type=int, default=6144)
    ap.add_argument("--max_len", type=int, default=2600)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    tok = AutoTokenizer.from_pretrained(a.base)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["ok"]]
    seqs = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"] + [end]
        if len(p) + len(c) <= a.max_len:
            seqs.append((p, c))
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    loras = attach(model, a.rank, kinds=tuple(a.kinds.split(",")), device="cuda")
    params = [p for w in loras.values() for p in (w.A, w.B)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    log = out.with_suffix(".steps.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"train_seqs": len(seqs), "tokens": sum(len(c) for _, c in seqs),
                      "lora_params": sum(p.numel() for p in params)}), flush=True)
    rng, step, t0 = random.Random(0), 0, time.time()
    for ep in range(a.epochs):
        order = list(range(len(seqs)))
        rng.shuffle(order)
        for s in range(0, len(order), a.seqs_per_step):
            group = sorted((seqs[i] for i in order[s:s + a.seqs_per_step]), key=lambda x: len(x[0]) + len(x[1]))
            ntok = sum(len(c) for _, c in group)
            opt.zero_grad(set_to_none=True)
            micro, tot = [], 0.0
            for q in group + [None]:      # micro-batches by padded-token budget; loss weighted by tokens
                if q is not None and (not micro or (len(micro) + 1) * max(len(x[0]) + len(x[1]) for x in micro + [q])
                                      <= a.tokens_per_micro):
                    micro.append(q)
                    continue
                w = sum(len(c) for _, c in micro) / ntok
                tot += batch_backward(model, micro, weight=w) * w
                micro = [q] if q is not None else []
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            rec = {"epoch": ep, "step": step, "loss": round(tot, 4), "seqs": len(group), "tokens": ntok,
                   "seconds": round(time.time() - t0)}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    export(loras, out, a.rank)
    print(json.dumps({"out": str(out), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
