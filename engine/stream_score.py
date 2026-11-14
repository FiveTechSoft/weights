"""engine/stream_score.py - teacher-forced scoring of a Qwen3.5-style model one layer at a time.

Only one decoder layer is in memory. Hidden states of every prompt move through the layers together, so each layer's
weights are loaded once per scoring pass. This gives per-token negative log-likelihood of given text. It cannot generate.
Weights come from a loader, so the same code runs on a tiny random model (parity test) or on streamed checkpoints.

  stream_nll(cfg, ids_list, embed, load_layer, final_norm, head_chunks, starts) -> list of per-token NLL tensors
    embed(ids)            -> (T, d) float32 rows for the token ids
    load_layer(L, layer)  -> fills the decoder-layer module's parameters in place (tensor by tensor, so no second full copy of the layer
                             exists in memory: a float32 layer is about 0.9 GB and this machine has about 2 GB); `fill_from_state` does it from a dict
    final_norm            -> (d,) weight of model.norm
    head_chunks()         -> iterator of (start, (rows, d) float32) pieces of lm_head covering the vocabulary
    starts[i]             -> first token position of prompt i whose NLL is wanted (the reference part)
Float32 compute, transformers' own decoder layer (eager attention with an explicit causal mask; torch fallback kernels).
"""
from typing import Callable, Dict, Iterable, List, Tuple

import torch


def _causal(t: int) -> torch.Tensor:
    return torch.full((t, t), float("-inf")).triu(1)[None, None]


def fill_from_state(layer, sd: Dict[str, torch.Tensor]) -> None:
    params = dict(layer.named_parameters())
    params.update(dict(layer.named_buffers()))
    if set(sd) != set(params):
        raise KeyError(f"layer tensors differ: missing {sorted(set(params) - set(sd))}, extra {sorted(set(sd) - set(params))}")
    for k, v in sd.items():
        params[k].data.copy_(v)


def stream_nll(cfg, ids_list: List[List[int]], embed: Callable, load_layer: Callable,
               final_norm: torch.Tensor, head_chunks: Callable[[], Iterable[Tuple[int, torch.Tensor]]],
               starts: List[int], spill_dir: str = None) -> List[torch.Tensor]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    # spill_dir: hidden states of every prompt live in files there and only one prompt is in memory at a time
    # (peak memory then does not grow with the number of prompts). Without it they stay in memory, as before.
    import os
    def put(i, h):
        if spill_dir is None:
            hs[i] = h
        else:
            torch.save(h, os.path.join(spill_dir, f"h{i}.pt"))
    def get(i):
        return hs[i] if spill_dir is None else torch.load(os.path.join(spill_dir, f"h{i}.pt"))
    hs = [None] * len(ids_list)
    for i, ids in enumerate(ids_list):
        put(i, embed(torch.tensor(ids)).float()[None])
    for L in range(cfg.num_hidden_layers):
        with torch.device("meta"):
            layer = m.Qwen3_5DecoderLayer(cfg, L)
        layer = layer.to_empty(device="cpu").float().eval()
        load_layer(L, layer)
        full = cfg.layer_types[L] == "full_attention"
        with torch.no_grad():
            for i in range(len(ids_list)):
                h = get(i)
                t = h.shape[1]
                pos = torch.arange(t)[None]
                put(i, layer(h, position_embeddings=rot(h, pos), attention_mask=_causal(t) if full else None, position_ids=pos))
                del h
        del layer
    norm = m.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float()
    norm.load_state_dict({"weight": final_norm.float()})
    # head_chunks() is iterated ONCE: the vocabulary is the outer loop and prompts the inner one, so lm_head is read once
    # per pass however many prompts there are.
    with torch.no_grad():
        n = len(ids_list)
        tgts = [torch.tensor(ids[1:]) for ids in ids_list]
        lse = [torch.full((len(t),), float("-inf")) for t in tgts]
        tl = [torch.zeros(len(t)) for t in tgts]
        for s, w in head_chunks():
            w = w.float()
            for i in range(n):
                h = norm(get(i))[0][:-1]  # position t predicts token t+1
                logits = h @ w.T
                lse[i] = torch.logaddexp(lse[i], torch.logsumexp(logits, dim=1))
                inside = (tgts[i] >= s) & (tgts[i] < s + w.shape[0])
                tl[i][inside] = logits[inside, tgts[i][inside] - s]
        return [(lse[i] - tl[i])[max(starts[i] - 1, 0):] for i in range(n)]  # NLL of tokens starts[i]..T-1
