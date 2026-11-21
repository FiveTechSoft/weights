"""Standalone helper for the behavior-extraction probe: embedding rows + head rows only, no decoder layers.

Same pattern as skill_delta/train_down23.py phase 2: rows are read on demand from the safetensors file, so
nothing but the tokenizer and a few row slices is in memory. Used by train_aux.py (all arms) and by
eval_aux.py for arms X and C. Arm H evaluation still needs the full QwenText.
Authorship: written by the project maintainer's task, not by the prototype's original author.
Status: checked only with synthetic tensors (tests/test_behavior_extraction_lite.py); NOT run on the real model.
"""
import torch
from safetensors import safe_open

PFX = "model.language_model."


class LiteText:
    def __init__(self, path: str, tokenizer: bool = True):
        self.f = safe_open(path + "/model.safetensors", "pt")
        self.sl = self.f.get_slice(PFX + "embed_tokens.weight")
        if tokenizer:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            self.tok.padding_side = "left"

    def embed(self, ids: torch.Tensor) -> torch.Tensor:           # identical to QwenText.embed
        flat = ids.reshape(-1).tolist()
        rows = {u: self.sl[u:u + 1] for u in sorted(set(flat))}
        return torch.cat([rows[i] for i in flat], 0).reshape(*ids.shape, -1).to(torch.bfloat16)

    def logits_last(self, h_last: torch.Tensor, chunk: int = 8192) -> torch.Tensor:   # identical to QwenText.logits_last
        out = []
        V = self.sl.get_shape()[0]
        for s in range(0, V, chunk):
            E = self.sl[s:min(V, s + chunk)].to(torch.bfloat16)
            out.append((h_last.to(torch.bfloat16) @ E.T).float())
        return torch.cat(out, -1)


def prefix_mean_bf16(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Arm X feature at evaluation: fp32 mean over non-pad tokens, rounded to bf16 exactly as the training cache
    stores it (extract_hidden.py), then returned as fp32 for the aux."""
    e = emb.float() * mask[..., None]
    m = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float()
    return m.to(torch.bfloat16).float()
