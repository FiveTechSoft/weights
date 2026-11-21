"""Low-memory text-only runner for Qwen3.5-0.8B (CPU, bf16): layers in memory, embedding rows read on demand from the safetensors file.

Why: the box has ~1.7 GB free RAM; the full model (with vision tower and a 0.5 GB tied embedding) does not fit as a loaded model.
Generation is real greedy decoding (no KV cache: the short prompt is re-run each step), so a hit is an exact generated string.
"""
import json
import struct
import resource
import time
from typing import Dict, List

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

PFX = "model.language_model."


class QwenText:
    def __init__(self, path: str):
        self.path = path
        cfg = AutoConfig.from_pretrained(path).text_config
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.f = safe_open(path + "/model.safetensors", "pt")
        cfg.vocab_size = 8                       # tiny placeholder embedding; real rows come from the file
        with torch.device("meta"):
            self.m = Qwen3_5TextModel(cfg)
        self.m.embed_tokens = torch.nn.Embedding(1, cfg.hidden_size)
        self.m = self.m.to_empty(device="cpu").to(torch.bfloat16)
        params = dict(self.m.named_parameters()); params.update(dict(self.m.named_buffers()))
        loaded = set()
        for k in self.f.keys():                  # one tensor at a time: no second full copy of the layers in memory
            if k.startswith(PFX) and not k.startswith(PFX + "embed_tokens"):
                name = k[len(PFX):]
                if name not in params:
                    raise RuntimeError("unexpected tensor " + name)
                params[name].data.copy_(self.f.get_tensor(k).to(params[name].dtype))
                loaded.add(name)
        missing = [n for n, _ in self.m.named_parameters() if n not in loaded and not n.startswith("embed_tokens")]
        if missing:
            raise RuntimeError("missing tensors: %s" % missing[:5])
        # non-persistent buffers (rotary inv_freq) were emptied by to_empty: rebuild
        self.m.rotary_emb = type(self.m.rotary_emb)(config=cfg)
        self.m.eval()
        self.sl = self.f.get_slice(PFX + "embed_tokens.weight")
        self.cfg = cfg

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        uniq = sorted(set(flat))
        rows = {u: self.sl[u:u + 1] for u in uniq}
        return torch.cat([rows[i] for i in flat], 0).reshape(*ids.shape, -1).to(torch.bfloat16)

    def hidden(self, ids, mask, grad=False):
        with torch.set_grad_enabled(grad):
            return self.m(inputs_embeds=self.embed(ids), attention_mask=mask).last_hidden_state

    def logits_last(self, h_last: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        out = []
        V = self.sl.get_shape()[0]
        for s in range(0, V, chunk):
            E = self.sl[s:min(V, s + chunk)].to(torch.bfloat16)
            out.append((h_last.to(torch.bfloat16) @ E.T).float())
        return torch.cat(out, -1)

    def generate(self, prompts: List[str], max_new: int) -> List[str]:
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        gen = [[] for _ in prompts]
        for _ in range(max_new):
            h = self.hidden(ids, mask)[:, -1]
            nxt = self.logits_last(h).argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                gen[i].append(t)
            ids = torch.cat([ids, nxt[:, None]], 1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
