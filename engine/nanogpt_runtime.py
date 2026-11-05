"""engine/nanogpt_runtime.py — Karpathy nanoGPT Reference Implementation & WSPLang Runtime.

Implementa la arquitectura canónica de nanoGPT (Andrej Karpathy) con soporte completo
para serialización SafeTensors, atado de pesos (weight tying), inicialización escalada
GPT-2 y paridad numérica exacta a precisión float32.
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# ==============================================================================
# 1. Configuración Canónica de nanoGPT
# ==============================================================================

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65      # Caracteres de Shakespeare por defecto
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    d_ff: Optional[int] = None # Si es None, computa dinámicamente 4 * n_embd
    dropout: float = 0.0
    bias: bool = True         # True: estándar GPT-2 (con bias en linear y layernorm)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.n_embd

    def total_parameters(self) -> int:
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        # Por capa: ln_1 (2*d), c_attn (d * 3d + 3d), c_proj (d*d + d), ln_2 (2*d), c_fc (d * d_ff + d_ff), c_proj (d_ff * d + d)
        per_layer = (2 * self.n_embd) + (self.n_embd * 3 * self.n_embd + 3 * self.n_embd) + \
                    (self.n_embd * self.n_embd + self.n_embd) + (2 * self.n_embd) + \
                    (self.n_embd * self.d_ff + self.d_ff) + (self.d_ff * self.n_embd + self.n_embd)
        ln_f = 2 * self.n_embd
        # lm_head tied to wte: 0 params adicionales
        return emb + (self.n_layer * per_layer) + ln_f


# ==============================================================================
# 2. Módulos Canónicos de nanoGPT (Karpathy)
# ==============================================================================

class LayerNorm(nn.Module):
    """LayerNorm con soporte opcional de bias (exacto a Karpathy)."""
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """Atención causal multi-cabeza canónica de Karpathy."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Proyecciones agrupadas Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Proyección de salida
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        # Q, K, V simultáneos
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention idéntica a Karpathy
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Capa Feed-Forward con activación GELU exacta a Karpathy."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_ff = getattr(config, "d_ff", 4 * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Bloque Transformer Transformer pre-LayerNorm canónico."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class KarpathyGPT(nn.Module):
    """Implementación oficial de Karpathy nanoGPT como módulo de PyTorch puro."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Atado de pesos de Karpathy (Weight Tying)
        self.transformer.wte.weight = self.lm_head.weight

        # Inicialización de pesos idéntica a model.py de Karpathy
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # Solo último logit durante generación
            loss = None

        return logits, loss


# ==============================================================================
# 3. Runtime WSPLang para nanoGPT (Serialización, Inspección y Paridad)
# ==============================================================================

class NanoGPT:
    """Runtime interoperable de nanoGPT para el ecosistema WSPLang."""

    def __init__(self, config: Optional[GPTConfig] = None, seed: int = 1337):
        self.config = config or GPTConfig()
        self.seed = seed
        self.model = KarpathyGPT(self.config)
        self.init_weights(seed=seed)

    def init_weights(self, seed: int = 1337):
        """Inicialización canónica de Karpathy con escalado residual GPT-2."""
        torch.manual_seed(seed)
        c = self.config

        def _init_mod(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.model.apply(_init_mod)

        # Escalado residual 1 / sqrt(2 * n_layer) por paper de GPT-2
        scale = 0.02 / math.sqrt(2 * c.n_layer)
        for pn, p in self.model.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=scale)

        # Asegurar puntero atado wte == lm_head
        self.model.transformer.wte.weight = self.model.lm_head.weight

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        """Exporta diccionario de tensores con claves exactas de HuggingFace / nanoGPT."""
        sd = self.model.state_dict()
        # Asegurar clave lm_head.weight
        sd["lm_head.weight"] = self.model.lm_head.weight.detach().clone()
        return sd

    def save_checkpoint(self, path: Union[str, Path]):
        """Guarda los pesos del modelo en formato SafeTensors."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sd = self.export_state_dict()
        save_file(sd, str(p))

    def load_checkpoint(self, path: Union[str, Path]):
        """Carga pesos desde SafeTensors y enlaza punteros."""
        sd = load_file(str(path))
        self.model.load_state_dict(sd, strict=False)
        self.model.transformer.wte.weight = self.model.lm_head.weight

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.model(idx, targets=targets)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generación autorregresiva canónica coincidente paso a paso con Karpathy."""
        self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass para obtener logits completos de la secuencia
            b, t = idx_cond.size()
            pos = torch.arange(0, t, dtype=torch.long, device=idx_cond.device)
            tok_emb = self.model.transformer.wte(idx_cond)
            pos_emb = self.model.transformer.wpe(pos)
            x = self.model.transformer.drop(tok_emb + pos_emb)
            for block in self.model.transformer.h:
                x = block(x)
            x = self.model.transformer.ln_f(x)
            logits = self.model.lm_head(x[:, -1, :]) # Logit de última posición

            if temperature == 0.0:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
