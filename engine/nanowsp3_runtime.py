"""nanowsp3_runtime.py — NanoWSP-3: 6-Layer Modular Thematic Micro-LLM Synthesized in Weight Space.

Architecture Specifications:
- 6 Layers, d_model=256, 8 Attention Heads (d_head=32), d_ff=512, Vocab=256.
- Total Parameters: 3,280,128 Float32 parameters (~12.51 MB on disk).
- Pure Weight-Space Construction (Zero backpropagation / autograd).
- Thematic Knowledge Banking:
    * Domain 1: Physical Science & Elements (SCI: Z79, Z47, Z26, H2O, C)
    * Domain 2: Global Geography & Landmarks (GEO: ESP, FRA, JPN, BFA, KAZ)
    * Domain 3: Computer Science & Pioneers (CS: LNX, PYT, WWW, LAN, ATT)
    * Domain 4: Humanities & Literature (LIT: HAM, QUI, HOB, REP)
    * Domain 5: Agentic Execution & Calculation (ACT: 3+4, 9-5, 8/2, 6+2)
"""

from __future__ import annotations
import math
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# ==============================================================================
# 1. Full 256-Byte Tokenizer
# ==============================================================================

class ByteTokenizer:
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3
    VOCAB_SIZE = 256

    def __init__(self):
        self.char_to_id: Dict[str, int] = {
            "<pad>": self.PAD_TOKEN,
            "<bos>": self.BOS_TOKEN,
            "<eos>": self.EOS_TOKEN,
            "<unk>": self.UNK_TOKEN,
        }
        self.id_to_char: Dict[int, str] = {
            self.PAD_TOKEN: "<pad>",
            self.BOS_TOKEN: "<bos>",
            self.EOS_TOKEN: "<eos>",
            self.UNK_TOKEN: "<unk>",
        }
        for b in range(4, 256):
            ch = chr(b)
            self.char_to_id[ch] = b
            self.id_to_char[b] = ch

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens = [self.BOS_TOKEN] if add_bos else []
        for ch in text:
            code = ord(ch)
            if code < 256:
                tokens.append(code if code >= 4 else self.UNK_TOKEN)
            else:
                for byte in ch.encode("utf-8"):
                    tokens.append(int(byte))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        chars = []
        for t in tokens:
            if skip_special and t in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN):
                continue
            chars.append(self.id_to_char.get(t, chr(t) if t < 256 else ""))
        return "".join(chars)


# ==============================================================================
# 2. NanoWSP-3 Architecture Configuration
# ==============================================================================

@dataclass
class NanoWSP3Config:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 32
    d_ff: int = 512
    max_seq_len: int = 128
    eps: float = 1e-5
    rope_theta: float = 10000.0


# ==============================================================================
# 3. Positional Encoding & Normalization Primitives
# ==============================================================================

def precompute_rope_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    b, s, h, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, s, h, -1, 2))
    freqs_cis = freqs_cis[:s].unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


# ==============================================================================
# 4. NanoWSP-3 Runtime Model Class
# ==============================================================================

class NanoWSP3:
    """6-Layer Modular Thematic Micro-LLM Synthesized in Weight Space."""

    def __init__(self, config: Optional[NanoWSP3Config] = None):
        self.config = config or NanoWSP3Config()
        self.tokenizer = ByteTokenizer()
        self.freqs_cis = precompute_rope_cis(
            dim=self.config.d_head,
            end=self.config.max_seq_len,
            theta=self.config.rope_theta
        )
        self.weights: Dict[str, torch.Tensor] = {}
        self.thematic_registry: Dict[str, Dict[str, Any]] = {}
        self._init_zero_backbone()

    def _init_zero_backbone(self) -> None:
        c = self.config
        rng = torch.Generator().manual_seed(42)

        # 1. Embedding and Output matrices
        embed = torch.randn(c.vocab_size, c.d_model, generator=rng)
        embed = F.normalize(embed, p=2, dim=1)
        self.weights["embedding.weight"] = embed
        self.weights["output.weight"] = embed.clone()

        # 2. 6 Transformer Layers
        for l in range(c.n_layers):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                w = torch.empty(c.d_model, c.d_model)
                nn.init.orthogonal_(w, gain=1.0, generator=rng)
                self.weights[f"layers.{l}.attention.{proj}.weight"] = w

            w_in = torch.empty(c.d_ff, c.d_model)
            nn.init.orthogonal_(w_in, gain=1.0, generator=rng)
            self.weights[f"layers.{l}.mlp_in.weight"] = w_in

            w_out = torch.empty(c.d_model, c.d_ff)
            nn.init.orthogonal_(w_out, gain=1.0, generator=rng)
            self.weights[f"layers.{l}.mlp_out.weight"] = w_out

            self.weights[f"layers.{l}.attn_norm.weight"] = torch.ones(c.d_model)
            self.weights[f"layers.{l}.mlp_norm.weight"] = torch.ones(c.d_model)

        self.weights["final_norm.weight"] = torch.ones(c.d_model)

    def count_parameters(self) -> int:
        return sum(w.numel() for w in self.weights.values())

    def size_in_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        elem_size = 4 if dtype == torch.float32 else 2
        return self.count_parameters() * elem_size

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.weights, str(p))

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        loaded = load_file(str(p))
        for k, v in loaded.items():
            self.weights[k] = v.float()

    # ==========================================================================
    # Forward Pass
    # ==========================================================================

    def forward(
        self,
        token_ids: List[int],
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        c = self.config
        seq_len = len(token_ids)
        if seq_len == 0:
            raise ValueError("Token IDs sequence cannot be empty.")
        if seq_len > c.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {c.max_seq_len}.")

        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        h = F.embedding(tokens, self.weights["embedding.weight"])

        traces: Dict[str, torch.Tensor] = {}
        if return_hidden:
            traces["embed"] = h.clone()

        for l in range(c.n_layers):
            h_norm_attn = rms_norm(h, self.weights[f"layers.{l}.attn_norm.weight"], eps=c.eps)

            wq = self.weights[f"layers.{l}.attention.q_proj.weight"]
            wk = self.weights[f"layers.{l}.attention.k_proj.weight"]
            wv = self.weights[f"layers.{l}.attention.v_proj.weight"]
            wo = self.weights[f"layers.{l}.attention.out_proj.weight"]

            q = torch.matmul(h_norm_attn, wq.t()).view(1, seq_len, c.n_heads, c.d_head)
            k = torch.matmul(h_norm_attn, wk.t()).view(1, seq_len, c.n_heads, c.d_head)
            v = torch.matmul(h_norm_attn, wv.t()).view(1, seq_len, c.n_heads, c.d_head)

            q = apply_rope(q, self.freqs_cis)
            k = apply_rope(k, self.freqs_cis)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c.d_head)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            attn_probs = F.softmax(scores, dim=-1)

            attn_out = torch.matmul(attn_probs, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, c.d_model)
            h = h + torch.matmul(attn_out, wo.t())

            h_norm_mlp = rms_norm(h, self.weights[f"layers.{l}.mlp_norm.weight"], eps=c.eps)
            w_in = self.weights[f"layers.{l}.mlp_in.weight"]
            w_out = self.weights[f"layers.{l}.mlp_out.weight"]

            mlp_hidden = F.gelu(torch.matmul(h_norm_mlp, w_in.t()))
            if return_hidden:
                traces[f"layer_{l}_mlp_hidden"] = mlp_hidden.clone()

            mlp_out = torch.matmul(mlp_hidden, w_out.t())
            h = h + mlp_out
            if return_hidden:
                traces[f"layer_{l}_after_mlp"] = h.clone()

        h_final = rms_norm(h, self.weights["final_norm.weight"], eps=c.eps)
        if return_hidden:
            traces["final_hidden"] = h_final.clone()
        logits = torch.matmul(h_final, self.weights["output.weight"].t())

        return logits, (traces if return_hidden else None)

    # ==========================================================================
    # 5. Multi-Thematic Corpus Synthesis Engine
    # ==========================================================================

    def imprint_thematic_corpus(
        self,
        thematic_domains: List[Dict[str, Any]],
        target_layer: int = 3,
        eta: float = 3.5,
        reg_lambda: float = 0.01
    ) -> Dict[str, Any]:
        """Synthesizes all thematic domains into the designated memory routing layer.
        Uses Full Gram-Matrix Orthogonalization to invert all cross-talk simultaneously:
            Delta W = eta * (V - W0 K) (K^T K + lambda I)^-1 K^T
        """
        keys = []
        values = []
        domain_slices = {}
        current_idx = 0

        target_tensor_key = f"layers.{target_layer}.mlp_out.weight"
        W0 = self.weights[target_tensor_key]

        for domain in thematic_domains:
            name = domain["name"]
            facts = domain["facts"]
            start_i = current_idx

            for p, t in facts:
                token_ids = self.tokenizer.encode(p, add_bos=True)
                _, traces = self.forward(token_ids, return_hidden=True)
                assert traces is not None
                k = traces[f"layer_{target_layer}_mlp_hidden"][0, -1, :].clone()
                k = k / max(float(torch.norm(k).item()), 1e-6)
                keys.append(k)

                target_id = self.tokenizer.encode(t, add_bos=False)[0]
                v = self.weights["output.weight"][target_id].clone()
                v = v / max(float(torch.norm(v).item()), 1e-6)
                values.append(v)
                current_idx += 1

            domain_slices[name] = (start_i, current_idx)

        K = torch.stack(keys, dim=1)    # [d_ff, M]
        V = torch.stack(values, dim=1)  # [d_model, M]
        M = K.shape[1]

        # Inversion of multi-theme Gram matrix
        C0 = torch.eye(M, dtype=torch.float32)
        KtK = torch.matmul(K.t(), K)
        inv_gram = torch.linalg.inv(KtK + reg_lambda * C0)

        residual = V - torch.matmul(W0, K)
        delta = eta * torch.matmul(torch.matmul(residual, inv_gram), K.t())

        before_norm = float(torch.norm(self.weights[target_tensor_key]).item())
        self.weights[target_tensor_key] += delta
        after_norm = float(torch.norm(self.weights[target_tensor_key]).item())
        delta_norm = float(torch.norm(delta).item())
        max_delta = float(torch.max(torch.abs(delta)).item())

        self.thematic_registry = domain_slices

        return {
            "target_layer": target_layer,
            "total_facts_imprinted": M,
            "domains_registered": list(domain_slices.keys()),
            "weight_norm_before": round(before_norm, 4),
            "weight_norm_after": round(after_norm, 4),
            "delta_frobenius": round(delta_norm, 4),
            "perturbation_pct": round(delta_norm / before_norm * 100, 3),
            "max_abs_delta": round(max_delta, 6)
        }

    # ==========================================================================
    # 6. Evaluation Helpers
    # ==========================================================================

    def evaluate_recall(self, prompt: str, expected_target: str = "") -> Tuple[bool, str, float]:
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        logits, _ = self.forward(tokens)
        pred_token = int(torch.argmax(logits[0, -1, :]).item())
        pred_char = self.tokenizer.decode([pred_token], skip_special=False)
        probs = F.softmax(logits[0, -1, :], dim=-1)

        if expected_target:
            enc = self.tokenizer.encode(expected_target, add_bos=False)
            expected_token = enc[0] if enc else pred_token
            prob = float(probs[expected_token].item())
            is_hit = (pred_token == expected_token)
        else:
            prob = float(probs[pred_token].item())
            is_hit = True
        return is_hit, pred_char, prob

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 15,
        temperature: float = 0.0
    ) -> str:
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        for _ in range(max_new_tokens):
            logits, _ = self.forward(tokens)
            next_logits = logits[0, -1, :]
            if temperature == 0.0:
                next_token = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == self.tokenizer.EOS_TOKEN:
                break
            tokens.append(next_token)
        return self.tokenizer.decode(tokens)
