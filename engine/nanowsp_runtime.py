"""nanowsp_runtime.py — Standalone Micro-Transformer LLM Runtime for Weight-Space Programming.

A lightweight, fully introspectable 2-layer autoregressive Transformer (~80k parameters)
designed to study weight-space programming, Fast Weight associative memory imprinting,
and rank-1 projection algebra from scratch without PyTorch autograd / SGD.
"""

from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


# ==============================================================================
# 1. ASCII Tokenizer (Vocab Size = 100)
# ==============================================================================

class ASCIITokenizer:
    """Byte/ASCII char tokenizer for NanoWSP.
    0: <pad>, 1: <bos>, 2: <eos>, 3: \\n, 4..98: ASCII 32..126, 99: <unk>
    """
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    NL_TOKEN = 3
    UNK_TOKEN = 99
    VOCAB_SIZE = 100

    def __init__(self):
        self.char_to_id: Dict[str, int] = {"\n": self.NL_TOKEN}
        self.id_to_char: Dict[int, str] = {
            self.PAD_TOKEN: "<pad>",
            self.BOS_TOKEN: "<bos>",
            self.EOS_TOKEN: "<eos>",
            self.NL_TOKEN: "\n",
            self.UNK_TOKEN: "<unk>"
        }
        # ASCII printable characters (32 to 126 inclusive -> 95 chars)
        for code in range(32, 127):
            ch = chr(code)
            idx = 4 + (code - 32)
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens = [self.BOS_TOKEN] if add_bos else []
        for ch in text:
            tokens.append(self.char_to_id.get(ch, self.UNK_TOKEN))
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        chars = []
        for t in tokens:
            if skip_special and t in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                continue
            chars.append(self.id_to_char.get(t, ""))
        return "".join(chars)


# ==============================================================================
# 2. NanoWSP Configuration & Weights
# ==============================================================================

@dataclass
class NanoWSPConfig:
    vocab_size: int = 100
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_head: int = 32
    d_ff: int = 128
    max_seq_len: int = 32
    eps: float = 1e-5

    def total_parameters(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        layer_attn = 4 * (self.d_model * self.d_model) + self.d_model
        layer_mlp = (self.d_model * self.d_ff) + (self.d_ff * self.d_model) + self.d_model
        per_layer = layer_attn + layer_mlp
        head = self.vocab_size * self.d_model + self.d_model
        return emb + (self.n_layers * per_layer) + head


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


# ==============================================================================
# 3. NanoWSP Model Class
# ==============================================================================

class NanoWSP:
    """Self-contained Pure-PyTorch/NumPy Micro-Transformer for WSP Research."""

    def __init__(self, config: Optional[NanoWSPConfig] = None, seed: int = 42):
        self.config = config or NanoWSPConfig()
        self.tokenizer = ASCIITokenizer()
        self.weights: Dict[str, torch.Tensor] = {}
        self.init_weights(seed=seed)

    def init_weights(self, seed: int = 42):
        """Orthogonal & spectral initialization of all parameters."""
        torch.manual_seed(seed)
        c = self.config

        # Embeddings
        self.weights["token_embd.weight"] = torch.randn(c.vocab_size, c.d_model) * (1.0 / math.sqrt(c.d_model))
        self.weights["pos_embd.weight"] = torch.randn(c.max_seq_len, c.d_model) * 0.02

        # Layers
        for l in range(c.n_layers):
            prefix = f"layers.{l}."
            # Attention
            self.weights[f"{prefix}attn_norm.weight"] = torch.ones(c.d_model)
            q_w = torch.empty(c.d_model, c.d_model)
            nn.init.orthogonal_(q_w)
            self.weights[f"{prefix}attn_q.weight"] = q_w

            k_w = torch.empty(c.d_model, c.d_model)
            nn.init.orthogonal_(k_w)
            self.weights[f"{prefix}attn_k.weight"] = k_w

            v_w = torch.empty(c.d_model, c.d_model)
            nn.init.orthogonal_(v_w)
            self.weights[f"{prefix}attn_v.weight"] = v_w

            out_w = torch.empty(c.d_model, c.d_model)
            nn.init.orthogonal_(out_w)
            self.weights[f"{prefix}attn_out.weight"] = out_w * (1.0 / math.sqrt(2 * c.n_layers))

            # MLP (in: d_model -> d_ff, out: d_ff -> d_model)
            self.weights[f"{prefix}mlp_norm.weight"] = torch.ones(c.d_model)
            in_w = torch.empty(c.d_ff, c.d_model)
            nn.init.kaiming_uniform_(in_w, a=math.sqrt(5))
            self.weights[f"{prefix}mlp_in.weight"] = in_w

            out_w_mlp = torch.empty(c.d_model, c.d_ff)
            nn.init.kaiming_uniform_(out_w_mlp, a=math.sqrt(5))
            self.weights[f"{prefix}mlp_out.weight"] = out_w_mlp * (1.0 / math.sqrt(2 * c.n_layers))

        # Final Norm & LM Head
        self.weights["final_norm.weight"] = torch.ones(c.d_model)
        head_w = torch.empty(c.vocab_size, c.d_model)
        nn.init.normal_(head_w, std=0.02)
        self.weights["output.weight"] = head_w

    def count_parameters(self) -> int:
        return sum(w.numel() for w in self.weights.values())

    def save_checkpoint(self, path: Union[str, Path]):
        """Save model weights to SafeTensors format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().float() for k, v in self.weights.items()}
        save_file(tensors, str(path))

    def load_checkpoint(self, path: Union[str, Path]):
        """Load model weights from SafeTensors format."""
        path = Path(path)
        tensors = load_file(str(path))
        for k, v in tensors.items():
            self.weights[k] = v.float()

    def forward(
        self,
        token_ids: List[int],
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Autoregressive causal forward pass."""
        c = self.config
        T = len(token_ids)
        if T > c.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {c.max_seq_len}")

        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # [1, T]
        pos = torch.arange(0, T, dtype=torch.long)  # [T]

        tok_emb = self.weights["token_embd.weight"][x]
        pos_emb = self.weights["pos_embd.weight"][pos].unsqueeze(0)
        h = tok_emb + pos_emb

        traces: Dict[str, Any] = {"token_emb": tok_emb, "initial_residual": h.clone()}
        causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

        for l in range(c.n_layers):
            prefix = f"layers.{l}."
            # Attention sublayer
            h_norm = rms_norm(h, self.weights[f"{prefix}attn_norm.weight"], eps=c.eps)
            traces[f"layer_{l}_attn_norm"] = h_norm.clone()

            Q = torch.matmul(h_norm, self.weights[f"{prefix}attn_q.weight"].t())
            K = torch.matmul(h_norm, self.weights[f"{prefix}attn_k.weight"].t())
            V = torch.matmul(h_norm, self.weights[f"{prefix}attn_v.weight"].t())

            Q = Q.view(1, T, c.n_heads, c.d_head).transpose(1, 2)
            K = K.view(1, T, c.n_heads, c.d_head).transpose(1, 2)
            V = V.view(1, T, c.n_heads, c.d_head).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(c.d_head)
            scores = scores + causal_mask
            attn_probs = F.softmax(scores, dim=-1)
            traces[f"layer_{l}_attn_probs"] = attn_probs.clone()

            attn_ctx = torch.matmul(attn_probs, V)
            attn_ctx = attn_ctx.transpose(1, 2).contiguous().view(1, T, c.d_model)
            attn_out = torch.matmul(attn_ctx, self.weights[f"{prefix}attn_out.weight"].t())

            h = h + attn_out
            traces[f"layer_{l}_after_attn"] = h.clone()

            # MLP sublayer
            h_norm_mlp = rms_norm(h, self.weights[f"{prefix}mlp_norm.weight"], eps=c.eps)
            traces[f"layer_{l}_mlp_norm"] = h_norm_mlp.clone()

            mlp_hidden = F.gelu(torch.matmul(h_norm_mlp, self.weights[f"{prefix}mlp_in.weight"].t()))
            traces[f"layer_{l}_mlp_hidden"] = mlp_hidden.clone()

            mlp_out = torch.matmul(mlp_hidden, self.weights[f"{prefix}mlp_out.weight"].t())
            h = h + mlp_out
            traces[f"layer_{l}_after_mlp"] = h.clone()

        h_final = rms_norm(h, self.weights["final_norm.weight"], eps=c.eps)
        traces["final_hidden"] = h_final.clone()
        logits = torch.matmul(h_final, self.weights["output.weight"].t())

        return logits, (traces if return_hidden else None)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0
    ) -> str:
        """Autoregressively generate next characters from prompt."""
        token_ids = self.tokenizer.encode(prompt, add_bos=True)

        for _ in range(max_new_tokens):
            if len(token_ids) >= self.config.max_seq_len:
                break
            logits, _ = self.forward(token_ids)
            next_token_logits = logits[0, -1, :]

            if temperature <= 0.0:
                next_token = int(torch.argmax(next_token_logits).item())
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == self.tokenizer.EOS_TOKEN:
                break
            token_ids.append(next_token)

        return self.tokenizer.decode(token_ids, skip_special=True)

    # ==========================================================================
    # 4. Weight-Space Programming Primitives (No Backpropagation)
    # ==========================================================================

    def imprint_associative_rule(
        self,
        prompt: str,
        target_char: str,
        layer_idx: int = 1,
        eta: float = 4.0
    ) -> Dict[str, Any]:
        """Imprint a single factual association directly in weight space via outer product.
        Delta W = eta * (v_target outer k_hidden)
        """
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        _, traces = self.forward(token_ids, return_hidden=True)
        assert traces is not None

        k_hidden = traces[f"layer_{layer_idx}_mlp_hidden"][0, -1, :].clone()
        k_norm = torch.norm(k_hidden)
        if k_norm > 1e-6:
            k_hidden = k_hidden / k_norm

        target_id = self.tokenizer.encode(target_char, add_bos=False)[0]
        v_target = self.weights["output.weight"][target_id].clone()
        v_norm = torch.norm(v_target)
        if v_norm > 1e-6:
            v_target = v_target / v_norm

        delta = eta * torch.outer(v_target, k_hidden)
        target_tensor_key = f"layers.{layer_idx}.mlp_out.weight"
        before_norm = float(torch.norm(self.weights[target_tensor_key]).item())
        self.weights[target_tensor_key] += delta
        after_norm = float(torch.norm(self.weights[target_tensor_key]).item())

        return {
            "imprint_status": "ok",
            "prompt": prompt,
            "target_char": target_char,
            "layer": layer_idx,
            "target_tensor": target_tensor_key,
            "delta_frobenius": float(torch.norm(delta).item()),
            "weight_norm_before": before_norm,
            "weight_norm_after": after_norm
        }

    def imprint_associative_batch(
        self,
        pairs: List[Tuple[str, str]],
        layer_idx: int = 1,
        eta: float = 4.0,
        ridge_lambda: float = 1e-4
    ) -> Dict[str, Any]:
        """Imprint multiple associations simultaneously via Moore-Penrose pseudo-inverse.
        Cancels out cross-talk between memories via (X^T X + lambda I)^-1.
        Delta W = eta * Y @ (X^T X + lambda I)^-1 X^T
        """
        keys = []
        values = []

        for p, t in pairs:
            token_ids = self.tokenizer.encode(p, add_bos=True)
            _, traces = self.forward(token_ids, return_hidden=True)
            assert traces is not None
            k = traces[f"layer_{layer_idx}_mlp_hidden"][0, -1, :].clone()
            k = k / max(torch.norm(k).item(), 1e-6)
            keys.append(k)

            target_id = self.tokenizer.encode(t, add_bos=False)[0]
            v = self.weights["output.weight"][target_id].clone()
            v = v / max(torch.norm(v).item(), 1e-6)
            values.append(v)

        X = torch.stack(keys, dim=1)    # [d_ff, M]
        Y = torch.stack(values, dim=1)  # [d_model, M]

        cond = float(torch.linalg.cond(X).item())

        if ridge_lambda > 0.0:
            M = X.shape[1]
            XtX = torch.matmul(X.t(), X) + ridge_lambda * torch.eye(M)
            X_pinv = torch.matmul(torch.linalg.inv(XtX), X.t())  # [M, d_ff]
        else:
            X_pinv = torch.pinverse(X)  # [M, d_ff]

        delta = eta * torch.matmul(Y, X_pinv)  # [d_model, d_ff]

        target_tensor_key = f"layers.{layer_idx}.mlp_out.weight"
        before_norm = float(torch.norm(self.weights[target_tensor_key]).item())
        self.weights[target_tensor_key] += delta
        after_norm = float(torch.norm(self.weights[target_tensor_key]).item())

        return {
            "batch_status": "ok",
            "pairs_count": len(pairs),
            "layer": layer_idx,
            "condition_number": cond,
            "delta_frobenius": float(torch.norm(delta).item()),
            "weight_norm_before": before_norm,
            "weight_norm_after": after_norm
        }

    def unlearn_attractor(
        self,
        prompt: str,
        layer_idx: int = 1,
        eta: float = 1.0
    ) -> Dict[str, Any]:
        """Cancel out a degenerative output/attractor via Delta-Rule rank-1 subtraction."""
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        _, traces = self.forward(token_ids, return_hidden=True)
        assert traces is not None

        k_hidden = traces[f"layer_{layer_idx}_mlp_hidden"][0, -1, :].clone()
        k_norm = torch.norm(k_hidden)
        if k_norm > 1e-6:
            k_hidden = k_hidden / k_norm

        target_tensor_key = f"layers.{layer_idx}.mlp_out.weight"
        W = self.weights[target_tensor_key]

        current_y = torch.matmul(W, k_hidden)
        before_energy = float(torch.norm(current_y).item())

        delta = eta * torch.outer(current_y, k_hidden)
        self.weights[target_tensor_key] -= delta

        after_y = torch.matmul(self.weights[target_tensor_key], k_hidden)
        after_energy = float(torch.norm(after_y).item())

        return {
            "unlearn_status": "ok",
            "prompt": prompt,
            "layer": layer_idx,
            "target_tensor": target_tensor_key,
            "energy_before": before_energy,
            "energy_after": after_energy,
            "reduction_pct": f"{((before_energy - after_energy) / max(before_energy, 1e-12)):.2%}"
        }

    def evaluate_recall(self, pairs: List[Tuple[str, str]]) -> float:
        """Measure exact character accuracy across test prompt-completion pairs."""
        correct = 0
        for p, t in pairs:
            out = self.generate(p, max_new_tokens=1)
            if out.endswith(t):
                correct += 1
        return correct / max(len(pairs), 1)


# ==============================================================================
# 5. CLI Smoke Runner
# ==============================================================================

if __name__ == "__main__":
    print("[*] Initializing NanoWSP Micro-Transformer...")
    model = NanoWSP(seed=42)
    print(f"    Total parameters: {model.count_parameters():,}")

    test_pairs = [("A->", "1"), ("B->", "2"), ("C->", "3"), ("D->", "4")]
    print("[*] Baseline evaluation (untrained):")
    for p, t in test_pairs:
        print(f"    {p} -> Expected: {t} | Raw: {model.generate(p, max_new_tokens=1)}")

    print("[*] Imprinting 4 factual associations via Moore-Penrose pseudo-inverse...")
    res = model.imprint_associative_batch(test_pairs, layer_idx=1, eta=5.0)
    print(f"    Batch result: {res}")

    print("[*] Post-programming evaluation:")
    for p, t in test_pairs:
        print(f"    {p} -> Expected: {t} | Programmed: {model.generate(p, max_new_tokens=1)}")
    acc = model.evaluate_recall(test_pairs)
    print(f"    Recall accuracy: {acc:.1%}")
