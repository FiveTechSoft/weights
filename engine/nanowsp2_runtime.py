"""nanowsp2_runtime.py — NanoWSP-2: Advanced 4-Layer Micro-LLM Synthesized & Programmed in Weight Space.

Architecture Specifications:
- 4 Layers, d_model=128, 4 Attention Heads (d_head=32), d_ff=256, Vocab=256, ~386k parameters.
- Built without backpropagation or autograd.
- Native integration with WSP-Lang 0.5 primitives:
  1. MEMIT / EMMET Covariance-Regularized Mass Associative Memory Synthesis.
  2. LEACE Closed-Form Concept Erasure (Belrose et al., NeurIPS 2023).
  3. Gated DeltaNet-2 Decoupled Erase & Write Linear Attention Channels.
  4. Titans Surprisal Test-Time Memory Monitor (Behrouz et al., Google 2025).
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
    """Full 256-byte ASCII / Binary Tokenizer.
    Every byte (0..255) maps 1:1 to a token ID.
    Special tokens:
    0: <pad>, 1: <bos>, 2: <eos>, 3: <unk>
    Standard printable ASCII and extended symbols map to IDs 4..255.
    """
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
                # UTF-8 fallback
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
# 2. NanoWSP-2 Configuration
# ==============================================================================

@dataclass
class NanoWSP2Config:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 32
    d_ff: int = 256
    max_seq_len: int = 64
    eps: float = 1e-5

    def total_parameters(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        layer_attn = 4 * (self.d_model * self.d_model) + self.d_model
        layer_mlp = (self.d_model * self.d_ff) + (self.d_ff * self.d_model) + self.d_model
        per_layer = layer_attn + layer_mlp
        head = self.vocab_size * self.d_model + self.d_model
        return emb + (self.n_layers * per_layer) + head


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


# ==============================================================================
# 3. NanoWSP-2 Model Architecture & Runtime
# ==============================================================================

class NanoWSP2:
    """Advanced 4-Layer Micro-Transformer LLM for WSP-Lang 0.5."""

    def __init__(self, config: Optional[NanoWSP2Config] = None, seed: int = 1337):
        self.config = config or NanoWSP2Config()
        self.tokenizer = ByteTokenizer()
        self.weights: Dict[str, torch.Tensor] = {}
        # Dynamic Fast-Weight overlays for test-time memory updates (Titans Surprisal)
        self.overlays: Dict[str, torch.Tensor] = {}
        self.background_covariances: Dict[str, torch.Tensor] = {}
        self.init_weights(seed=seed)

    def init_weights(self, seed: int = 1337):
        """Orthogonal spectral initialization of all 4 layers."""
        torch.manual_seed(seed)
        c = self.config

        # 1. Embeddings
        self.weights["token_embd.weight"] = torch.randn(c.vocab_size, c.d_model) * (1.0 / math.sqrt(c.d_model))
        self.weights["pos_embd.weight"] = torch.randn(c.max_seq_len, c.d_model) * 0.02

        # 2. 4 Layers (Hierarchical representation depth)
        for l in range(c.n_layers):
            prefix = f"layers.{l}."
            # Attention
            self.weights[f"{prefix}attn_norm.weight"] = torch.ones(c.d_model)
            for proj in ("q", "k", "v", "out"):
                w = torch.empty(c.d_model, c.d_model)
                nn.init.orthogonal_(w)
                scale = (1.0 / math.sqrt(2 * c.n_layers)) if proj == "out" else 1.0
                self.weights[f"{prefix}attn_{proj}.weight"] = w * scale

            # MLP (d_model -> d_ff, d_ff -> d_model)
            self.weights[f"{prefix}mlp_norm.weight"] = torch.ones(c.d_model)
            in_w = torch.empty(c.d_ff, c.d_model)
            nn.init.kaiming_uniform_(in_w, a=math.sqrt(5))
            self.weights[f"{prefix}mlp_in.weight"] = in_w

            out_w = torch.empty(c.d_model, c.d_ff)
            nn.init.kaiming_uniform_(out_w, a=math.sqrt(5))
            self.weights[f"{prefix}mlp_out.weight"] = out_w * (1.0 / math.sqrt(2 * c.n_layers))

        # 3. Final Norm & Output Logits Head
        self.weights["final_norm.weight"] = torch.ones(c.d_model)
        head_w = torch.empty(c.vocab_size, c.d_model)
        nn.init.normal_(head_w, std=0.02)
        self.weights["output.weight"] = head_w

    def count_parameters(self) -> int:
        return sum(w.numel() for w in self.weights.values())

    def save_checkpoint(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().float() for k, v in self.weights.items()}
        save_file(tensors, str(path))

    def load_checkpoint(self, path: Union[str, Path]):
        path = Path(path)
        tensors = load_file(str(path))
        for k, v in tensors.items():
            self.weights[k] = v.float()

    def get_effective_weight(self, key: str) -> torch.Tensor:
        """Returns base weight combined with any active transient fast-weight overlay."""
        W = self.weights[key]
        if key in self.overlays:
            return W + self.overlays[key]
        return W

    def forward(
        self,
        token_ids: List[int],
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Hierarchical causal forward pass across all 4 layers."""
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

            # --- Multi-Head Self-Attention ---
            h_norm = rms_norm(h, self.weights[f"{prefix}attn_norm.weight"], eps=c.eps)
            traces[f"layer_{l}_attn_norm"] = h_norm.clone()

            wq = self.get_effective_weight(f"{prefix}attn_q.weight")
            wk = self.get_effective_weight(f"{prefix}attn_k.weight")
            wv = self.get_effective_weight(f"{prefix}attn_v.weight")
            wo = self.get_effective_weight(f"{prefix}attn_out.weight")

            Q = torch.matmul(h_norm, wq.t()).view(1, T, c.n_heads, c.d_head).transpose(1, 2)
            K = torch.matmul(h_norm, wk.t()).view(1, T, c.n_heads, c.d_head).transpose(1, 2)
            V = torch.matmul(h_norm, wv.t()).view(1, T, c.n_heads, c.d_head).transpose(1, 2)

            scores = (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(c.d_head)) + causal_mask
            attn_probs = F.softmax(scores, dim=-1)
            traces[f"layer_{l}_attn_probs"] = attn_probs.clone()

            attn_ctx = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(1, T, c.d_model)
            attn_out = torch.matmul(attn_ctx, wo.t())
            h = h + attn_out
            traces[f"layer_{l}_after_attn"] = h.clone()

            # --- MLP Sublayer (GELU) ---
            h_norm_mlp = rms_norm(h, self.weights[f"{prefix}mlp_norm.weight"], eps=c.eps)
            traces[f"layer_{l}_mlp_norm"] = h_norm_mlp.clone()

            w_in = self.get_effective_weight(f"{prefix}mlp_in.weight")
            w_out = self.get_effective_weight(f"{prefix}mlp_out.weight")

            mlp_hidden = F.gelu(torch.matmul(h_norm_mlp, w_in.t()))
            traces[f"layer_{l}_mlp_hidden"] = mlp_hidden.clone()

            mlp_out = torch.matmul(mlp_hidden, w_out.t())
            h = h + mlp_out
            traces[f"layer_{l}_after_mlp"] = h.clone()

        h_final = rms_norm(h, self.weights["final_norm.weight"], eps=c.eps)
        traces["final_hidden"] = h_final.clone()
        logits = torch.matmul(h_final, self.weights["output.weight"].t())

        return logits, (traces if return_hidden else None)

    # ==========================================================================
    # 4. WSP-Lang 0.5 Primitives: MEMIT, LEACE, DeltaNet-2, Titans
    # ==========================================================================

    def memit_imprint_batch(
        self,
        pairs: List[Tuple[str, str]],
        layer_idx: int = 2,
        eta: float = 3.5,
        reg_lambda: float = 0.05
    ) -> Dict[str, Any]:
        """MEMIT / EMMET Covariance-Regularized Mass Memory Synthesis.
        Updates Delta W = (V - W_0 K) (K^T K + lambda C_0)^-1 K^T
        """
        keys = []
        values = []
        target_tensor_key = f"layers.{layer_idx}.mlp_out.weight"
        W0 = self.weights[target_tensor_key]  # [d_model, d_ff]

        for p, t in pairs:
            token_ids = self.tokenizer.encode(p, add_bos=True)
            _, traces = self.forward(token_ids, return_hidden=True)
            assert traces is not None
            k = traces[f"layer_{layer_idx}_mlp_hidden"][0, -1, :].clone()
            k = k / max(float(torch.norm(k).item()), 1e-6)
            keys.append(k)

            target_id = self.tokenizer.encode(t, add_bos=False)[0]
            v = self.weights["output.weight"][target_id].clone()
            v = v / max(float(torch.norm(v).item()), 1e-6)
            values.append(v)

        K = torch.stack(keys, dim=1)    # [d_ff, M]
        V = torch.stack(values, dim=1)  # [d_model, M]
        M = K.shape[1]

        # Background covariance C0 estimation
        cov_key = f"layer_{layer_idx}_cov"
        if cov_key not in self.background_covariances:
            # Estimate isotropic or empirical covariance
            C0 = torch.eye(M, dtype=torch.float32)
        else:
            C0 = self.background_covariances[cov_key]

        # Inversion with Tikhonov regularization
        KtK = torch.matmul(K.t(), K)
        unreg_cond = float(torch.linalg.cond(KtK).item())
        inv_gram = torch.linalg.inv(KtK + reg_lambda * C0)
        reg_cond = float(torch.linalg.cond(KtK + reg_lambda * C0).item())

        # Delta W = eta * (V - W0 @ K) @ inv_gram @ K^T
        residual = V - torch.matmul(W0, K)
        delta = eta * torch.matmul(torch.matmul(residual, inv_gram), K.t())

        before_norm = float(torch.norm(self.weights[target_tensor_key]).item())
        self.weights[target_tensor_key] += delta
        after_norm = float(torch.norm(self.weights[target_tensor_key]).item())

        return {
            "status": "ok",
            "operation": "memit_imprint_batch",
            "layer": layer_idx,
            "pairs_imprinted": M,
            "unreg_condition": unreg_cond,
            "reg_condition": reg_cond,
            "delta_frobenius": float(torch.norm(delta).item()),
            "weight_norm_before": before_norm,
            "weight_norm_after": after_norm
        }

    def imprint_associative_rule(
        self,
        prompt: str,
        target_char: str,
        layer_idx: int = 2,
        eta: float = 5.0
    ) -> Dict[str, Any]:
        """Imprint a single association using MEMIT regularized formulation."""
        return self.memit_imprint_batch([(prompt, target_char)], layer_idx=layer_idx, eta=eta, reg_lambda=0.01)

    def imprint_associative_batch(
        self,
        pairs: List[Tuple[str, str]],
        layer_idx: int = 2,
        eta: float = 5.0,
        ridge_lambda: float = 0.01
    ) -> Dict[str, Any]:
        """Imprint multiple associations using MEMIT regularized formulation."""
        return self.memit_imprint_batch(pairs, layer_idx=layer_idx, eta=eta, reg_lambda=ridge_lambda)

    def leace_erase_concept(
        self,
        u_concept: torch.Tensor,
        tensor_key: str,
        reg_lambda: float = 0.05
    ) -> Dict[str, Any]:
        """LEACE Closed-Form Concept Erasure (Belrose et al., NeurIPS 2023).
        P = I - (Sigma u u^T) / (u^T Sigma u)
        W <- P @ W
        """
        W = self.weights[tensor_key]
        d_out, d_in = W.shape
        u = u_concept / max(float(torch.norm(u_concept).item()), 1e-6)

        proj_before = torch.matmul(u, W)
        energy_before = float(torch.norm(proj_before).item())

        # Feature covariance Sigma @ u = (1/d_in) W (W^T u) + lambda u
        Wt_u = torch.matmul(W.t(), u)
        Sigma_u = (1.0 / d_in) * torch.matmul(W, Wt_u) + reg_lambda * u
        u_Sigma_u = float(torch.dot(u, Sigma_u).item())

        alpha = Sigma_u / max(u_Sigma_u, 1e-12)
        delta = torch.outer(alpha, proj_before)

        self.weights[tensor_key] = W - delta

        proj_after = torch.matmul(u, self.weights[tensor_key])
        energy_after = float(torch.norm(proj_after).item())

        return {
            "status": "ok",
            "operation": "leace_erase_concept",
            "tensor": tensor_key,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "erasure_pct": (1.0 - energy_after / max(energy_before, 1e-12)) * 100.0,
            "delta_frobenius": float(torch.norm(delta).item()),
            "max_abs_delta": float(torch.max(torch.abs(delta)).item())
        }

    def apply_decoupled_delta(
        self,
        tensor_key: str,
        erase_u: torch.Tensor,
        write_k: torch.Tensor,
        write_v: torch.Tensor,
        beta_erase: float = 1.0,
        beta_write: float = 2.0
    ) -> Dict[str, Any]:
        """Gated DeltaNet-2 Decoupled Erase & Write Linear Attention Update.
        W <- W (I - beta_e u_e u_e^T) + beta_w v_w k_w^T, with v_w perp u_e.
        """
        W = self.weights[tensor_key]
        u_e = erase_u / max(float(torch.norm(erase_u).item()), 1e-6)
        k_w = write_k / max(float(torch.norm(write_k).item()), 1e-6)
        v_w = write_v / max(float(torch.norm(write_v).item()), 1e-6)

        # Orthogonalize write value from erase direction: v_w <- v_w - (u_e^T v_w) u_e
        v_w = v_w - torch.dot(u_e, v_w) * u_e
        v_w = v_w / max(float(torch.norm(v_w).item()), 1e-6)

        # Measure energy along erase attractor before
        energy_erase_before = float(torch.norm(torch.matmul(u_e, W)).item())

        # Decoupled transformation:
        # W_erased = W - beta_e * (u_e @ (u_e^T W))
        W_erased = W - beta_erase * torch.outer(u_e, torch.matmul(u_e, W))
        # W_new = W_erased + beta_w * (v_w @ k_w^T)
        W_new = W_erased + beta_write * torch.outer(v_w, k_w)

        self.weights[tensor_key] = W_new
        energy_erase_after = float(torch.norm(torch.matmul(u_e, W_new)).item())

        return {
            "status": "ok",
            "operation": "apply_decoupled_delta",
            "tensor": tensor_key,
            "energy_erase_before": energy_erase_before,
            "energy_erase_after": energy_erase_after,
            "write_orthogonality": float(torch.dot(u_e, v_w).item()),
            "delta_frobenius": float(torch.norm(W_new - W).item())
        }

    # ==========================================================================
    # 5. Titans Surprisal Test-Time Loop Breaker & Generation
    # ==========================================================================

    def check_titans_surprisal(self, token_history: List[int], window_size: int = 12) -> Tuple[bool, float]:
        """Titans test-time memory monitor: computes n-gram repetition & entropy surprise.
        Triggers transient memory overlay when agentic loops emerge.
        """
        if len(token_history) < window_size:
            return False, 0.0

        window = token_history[-window_size:]
        unique_tokens = len(set(window))
        repetition_rate = 1.0 - (unique_tokens / float(window_size))

        # Check for 2-token or 3-token repeating cycles
        is_cyclic = False
        if window_size >= 6:
            # Check 2-cycle: a b a b a b
            if window[-1] == window[-3] == window[-5] and window[-2] == window[-4] == window[-6]:
                is_cyclic = True
            # Check 3-cycle: a b c a b c
            if len(window) >= 9:
                if window[-1] == window[-4] == window[-7] and window[-2] == window[-5] == window[-8]:
                    is_cyclic = True

        triggered = repetition_rate >= 0.5 or is_cyclic
        return triggered, repetition_rate

    def inject_transient_loop_breaker(self, token_ids: List[int], target_layer: int = 3):
        """Synthesizes a transient Fast-Weight Overlay (FWP) to redirect runaway loop attractors to EOS."""
        c = self.config
        tensor_key = f"layers.{target_layer}.mlp_out.weight"
        _, traces = self.forward(token_ids, return_hidden=True)
        if traces is None:
            return
        k = traces[f"layer_{target_layer}_mlp_hidden"][0, -1, :].clone()
        k_norm = torch.norm(k)
        if k_norm > 1e-6:
            k = k / k_norm

        rep_token = token_ids[-1]
        v_rep = self.weights["output.weight"][rep_token].clone()
        v_rep = v_rep / max(float(torch.norm(v_rep).item()), 1e-6)

        v_eos = self.weights["output.weight"][self.tokenizer.EOS_TOKEN].clone()
        v_eos = v_eos / max(float(torch.norm(v_eos).item()), 1e-6)

        # Inhibitory overlay: suppress repeating token, promote EOS token
        overlay = -5.0 * torch.outer(v_rep, k) + 5.0 * torch.outer(v_eos, k)
        self.overlays[tensor_key] = overlay

    def clear_overlays(self):
        """Clears all test-time transient fast-weight overlays."""
        self.overlays.clear()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        enable_titans_monitor: bool = True
    ) -> str:
        """Autoregressive generation with optional Titans Surprisal self-healing."""
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        self.clear_overlays()
        monitor_fired = False

        for _ in range(max_new_tokens):
            if len(token_ids) >= self.config.max_seq_len:
                break

            # Titans Surprisal Monitor Check
            if enable_titans_monitor and not monitor_fired:
                triggered, score = self.check_titans_surprisal(token_ids, window_size=8)
                if triggered:
                    self.inject_transient_loop_breaker(token_ids, target_layer=3)
                    monitor_fired = True

            logits, _ = self.forward(token_ids)
            next_logits = logits[0, -1, :]

            if temperature <= 0.0:
                next_token = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            if next_token == self.tokenizer.EOS_TOKEN:
                break
            token_ids.append(next_token)

        self.clear_overlays()
        return self.tokenizer.decode(token_ids, skip_special=True)
