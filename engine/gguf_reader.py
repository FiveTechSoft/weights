"""engine/gguf_reader.py - a local GGUF file behind the reader interface of engine/weight_diff.py, and an HF-layout
view of it, so model_report (and anything that takes a reader) can compare a GGUF model with a safetensors base.

  LocalGGUF(path)         keys() / shape(name) / get(name) / get_rows(name, start, count) / close(), shapes in torch
                          order (GGUF stores ne reversed); unquantised tensors come back in the file's dtype (BF16,
                          F16, F32), quantised ones (Q6_K, Q8_0, Q4_K_M, ...) dequantised to float32 by gguf-py's
                          gguf.quants.dequantize, row blocks only for get_rows.
  open_gguf_as_hf(path)   the same file seen with Hugging Face names and layout:
                            names      token_embd -> model.embed_tokens, blk.N.* -> model.layers.L.* (L = the rank
                                       of N among the non-MTP blocks: SenseNova's fork numbers blocks by execution
                                       index, 0-27 then 52-55), nextn.* -> mtp.* (hidden, like text_model_name does
                                       on the safetensors side), unknown names hidden (mHC routing, rope_freqs);
                            fused      attn_qkv_gate = [in_proj_qkv ; in_proj_z] and ssm_beta_alpha = [in_proj_b ;
                                       in_proj_a] (SenseNova's fork) split back into their HF tensors;
                            values     conv1d back to (C, 1, K); A_log = log(-ssm_a); the attention / MLP / q, k and
                                       final norms minus 1 for architectures whose RMSNorm scales by (1 + w)
                                       (Qwen3.5, Qwen3-Next, Gemma, SenseNova-Loop: llama.cpp stores w + 1);
                            heads      the linear-attention value heads back in HF order: llama.cpp interleaves them
                                       by key group (GGUF head n = HF head value_head_order(n_v, n_k)[n]) on ssm_a,
                                       ssm_dt.bias, in_proj_a / in_proj_b rows, in_proj_z rows, the v rows of
                                       in_proj_qkv and conv1d, and the columns of out_proj.
                          Geometry (value heads, key heads, head dim) is read from each block's tensor shapes, never
                          from a config.

Verified on hf_cache/h4/A0-Q6_K.gguf against hf_cache/qwen35-9b-hf (tests/test_gguf_reader.py): all 427 text-model
tensors map to an HF tensor of the same shape, F32 tensors are equal after bf16 rounding, Q6_K matrices have cosine
> 0.999 with the HF ones once the head order is undone. Through model_report.fingerprint the quantised matrices show
rel_delta 0.018 (the Q6_K floor) and every F32 tensor 0.

Windows: the GGUF is memory-mapped; call close() (or drop the object and gc.collect()) before renaming or deleting
the file.
"""
import gc
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

import gguf

from engine import weight_diff as wd

_F32, _F16, _BF16 = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
_PLAIN = {_F32: torch.float32, _F16: torch.float16, _BF16: torch.bfloat16}


class LocalGGUF:
    """A GGUF file as a weight_diff-style reader (torch-order shapes, lazy per-tensor reads)."""

    def __init__(self, path):
        self.path = Path(path)
        self._reader = gguf.GGUFReader(str(self.path))
        self._t = {t.name: t for t in self._reader.tensors}
        self.architecture = self._field_str("general.architecture")

    def _field_str(self, key: str) -> Optional[str]:
        f = self._reader.fields.get(key)
        if f is None:
            return None
        try:
            v = f.contents()
            return v if isinstance(v, str) else bytes(f.parts[-1]).decode("utf-8", "replace")
        except Exception:
            return bytes(f.parts[-1]).decode("utf-8", "replace")

    # -- interface -------------------------------------------------------------------------------------------------
    def keys(self) -> List[str]:
        return sorted(self._t)

    def shape(self, name: str) -> tuple:
        return tuple(int(v) for v in reversed(self._t[name].shape))

    def tensor_type(self, name: str) -> gguf.GGMLQuantizationType:
        return self._t[name].tensor_type

    def dtype(self, name: str) -> torch.dtype:
        """The dtype get() returns: the file's for F32 / F16 / BF16, float32 for anything quantised."""
        return _PLAIN.get(self._t[name].tensor_type, torch.float32)

    def get(self, name: str) -> torch.Tensor:
        t = self._t[name]
        return self._to_torch(t, np.asarray(t.data), self.shape(name))

    def get_rows(self, name: str, start: int, count: int) -> torch.Tensor:
        """Rows start .. start+count-1 (clipped) of the torch-order tensor; only those bytes are touched."""
        t = self._t[name]
        shape = self.shape(name)
        n = shape[0] if shape else 1
        start, stop = max(0, start), min(n, start + count)
        sub = (max(stop - start, 0),) + tuple(shape[1:])
        if stop <= start:
            return torch.empty(sub, dtype=self.dtype(name))
        tt = t.tensor_type
        if tt in _PLAIN:
            if tt == _BF16:
                data = np.ascontiguousarray(t.data).view(np.int16).reshape(shape)[start:stop]
            else:
                data = np.asarray(t.data).reshape(shape)[start:stop]
            return self._to_torch(t, data, sub)
        if len(shape) >= 2:                       # quantised: data is (rows, bytes per row); dequantise the block
            return self._to_torch(t, np.asarray(t.data)[start:stop], sub)
        return self.get(name)[start:stop]         # 1-D quantised (rare): whole tensor, then slice

    def close(self):
        self._t = {}
        self._reader = None
        gc.collect()

    # -- helpers ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _to_torch(t, data: np.ndarray, shape: tuple) -> torch.Tensor:
        tt = t.tensor_type
        if tt == _BF16:
            raw = np.ascontiguousarray(data)
            if raw.dtype != np.int16:
                raw = raw.view(np.int16)
            return torch.from_numpy(raw.reshape(shape).copy()).view(torch.bfloat16)
        if tt in (_F32, _F16):
            return torch.from_numpy(np.array(data).reshape(shape))
        return torch.from_numpy(np.array(gguf.quants.dequantize(np.ascontiguousarray(data), tt),
                                         dtype=np.float32).reshape(shape))


# -- names ---------------------------------------------------------------------------------------------------------
_TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight"}
_LAYER = {
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    # Qwen3.5 / Qwen3-Next Gated DeltaNet layers, as llama.cpp's converter names them
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight", "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log", "ssm_norm.weight": "linear_attn.norm.weight",
}
# SenseNova's fork fuses two pairs; the parts are listed in stored row order and split by geometry
_FUSED = {"attn_qkv_gate.weight": ("attn_qkv.weight", "attn_gate.weight"),
          "ssm_beta_alpha.weight": ("ssm_beta.weight", "ssm_alpha.weight")}
_NEXTN = {"nextn.eh_proj.weight": "mtp.fc.weight", "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
          "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight", "nextn.shared_head_norm.weight": "mtp.norm.weight"}
_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def mtp_blocks(names: Iterable[str]) -> Set[int]:
    """Block indices that hold a nextn.* tensor (llama.cpp stores Qwen3.5's MTP head as one extra block)."""
    out = set()
    for n in names:
        m = _BLK.match(n)
        if m and m.group(2).startswith("nextn."):
            out.add(int(m.group(1)))
    return out


def layer_map(names: Iterable[str]) -> Dict[int, int]:
    """GGUF block index -> HF layer index: the non-MTP blocks in order (identity for the usual 0..n-1 numbering;
    SenseNova's fork numbers by execution index, 0-27 and 52-55, and its nextn block is 56)."""
    names = list(names)
    mb = mtp_blocks(names)
    blocks = sorted({int(m.group(1)) for m in (_BLK.match(n) for n in names) if m} - mb)
    return {b: i for i, b in enumerate(blocks)}


def gguf_to_hf_name(name: str, mtp_blocks: Iterable[int] = (), layers: Optional[Dict[int, int]] = None) -> Optional[str]:
    """The Hugging Face (Qwen3.5 layout) name of a GGUF tensor, or None when it has no HF counterpart. Fused tensors
    (attn_qkv_gate, ssm_beta_alpha) have two counterparts and are handled by GGUFAsHF, not here."""
    if name in _TOP:
        return _TOP[name]
    m = _BLK.match(name)
    if not m:
        return None
    b, rest = int(m.group(1)), m.group(2)
    if rest.startswith("nextn."):
        return _NEXTN.get(rest)
    if rest not in _LAYER:
        return None
    if b in set(mtp_blocks):
        return "mtp.layers.0." + _LAYER[rest]
    return f"model.layers.{layers.get(b, b) if layers else b}." + _LAYER[rest]


_PLUS_ONE_NORM_ARCHS = ("qwen35", "qwen3next", "gemma", "sensenova_loop")
_OFFSET_NORMS = ("attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "attn_q_norm.weight",
                 "attn_k_norm.weight")                     # + output_norm.weight; not ssm_norm (stored as is)


def norms_stored_plus_one(arch: Optional[str]) -> bool:
    """Architectures whose RMSNorm scales by (1 + w): llama.cpp's converter stores w + 1 for the attention, MLP,
    q/k and final norms (checked on Qwen3.5-9B: GGUF attn_norm = HF input_layernorm + 1 exactly; linear_attn.norm
    is stored unchanged; SenseNova-Loop-9B's BF16 GGUF follows the same convention)."""
    return bool(arch) and arch.startswith(_PLUS_ONE_NORM_ARCHS)


def value_head_order(n_v_heads: int, n_k_heads: int) -> List[int]:
    """GGUF value-head n holds HF value-head order[n]: llama.cpp interleaves the value heads so that the heads sharing
    a key head are spread (HF head g*r + j -> GGUF head j*n_k + g, with r = n_v / n_k heads per key head)."""
    if n_k_heads <= 0 or n_v_heads % n_k_heads:
        raise ValueError(f"value heads {n_v_heads} not a multiple of key heads {n_k_heads}")
    r = n_v_heads // n_k_heads
    return [(n % n_k_heads) * r + n // n_k_heads for n in range(n_v_heads)]


# -- the HF-layout view --------------------------------------------------------------------------------------------
class GGUFAsHF:
    """A LocalGGUF seen with HF names, HF shapes and HF head order. Same interface as weight_diff.Renamed, plus
    geometry(block), architecture, norm_offset and close()."""

    _ROWS_HEAD = ("ssm_a", "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight")   # one row per value head
    _ROWS_BLOCK = ("attn_gate.weight",)                                            # head_dim rows per value head
    _ROWS_V_TAIL = ("attn_qkv.weight", "ssm_conv1d.weight")                        # [q | k | v]: v rows permuted
    _COLS_BLOCK = ("ssm_out.weight",)                                              # columns = value heads x head_dim

    def __init__(self, g: LocalGGUF, norm_offset: Optional[bool] = None):
        self._g = g
        self.architecture = g.architecture
        self.norm_offset = norms_stored_plus_one(g.architecture) if norm_offset is None else bool(norm_offset)
        names = g.keys()
        mb = mtp_blocks(names)
        self._layers = layer_map(names)
        self._blocks: Dict[int, Set[str]] = {}
        for n in names:
            m = _BLK.match(n)
            if m:
                self._blocks.setdefault(int(m.group(1)), set()).add(m.group(2))
        self._geom: Dict[int, Optional[dict]] = {}
        # hf name -> (stored name, effective part name, block, (row start, row stop) or None)
        self._map: Dict[str, Tuple[str, str, int, Optional[Tuple[int, int]]]] = {}
        self._hidden: List[str] = []
        for n in names:
            m = _BLK.match(n)
            rest, block = (m.group(2), int(m.group(1))) if m else (n, -1)   # top-level: the part is the name itself
            if rest in _FUSED:
                parts = self._split(block, rest)
                if parts is None:
                    self._hidden.append(n)
                    continue
                for part, rows in parts:
                    self._add(gguf_to_hf_name(f"blk.{block}.{part}", mb, self._layers), n, part, block, rows)
                continue
            h = gguf_to_hf_name(n, mb, self._layers)
            if h is None:
                self._hidden.append(n)
                continue
            self._add(h, n, rest, block, None)

    def _add(self, h: Optional[str], stored: str, part: str, block: int, rows):
        h = wd.text_model_name(h) if h else None
        if h is None:
            if stored not in self._hidden:
                self._hidden.append(stored)
            return
        if h in self._map:
            raise ValueError(f"two tensors renamed to {h}: {self._map[h][0]}, {stored}")
        self._map[h] = (stored, part, block, rows)

    def _split(self, block: int, fused: str):
        geo = self.geometry(block)
        if geo is None:
            return None
        n_v, hd = geo["n_v_heads"], geo["head_dim"]
        rows = self._g.shape(f"blk.{block}.{fused}")[0]
        if fused == "attn_qkv_gate.weight":
            z = n_v * hd
            return [("attn_qkv.weight", (0, rows - z)), ("attn_gate.weight", (rows - z, rows))]
        return [("ssm_beta.weight", (0, n_v)), ("ssm_alpha.weight", (n_v, 2 * n_v))]

    # -- interface -------------------------------------------------------------------------------------------------
    def keys(self) -> List[str]:
        return sorted(self._map)

    def hidden(self) -> List[str]:
        return sorted(self._hidden)

    def original(self, name: str) -> str:
        return self._map[name][0]

    def shape(self, name: str) -> tuple:
        stored, part, block, rows = self._map[name]
        s = self._g.shape(stored)
        if rows is not None:
            s = (rows[1] - rows[0],) + tuple(s[1:])
        if part == "ssm_conv1d.weight" and len(s) == 2:
            return (s[0], 1, s[1])
        return s

    def get(self, name: str) -> torch.Tensor:
        stored, part, block, rows = self._map[name]
        x = self._g.get_rows(stored, rows[0], rows[1] - rows[0]) if rows else self._g.get(stored)
        return self._present(part, block, x)

    def get_rows(self, name: str, start: int, count: int) -> torch.Tensor:
        stored, part, block, rows = self._map[name]
        if self._permuted_rows(part, block) or rows is not None:
            x = self.get(name)
            n = x.shape[0]
            start, stop = max(0, start), min(n, start + count)
            return x[start:stop] if stop > start else x[:0]
        return self._present(part, block, self._g.get_rows(stored, start, count))

    def geometry(self, block: int) -> Optional[dict]:
        """Value heads, key heads and head dim of a linear-attention block, from its tensor shapes; None if the block
        has no complete linear-attention set (a full-attention layer, or a converter that names things differently)."""
        if block in self._geom:
            return self._geom[block]
        out = None
        parts = self._blocks.get(block, set())
        if "ssm_a" in parts:
            n_v = int(np.prod(self._g.shape(f"blk.{block}.ssm_a")))
            hd = qkv_rows = None
            if "attn_gate.weight" in parts and "attn_qkv.weight" in parts:
                z_rows = self._g.shape(f"blk.{block}.attn_gate.weight")[0]
                hd = z_rows // n_v if n_v and z_rows % n_v == 0 else None
                qkv_rows = self._g.shape(f"blk.{block}.attn_qkv.weight")[0]
            elif "attn_qkv_gate.weight" in parts and "ssm_norm.weight" in parts:
                hd = int(np.prod(self._g.shape(f"blk.{block}.ssm_norm.weight")))
                qkv_rows = self._g.shape(f"blk.{block}.attn_qkv_gate.weight")[0] - n_v * hd
            if hd and qkv_rows:
                kv = qkv_rows - n_v * hd
                if kv > 0 and kv % (2 * hd) == 0 and n_v % (kv // (2 * hd)) == 0:
                    out = {"n_v_heads": n_v, "n_k_heads": kv // (2 * hd), "head_dim": hd}
        self._geom[block] = out
        return out

    def close(self):
        self._g.close()

    # -- layout ----------------------------------------------------------------------------------------------------
    def _permuted_rows(self, part: str, block: int) -> bool:
        return part in self._ROWS_HEAD + self._ROWS_BLOCK + self._ROWS_V_TAIL and self.geometry(block) is not None

    def _inverse(self, block: int):
        """HF value head j is GGUF head inv[j] (the inverse of value_head_order)."""
        geo = self.geometry(block)
        order = value_head_order(geo["n_v_heads"], geo["n_k_heads"])
        inv = [0] * len(order)
        for n, j in enumerate(order):
            inv[j] = n
        return torch.tensor(inv), geo

    def _present(self, part: str, block: int, x: torch.Tensor) -> torch.Tensor:
        """A stored tensor (or a row block of one that is not head-permuted) -> HF layout."""
        if part == "ssm_a":
            x = torch.log(-x.float())
        if self.norm_offset and (part in _OFFSET_NORMS or part == "output_norm.weight"):
            x = x - 1
        geo = self.geometry(block) if block >= 0 else None
        if geo is not None and (self._permuted_rows(part, block) or part in self._COLS_BLOCK):
            inv, geo = self._inverse(block)
            hd, n_v = geo["head_dim"], geo["n_v_heads"]
            block_inv = (inv[:, None] * hd + torch.arange(hd)[None, :]).flatten()
            if part in self._ROWS_HEAD:
                x = x[inv]
            elif part in self._ROWS_BLOCK:
                x = x[block_inv]
            elif part in self._ROWS_V_TAIL:
                kv = x.shape[0] - n_v * hd
                x = torch.cat([x[:kv], x[kv:][block_inv]])
            elif part in self._COLS_BLOCK:
                x = x[:, block_inv]
        if part == "ssm_conv1d.weight" and x.dim() == 2:
            x = x.unsqueeze(1)
        return x


def open_gguf_as_hf(path, norm_offset: Optional[bool] = None) -> GGUFAsHF:
    """A local GGUF as an HF-layout reader (what model_report.reader_for should return for a .gguf path).
    norm_offset=None decides from general.architecture (norms_stored_plus_one); pass True / False to override."""
    return GGUFAsHF(LocalGGUF(path), norm_offset=norm_offset)
