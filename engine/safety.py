"""Canonical forbidden-tensor policy shared by every WSP-Lang write path.

This is the single source of truth for which tensor names may never be edited,
shared by the CLI (wsp.py), the engine (engine/wsplang.py), the per-op handlers
(engine/wsplang_ops.py) and the MCP poke tool (engine/wsp_mcp_server.py).
Do not re-declare these lists anywhere else: import them from this module.

Policy: embeddings, the readout head and final/embedding norms change only when
a program unlocks them by name (``unlock`` parameter, see ``is_protected_tensor``).
"""
from __future__ import annotations

FORBIDDEN_TENSORS = frozenset({
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
})

FORBIDDEN_PATTERNS = (
    "embed",
    "lm_head",
    "norm.weight",
    "output_norm",
    "wte",
    "ln_f",
    "token_types",
    ".norm",
)


def is_protected_tensor(name, unlock=()):
    """True when a write to ``name`` would break the WSP-Lang safety policy.

    ``unlock`` holds tensor names a program explicitly listed in ``unlock_tensors``;
    those are the only protected tensors allowed through by name.
    """
    if name in unlock:
        return False
    return name in FORBIDDEN_TENSORS or any(pat in name for pat in FORBIDDEN_PATTERNS)
