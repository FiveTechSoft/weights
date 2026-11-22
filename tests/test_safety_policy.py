"""Regression tests: one canonical forbidden-tensor policy across every write path.

The engine (engine/wsplang.py), the CLI (wsp.py) and the MCP poke tool
(engine/wsp_mcp_server.py) must reject the same tensors. Before the unified
engine/safety.py, the three paths disagreed: e.g. model.embed_tokens.weight
was blocked by the engine but allowed by the CLI and the poke tool.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.safety as safety
import engine.wsplang as wsplang
import wsp
import engine.wsp_mcp_server as mcp_server


class TestSafetyPolicy(unittest.TestCase):

    def test_canonical_policy_is_superset_of_old_surfaces(self):
        """The canonical set covers what every pre-unification surface forbade."""
        for t in ("token_embd.weight", "output.weight", "output_norm.weight",
                  "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
            self.assertIn(t, safety.FORBIDDEN_TENSORS)
        for p in ("embed", "lm_head", "norm.weight", "output_norm", "wte", "ln_f",
                  "token_types", ".norm"):
            self.assertIn(p, safety.FORBIDDEN_PATTERNS)

    def test_protects_embeddings_the_cli_used_to_allow(self):
        for surface_name, blocked in (("engine", wsplang.is_protected_tensor("model.embed_tokens.weight")),
                                      ("wsp", any(p in "model.embed_tokens.weight" for p in wsp.FORBIDDEN_PATTERNS)),
                                      ):
            self.assertTrue(blocked, f"{surface_name} no bloquea model.embed_tokens.weight")

    def test_engine_unlock_flow_still_works(self):
        self.assertFalse(wsplang.is_protected_tensor("lm_head.weight", unlock=("lm_head.weight",)))
        self.assertFalse(safety.is_protected_tensor("lm_head.weight", unlock=("lm_head.weight",)))

    def test_cli_check_tensors_rejects_protected(self):
        with self.assertRaises(SystemExit):
            wsp._check_tensors([{"type": "add-scalar", "tensor": "model.embed_tokens.weight", "value": 0.5}],
                               {"model.embed_tokens.weight": None})

    def test_mcp_poke_rejects_protected(self):
        res = mcp_server.poke_weight("", "model.embed_tokens.weight", 0, 0, 0.5)
        self.assertEqual(res["status"], "security_rejection")

    def test_mcp_poke_rejects_engine_patterns(self):
        for tensor in ("lm_head.weight", "blk.0.norm.weight", "x.ln_f.weight"):
            res = mcp_server.poke_weight("", tensor, 0, 0, 0.5)
            self.assertEqual(res["status"], "security_rejection", tensor)

    def test_ordinary_tensors_are_allowed(self):
        self.assertFalse(safety.is_protected_tensor("blk.5.ffn_down.weight"))
        self.assertFalse(wsplang.is_protected_tensor("blk.5.ffn_down.weight"))
        self.assertNotIn("blk.5.ffn_down.weight", safety.FORBIDDEN_TENSORS)


if __name__ == "__main__":
    unittest.main()
