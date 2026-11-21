"""Unit tests for wsp_mcp_server: validates all registered MCP tools, security gates, inline code execution, and WSP-Lang 0.6/0.7 NanoWSP tools."""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import wsp_mcp_server as server


class TestWspMcpServer(unittest.TestCase):

    def test_registered_tools_count(self):
        """Verify that all core MCP tools including WSP-Lang 0.6/0.7 additions are exposed."""
        tools = [
            "run_challenge", "inspect_feedback", "experiment_history",
            "sell_holdout", "record_verdict", "engine_status",
            "peek_weight", "poke_weight", "get_semantic_areas",
            "run_wsplang", "execute_wsplang_code", "run_agent_loop_probe",
            "run_logic_probe",
            # WSP-Lang 0.6/0.7 & NanoWSP Additions:
            "run_nanowsp_benchmark", "query_nanowsp3",
            "run_cross_model_procrustes", "run_causal_contract"
        ]
        for t in tools:
            self.assertTrue(hasattr(server, t), f"MCP tool function '{t}' should exist in server")

    def test_poke_security_rejection(self):
        """poke_weight must reject attempts on forbidden tensors directly at MCP level."""
        res = server.poke_weight("dummy.gguf", "output.weight", 0, 0, 1.0)
        self.assertEqual(res.get("status"), "security_rejection")
        self.assertIn("forbidden", res.get("error", "").lower())

        res_norm = server.poke_weight("dummy.gguf", "blk.0.attn_norm.weight", 0, 0, 1.0)
        self.assertEqual(res_norm.get("status"), "security_rejection")

    def test_execute_wsplang_code_validation_gate(self):
        """execute_wsplang_code must reject invalid code without executing."""
        invalid_wspl = """
        experiment broken:
            step totally_invalid_op_xyz 123
        """
        res = server.execute_wsplang_code(invalid_wspl)
        self.assertFalse(res.get("valid"))
        self.assertTrue(len(res.get("errors", [])) > 0)
        exec_step = res.get("ejecucion", [{}])[0]
        self.assertEqual(exec_step.get("status"), "aborted")

    @unittest.skipUnless((ROOT / "data" / "nanowsp3_weights.safetensors").exists(),
                         "data/nanowsp3_weights.safetensors is local (not in git): without it the tool answers from "
                         "an untrained model (CI: 'å' instead of 'A')")
    def test_query_nanowsp3_tool(self):
        """query_nanowsp3 must return accurate next token prediction."""
        res = server.query_nanowsp3("SCI: Z79=")
        self.assertIn("next_token_pred", res)
        self.assertEqual(res["next_token_pred"], "A")
        self.assertGreater(res["confidence_pct"], 90.0)


if __name__ == "__main__":
    unittest.main()
