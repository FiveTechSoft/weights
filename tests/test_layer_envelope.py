"""test_layer_envelope.py — Unit test for WSP-Lang 0.7 layer_envelope operator.
"""
import unittest
from pathlib import Path
import tempfile
import gc
import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent

from engine.wsplang import load_vector_numpy, execute_plan_v4


class TestLayerEnvelope(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)
        # Create dummy vector
        self.vec_file = self.tmppath / "dummy_u.npz"
        dummy_vec = np.random.randn(256).astype(np.float32)
        dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
        np.savez(self.vec_file, u=dummy_vec)

    def tearDown(self):
        gc.collect()
        try:
            self.tmpdir.cleanup()
        except Exception:
            pass

    def test_layer_envelope_gaussian_execution(self):
        out_gguf = self.tmppath / "envelope_test.gguf"
        plan = {
            "pasos": [
                {
                    "op": "layer_envelope",
                    "args": f"10..22 peak=16 bandwidth=2.5 envelope=gaussian vector={self.vec_file} out_gguf={out_gguf} n_layers=32"
                }
            ]
        }
        prog = {"__root__": [("model", "MiMo-9B")]}
        results = execute_plan_v4(plan, prog, model_path=str(self.tmppath))
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["execution_state"], "envelope_computed")
        self.assertTrue(res["result"]["gguf_exported"])
        self.assertEqual(res["result"]["peak_layer"], 16)
        
        # Verify envelope weights peak at 16
        weights = res["result"]["weights_distribution"]
        self.assertEqual(weights[16], 1.0)
        self.assertGreater(weights[16], weights[15])
        self.assertGreater(weights[15], weights[14])
        self.assertAlmostEqual(weights[15], weights[17], places=4)

        # Verify GGUF output (1-based tensor indexing for llama.cpp: direction.1..direction.32)
        self.assertTrue(out_gguf.exists())
        reader = gguf.GGUFReader(str(out_gguf))
        self.assertEqual(len(reader.tensors), 32)
        # direction.17 corresponds to layer_idx=16 (peak)
        t_peak = [t for t in reader.tensors if t.name == "direction.17"][0]
        # direction.1 corresponds to layer_idx=0 (zero scale)
        t_base = [t for t in reader.tensors if t.name == "direction.1"][0]
        self.assertGreater(np.linalg.norm(t_peak.data), 0.9)
        self.assertEqual(np.linalg.norm(t_base.data), 0.0)
        del reader
        gc.collect()

    def test_layer_envelope_cosine_execution(self):
        plan = {
            "pasos": [
                {
                    "op": "layer_envelope",
                    "args": f"12..20 peak=16 envelope=cosine vector={self.vec_file}"
                }
            ]
        }
        prog = {"__root__": [("model", "MiMo-9B")]}
        results = execute_plan_v4(plan, prog, model_path=str(self.tmppath))
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["result"]["envelope_type"], "cosine")
        weights = res["result"]["weights_distribution"]
        self.assertAlmostEqual(weights[16], 1.0, places=4)
        self.assertAlmostEqual(weights[12], 0.0, places=4)
        self.assertAlmostEqual(weights[20], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
