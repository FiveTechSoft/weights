"""Test apply_wsp_to_safetensors on a small synthetic SafeTensors shard."""
import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import save_file, load_file

from scripts.apply_wsp_to_safetensors import apply_wsp_edit


class TestSafeTensorsEdit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()

        # Create dummy weights: hidden_dim=64, inter_dim=128
        torch.manual_seed(42)
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        save_file({"model.layers.14.mlp.down_proj.weight": w}, os.path.join(self.test_dir, "model.safetensors"))

        # Create dummy vector
        np.random.seed(42)
        u = np.random.randn(64).astype(np.float32)
        u = u / np.linalg.norm(u)
        self.vec_path = os.path.join(self.test_dir, "u_test.npz")
        np.savez(self.vec_path, u=u)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_apply_edit_modifies_only_k_columns(self):
        apply_wsp_edit(
            model_dir=self.test_dir,
            out_dir=self.out_dir,
            layer_idx=14,
            vector_path=self.vec_path,
            k=8,
            target_frac=0.04
        )

        orig_w = load_file(os.path.join(self.test_dir, "model.safetensors"))["model.layers.14.mlp.down_proj.weight"].to(torch.float32)
        new_w = load_file(os.path.join(self.out_dir, "model.safetensors"))["model.layers.14.mlp.down_proj.weight"].to(torch.float32)

        diff = (orig_w - new_w).abs()
        # Find which columns changed
        col_changed = (diff.sum(dim=0) > 1e-6).nonzero(as_tuple=True)[0]
        self.assertEqual(len(col_changed), 8, f"Expected exactly 8 columns to change, got {len(col_changed)}")

        # Check manifest was written
        manifest_path = os.path.join(self.out_dir, "wsp_edit_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))


if __name__ == "__main__":
    unittest.main()
