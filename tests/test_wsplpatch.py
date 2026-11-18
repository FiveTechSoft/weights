"""engine/wsplpatch.py (lab/probes/column_edit/PLAN.md): .weightpatch.json v1.0 bound to the base tensor's sha256 and
to the delta's sha256, applied into a new safetensors dir (never over an input), a PatchManager with a session stack
on disk (apply pushes, rollback_last pops and returns the previous state dir, list_patches, commit_manifest bundle),
exact reconstruction from base + patches, and byte-equality checks. Written before the code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import wsplpatch as wp  # noqa: E402

T = "model.language_model.layers.0.mlp.down_proj.weight"
OTHER = "model.language_model.layers.0.mlp.up_proj.weight"


def make_base(d: Path, seed=0):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    (d / "base").mkdir()
    save_file({T: torch.randn(8, 16, generator=g).to(torch.bfloat16),
               OTHER: torch.randn(16, 8, generator=g).to(torch.bfloat16)},
              str(d / "base" / "model.safetensors"), metadata={"format": "pt"})
    (d / "base" / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    return d / "base"


def write_npz(path: Path, cols, V):
    np.savez(path, cols_0=np.asarray(cols), vec_0=np.asarray(V, dtype=np.float32))


class PatchFormatTest(unittest.TestCase):
    def test_fields_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); base = make_base(d)
            write_npz(d / "p.npz", [1], np.ones((8, 1), dtype=np.float32))
            p = wp.make_patch(base, T, [1], d / "p.npz", alpha=1.0, mode="add", direction_hash="abc",
                              normalize_direction=True, experiment_id="exp", notes="n")
            self.assertEqual(p["wsplang_patch_version"], "1.0")
            self.assertEqual(p["tensor_name"], T)
            self.assertEqual(p["op"], "edit_column")
            self.assertEqual(p["params"]["column"], [1])
            self.assertEqual(p["params"]["mode"], "add")
            self.assertEqual(p["params"]["direction_hash"], "abc")
            self.assertEqual(p["delta_compact"]["type"], "column_vector")
            self.assertEqual(p["delta_full_hash"], wp.sha256_file(d / "p.npz"))
            self.assertEqual(p["base_tensor_hash"], wp.tensor_hash(base, T))
            self.assertEqual(p["base_checkpoint_hash"], wp.sha256_file(base / "model.safetensors"))
            self.assertEqual(p["parent_patches"], [])
            for k in ("created", "experiment_id", "notes"):
                self.assertIn(k, p)


class PatchManagerTest(unittest.TestCase):
    def test_apply_rollback_reconstruct_commit_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); base = make_base(d)
            pm = wp.PatchManager(d / "session")
            state, patches = base, []
            for i in range(3):
                cols = [i, i + 5]
                write_npz(d / f"p{i}.npz", cols, np.full((8, 2), 0.5 * (i + 1), dtype=np.float32))
                p = wp.make_patch(state, T, cols, d / f"p{i}.npz", alpha=1.0, mode="add",
                                  parent_patches=[str(x) for x in patches])
                pp = d / f"p{i}.weightpatch.json"
                pp.write_text(json.dumps(p, indent=1), encoding="utf-8")
                patches.append(pp)
                state = pm.apply(pp, state, d / f"s{i}")
            self.assertEqual([Path(x["dst"]).name for x in pm.list_patches()], ["s0", "s1", "s2"])
            self.assertEqual(len(json.loads((d / "session" / "stack.json").read_text(encoding="utf-8"))["applied"]), 3)
            self.assertTrue(wp.tensors_equal(base, d / "s2", [OTHER]))          # untouched tensor travels unchanged
            self.assertTrue((d / "s2" / "config.json").exists())
            # A then B then rollback B equals the state after A, by hash; nothing is deleted
            h_after_1 = wp.tensor_hash(d / "s1", T)
            cur = pm.rollback_last()
            self.assertEqual(Path(cur).name, "s1")
            self.assertEqual(wp.tensor_hash(Path(cur), T), h_after_1)
            cur = pm.rollback_last()
            self.assertEqual(Path(cur).name, "s0")
            self.assertTrue((d / "s2" / "model.safetensors").exists())
            # the stack survives a process restart (re-open from disk)
            pm2 = wp.PatchManager(d / "session")
            self.assertEqual(len(pm2.list_patches()), 1)
            self.assertEqual(Path(pm2.current()).name, "s0")
            # reconstruct: base + first patch == s0; base + all == s2 (byte for byte)
            self.assertTrue(wp.tensors_equal(wp.reconstruct(base, patches[:1], d / "r1"), d / "s0"))
            r3 = wp.reconstruct(base, patches, d / "r3")
            self.assertTrue(wp.tensors_equal(r3, d / "s2"))
            self.assertFalse(wp.tensors_equal(r3, d / "s0"))
            # values after patch 0: base + 0.5 on columns 0 and 5 (bf16)
            expect = wp.read_tensor(base, T).float(); expect[:, [0, 5]] += 0.5
            self.assertTrue(torch.equal(wp.read_tensor(d / "s0", T), expect.to(torch.bfloat16)))
            # commit_manifest: bundle with base hash, ordered patches, final tensor hashes
            bundle = pm2.commit_manifest("demo", include_patches=True)
            b = json.loads(Path(bundle).read_text(encoding="utf-8"))
            self.assertEqual(b["name"], "demo")
            self.assertEqual(b["base_checkpoint_hash"], wp.sha256_file(base / "model.safetensors"))
            self.assertEqual(len(b["patches"]), 1)
            self.assertEqual(b["final_tensor_hashes"][T], wp.tensor_hash(d / "s0", T))
            self.assertIn("wsplang_patch_version", b["patches"][0])          # inlined when include_patches
            # guards: base-hash mismatch, delta tampering, writing over an input, empty rollback
            with self.assertRaises(wp.PatchError):
                pm2.apply(patches[2], base, d / "bad")
            write_npz(d / "p0.npz", [0, 5], np.zeros((8, 2), dtype=np.float32))
            with self.assertRaises(wp.PatchError):
                pm2.apply(patches[0], base, d / "bad2")
            with self.assertRaises(wp.PatchError):
                wp.apply_patch(patches[1], d / "s0", d / "s0")
            pm2.rollback_last()
            with self.assertRaises(wp.PatchError):
                pm2.rollback_last()
            self.assertFalse((d / "bad").exists() and (d / "bad" / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
