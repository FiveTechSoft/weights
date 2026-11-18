"""engine/ternary.py and the WSP-Lang op ternary_survival (lab/probes/ternary/PLAN.md): ternarisation under ggml's
TQ2_0 rule and BitNet's absmean rule, and how much of a small edit survives it. Written before the code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import ternary as tn  # noqa: E402
import engine.wsplang as wsplang  # noqa: E402


def gauss(shape, seed=0, std=0.02):
    return (np.random.default_rng(seed).normal(size=shape) * std).astype(np.float32)


class TernarizeTest(unittest.TestCase):
    def test_absmean_codes_scales_and_dequant(self):
        W = np.array([[0.1, -0.3, 0.0, 0.02], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        codes, scales = tn.ternarize(W, "absmean")
        self.assertEqual(codes.dtype, np.int8)
        self.assertTrue(set(np.unique(codes)) <= {-1, 0, 1})
        np.testing.assert_allclose(scales, [0.105, 1.0, 0.0], atol=1e-6)  # mean |w| per row; zero row stays zero
        np.testing.assert_array_equal(codes[0], [1, -1, 0, 0])             # 0.1/0.105 -> 1, -0.3 -> -1, 0.02/0.105 -> 0
        np.testing.assert_array_equal(codes[1], [1, 1, 1, 1])
        np.testing.assert_array_equal(codes[2], [0, 0, 0, 0])
        D = tn.dequant(codes, scales, "absmean")
        np.testing.assert_allclose(D[0], [0.105, -0.105, 0.0, 0.0], atol=1e-6)

    def test_tq2_0_matches_gguf_dequantize_and_is_absmax_per_block(self):
        from gguf import quants, GGMLQuantizationType as T
        W = gauss((4, 512), seed=1)
        codes, scales = tn.ternarize(W, "tq2_0")
        D = tn.dequant(codes, scales, "tq2_0")
        ref = quants.dequantize(quants.quantize(W, T.TQ2_0), T.TQ2_0)
        np.testing.assert_allclose(D, ref, atol=1e-6)
        self.assertEqual(scales.shape, (4, 2))                              # one scale per 256-element block
        self.assertTrue(np.all(np.abs(codes) <= 1))
        # codes = round(w / absmax) over the block: the block's largest entry is always +-1
        for r in range(4):
            for b in range(2):
                blk = W[r, b * 256:(b + 1) * 256]
                self.assertEqual(abs(int(codes[r, b * 256 + int(np.argmax(np.abs(blk)))])), 1)

    def test_rules_rejected(self):
        with self.assertRaises(ValueError):
            tn.ternarize(gauss((2, 256)), "int4")


class SurvivalTest(unittest.TestCase):
    def test_zero_edit_gives_zero_flips_and_zero_cos(self):
        W0 = gauss((8, 512), seed=2)
        s = tn.survival(W0, np.zeros_like(W0), "absmean", "requant", scales=(1,))
        self.assertEqual(s["cells"][0]["flip_touched"], 0.0)
        self.assertEqual(s["cells"][0]["cos"], 0.0)
        self.assertEqual(s["k70"], "> 1")

    def test_huge_edit_is_kept_and_k70_found(self):
        W0 = gauss((8, 512), seed=3)
        dW = gauss((8, 512), seed=4, std=0.002)  # a tenth of the weights: small at k = 1, dominant at k = 20
        s = tn.survival(W0, dW, "absmean", "requant", scales=(1, 20))
        self.assertGreater(s["cells"][1]["cos"], 0.8)
        self.assertGreater(s["cells"][1]["flip_touched"], 0.3)
        self.assertEqual(s["k70"], 20)
        f = tn.survival(W0, dW, "absmean", "fixed", scales=(1, 20))   # base scales: codes saturate at +-1, cos is capped
        self.assertGreater(f["cells"][1]["cos"], 0.6)
        self.assertLess(f["cells"][1]["cos"], s["cells"][1]["cos"])
        self.assertGreater(f["cells"][1]["flip_touched"], 0.3)

    def test_alpha_and_noise_definitions(self):
        dW = gauss((4, 8), seed=5)
        D = 2.0 * dW
        a, n, c = tn.overlap(D, dW)
        self.assertAlmostEqual(a, 2.0, places=5)
        self.assertAlmostEqual(n, 0.0, places=5)
        self.assertAlmostEqual(c, 1.0, places=5)

    def test_fixed_mode_keeps_base_scales(self):
        W0 = gauss((4, 256), seed=6)
        dW = gauss((4, 256), seed=7, std=0.5)
        c0, s0 = tn.ternarize(W0, "tq2_0")
        c1, s1 = tn.ternarize_fixed(W0 + dW, s0, "tq2_0")
        np.testing.assert_array_equal(s1, s0)
        self.assertTrue(np.all(np.abs(c1) <= 1))
        self.assertGreater(int((c1 != c0).sum()), 0)


class LoadersTest(unittest.TestCase):
    def test_npz_cols_to_dense(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.npz"
            np.savez(p, cols_0=np.array([1, 3]), vec_0=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                     cols_1=np.array([0]), vec_1=np.array([[5.0], [6.0]], dtype=np.float32))
            with np.load(p) as z:
                dw = tn.delta_from_npz(z, 0, (2, 4))
                np.testing.assert_array_equal(dw, [[0, 1, 0, 2], [0, 3, 0, 4]])
                dw1 = tn.delta_from_npz(z, 1, (2, 4))
                np.testing.assert_array_equal(dw1, [[5, 0, 0, 0], [6, 0, 0, 0]])

    def test_fit_to_dense_is_outer_v_u(self):
        fit = {"u": {3: np.array([1.0, 2.0, 3.0])}, "v": {3: np.array([1.0, -1.0])}}
        dw = tn.delta_from_fit(fit, 3)
        np.testing.assert_array_equal(dw, [[1, 2, 3], [-1, -2, -3]])  # rows = output dim (v), cols = neurons (u)


class WsplangOpTest(unittest.TestCase):
    def test_registered_as_read_only(self):
        self.assertIn("ternary_survival", wsplang.OPS_ALLOWED)
        self.assertIn("ternary_survival", wsplang.READ_ONLY_OPS)

    def test_report_on_tiny_safetensors(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors / torch not installed")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "m").mkdir()
            W = torch.from_numpy(gauss((8, 512), seed=8))
            save_file({"model.language_model.layers.0.mlp.down_proj.weight": W}, str(d / "m" / "model.safetensors"))
            np.savez(d / "e.npz", cols_0=np.arange(4), vec_0=gauss((8, 4), seed=9, std=0.2))
            rep = tn.report(str(d / "m"), str(d / "e.npz"), layers=[0], rule="absmean", mode="requant",
                            scales=(1, 5), out=str(d / "r.json"))
            saved = json.loads((d / "r.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["rule"], "absmean")
            self.assertEqual(list(saved["tensors"]), ["model.language_model.layers.0.mlp.down_proj.weight"])
            cell = saved["tensors"]["model.language_model.layers.0.mlp.down_proj.weight"]["cells"][0]
            self.assertEqual(cell["k"], 1)
            for key in ("flip_touched", "cos", "alpha", "noise"):
                self.assertIn(key, cell)
            self.assertEqual(rep["tensors"][list(rep["tensors"])[0]]["k70"], saved["tensors"][list(saved["tensors"])[0]]["k70"])


if __name__ == "__main__":
    unittest.main()
