"""lab/probes/cc_phi/scale_vector.py: a control-vector GGUF multiplied by a scale, written as a new GGUF. The compile
op writes the vector at x1.0 (cc_weights/PLAN.md); Phi's chosen scale is 0.5, so the op gets the scaled file."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestScaleVector(unittest.TestCase):
    def test_scaled_copy_has_every_layer_times_scale_and_keeps_the_hint(self):
        try:
            import gguf  # noqa: F401
        except ImportError:
            self.skipTest("gguf not installed")
        from engine import contrast_vector as cvx
        from engine.cc_compile import read_vector
        from lab.probes.cc_phi.scale_vector import scale_vector
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "v.gguf", Path(d) / "v_x0.5.gguf"
            vecs = {1: np.array([2.0, -4.0, 6.0], dtype=np.float32), 2: np.array([1.0, 1.0, 1.0], dtype=np.float32)}
            cvx.write_gguf(vecs, src, "phi3")
            meta = scale_vector(src, dst, 0.5)
            got = read_vector(dst)
            self.assertEqual(sorted(got), [1, 2])
            np.testing.assert_allclose(got[1].numpy(), [1.0, -2.0, 3.0])
            np.testing.assert_allclose(got[2].numpy(), [0.5, 0.5, 0.5])
            self.assertEqual(meta["scale"], 0.5)
            self.assertEqual(meta["layers"], 2)
            self.assertAlmostEqual(meta["norms"]["1"], float(np.linalg.norm([1.0, -2.0, 3.0])), places=3)
            self.assertEqual(meta["model_hint"], "phi3")
            import gc
            r = gguf.GGUFReader(str(dst))
            hint = [f for f in r.fields.values() if f.name == "controlvector.model_hint"][0]
            self.assertEqual(bytes(hint.parts[-1]).decode(), "phi3")
            del r, hint
            gc.collect()                       # the reader memory-maps the file; Windows keeps it locked while mapped

    def test_existing_output_is_never_overwritten(self):
        try:
            import gguf  # noqa: F401
        except ImportError:
            self.skipTest("gguf not installed")
        from engine import contrast_vector as cvx
        from lab.probes.cc_phi.scale_vector import scale_vector
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "v.gguf", Path(d) / "out.gguf"
            cvx.write_gguf({1: np.ones(3, dtype=np.float32)}, src, "phi3")
            dst.write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                scale_vector(src, dst, 0.5)


if __name__ == "__main__":
    unittest.main()
