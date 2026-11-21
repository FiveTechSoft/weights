"""wsp.py ops add-scalar / add-scalar-signed (lab/probes/dt_bias/PLAN.md): add a constant to 1-D F32 tensors matched by a
glob (the Gated DeltaNet time-step biases blk.*.ssm_dt.bias), and the placebo with a random sign per tensor from a seed.
Written before the code."""
import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wsp  # noqa: E402

BIAS = {0: [-3.0, -2.0, 0.5, 1.0], 1: [-4.0, -1.0, 0.0, 2.0], 2: [0.1, 0.2, 0.3, 0.4], 3: [-2.5, -2.5, -2.5, -2.5]}


def tiny_gguf(path):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_block_count(4)
    for L, v in BIAS.items():
        w.add_tensor(f"blk.{L}.ssm_dt.bias", np.array(v, dtype=np.float32))
    w.add_tensor("blk.0.ffn_down.weight", np.arange(16, dtype=np.float32).reshape(4, 4))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read_all(path):
    r = gguf.GGUFReader(str(path))
    out = {t.name: np.array(t.data, dtype=np.float32).copy() for t in r.tensors}
    del r
    gc.collect()
    return out


class AddScalarTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.src = self.d / "base.gguf"
        tiny_gguf(self.src)
        self._jrnl = wsp.PATH_JRNL
        wsp.PATH_JRNL = self.d / "journal.jsonl"

    def tearDown(self):
        wsp.PATH_JRNL = self._jrnl
        wsp._READER_CACHE.clear()
        gc.collect()
        shutil.rmtree(self.d, ignore_errors=True)

    def run_edit(self, ops, name):
        prog = self.d / f"{name}.json"
        out = self.d / f"{name}.gguf"
        prog.write_text(json.dumps({"out": str(out), "ops": ops}), encoding="utf-8")
        wsp.cmd_edit(argparse.Namespace(gguf=str(self.src), prog=str(prog), out=None))
        return out, json.loads(Path(str(out) + ".manifest.json").read_text(encoding="utf-8"))

    def test_add_scalar_on_glob_adds_to_every_matched_bias_and_nothing_else(self):
        out, man = self.run_edit([{"type": "add-scalar", "tensor": "blk.*.ssm_dt.bias", "value": 0.5}], "plus")
        got = read_all(out)
        for L, v in BIAS.items():
            np.testing.assert_allclose(got[f"blk.{L}.ssm_dt.bias"], np.array(v) + 0.5, atol=1e-6)
        np.testing.assert_array_equal(got["blk.0.ffn_down.weight"], np.arange(16, dtype=np.float32).reshape(4, 4))
        self.assertEqual(len(man["ops"]), 4)                      # the glob is expanded into one op per tensor
        self.assertTrue(all(o["type"] == "add-scalar" and o["value"] == 0.5 for o in man["ops"]))
        self.assertEqual(man["output_sha256"], hashlib.sha256(out.read_bytes()).hexdigest())

    def test_signed_placebo_is_plus_or_minus_value_per_tensor_and_deterministic(self):
        out1, man1 = self.run_edit([{"type": "add-scalar-signed", "tensor": "blk.*.ssm_dt.bias", "value": 0.5,
                                     "seed": 2026}], "pm1")
        out2, man2 = self.run_edit([{"type": "add-scalar-signed", "tensor": "blk.*.ssm_dt.bias", "value": 0.5,
                                     "seed": 2026}], "pm2")
        g1, g2 = read_all(out1), read_all(out2)
        signs = []
        for L, v in BIAS.items():
            d = g1[f"blk.{L}.ssm_dt.bias"] - np.array(v, dtype=np.float32)
            self.assertTrue(np.allclose(d, 0.5) or np.allclose(d, -0.5), f"layer {L}: {d}")
            signs.append(float(np.sign(d[0])))
            np.testing.assert_allclose(g1[f"blk.{L}.ssm_dt.bias"], g2[f"blk.{L}.ssm_dt.bias"])
        self.assertEqual([o["value"] for o in man1["ops"]], [0.5 * s for s in signs])
        self.assertTrue(all(o.get("signed_seed") == 2026 for o in man1["ops"]))

    def test_glob_without_match_and_quantised_target_are_refused(self):
        with self.assertRaises(SystemExit):
            self.run_edit([{"type": "add-scalar", "tensor": "blk.*.nothing.bias", "value": 0.5}], "nomatch")
        with self.assertRaises(SystemExit):
            self.run_edit([{"type": "add-scalar", "tensor": "blk.0.ssm_dt.bias", "value": 9.0}], "toolarge")

    def test_existing_output_is_never_overwritten(self):
        self.run_edit([{"type": "add-scalar", "tensor": "blk.0.ssm_dt.bias", "value": 0.5}], "once")
        with self.assertRaises(SystemExit):
            self.run_edit([{"type": "add-scalar", "tensor": "blk.0.ssm_dt.bias", "value": 0.5}], "once")


if __name__ == "__main__":
    unittest.main()
