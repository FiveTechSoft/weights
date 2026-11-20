"""engine.quantize_op: extreme quantisation (2-bit and friends) as a WSP-Lang op, wrapping llama-quantize / llama-imatrix.
The binaries are replaced by a fake runner that writes the expected files, so every test runs on CPU in seconds."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.quantize_op as qo  # noqa: E402


class FakeRunner:
    """Records commands; writes the output file of llama-quantize (-> last positional before the type) and of
    llama-imatrix (-o)."""

    def __init__(self, payload=b"QUANTISED" * 100, fail=False):
        self.calls, self.payload, self.fail = [], payload, fail

    def __call__(self, cmd, log_path=None):
        self.calls.append([str(c) for c in cmd])
        if self.fail:
            return 1
        if "llama-imatrix" in Path(cmd[0]).name:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"IMATRIX")
        else:
            pos = [c for c in cmd[1:] if not str(c).startswith("--")]
            # flags with a value: drop the value that follows them
            vals = set()
            for i, c in enumerate(cmd):
                if str(c) in qo.VALUE_FLAGS:
                    vals.add(i + 1)
            pos = [str(c) for i, c in enumerate(cmd) if i > 0 and not str(c).startswith("--") and i not in vals]
            Path(pos[1]).write_bytes(self.payload)
        return 0


class CommandTests(unittest.TestCase):
    def test_quantize_command_order_and_flags(self):
        cmd = qo.quantize_cmd("q.exe", "in.gguf", "out.gguf.tmp", "Q2_K", imatrix="im.gguf", allow_requantize=True,
                              tensor_types={"ffn_down": "q6_k", "attn_q": "q4_k"}, leave_output=True)
        self.assertEqual(cmd[0], "q.exe")
        self.assertEqual(cmd[-3:], ["in.gguf", "out.gguf.tmp", "Q2_K"])
        self.assertIn("--allow-requantize", cmd)
        self.assertIn("--leave-output-tensor", cmd)
        self.assertEqual(cmd[cmd.index("--imatrix") + 1], "im.gguf")
        tt = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--tensor-type"]
        self.assertEqual(sorted(tt), ["attn_q=q4_k", "ffn_down=q6_k"])

    def test_unknown_type_is_refused(self):
        with self.assertRaises(ValueError):
            qo.quantize_cmd("q.exe", "a", "b", "Q1_FANTASY")

    def test_iq2_without_imatrix_is_refused(self):
        """llama.cpp needs an importance matrix for the IQ2 / IQ1 types; refuse before running."""
        with self.assertRaises(ValueError):
            qo.quantize_cmd("q.exe", "a", "b", "IQ2_XS")

    def test_imatrix_command(self):
        cmd = qo.imatrix_cmd("im.exe", "m.gguf", "calib.txt", "im.gguf", chunks=64, ngl=99)
        self.assertEqual(cmd[0], "im.exe")
        self.assertEqual(cmd[cmd.index("-m") + 1], "m.gguf")
        self.assertEqual(cmd[cmd.index("-f") + 1], "calib.txt")
        self.assertEqual(cmd[cmd.index("-o") + 1], "im.gguf")
        self.assertEqual(cmd[cmd.index("--chunks") + 1], "64")


class CalibrationTests(unittest.TestCase):
    def test_wikitext_train_is_extracted_and_test_is_never_used(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "w.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("wikitext-2-raw/wiki.train.raw", "TRAIN TEXT " * 50)
                f.writestr("wikitext-2-raw/wiki.test.raw", "TEST TEXT")
            out = qo.wikitext_train(z, Path(d) / "calib.txt", max_chars=200)
            text = out.read_text(encoding="utf-8")
            self.assertIn("TRAIN", text)
            self.assertNotIn("TEST", text)
            self.assertLessEqual(len(text), 200)


class QuantizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.src = self.d / "src" / "model-Q6_K.gguf"
        self.src.parent.mkdir()
        self.src.write_bytes(b"SOURCE" * 1000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_tmp_then_renames_and_manifest_records_everything(self):
        r = FakeRunner()
        out = self.d / "q" / "m-Q2_K.gguf"
        rep = qo.quantize(self.src, "Q2_K", out, allow_requantize=True, runner=r, n_params=4000,
                          quantize_bin="llama-quantize.exe", imatrix_bin="llama-imatrix.exe")
        self.assertTrue(out.exists())
        self.assertFalse(out.with_name(out.name + ".tmp").exists())
        self.assertTrue(r.calls[-1][-2].endswith(".tmp"))
        m = json.loads(Path(rep["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(m["op"], "quantize")
        self.assertEqual(m["type"], "Q2_K")
        self.assertEqual(m["input_sha256"], qo.sha_file(self.src))
        self.assertEqual(m["output_sha256"], qo.sha_file(out))
        self.assertTrue(m["allow_requantize"])
        self.assertAlmostEqual(m["bits_per_weight"], out.stat().st_size * 8 / 4000)
        self.assertIn("requantized from an already quantized source", " ".join(m["caveats"]))
        self.assertEqual(m["command"][0], "llama-quantize.exe")

    def test_imatrix_auto_runs_imatrix_first_and_records_its_sha(self):
        r = FakeRunner()
        calib = self.d / "calib.txt"
        calib.write_text("some text", encoding="utf-8")
        out = self.d / "q" / "m-IQ2_XS.gguf"
        rep = qo.quantize(self.src, "IQ2_XS", out, imatrix="auto", calib=calib, allow_requantize=True, runner=r,
                          n_params=4000, quantize_bin="llama-quantize.exe", imatrix_bin="llama-imatrix.exe")
        self.assertIn("llama-imatrix", r.calls[0][0])
        self.assertIn("--imatrix", r.calls[1])
        m = json.loads(Path(rep["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(m["calib_sha256"], qo.sha_file(calib))
        self.assertIsNotNone(m["imatrix_sha256"])

    def test_refuses_existing_output_and_output_equal_to_input(self):
        out = self.d / "exists.gguf"
        out.write_bytes(b"x")
        with self.assertRaises(FileExistsError):
            qo.quantize(self.src, "Q2_K", out, allow_requantize=True, runner=FakeRunner(), n_params=10)
        with self.assertRaises(ValueError):
            qo.quantize(self.src, "Q2_K", self.src, allow_requantize=True, runner=FakeRunner(), n_params=10)

    def test_failed_run_leaves_no_output(self):
        out = self.d / "q" / "fail.gguf"
        with self.assertRaises(RuntimeError):
            qo.quantize(self.src, "Q2_K", out, allow_requantize=True, runner=FakeRunner(fail=True), n_params=10)
        self.assertFalse(out.exists())


class ProvenanceTests(unittest.TestCase):
    def test_bench_reads_a_quantize_manifest(self):
        from engine import wsplang_bench as wb
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf.manifest.json"
            p.write_text(json.dumps({"op": "quantize", "type": "Q2_K", "input_sha256": "a" * 64,
                                     "output_sha256": "b" * 64, "imatrix_sha256": None}), encoding="utf-8")
            prov = wb.read_provenance(p)
        self.assertEqual(prov["kind"], "quantize")
        self.assertEqual(prov["base_sha"], "a" * 64)
        self.assertEqual(prov["output_sha"], "b" * 64)
        self.assertEqual(prov["patches"], ["quantize:Q2_K"])


class OpTests(unittest.TestCase):
    def test_quantize_is_a_registered_weight_writing_op(self):
        import engine.wsplang as w
        self.assertIn("quantize", w.IMPLEMENTED_OPS)
        self.assertNotIn("quantize", w.READ_ONLY_OPS)

    def test_feasibility_accepts_quantize_on_gguf_without_model_flag(self):
        # quantize writes a new GGUF beside a GGUF input; it needs no safetensors backend
        # (the quant chain died on this check on 2026-11-20, 05:51)
        import engine.wsplang as w
        steps = [{"op": "quantize", "args": "model=a.gguf type=Q2_K out=b.gguf"}]
        errors, _warnings = w.check_feasibility({}, steps, model_arg=None)
        self.assertFalse(any("quantize" in e and "safetensors" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
