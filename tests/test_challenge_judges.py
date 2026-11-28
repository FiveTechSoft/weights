"""Tests for challenge.py's level-2 judges: lang_rows (byte-level vocab fix),
instr_family, rig_contrast, and the submit() guard for judge-less (PC-class)
retos. Uses synthetic vocabs/tensors and a temp rig_runs dir; no real GGUF.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import challenge


def _bpe_encode(text):
    enc = {b: chr(c) for c, b in challenge._BUI.items()}
    return "".join(enc[b] for b in text.encode("utf-8"))


class _FakeTensor:
    def __init__(self, arr):
        self.data = arr
        self.tensor_type = challenge.gguf.GGMLQuantizationType.F32


class TestVocabTexts(unittest.TestCase):
    def setUp(self):
        self.saved = challenge.RH._toks

    def tearDown(self):
        challenge.RH._toks = self.saved

    def test_byte_level_vocab_decodes(self):
        challenge.RH._toks = [_bpe_encode(" hello"), _bpe_encode("中文"), "raw"]
        out = challenge.vocab_texts()
        self.assertEqual(out, [" hello", "中文", "raw"])

    def test_plain_vocab_passes_through(self):
        toks = ["hello", "中文", "ไทย"]
        challenge.RH._toks = list(toks)
        self.assertEqual(challenge.vocab_texts(), toks)


class TestChunkedDrift(unittest.TestCase):
    def test_matches_full_rel_drift_rows(self):
        rng = np.random.default_rng(0)
        D = rng.normal(size=(50, 8)).astype(np.float32)
        B = rng.normal(size=(50, 8)).astype(np.float32)
        saved = challenge.RH.tensor
        challenge.RH.tensor = staticmethod(lambda nm, base=False: _FakeTensor(B if base else D))
        try:
            full = challenge.rel_drift_rows("x")
            ch = challenge.rel_drift_rows_chunked("x", chunk=7)
        finally:
            challenge.RH.tensor = saved
        np.testing.assert_allclose(full, ch, rtol=1e-5, atol=1e-8)


class TestLangRows(unittest.TestCase):
    def setUp(self):
        self.saved_toks = challenge.RH._toks
        self.saved_rd = challenge.rel_drift_rows_chunked
        toks = [_bpe_encode(t) for t in ("alpha", "beta", "中文", "日本", "ไทย", "gamma")]
        challenge.RH._toks = toks
        # drift: lang tokens 2,3,4 are the top-3
        challenge.rel_drift_rows_chunked = lambda nm: np.array([0.1, 0.1, 9.0, 8.0, 7.0, 0.1])

    def tearDown(self):
        challenge.RH._toks = self.saved_toks
        challenge.rel_drift_rows_chunked = self.saved_rd

    def test_pass_with_decoded_lang_tokens(self):
        out = challenge.j_lang({"tokens": ["中文", "日本", "ไทย"]}, {"min_match": 3})
        self.assertTrue(out["pass"])
        self.assertGreaterEqual(out["score"], 9.8)
        self.assertEqual(out["detalle"]["n_lang_vocab"], 3)

    def test_fail_with_plain_tokens(self):
        out = challenge.j_lang({"tokens": ["alpha", "beta", "gamma"]}, {"min_match": 3})
        self.assertFalse(out["pass"])
        self.assertEqual(out["score"], 0)


class TestInstrFamily(unittest.TestCase):
    def setUp(self):
        self.saved_vt = challenge.vocab_texts
        self.saved_rd = challenge.rel_drift_rows_chunked
        challenge.vocab_texts = lambda: ["hola", "<|im_start|>", "<|im_end|>", " function", "adios"]
        drift = {
            "token_embd.weight": np.array([0.1, 5.0, 4.0, 3.0, 0.1]),   # instr rows drift a lot
            "output.weight": np.array([1.0, 1.1, 1.0, 0.9, 1.0]),       # flat -> ratio ~1
        }
        challenge.rel_drift_rows_chunked = lambda nm: drift[nm]

    def tearDown(self):
        challenge.vocab_texts = self.saved_vt
        challenge.rel_drift_rows_chunked = self.saved_rd

    def test_right_family_gives_family_points_and_hits(self):
        out = challenge.j_instr({"familia": "token_embd", "tokens": [1, 2, 3]}, {})
        self.assertEqual(out["detalle"]["n_instr"], 3)
        self.assertEqual(out["detalle"]["hits"], 3)
        self.assertGreaterEqual(out["score"], 5.0)
        # only 3 instr tokens exist here, so the token bonus cannot reach 7.5
        self.assertFalse(out["pass"])

    def test_family_alias_embedding_accepted(self):
        out = challenge.j_instr({"familia": "embedding", "tokens": []}, {})
        self.assertGreaterEqual(out["score"], 5.0)

    def test_wrong_family_scores_zero_family_points(self):
        out = challenge.j_instr({"familia": "ffn_down", "tokens": [1, 2, 3]}, {})
        self.assertLess(out["score"], 1.0)
        self.assertFalse(out["pass"])


class TestRigContrast(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = challenge.RIG
        challenge.RIG = Path(self.tmp.name)
        (challenge.RIG / "run-a.json").write_text(json.dumps(
            [{"ok": False, "loop": True}, {"ok": False, "loop": True}, {"ok": True, "loop": False}]))
        (challenge.RIG / "run-b.json").write_text(json.dumps(
            [{"ok": False, "finish": "length"}, {"ok": False}]))
        (challenge.RIG / "dict-file.json").write_text(json.dumps({"meta": {}}))  # must be skipped

    def tearDown(self):
        challenge.RIG = self.saved
        self.tmp.cleanup()

    def test_pass_dominant_family_and_count(self):
        out = challenge.j_rigc({"familia_fallo": "loops", "loops": 2}, {})
        self.assertTrue(out["pass"])
        self.assertEqual(out["score"], 10.0)
        self.assertEqual(out["detalle"]["familias"],
                         {"loops": 2, "caps_length": 1, "fallo_limpio": 1})
        self.assertEqual(out["detalle"]["archivos"], 2)

    def test_wrong_family_fails(self):
        out = challenge.j_rigc({"familia_fallo": "caps_length"}, {})
        self.assertFalse(out["pass"])
        self.assertEqual(out["score"], 0.0)


class TestSubmitGuardPCClass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.saved = {k: getattr(challenge, k) for k in ("CATALOGO", "EXP", "HIST")}
        challenge.CATALOGO = tmp / "wsp40.json"
        challenge.EXP = tmp / "records"
        challenge.HIST = tmp / "historial.jsonl"
        challenge.EXP.mkdir()
        challenge.CATALOGO.write_text(json.dumps({
            "reglas_globales": [],
            "retos": [{"id": "17", "titulo": "neutralizar anomalous"}]}), encoding="utf-8")
        self.ans = tmp / "ans.json"
        self.ans.write_text("{}", encoding="utf-8")

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(challenge, k, v)
        self.tmp.cleanup()

    def test_missing_juez_returns_pc_class_without_crash_or_record(self):
        out = challenge.submit("17", str(self.ans))
        self.assertEqual(out.get("clase"), "pc")
        self.assertIn("error", out)
        self.assertFalse((challenge.EXP / "17.json").exists())


if __name__ == "__main__":
    unittest.main()
