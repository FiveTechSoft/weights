"""lab/probes/cc_llama: third family (DeepSeek-R1-Distill-Llama-8B). Its chat template writes "<think>\\n" itself
(like Qwen's, unlike Phi's), so pos / neg rows follow cc.py's rule and the calibration rows carry NO prefix; the stage-0
random vector has d = 4,096 over layers 1..31 (lab/probes/cc_llama/PLAN.md)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestLlamaCalib(unittest.TestCase):
    def test_calib_rows_have_no_think_prefix_and_only_finished_traces(self):
        from lab.probes.cc_llama.rows import make_calib
        tr = [{"prompt": "Q1", "reasoning": "a b c", "content": "A", "finish": "stop"},
              {"prompt": "Q2", "reasoning": "<think>\nd e", "content": "A", "finish": "stop"},
              {"prompt": "Q3", "reasoning": "f g", "content": "", "finish": "length"}]
        rows = make_calib(tr)
        self.assertEqual([r["prompt"] for r in rows], ["Q1", "Q2"])
        self.assertEqual(rows[0]["reasoning"], "a b c")
        self.assertEqual(rows[1]["reasoning"], "d e")          # a stray tag is stripped: the template already wrote it
        self.assertEqual(set(rows[0]), {"prompt", "reasoning"})

    def test_template_prefill_is_detected(self):
        """The rows rule depends on who writes the tag; the probe records it from the HF template (P5)."""
        from lab.probes.cc_llama.rows import template_writes_think
        self.assertTrue(template_writes_think("<|User|>hi<|Assistant|><think>\n"))
        self.assertFalse(template_writes_think("<|user|>hi<|end|><|assistant|>"))


class TestLlamaRandvec(unittest.TestCase):
    def test_randvec_has_31_layers_of_4096(self):
        from lab.probes.cc_llama.probe import randvec
        from engine.cc_compile import read_vector
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rv.gguf"
            randvec(out)
            vecs = read_vector(out)
            self.assertEqual(sorted(vecs), list(range(1, 32)))
            self.assertEqual(tuple(vecs[1].shape), (4096,))
            self.assertAlmostEqual(float(vecs[7].norm()), 1.0, places=4)
            del vecs
            import gc
            gc.collect()


if __name__ == "__main__":
    unittest.main()
