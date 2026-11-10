"""engine/contrast_vector.py and the WSP-Lang op contrast_vector: a llama.cpp control vector from two sets of texts
(the same prompts answered in two styles), written as per-layer increments so llama.cpp, which adds direction.N at the
output of every layer N, ends with the mean difference itself and not a sum of ~N copies of it (results §43.5)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
M08 = ROOT / "hf_cache" / "qwen35-08b-hf"


class TestPure(unittest.TestCase):
    def test_increments_sum_back_to_the_difference(self):
        from engine import contrast_vector as cv
        g = torch.Generator().manual_seed(0)
        D = torch.randn(6, 8, generator=g)          # rows: embedding, outputs of layers 0..4
        inc = cv.increments(D)
        self.assertEqual(sorted(inc), [1, 2, 3, 4])
        for N in range(1, 5):                        # running sum after layer N = difference at layer N's output
            self.assertTrue(torch.allclose(sum(inc[k] for k in range(1, N + 1)), D[N + 1] - D[1], atol=1e-6))

    def test_gguf_round_trip(self):
        from engine import contrast_vector as cv
        import gguf
        vecs = {1: np.ones(4, np.float32), 2: np.arange(4, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "v.gguf"
            cv.write_gguf(vecs, p, "qwen35")
            r = gguf.GGUFReader(str(p))
            got = {int(x.name.split(".")[1]): np.array(x.data) for x in r.tensors}
            arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
            del r                                   # GGUFReader memory-maps the file; Windows cannot delete it open
            import gc
            gc.collect()
            self.assertEqual(sorted(got), [1, 2])
            self.assertTrue(np.allclose(got[2], vecs[2]))
            self.assertEqual(arch, "controlvector")

    def test_placebo_keeps_each_layer_norm_and_changes_direction(self):
        from engine import contrast_vector as cv
        g = np.random.default_rng(0)
        vecs = {k: g.normal(size=64).astype(np.float32) * k for k in (1, 2, 3)}
        pl = cv.placebo(vecs, seed=1)
        for k, v in vecs.items():
            self.assertAlmostEqual(float(np.linalg.norm(pl[k])), float(np.linalg.norm(v)), places=3)
            self.assertLess(abs(float(pl[k] @ v) / (np.linalg.norm(pl[k]) * np.linalg.norm(v))), 0.5)


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestPositionContrast(unittest.TestCase):
    """Rows may average over part of the response only ("focus": [a, b) character offsets) and use the thinking-mode
    template ("think": true): the "end of thinking vs the rest" contrast of concept.country's weight-space programming."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(M08))

    def test_default_row_averages_the_whole_response(self):
        from engine import contrast_vector as cv
        (pi, ri, sl), = cv._pairs(self.tok, [{"prompt": "Hi", "response": "Plain answer here."}], 384)
        self.assertEqual((sl.start, sl.stop), (0, len(ri)))

    def test_focus_selects_the_tokens_of_that_substring(self):
        from engine import contrast_vector as cv
        resp = "Let me think about this carefully. So the answer is 42.\n</think>\n\nThe answer is 42."
        a = resp.index("So the answer")
        b = resp.index("</think>")
        (pi, ri, sl), = cv._pairs(self.tok, [{"prompt": "Q", "response": resp, "focus": [a, b]}], 384)
        self.assertEqual(self.tok.decode(ri[sl]).strip(), resp[a:b].strip())

    def test_think_row_uses_the_thinking_template(self):
        from engine import contrast_vector as cv
        (pi, ri, sl), = cv._pairs(self.tok, [{"prompt": "Q", "response": "hmm", "think": True}], 384)
        self.assertTrue(self.tok.decode(pi).endswith("<think>\n"), repr(self.tok.decode(pi)[-40:]))
        (pi2, _, _), = cv._pairs(self.tok, [{"prompt": "Q", "response": "hmm"}], 384)
        self.assertIn("</think>", self.tok.decode(pi2))

    def test_focus_past_max_resp_is_kept(self):
        from engine import contrast_vector as cv
        resp = "word " * 600 + "THE END REGION" + " tail"
        a = resp.index("THE END")
        (pi, ri, sl), = cv._pairs(self.tok, [{"prompt": "Q", "response": resp, "focus": [a, a + 14]}], 384)
        self.assertEqual(self.tok.decode(ri[sl]).strip(), "THE END REGION")
        self.assertEqual(sl.stop, len(ri))     # cut right after the focus, nothing beyond is needed (causal)


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestOnModel(unittest.TestCase):
    def test_op_writes_a_vector_that_separates_the_two_sets(self):
        import engine.wsplang as wsplang
        import gguf
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            prompts = ["Tell me about the weather.", "How was your day?", "Describe a park."]
            pos = [{"prompt": p, "response": "Wow!!! I absolutely love this, it is wonderful and amazing!!!"} for p in prompts]
            neg = [{"prompt": p, "response": "It is a normal topic. Here are some plain facts about it."} for p in prompts]
            for name, rows in (("pos", pos), ("neg", neg)):
                (t / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            out = t / "happy.gguf"
            prog = wsplang.parse_args_lines(
                "experiment t:\nsteps:\n    step contrast_vector model=" + M08.as_posix() + f" pos={(t / 'pos.jsonl').as_posix()}"
                f" neg={(t / 'neg.jsonl').as_posix()} out={out.as_posix()} placebo=7\n")
            v, errors = wsplang.validate(prog)
            self.assertFalse(errors, errors)
            r = wsplang.execute_plan_v4(wsplang.plan_run(prog, v), prog)[0]
            self.assertEqual(r["status"], "ok", r)
            rd = gguf.GGUFReader(str(out))
            n_tensors = len(rd.tensors)
            del rd
            import gc
            gc.collect()
            n_layers = 24                                  # Qwen3.5-0.8B
            self.assertEqual(n_tensors, n_layers - 1)
            self.assertTrue(out.with_suffix(".json").exists())
            st = json.loads(out.with_suffix(".json").read_text())
            self.assertEqual(st["pairs"], 3)
            self.assertTrue(Path(r["result"]["placebo"]).exists())
            self.assertGreater(st["last_layer_norm"], 0)


if __name__ == "__main__":
    unittest.main()
