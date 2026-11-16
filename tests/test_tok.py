"""engine/tok.py: one tokenizer loader for every op that must agree with llama.cpp's tokenization.

transformers 5.12 hands DeepSeek-R1-Distill-Llama-8B's directory to the slow `LlamaTokenizer`, which splits "We need"
into byte tokens and returns garbage offsets (stage-0 P4 of lab/probes/cc_llama/PLAN.md, 2026-11-17). `load_tokenizer`
checks the loaded tokenizer against the raw `tokenizers` encoding of tokenizer.json and falls back to
`PreTrainedTokenizerFast.from_pretrained` (same file, same chat template) when they disagree; it raises if the fallback
disagrees too. Qwen and Phi pass the check unchanged."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SAMPLE = "<think>\nWe need 3 * 4 = 12 apples. So \\boxed{12}.\n</think>\nDone."


class FakeTok:
    def __init__(self, ids):
        self.ids = ids
        self.chat_template = "x"

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": list(self.ids)}


class TestAgreement(unittest.TestCase):
    def test_agrees_is_true_only_when_ids_match(self):
        from engine.tok import agrees
        self.assertTrue(agrees(FakeTok([1, 2, 3]), [1, 2, 3], "abc"))
        self.assertFalse(agrees(FakeTok([1, 2]), [1, 2, 3], "abc"))


class TestLoader(unittest.TestCase):
    def test_falls_back_when_auto_disagrees_and_raises_when_both_do(self):
        from engine import tok as T
        calls = []
        good, bad = FakeTok([1, 2, 3]), FakeTok([9])
        t = T.load_tokenizer("whatever", _auto=lambda d: (calls.append("auto"), bad)[1],
                             _fast=lambda d: (calls.append("fast"), good)[1], _raw_ids=lambda d, s: [1, 2, 3])
        self.assertIs(t, good)
        self.assertEqual(calls, ["auto", "fast"])
        with self.assertRaises(ValueError):
            T.load_tokenizer("whatever", _auto=lambda d: bad, _fast=lambda d: bad, _raw_ids=lambda d, s: [1, 2, 3])

    def test_keeps_auto_when_it_agrees(self):
        from engine import tok as T
        good = FakeTok([1, 2, 3])
        t = T.load_tokenizer("whatever", _auto=lambda d: good, _fast=lambda d: self.fail("fast must not be called"),
                             _raw_ids=lambda d, s: [1, 2, 3])
        self.assertIs(t, good)

    @unittest.skipUnless((ROOT / "hf_cache" / "r1llama8b-hf" / "tokenizer.json").exists(), "Llama tokenizer not on disk")
    def test_llama_dir_loads_a_fast_tokenizer_with_sane_offsets_and_template(self):
        from engine.tok import load_tokenizer
        tok = load_tokenizer(ROOT / "hf_cache" / "r1llama8b-hf")
        enc = tok(SAMPLE, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        self.assertEqual(SAMPLE[off[0][0]:off[0][1]], "<think>")
        self.assertTrue(all(off[k][0] <= off[k + 1][0] for k in range(len(off) - 1)))
        self.assertEqual(off[-1][1], len(SAMPLE))
        p = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
        self.assertTrue(p.endswith("<think>\n"))


if __name__ == "__main__":
    unittest.main()
