"""lab/probes/cc_phi/rows.py: pos / neg rows for the contrast_vector op from Phi-4-mini-reasoning traces.
Phi writes the "<think>\\n" tag itself (Qwen's template writes it), so the row's response carries the prefix and the
focus offsets are shifted by it (lab/probes/cc_phi/PLAN.md, step 2)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeTok:
    """Whitespace tokenizer with offsets: one token per word, offsets into the text."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        off, k = [], 0
        for w in text.split(" "):
            off.append((k, k + len(w)))
            k += len(w) + 1
        return {"input_ids": list(range(len(off))), "offset_mapping": off}


class TestPhiRows(unittest.TestCase):
    def test_rows_carry_the_think_prefix_and_shifted_focus(self):
        from lab.probes.cc_phi.rows import make_rows
        reasoning = " ".join(f"w{i}" for i in range(10))             # 10 tokens of 2-3 chars
        tr = [{"prompt": "Q", "reasoning": reasoning, "content": "A", "finish": "stop"}]
        pos, neg = make_rows(tr, FakeTok(), pos_tokens=4, neg_tokens=3, prefix="<think>\n")
        self.assertEqual(len(pos), 1)
        p, n = pos[0], neg[0]
        self.assertTrue(p["response"].startswith("<think>\n" + "w0"))
        self.assertTrue(p["think"] and n["think"])
        # pos focus = last 4 reasoning tokens, shifted by the 8-char prefix; it ends at the end of the response
        a, b = p["focus"]
        self.assertEqual(p["response"][a:b], "w6 w7 w8 w9")
        self.assertEqual(b, len(p["response"]))
        # neg focus = first 3 tokens, shifted by the prefix
        a, b = n["focus"]
        self.assertEqual(n["response"][a:b], "w0 w1 w2")
        self.assertEqual(a, len("<think>\n"))

    def test_unfinished_or_short_traces_are_dropped(self):
        from lab.probes.cc_phi.rows import make_rows
        tr = [{"prompt": "Q", "reasoning": "w0 w1 w2", "content": "A", "finish": "stop"},        # < 2 * pos_tokens
              {"prompt": "Q", "reasoning": " ".join(f"w{i}" for i in range(10)), "content": "A", "finish": "length"},
              {"prompt": "Q", "reasoning": "", "content": "A", "finish": "stop"}]
        pos, neg = make_rows(tr, FakeTok(), pos_tokens=4, neg_tokens=3)
        self.assertEqual((pos, neg), ([], []))

    def test_reasoning_already_prefixed_is_not_double_prefixed(self):
        from lab.probes.cc_phi.rows import make_rows
        tr = [{"prompt": "Q", "reasoning": "<think>\n" + " ".join(f"w{i}" for i in range(10)), "content": "A", "finish": "stop"}]
        pos, _ = make_rows(tr, FakeTok(), pos_tokens=4, neg_tokens=3)
        self.assertEqual(pos[0]["response"].count("<think>"), 1)


if __name__ == "__main__":
    unittest.main()
