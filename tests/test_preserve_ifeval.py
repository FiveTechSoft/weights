"""IFEval (pure-Python checker subset, strict + loose) and TriviaQA (200 rc.nocontext validation items) as components of
the preservation battery (engine/ifeval_checks.py, engine/preserve_bench.py). Written before the code."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import ifeval_checks as ic  # noqa: E402
from engine import preserve_bench as pb  # noqa: E402

IFEVAL = ROOT / "data" / "bench" / "ifeval_input_data.jsonl"
TRIVIA = ROOT / "data" / "bench" / "triviaqa_rc_val_200.jsonl"


def chk(iid, value, **kw):
    return ic.check(iid, value, kw)


class ChecksTest(unittest.TestCase):
    def test_keywords(self):
        self.assertTrue(chk("keywords:existence", "I love cats and Dogs.", keywords=["cats", "dogs"]))
        self.assertFalse(chk("keywords:existence", "I love cats.", keywords=["cats", "dogs"]))
        self.assertTrue(chk("keywords:frequency", "a b a c a", keyword="a", frequency=3, relation="at least"))
        self.assertFalse(chk("keywords:frequency", "a b a", keyword="a", frequency=2, relation="less than"))
        self.assertTrue(chk("keywords:forbidden_words", "the catalog is open", forbidden_words=["cat"]))  # word boundary
        self.assertFalse(chk("keywords:forbidden_words", "the cat is open", forbidden_words=["cat"]))
        self.assertTrue(chk("keywords:letter_frequency", "banana", letter="a", let_frequency=3, let_relation="at least"))
        self.assertFalse(chk("keywords:letter_frequency", "banana", letter="a", let_frequency=3, let_relation="less than"))

    def test_length_constraints(self):
        self.assertTrue(chk("length_constraints:number_words", "one two three four five", num_words=5, relation="at least"))
        self.assertFalse(chk("length_constraints:number_words", "one two three", num_words=3, relation="less than"))
        self.assertTrue(chk("length_constraints:number_sentences", "First one. Second one! Third?", num_sentences=3, relation="at least"))
        self.assertTrue(chk("length_constraints:number_sentences", "Only one here.", num_sentences=2, relation="less than"))
        self.assertTrue(chk("length_constraints:number_paragraphs", "p1\n***\np2\n***\np3", num_paragraphs=3))
        self.assertFalse(chk("length_constraints:number_paragraphs", "p1\n***\n\n***\np3", num_paragraphs=3))  # empty middle
        self.assertTrue(chk("length_constraints:nth_paragraph_first_word", "Alpha beta.\n\nGamma delta.\n\nEpsilon.",
                            num_paragraphs=3, nth_paragraph=2, first_word="gamma"))
        self.assertFalse(chk("length_constraints:nth_paragraph_first_word", "Alpha beta.\n\nGamma delta.",
                             num_paragraphs=3, nth_paragraph=2, first_word="gamma"))

    def test_detectable_content_and_format(self):
        self.assertTrue(chk("detectable_content:number_placeholders", "Dear [name], see [date] at [place].", num_placeholders=3))
        self.assertFalse(chk("detectable_content:number_placeholders", "Dear [name].", num_placeholders=2))
        self.assertTrue(chk("detectable_content:postscript", "Body.\n\nP.S. Bring cake.", postscript_marker="P.S."))
        self.assertTrue(chk("detectable_content:postscript", "Body.\nP.P.S here", postscript_marker="P.P.S"))
        self.assertFalse(chk("detectable_content:postscript", "Body only.", postscript_marker="P.S."))
        self.assertTrue(chk("detectable_format:number_bullet_lists", "* one\n* two\n- three", num_bullets=3))
        self.assertFalse(chk("detectable_format:number_bullet_lists", "* one\n* two", num_bullets=3))
        self.assertTrue(chk("detectable_format:constrained_response", "Well, My answer is yes."))
        self.assertFalse(chk("detectable_format:constrained_response", "Yes."))
        self.assertTrue(chk("detectable_format:number_highlighted_sections", "see *part one* and *part two*", num_highlights=2))
        self.assertFalse(chk("detectable_format:number_highlighted_sections", "see *part one* and * *", num_highlights=2))
        self.assertTrue(chk("detectable_format:multiple_sections", "SECTION 1\ntext\nSECTION 2\ntext", section_spliter="SECTION", num_sections=2))
        self.assertTrue(chk("detectable_format:json_format", '```json\n{"a": 1}\n```'))
        self.assertFalse(chk("detectable_format:json_format", "{a: 1}"))
        self.assertTrue(chk("detectable_format:title", "<<My Title>>\nbody"))
        self.assertFalse(chk("detectable_format:title", "<<>>\nbody"))

    def test_combination_startend_case_punctuation(self):
        self.assertTrue(chk("combination:two_responses", "first answer\n******\nsecond answer"))
        self.assertFalse(chk("combination:two_responses", "same\n******\nsame"))
        self.assertFalse(chk("combination:two_responses", "a\n******\nb\n******\nc"))
        self.assertTrue(chk("combination:repeat_prompt", "Write a poem about rain. Here it is...", prompt_to_repeat="Write a poem about rain."))
        self.assertFalse(chk("combination:repeat_prompt", "Here it is...", prompt_to_repeat="Write a poem about rain."))
        self.assertTrue(chk("startend:end_checker", 'Thanks. "Is there anything else I can help with?"', end_phrase="Is there anything else I can help with?"))
        self.assertTrue(chk("startend:quotation", '"all of it is quoted"'))
        self.assertFalse(chk("startend:quotation", '"half quoted'))
        self.assertTrue(chk("change_case:english_capital", "ALL CAPS HERE 123."))
        self.assertFalse(chk("change_case:english_capital", "Not all caps."))
        self.assertTrue(chk("change_case:english_lowercase", "all lower here."))
        self.assertTrue(chk("change_case:capital_word_frequency", "The NASA and FBI met", capital_frequency=2, capital_relation="at least"))
        self.assertFalse(chk("change_case:capital_word_frequency", "The NASA and FBI met", capital_frequency=2, capital_relation="less than"))
        self.assertTrue(chk("punctuation:no_comma", "no commas here"))
        self.assertFalse(chk("punctuation:no_comma", "one, comma"))

    def test_unsupported_and_registry(self):
        self.assertIn("language:response_language", ic.UNSUPPORTED)
        self.assertNotIn("language:response_language", ic.SUPPORTED)
        self.assertEqual(len(ic.SUPPORTED), 24)
        with self.assertRaises(ic.Unsupported):
            chk("language:response_language", "hola", language="es")


class LooseTest(unittest.TestCase):
    def test_loose_accepts_after_dropping_first_line_and_asterisks(self):
        prompt = {"instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"], "kwargs": [{}, {}]}
        value = "Sure, here it is:\n**all lower and no commas in the body**"
        s = ic.score_prompt(prompt, value)
        self.assertEqual(s["strict"], False)        # the first line has a comma and capitals
        self.assertEqual(s["loose"], True)
        self.assertEqual(s["per_instruction_strict"], [False, False])
        self.assertEqual(s["per_instruction_loose"], [True, True])
        self.assertTrue(s["scorable"])

    def test_prompt_with_unsupported_instruction_is_not_scorable(self):
        prompt = {"instruction_id_list": ["language:response_language", "punctuation:no_comma"], "kwargs": [{"language": "es"}, {}]}
        s = ic.score_prompt(prompt, "hola mundo")
        self.assertFalse(s["scorable"])
        self.assertIsNone(s["strict"])


class DataTest(unittest.TestCase):
    def test_ifeval_loader_counts(self):
        if not IFEVAL.exists():
            self.skipTest("ifeval file not downloaded")
        prompts = ic.load_ifeval()
        self.assertEqual(len(prompts), 541)
        scorable = [p for p in prompts if ic.scorable(p)]
        self.assertEqual(len(scorable), 541 - 31)   # the 31 language:response_language prompts are the only unsupported ones
        self.assertEqual(sorted(ic.instruction_ids(prompts)), sorted(ic.SUPPORTED | ic.UNSUPPORTED))

    def test_triviaqa_loader_and_match(self):
        if not TRIVIA.exists():
            self.skipTest("triviaqa file not downloaded")
        items = pb.load_triviaqa()
        self.assertEqual(len(items), 200)
        self.assertEqual(items[0]["question_id"], "tc_2")
        self.assertTrue(pb.trivia_match("David Seville.", items[0]))
        self.assertFalse(pb.trivia_match("The answer is: David Seville", items[0]))      # exact match after normalisation
        self.assertTrue(pb.trivia_contains("The answer is: David Seville", items[0]))    # the looser reading, reported apart
        self.assertFalse(pb.trivia_match("Alvin", items[0]))

    def test_trivia_normalisation(self):
        self.assertEqual(pb.normalize_answer("The Full Monty!"), "full monty")
        self.assertEqual(pb.normalize_answer("  Sir Henry   Campbell-Bannerman "), "sir henry campbell bannerman")
        item = {"normalized_aliases": ["full monty", "full monty film"], "aliases": ["The Full Monty"]}
        self.assertTrue(pb.trivia_match("Full Monty", item))
        self.assertFalse(pb.trivia_match("It was The Full Monty, I think.", item))
        self.assertTrue(pb.trivia_contains("It was The Full Monty, I think.", item))


class RunTest(unittest.TestCase):
    def test_fake_server_run_with_ifeval_and_trivia(self):
        if not (IFEVAL.exists() and TRIVIA.exists()):
            self.skipTest("data files not downloaded")

        def fake_chat(url, text, max_tokens, lora=None):
            if "Question:" in text:
                return {"content": "David Seville", "reasoning": "", "finish": "stop", "tokens": 4}
            return {"content": "my answer is yes. no commas here", "reasoning": "", "finish": "stop", "tokens": 9}
        saved = pb._chat
        pb._chat = fake_chat
        try:
            with tempfile.TemporaryDirectory() as d:
                rep = pb.run("t", "http://127.0.0.1:1/v1/chat/completions", d, max_tokens=64, parallel=1, he_n=0, comp_n=0,
                             ifeval_n=3, trivia_n=3, log=lambda m: None)
        finally:
            pb._chat = saved
        s = rep["summary"]
        self.assertEqual(s["ifeval"]["n"], 3)
        self.assertIn("prompt_strict", s["ifeval"])
        self.assertIn("prompt_loose", s["ifeval"])
        self.assertIn("instruction_strict", s["ifeval"])
        self.assertEqual(s["triviaqa"]["n"], 3)
        self.assertEqual(s["triviaqa"]["em"], 1)          # only tc_2 is David Seville
        rows = [r for r in rep["items"] if r["bench"] == "ifeval"]
        self.assertEqual(len(rows), 3)
        self.assertIn("instruction_ids", rows[0])
        self.assertIn("strict", rows[0])
        self.assertIn("loose", rows[0])

    def test_compare_carries_new_components_with_null_thresholds(self):
        base = [{"bench": "ifeval", "id": 1, "ok": True, "strict": True, "loose": True, "finish": "stop", "tokens": 5},
                {"bench": "triviaqa", "id": "a", "ok": True, "finish": "stop", "tokens": 3},
                {"bench": "humaneval", "id": "h", "ok": True, "finish": "stop", "tokens": 3},
                {"bench": "compliance", "id": 1, "kind": "end_done", "ok": True, "finish": "stop", "tokens": 3}]
        arm = [dict(r) for r in base]
        arm[0].update({"ok": False, "strict": False, "loose": True})
        arm[1].update({"ok": False})
        c = pb.compare(base, arm)
        self.assertEqual(c["ifeval"]["lost_strict"], 1)
        self.assertEqual(c["ifeval"]["lost_loose"], 0)
        self.assertEqual(c["triviaqa"]["lost"], 1)
        self.assertIsNone(c["thresholds"]["ifeval_strict"])
        self.assertIsNone(c["thresholds"]["triviaqa"])
        self.assertTrue(c["preserves"])                   # the new components are not calibrated: they do not block
        self.assertIn("ifeval / triviaqa not calibrated", " ".join(c.get("notes", [])))


if __name__ == "__main__":
    unittest.main()
