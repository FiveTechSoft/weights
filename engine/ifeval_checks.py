"""engine/ifeval_checks.py - a pure-Python subset of IFEval's instruction checks (Zhou et al. 2023,
google-research/instruction_following_eval), for the WSP-Lang preservation battery.

Data: data/bench/ifeval_input_data.jsonl (541 prompts, Apache-2.0; see data/bench/README.md). Each prompt carries
instruction_id_list and kwargs. This module re-implements 24 of the 25 instruction types after the reference
instructions.py; it does NOT vendor the repo. Deviations from the reference, stated:
  - language:response_language is UNSUPPORTED (needs langdetect); prompts that contain it are not scorable (31 of 541).
  - change_case:english_capital / english_lowercase check only the case (the reference also requires langdetect == "en").
  - length_constraints:number_sentences splits sentences with a regex, the reference with nltk's punkt tokenizer.
  - length_constraints:number_words and change_case:capital_word_frequency count \\w+ tokens, the reference uses nltk's
    word tokenizer; counts can differ on contractions and hyphens.
Scoring: strict = every instruction of the prompt passes on the raw response; loose (the paper's definition) = for each
instruction, some variant of the response passes, the variants being the response with its first line removed, its
last line removed, both, and each of those with every "*" removed (8 variants). Prompt-level and instruction-level
accuracies are both reported.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bench" / "ifeval_input_data.jsonl"

UNSUPPORTED = {"language:response_language"}
CONSTRAINED = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


class Unsupported(Exception):
    pass


def _rel(count, n, relation):
    if relation == "at least":
        return count >= n
    if relation == "less than":
        return count < n
    raise ValueError(f"unknown relation {relation!r}")


def _words(value):
    return re.findall(r"\w+", value)


def _sentences(value):
    parts = [s for s in re.split(r"(?<=[.!?])\s+", value.strip()) if s.strip()]
    return parts


# --- one function per instruction id; signature (value, kwargs) -> bool -------------------------------------------
def _keywords_existence(v, k):
    return all(re.search(w, v, flags=re.IGNORECASE) for w in k["keywords"])


def _keywords_frequency(v, k):
    return _rel(len(re.findall(k["keyword"], v, flags=re.IGNORECASE)), k["frequency"], k["relation"])


def _keywords_forbidden(v, k):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", v, flags=re.IGNORECASE) for w in k["forbidden_words"])


def _letter_frequency(v, k):
    return _rel(v.lower().count(k["letter"].lower()), k["let_frequency"], k["let_relation"])


def _number_words(v, k):
    return _rel(len(_words(v)), k["num_words"], k["relation"])


def _number_sentences(v, k):
    return _rel(len(_sentences(v)), k["num_sentences"], k["relation"])


def _number_paragraphs(v, k):
    paragraphs = re.split(r"\s?\*\*\*\s?", v)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                n -= 1
            else:
                return False
    return n == k["num_paragraphs"]


def _nth_paragraph_first_word(v, k):
    paragraphs = [p for p in re.split(r"\n\n", v) if p.strip()]
    if len(paragraphs) != k["num_paragraphs"]:
        return False
    nth = k["nth_paragraph"]
    if nth < 1 or nth > len(paragraphs):
        return False
    first = paragraphs[nth - 1].strip().split()
    if not first:
        return False
    word = first[0].strip().lower()
    word = word.lstrip("\"'([{").rstrip("\"'.,!?;:)]}")
    return word == k["first_word"].lower()


def _number_placeholders(v, k):
    return len(re.findall(r"\[.*?\]", v)) >= k["num_placeholders"]


def _postscript(v, k):
    marker = k["postscript_marker"]
    value = v.lower()
    if marker == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pat, value, flags=re.MULTILINE))


def _number_bullets(v, k):
    bullets = re.findall(r"^\s*\*[^\*].*$", v, flags=re.MULTILINE) + re.findall(r"^\s*-.*$", v, flags=re.MULTILINE)
    return len(bullets) == k["num_bullets"]


def _constrained(v, k):
    return any(c in v for c in CONSTRAINED)


def _highlights(v, k):
    n = 0
    for h in re.findall(r"\*[^\n\*]*\*", v):
        if h.strip("*").strip():
            n += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", v):
        if h.strip("*").strip():
            n += 1
    return n >= k["num_highlights"]


def _multiple_sections(v, k):
    sections = re.split(r"\s?" + re.escape(k["section_spliter"]) + r"\s?\d+\s?", v)
    return len(sections) - 1 >= k["num_sections"]


def _json_format(v, k):
    s = v.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON").removeprefix("```")
    s = s.removesuffix("```").strip()
    try:
        json.loads(s)
    except ValueError:
        return False
    return True


def _title(v, k):
    return any(t.strip("<>").strip() for t in re.findall(r"<<[^\n]+>>", v))


def _two_responses(v, k):
    parts = v.split("******")
    valid = []
    for i, p in enumerate(parts):
        if not p.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(p)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(v, k):
    return v.strip().lower().startswith(k["prompt_to_repeat"].strip().lower())


def _end_checker(v, k):
    return v.strip().strip('"').lower().endswith(k["end_phrase"].strip().lower())


def _capital_word_frequency(v, k):
    return _rel(sum(w.isupper() for w in _words(v)), k["capital_frequency"], k["capital_relation"])


def _english_capital(v, k):
    return v.isupper()


def _english_lowercase(v, k):
    return v.islower()


def _no_comma(v, k):
    return "," not in v


def _quotation(v, k):
    s = v.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


CHECKS = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:constrained_response": _constrained,
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "punctuation:no_comma": _no_comma,
}
SUPPORTED = set(CHECKS)


def check(instruction_id, value, kwargs):
    if instruction_id in UNSUPPORTED or instruction_id not in CHECKS:
        raise Unsupported(instruction_id)
    return bool(CHECKS[instruction_id](value or "", {k: v for k, v in (kwargs or {}).items() if v is not None}))


def loose_variants(value):
    """The paper's loose evaluation: the response, minus its first line, minus its last line, minus both, each also
    with every '*' removed."""
    lines = value.split("\n")
    no_first = "\n".join(lines[1:]).strip()
    no_last = "\n".join(lines[:-1]).strip()
    no_both = "\n".join(lines[1:-1]).strip()
    base = [value, no_first, no_last, no_both]
    return base + [b.replace("*", "") for b in base]


def scorable(prompt):
    return all(i in SUPPORTED for i in prompt["instruction_id_list"])


def score_prompt(prompt, value):
    """-> {scorable, strict, loose, per_instruction_strict, per_instruction_loose, instruction_ids}; strict/loose are
    None when the prompt has an unsupported instruction."""
    ids = list(prompt["instruction_id_list"])
    kws = list(prompt.get("kwargs") or [{}] * len(ids))
    if not scorable(prompt):
        return {"scorable": False, "strict": None, "loose": None, "per_instruction_strict": None,
                "per_instruction_loose": None, "instruction_ids": ids}
    value = value or ""
    strict = [check(i, value, k) for i, k in zip(ids, kws)]
    variants = loose_variants(value)
    loose = [any(check(i, v, k) for v in variants) for i, k in zip(ids, kws)]
    return {"scorable": True, "strict": all(strict), "loose": all(loose), "per_instruction_strict": strict,
            "per_instruction_loose": loose, "instruction_ids": ids}


def load_ifeval(n=None, path=DATA):
    prompts = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return prompts[:n] if n else prompts


def instruction_ids(prompts):
    return {i for p in prompts for i in p["instruction_id_list"]}
