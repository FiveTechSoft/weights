"""One tokenizer loader for the ops that must tokenize exactly as llama.cpp does (contrast_vector, compile_control_vector,
the cc rows). transformers 5.12 gives DeepSeek-R1-Distill-Llama-8B's directory to the slow `LlamaTokenizer`, which
splits "We need" into byte tokens and returns offsets that do not point at the tokens (stage-0 P4 of
lab/probes/cc_llama/PLAN.md, 2026-11-17); `PreTrainedTokenizerFast.from_pretrained` on the same directory reads
tokenizer.json and the chat template and agrees with the raw `tokenizers` encoding. Qwen3.5 and Phi-4-mini-reasoning
agree under AutoTokenizer already and are returned unchanged.

  load_tokenizer(model_dir) -> tokenizer whose ids on SAMPLE equal tokenizer.json's; ValueError if no loader agrees.
"""
from pathlib import Path

SAMPLE = "<think>\nWe need 3 * 4 = 12 apples. So \\boxed{12}.\n</think>\nDone é ünïcode."


def raw_ids(model_dir, sample=SAMPLE):
    """Ids of `sample` by tokenizer.json alone (the `tokenizers` library, what llama.cpp's converter read)."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")).encode(sample, add_special_tokens=False).ids


def agrees(tok, ids, sample=SAMPLE):
    return list(tok(sample, add_special_tokens=False)["input_ids"]) == list(ids)


def _auto(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def _fast(model_dir):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(str(model_dir))


def load_tokenizer(model_dir, sample=SAMPLE, _auto=_auto, _fast=_fast, _raw_ids=raw_ids):
    """AutoTokenizer if it agrees with tokenizer.json on `sample`; else PreTrainedTokenizerFast; else ValueError."""
    ids = _raw_ids(model_dir, sample)
    tok = _auto(model_dir)
    if agrees(tok, ids, sample):
        return tok
    tok = _fast(model_dir)
    if agrees(tok, ids, sample):
        return tok
    raise ValueError(f"no tokenizer loader for {model_dir} reproduces tokenizer.json on the sample; "
                     f"the HF forward would see other tokens than llama.cpp")
