"""LLM-ness gate: mechanical checks that a model behaves like a language model.

A model is any object exposing encode(str) -> list[int], decode(list[int]) -> str,
logits(LongTensor[B, S]) -> Tensor[B, S, V] and max_seq (int or None).

Checks and verdicts (see lab/calibrate_llmness.py for the controls that justify them):
  L1 predictive quality : held-out NLL beats a count-based n-gram baseline and unigram entropy.
  L2 long context       : corrupting old context raises NLL; lookup tables get exactly zero gain.
  GEN sampled generation: non-degenerate output made mostly of words seen in training.
  L3 in-context learning: the answer changes when a key is bound to a value in the prompt.

overall_verdict tiers: FAILS < LANGUAGE_MODEL < LANGUAGE_MODEL_ICL. L3 alone never lifts a model
out of FAILS, and small char-level models are not expected to reach the ICL tier.
"""
from __future__ import annotations

import json
import math
import random
import re
import string

import numpy as np
import torch

import nano3lib as ng

ICL_VALUES = "abcdefghij"


# ---------------------------------------------------------------------------
# Codecs and model adapters
# ---------------------------------------------------------------------------

class ByteCodec:
    """ASCII byte-level tokenizer shared by the NanoWSP patients."""
    max_seq = None
    vocab_chars = None

    def encode(self, text):
        return list(text.encode("ascii", "ignore"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("ascii", "ignore")


class CharCodec:
    """Char-level tokenizer over a fixed alphabet; characters outside it are dropped."""
    max_seq = None

    def __init__(self, chars):
        self.vocab_chars = "".join(sorted(set(chars)))
        self._stoi = {c: i for i, c in enumerate(self.vocab_chars)}

    def encode(self, text):
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids):
        return "".join(self.vocab_chars[int(i)] for i in ids)


class NanoTraceModel(ByteCodec):
    """NanoWSP-3/4 patient rebuilt from a training trace; head count comes from trace metadata."""

    def __init__(self, trace_path, snap_idx=-1):
        tr = ng.load_trace(trace_path)
        self.params = ng.build_from_trace(tr, snap_idx)
        meta = json.loads(str(tr["meta"])) if "meta" in tr.files else {}
        self.n_heads = meta.get("n_heads")

    def logits(self, x):
        return ng.fwd(x, self.params, self.n_heads)[0]


class NanoGPTCharModel(CharCodec):
    """Wraps a nanoGPT-style module (official or runtime) returning all positions when given targets."""

    def __init__(self, module, chars):
        super().__init__(chars)
        self.module = module.eval()
        self.max_seq = module.config.block_size

    def logits(self, x):
        return self.module(x, torch.zeros_like(x))[0]


class CopyOracle(ByteCodec):
    """Positive control for L3: returns the value bound to the queried key, if any is bound."""

    def logits(self, x):
        out = torch.zeros(x.shape[0], x.shape[1], 256)
        for b in range(x.shape[0]):
            lines = self.decode(x[b].tolist()).split("\n")
            key = lines[-1].split(":")[0]
            for line in lines[:-1]:
                if line.startswith(f"{key}: "):
                    out[b, -1, ord(line[len(key) + 2])] = 10.0
        return out


class NGramModel:
    """Count-based add-k n-gram with backoff to shorter contexts: the no-gradient reference."""
    max_seq = None

    def __init__(self, train_text, order, vocab_chars=None, k=0.01):
        self._codec = CharCodec(vocab_chars) if vocab_chars else ByteCodec()
        self.vocab_chars = self._codec.vocab_chars
        self.order = order
        self.k = k
        self.vocab = len(vocab_chars and self.vocab_chars or range(256))
        ids = np.asarray(self.encode(train_text), dtype=np.int64)
        self._tables = {o: self._count(ids, o) for o in range(1, order + 1)}
        self._rows = {}

    def encode(self, text):
        return self._codec.encode(text)

    def decode(self, ids):
        return self._codec.decode(ids)

    def _count(self, ids, o):
        grams = np.zeros(len(ids) - o + 1, dtype=np.int64)
        for j in range(o):
            grams = grams * self.vocab + ids[j:len(ids) - o + 1 + j]
        uniq, counts = np.unique(grams, return_counts=True)
        table = {}
        ctx = uniq // self.vocab
        cuts = np.flatnonzero(np.diff(ctx)) + 1
        for chunk, cnt in zip(np.split(uniq, cuts), np.split(counts, cuts)):
            table[int(chunk[0] // self.vocab)] = (chunk % self.vocab, cnt)
        return table

    def _row(self, context):
        for o in range(min(len(context) + 1, self.order), 0, -1):
            ctx = tuple(context[len(context) - o + 1:]) if o > 1 else ()
            cached = self._rows.get(ctx)
            if cached is not None:
                return cached
            key = 0
            for tok in ctx:
                key = key * self.vocab + tok
            entry = self._tables[o].get(key)
            if entry is not None:
                idx, cnt = entry
                row = np.full(self.vocab, self.k / (cnt.sum() + self.k * self.vocab), dtype=np.float64)
                row[idx] = (cnt + self.k) / (cnt.sum() + self.k * self.vocab)
                log_row = torch.tensor(np.log(row), dtype=torch.float32)
                self._rows[ctx] = log_row
                return log_row
        return torch.full((self.vocab,), -math.log(self.vocab), dtype=torch.float32)

    def logits(self, x):
        B, S = x.shape
        out = torch.zeros(B, S, self.vocab)
        for b in range(B):
            seq = x[b].tolist()
            for t in range(S):
                out[b, t] = self._row(seq[max(0, t - self.order + 2):t + 1])
        return out


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _seq_limit(model, seq):
    return min(seq, model.max_seq) if model.max_seq else seq


@torch.no_grad()
def score_nll(model, text, seq=128, batch=8):
    """Mean NLL in nats of predicting each token from the tokens before it."""
    seq = _seq_limit(model, seq)
    ids = np.asarray(model.encode(text), dtype=np.int64)
    n_windows = (len(ids) - 1) // seq
    if n_windows < 1:
        raise ValueError(f"text too short: {len(ids)} tokens for seq={seq}")
    windows = torch.tensor(np.stack([ids[i * seq:i * seq + seq + 1] for i in range(n_windows)]))
    total, count = 0.0, 0
    for i in range(0, n_windows, batch):
        w = windows[i:i + batch]
        log_p = torch.log_softmax(model.logits(w[:, :-1]).double(), -1)
        total -= log_p.gather(2, w[:, 1:].unsqueeze(-1)).sum().item()
        count += w[:, 1:].numel()
    return total / count


def unigram_entropy(ids):
    counts = np.bincount(np.asarray(ids, dtype=np.int64))
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def check_l1(model, train_text, test_text, baseline_order=3, eph_factor=0.85):
    nll = score_nll(model, test_text)
    baseline = NGramModel(train_text, baseline_order, vocab_chars=model.vocab_chars)
    baseline.max_seq = model.max_seq  # same windows as the model, so both score the same tokens
    base_nll = score_nll(baseline, test_text)
    eph = unigram_entropy(model.encode(test_text))
    passes = nll < base_nll and nll < eph * eph_factor
    return {"nll": round(nll, 4), "ppl": round(math.exp(nll), 3), "unigram_entropy": round(eph, 4),
            "baseline_order": baseline_order, "baseline_nll": round(base_nll, 4),
            "verdict": "PASSES" if passes else "FAILS"}


@torch.no_grad()
def check_long_context(model, text, short=8, long=64, n=400, seed=0, n_boot=1000):
    """NLL gain at the same target from intact context versus context with old tokens replaced.

    Only the last `short` of `long` tokens are kept in the corrupted arm. Sequence length and
    positions are identical in both arms, so position embeddings cannot pass for context use
    (truncating instead gave 3/12 false positives on random nanoGPT inits; this design gave 0/12).
    """
    long = _seq_limit(model, long)
    ids = np.asarray(model.encode(text), dtype=np.int64)
    if len(ids) <= long + 1:
        raise ValueError(f"text too short: {len(ids)} tokens for long={long}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - long - 1, size=n)
    gains = []
    for i in range(0, n, 32):
        w = torch.tensor(np.stack([ids[s:s + long + 1] for s in starts[i:i + 32]]))
        intact, target = w[:, :-1], w[:, -1:]
        corrupted = intact.clone()
        corrupted[:, :long - short] = torch.tensor(ids[rng.integers(0, len(ids), size=(w.shape[0], long - short))])
        lp_intact = torch.log_softmax(model.logits(intact)[:, -1].double(), -1).gather(1, target)
        lp_corrupted = torch.log_softmax(model.logits(corrupted)[:, -1].double(), -1).gather(1, target)
        gains.append((lp_intact - lp_corrupted).squeeze(1).numpy())
    gains = np.concatenate(gains)
    boots = gains[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"short": short, "long": long, "n": n, "mean_gain_nats": round(float(gains.mean()), 4),
            "ci95_low": round(float(low), 4), "ci95_high": round(float(high), 4),
            "verdict": "PASSES" if low > 0 else "FAILS"}


@torch.no_grad()
def _sample(model, prompt, n_new, temperature, seed, ctx=128):
    ctx = _seq_limit(model, ctx)
    gen = torch.Generator().manual_seed(seed)
    ids = model.encode(prompt)
    new = []
    for _ in range(n_new):
        window = torch.tensor([(ids + new)[-ctx:]], dtype=torch.long)
        last = model.logits(window)[0, -1].float()
        if temperature == 0:
            new.append(int(last.argmax()))
        else:
            new.append(int(torch.multinomial(torch.softmax(last / temperature, -1), 1, generator=gen)))
    return model.decode(new)


def check_generation(model, prompt, train_text, n_new=300, temperature=0.8, seed=0,
                     min_distinct=0.35, min_letters=0.6, min_known_words=0.5):
    """Scores generated tokens only: the prompt must not be able to rescue a degenerate model."""
    text = _sample(model, prompt, n_new, temperature, seed)
    grams = {text[i:i + 3] for i in range(len(text) - 2)}
    distinct = len(grams) / max(len(text) - 2, 1)
    letters = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    known = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", train_text)}
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", text)]
    known_rate = sum(w in known for w in words) / max(len(words), 1)
    passes = distinct > min_distinct and letters > min_letters and known_rate >= min_known_words
    return {"sample": text[:200], "greedy_sample": _sample(model, prompt, 100, 0, seed),
            "distinct_3grams_ratio": round(distinct, 3), "letters_space_share": round(letters, 3),
            "known_word_rate": round(known_rate, 3), "n_words": len(words),
            "verdict": "PASSES" if passes else "FAILS"}


# ---------------------------------------------------------------------------
# L3: in-context learning
# ---------------------------------------------------------------------------

def icl_cases(n, seed=0):
    """Paired prompts with identical context values; only the key binding differs between arms."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        keys = set()
        while len(keys) < 4:
            keys.add(rng.choice(string.ascii_uppercase) + rng.choice(string.ascii_uppercase))
        keys = sorted(keys)
        rng.shuffle(keys)
        values = rng.sample(ICL_VALUES, 3)
        pos = i % 3
        query, decoy = keys[pos], keys[3]
        treated = [f"{keys[j]}: {values[j]}\n" for j in range(3)]
        control = list(treated)
        control[pos] = f"{decoy}: {values[pos]}\n"
        cases.append({"key": query, "target": values[pos],
                      "with_binding": "".join(treated) + f"{query}: ",
                      "without_binding": "".join(control) + f"{query}: "})
    return cases


def sign_test_p(wins, losses):
    """Exact two-sided sign test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@torch.no_grad()
def _icl_correct(model, cases, field):
    """Argmax restricted to the ten candidate values, so chance is 0.10 for any model."""
    candidates = torch.tensor([model.encode(v)[0] for v in ICL_VALUES])
    correct = []
    for i in range(0, len(cases), 32):
        chunk = cases[i:i + 32]
        x = torch.tensor([model.encode(c[field]) for c in chunk], dtype=torch.long)
        picks = model.logits(x)[:, -1, candidates].argmax(-1)
        correct += [ICL_VALUES[int(p)] == c["target"] for p, c in zip(picks, chunk)]
    return np.array(correct)


def check_icl(model, n=200, seed=0, min_delta=0.10, max_p=0.01):
    cases = icl_cases(n, seed)
    treated = _icl_correct(model, cases, "with_binding")
    control = _icl_correct(model, cases, "without_binding")
    delta = float(treated.mean() - control.mean())
    p = sign_test_p(int((treated & ~control).sum()), int((control & ~treated).sum()))
    return {"n": n, "acc_with_binding": round(float(treated.mean()), 4),
            "acc_without_binding": round(float(control.mean()), 4), "delta": round(delta, 4),
            "sign_test_p": round(p, 6),
            "verdict": "PASSES" if delta >= min_delta and p <= max_p else "FAILS"}


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def overall_verdict(l1, l2, gen, l3):
    if not all(c["verdict"] == "PASSES" for c in (l1, l2, gen)):
        return "FAILS"
    return "LANGUAGE_MODEL_ICL" if l3["verdict"] == "PASSES" else "LANGUAGE_MODEL"


def split_corpus(corpus, holdout_frac, eval_chars):
    """Baselines fit before the whole holdout, like the model; scoring uses the slice at its tail."""
    cut = int(len(corpus) * (1 - holdout_frac))
    if len(corpus) - cut < eval_chars:
        raise ValueError(f"eval slice of {eval_chars} does not fit in a holdout of {len(corpus) - cut}")
    return corpus[:cut], corpus[-eval_chars:]


def evaluate(model, train_text, test_text, ladder=(1, 2, 3, 4, 5)):
    """Runs every check. `test_text` must not overlap what the model or the baselines trained on."""
    l1 = check_l1(model, train_text, test_text)
    l1["ngram_ladder_nll"] = {str(o): round(score_nll(NGramModel(train_text, o, vocab_chars=model.vocab_chars),
                                                      test_text), 4) for o in ladder}
    l2 = check_long_context(model, test_text)
    gen = check_generation(model, test_text[:200], train_text)
    l3 = check_icl(model)
    return {"L1": l1, "L2_long_context": l2, "GEN": gen, "L3_icl": l3,
            "verdict": overall_verdict(l1, l2, gen, l3)}
