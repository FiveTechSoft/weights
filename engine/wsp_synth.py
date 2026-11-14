"""Closed-form weight synthesis: compiles an n-gram table into nanoGPT weights without any gradient.

Every weight is computed directly, and the compiled model reproduces the table's log-probabilities:
  - An orthonormal basis gives each role its own directions: output logits, a large constant "ballast",
    position codes and one token-code slot per context position.
  - Positions are Fourier codes, because shifting a position is then a linear rotation. Attention head j
    queries with code(t) and keys with code(s + j), so it retrieves exactly the token j steps back and
    writes its code into slot j.
  - Each MLP unit is a table entry: it fires only when every slot matches its context, and its c_proj
    column adds that context's correction. Active units telescope from shorter to longer contexts, so an
    unseen context backs off by itself.
  - The ballast dominates the vector norm, so LayerNorm acts as a nearly constant scale.
  - wte and lm_head stay tied, and ln_f rescales so that the logit directions read out exact log-probs.

This is a lookup table in a transformer's body: it has no long-range context and generalizes only as
far as its smoothing does. Attention cannot look further back than `heads` tokens with this layout.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch

from engine import wsp_arch

PROVENANCE = "closed_form"
BALLAST = 1.0e4
POSITION_NORM = 1.0
LONG_NORM = 1.0
WINDOW_SHARPNESS = 4.0
TOKEN_NORM = 1.0
SLOT_NORM = 1.0
UNIT_GAIN = 20.0  # gelu(+10) = 10; gelu(-10) is about 1e-22, so a mismatched unit is silent
ATTENTION_GAIN = 40.0
LAYER_NORM_EPS = 1e-5
MIN_POSITION_DIMS = 16
FEATURE_LONG_SCALE = 0.7
FEATURE_TOKEN_SCALE = 1.5
FEATURE_BIAS_SCALE = 0.7


class SynthesisError(ValueError):
    """The requested table cannot be compiled into the requested architecture."""


@contextlib.contextmanager
def forbid_gradients():
    """Makes any gradient computation inside the block an error, proving synthesis uses none."""
    def refuse(*_args, **_kwargs):
        raise SynthesisError("gradient computation attempted during closed-form synthesis")

    saved = torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad
    torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad = refuse, refuse, refuse
    try:
        with torch.no_grad():
            yield
    finally:
        torch.Tensor.backward, torch.autograd.backward, torch.autograd.grad = saved


class WittenBellTable:
    """Interpolated n-gram: P(x|c) = (count(c,x) + T(c) * P(x|shorter c)) / (n(c) + T(c))."""

    def __init__(self, ids, vocab, order):
        self.vocab = vocab
        self.order = order
        ids = np.asarray(ids, dtype=np.int64)
        counts = np.bincount(ids, minlength=vocab).astype(np.float64) + 0.5
        self._log = {(): np.log(counts / counts.sum())}
        self._count = {(): int(len(ids))}
        for o in range(2, order + 1):
            self._add_order(ids, o)

    def _add_order(self, ids, o):
        grams = np.zeros(len(ids) - o + 1, dtype=np.int64)
        for j in range(o):
            grams = grams * self.vocab + ids[j:len(ids) - o + 1 + j]
        unique, counts = np.unique(grams, return_counts=True)
        contexts = unique // self.vocab
        cuts = np.flatnonzero(np.diff(contexts)) + 1
        for chunk, chunk_counts in zip(np.split(unique, cuts), np.split(counts, cuts)):
            key, context = int(chunk[0] // self.vocab), []
            for _ in range(o - 1):
                context.append(key % self.vocab)
                key //= self.vocab
            context = tuple(reversed(context))
            n, followers = int(chunk_counts.sum()), len(chunk_counts)
            probs = np.exp(self._log[context[1:]]) * followers / (n + followers)
            probs[chunk % self.vocab] += chunk_counts / (n + followers)
            self._log[context] = np.log(probs)
            self._count[context] = n

    def log_prob(self, context):
        """Log-probabilities after `context`, backing off to its longest seen suffix."""
        context = tuple(context)[-(self.order - 1):] if self.order > 1 else ()
        while context not in self._log:
            context = context[1:]
        return self._log[context]

    def num_units(self):
        return len(self._log)

    def units(self, limit=None):
        """(context, log-prob correction) ordered by frequency, shorter contexts first on ties.

        A context never counts more than its suffixes, so any prefix of this order keeps the parents
        of everything it keeps, and truncating only means backing off earlier.
        """
        ranked = sorted(self._log, key=lambda c: (-self._count[c], len(c), c))
        for context in ranked[:limit]:
            parent = self._log[context[1:]] if context else 0.0
            yield context, self._log[context] - parent


def _basis(d, seed):
    """Orthonormal columns: col 0 is ballast (e0 - e1)/sqrt(2); all are orthogonal to the all-ones vector."""
    rng = np.random.default_rng(seed)
    start = np.zeros((d, 2))
    start[:, 0] = 1.0
    start[0, 1], start[1, 1] = 1.0, -1.0
    q, _ = np.linalg.qr(np.hstack([start, rng.standard_normal((d, d - 2))]))
    if np.dot(q[:, 1], start[:, 1]) < 0:
        q[:, 1] = -q[:, 1]
    return q[:, 1:]


def _simplex_codes(vocab):
    """Unit vectors in vocab-1 dimensions with equal pairwise dot product -1/(vocab-1)."""
    centered = np.eye(vocab) - 1.0 / vocab
    u, s, _ = np.linalg.svd(centered)
    codes = u[:, :vocab - 1] * s[:vocab - 1]
    return codes / np.linalg.norm(codes, axis=1, keepdims=True)


def _position_codes(block, dims):
    """cos/sin pairs at spread frequencies; returns codes and the angular frequencies."""
    pairs = dims // 2
    freqs = 2 * np.pi * np.linspace(1, block - 1, pairs) / block if block > pairs else (
        2 * np.pi * (np.arange(pairs) + 1) / max(block, pairs + 1))
    t = np.arange(block)[:, None]
    codes = np.concatenate([np.cos(t * freqs), np.sin(t * freqs)], axis=1) / math.sqrt(pairs)
    return codes, freqs


def _shift_matrix(freqs, steps):
    """Linear map taking the code of position t to the code of position t + steps."""
    pairs = len(freqs)
    c, s = np.cos(freqs * steps), np.sin(freqs * steps)
    rot = np.zeros((2 * pairs, 2 * pairs))
    rot[np.arange(pairs), np.arange(pairs)] = c
    rot[np.arange(pairs), pairs + np.arange(pairs)] = -s
    rot[pairs + np.arange(pairs), np.arange(pairs)] = s
    rot[pairs + np.arange(pairs), pairs + np.arange(pairs)] = c
    return rot


def compile_ngram(arch_path, table, overrides=None, seed=1337, windows=(), char_classes=None, free_blocks=0):
    """Returns a CompiledArch whose weights implement `table`; computes everything in closed form.

    `windows` adds attention heads writing recent character-class averages, and `free_blocks` leaves the
    last MLP blocks empty for fit_block(). Both exist for compile_long_context().
    """
    overrides = dict(overrides or {})
    vocab = table.vocab
    overrides["vocab"] = vocab
    layers = int(overrides.get("layers", 6))
    if "ff" not in overrides:
        overrides["ff"] = math.ceil(table.num_units() / layers)

    with forbid_gradients():
        model = wsp_arch.load_arch_file(arch_path, overrides=overrides, seed=seed)
        cfg = model.config
        d, heads, ff, block = cfg.d, cfg.heads, cfg.ff, cfg.block
        head_dim = d // heads
        slots = table.order - 1
        code_dim = vocab - 1

        if slots + len(windows) > heads:
            raise SynthesisError(f"order {table.order} and {len(windows)} windows need "
                                 f"{slots + len(windows)} attention heads, architecture has {heads}")
        classes = char_classes.shape[1] if len(windows) else 0
        long_dims = classes * len(windows)
        if code_dim > head_dim:
            raise SynthesisError(f"{vocab} tokens need head dimension {code_dim}, architecture has {head_dim}")
        free = d - 1 - vocab - 1 - slots * code_dim - long_dims
        position_dims = min(free - free % 2, head_dim - head_dim % 2)
        if position_dims < MIN_POSITION_DIMS:
            raise SynthesisError(f"only {free} free dimensions remain for position codes with d={d}")

        basis = _basis(d, seed)
        ballast = basis[:, 0]
        logit_dirs = basis[:, 1:1 + vocab]
        position_dirs = basis[:, 1 + vocab:1 + vocab + position_dims]
        slot_start = 1 + vocab + position_dims
        slot_dirs = [basis[:, slot_start + j * code_dim:slot_start + (j + 1) * code_dim] for j in range(slots)]
        long_dirs = basis[:, slot_start + slots * code_dim:slot_start + slots * code_dim + long_dims]
        codes = _simplex_codes(vocab)
        position_codes, freqs = _position_codes(block, position_dims)

        weights = {name: np.zeros(tuple(p.shape)) for name, p in model.named_parameters()}
        for name in weights:
            if ".ln_" in name and name.endswith(".weight"):
                weights[name] += 1.0

        # Tied embedding is the readout basis; positions add their code and the constant ballast.
        weights["transformer.wte.weight"] = TOKEN_NORM * logit_dirs.T
        weights["transformer.wpe.weight"] = POSITION_NORM * position_codes @ position_dirs.T + BALLAST * ballast

        def scale(norm_sq):
            return 1.0 / math.sqrt(norm_sq / d + LAYER_NORM_EPS)

        base_norm_sq = BALLAST ** 2 + TOKEN_NORM ** 2 + POSITION_NORM ** 2
        scale_attn = scale(base_norm_sq)
        scale_mlp = scale(base_norm_sq + slots * SLOT_NORM ** 2)

        # Block 0 attention: head j retrieves the code of the token j steps back into slot j.
        attn = "transformer.h.0.attn"
        c_attn = weights[f"{attn}.c_attn.weight"]
        attn_proj = weights[f"{attn}.c_proj.weight"]
        query_gain = ATTENTION_GAIN / (scale_attn * POSITION_NORM)
        for j in range(slots):
            lo = j * head_dim
            c_attn[lo:lo + position_dims] = query_gain * position_dirs.T
            c_attn[d + lo:d + lo + position_dims] = query_gain * _shift_matrix(freqs, j) @ position_dirs.T
            c_attn[2 * d + lo:2 * d + lo + code_dim] = codes.T @ logit_dirs.T / (scale_attn * TOKEN_NORM)
            attn_proj[:, lo:lo + code_dim] = SLOT_NORM * slot_dirs[j]

        # Window heads: a soft kernel over older tokens averages their character classes.
        pairs = position_dims // 2
        for i, width in enumerate(windows):
            lo = (slots + i) * head_dim
            chosen = np.flatnonzero(freqs * width / 2 <= np.pi)
            chosen = chosen if len(chosen) else np.array([0])
            dims = np.concatenate([chosen, pairs + chosen])
            gain = math.sqrt(WINDOW_SHARPNESS * math.sqrt(head_dim) * pairs / len(chosen)) / (scale_attn * POSITION_NORM)
            center = slots + width / 2
            c_attn[lo:lo + len(dims)] = gain * position_dirs.T[dims]
            c_attn[d + lo:d + lo + len(dims)] = gain * (_shift_matrix(freqs, center) @ position_dirs.T)[dims]
            c_attn[2 * d + lo:2 * d + lo + classes] = char_classes.T @ logit_dirs.T / (scale_attn * TOKEN_NORM)
            attn_proj[:, lo:lo + classes] = LONG_NORM * long_dirs[:, i * classes:(i + 1) * classes]

        # MLP units: one table entry each; higher-order corrections superpose on their suffixes.
        activation = float(torch.nn.functional.gelu(torch.tensor(UNIT_GAIN / 2, dtype=torch.float64)))
        input_gain = UNIT_GAIN / (scale_mlp * SLOT_NORM)
        capacity = (layers - free_blocks) * ff
        used = 0
        for context, correction in table.units(limit=capacity):
            layer, unit = divmod(used, ff)
            used += 1
            matched = len(context)
            row = weights[f"transformer.h.{layer}.mlp.c_fc.weight"][unit]
            row += (UNIT_GAIN / 2 - UNIT_GAIN * matched) / (scale_mlp * BALLAST) * ballast
            for steps_back, token in enumerate(reversed(context)):
                row += input_gain * slot_dirs[steps_back] @ codes[token]
            column = correction.copy()
            if matched == 1:
                column[context[0]] -= TOKEN_NORM  # cancels the embedding of the current token
            column -= column.mean()  # softmax ignores it; a smaller norm keeps LayerNorm scale steady
            weights[f"transformer.h.{layer}.mlp.c_proj.weight"][:, unit] = logit_dirs @ column / activation

        # Final LayerNorm: choose its gain so that logit directions read out exact log-probs.
        weights["transformer.ln_f.weight"] += 1.0 / (scale(base_norm_sq + slots * SLOT_NORM ** 2) * TOKEN_NORM) - 1.0

        for name, parameter in model.named_parameters():
            parameter.copy_(torch.from_numpy(weights[name]).to(parameter.dtype))

    model.synthesis_info = {"provenance": PROVENANCE, "backward_calls": 0, "method": "witten_bell_ngram",
                            "order": table.order, "units_available": table.num_units(), "units_used": used,
                            "ff": ff, "position_dims": position_dims}
    model.synthesis_layout = {"ballast": ballast, "logit_dirs": logit_dirs, "long_dirs": long_dirs,
                              "current_token_dirs": slot_dirs[0] if slot_dirs else None, "scale_mlp": scale_mlp}
    return model


CHAR_CLASSES = ("\n", " ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "aeiou", "bcdfghjklmnpqrstvwxyz", ",.;:!?'-&$3")


def _class_matrix(chars):
    matrix = np.zeros((len(chars), len(CHAR_CLASSES)))
    for j, members in enumerate(CHAR_CLASSES):
        for ch in members:
            if ch in chars:
                matrix[chars.index(ch), j] = 1.0
    return matrix


def fit_block(model, ids, layer, seed, ridge=1e3, batch=32, device="cpu"):
    """Fills one empty MLP block in closed form: random gated features, then ridge on the softmax residual.

    Each block is one boosting step. It uses no backpropagation: the residual comes from a forward pass and
    the output weights from a linear solve. The step size is chosen on the last 30% of `ids`.
    """
    layout = model.synthesis_layout
    block, vocab = model.config.block_size, model.config.vocab_size
    prefix = f"transformer.h.{layer}.mlp"
    c_fc, c_proj = model.get_parameter(f"{prefix}.c_fc.weight"), model.get_parameter(f"{prefix}.c_proj.weight")
    ff = c_fc.shape[0]
    count = (len(ids) - 1) // block
    windows = torch.tensor(np.stack([ids[i * block:i * block + block + 1] for i in range(count)]))
    cut = int(count * 0.7)
    model.to(device)

    long_dirs = torch.tensor(layout["long_dirs"], dtype=torch.float32, device=device)
    token_dirs = torch.tensor(layout["current_token_dirs"], dtype=torch.float32, device=device)
    ballast = torch.tensor(layout["ballast"], dtype=torch.float32, device=device)
    sample = torch.cat([model.input_of(f"{prefix}.c_fc", w[:, :-1].to(device)).flatten(0, 1)
                        for w in windows[:cut].split(batch)]) @ long_dirs
    mean, std = sample.mean(0), sample.std(0) + 1e-9

    rng = np.random.default_rng(seed + layer)
    longs = torch.tensor(rng.normal(0, FEATURE_LONG_SCALE, (ff, long_dirs.shape[1])), dtype=torch.float32, device=device)
    tokens = torch.tensor(rng.normal(0, FEATURE_TOKEN_SCALE, (ff, token_dirs.shape[1])), dtype=torch.float32, device=device)
    offset = torch.tensor(rng.normal(0, FEATURE_BIAS_SCALE, ff), dtype=torch.float32, device=device)
    scale_mlp = layout["scale_mlp"]
    constant = offset - (longs * (mean / std)).sum(1)
    rows = ((longs / std) @ long_dirs.T + tokens @ token_dirs.T / scale_mlp
            + torch.outer(constant, ballast) / (scale_mlp * BALLAST))
    c_fc.copy_(rows.to(c_fc.device))

    eye = torch.eye(vocab, device=device, dtype=torch.float64)
    gram = torch.zeros(ff, ff, device=device, dtype=torch.float64)
    cross = torch.zeros(ff, vocab, device=device, dtype=torch.float64)
    stored = []

    def features(x):
        inputs = model.input_of(f"{prefix}.c_fc", x)
        return torch.nn.functional.gelu(inputs @ c_fc.T)

    for start, w in enumerate(windows.split(batch)):
        x, y = w[:, :-1].to(device), w[:, 1:].to(device)
        phi, logits = features(x).flatten(0, 1), model(x, y.contiguous())[0].flatten(0, 1)
        if (start + 1) * batch <= cut:
            residual = eye[y.flatten()] - torch.softmax(logits.double(), -1)
            gram += (phi.T @ phi).double()
            cross += phi.double().T @ residual
        elif start * batch >= cut:
            stored.append((phi, logits, y.flatten()))

    solution = torch.linalg.solve(gram + ridge * torch.eye(ff, device=device, dtype=torch.float64), cross).float()
    solution -= solution.mean(1, keepdim=True)

    def holdout_nll(step):
        total, n = 0.0, 0
        for phi, logits, y in stored:
            total += torch.nn.functional.cross_entropy(logits + step * (phi @ solution), y, reduction="sum").item()
            n += len(y)
        return total / n

    step = min((0.0, 0.25, 0.5, 1, 2, 4, 8, 16, 32), key=holdout_nll)
    logit_dirs = torch.tensor(layout["logit_dirs"], dtype=torch.float32, device=device)
    c_proj.copy_(logit_dirs @ (step * solution).T)
    return {"layer": layer, "step": step, "holdout_nll_before": round(holdout_nll(0.0), 4),
            "holdout_nll_after": round(holdout_nll(step), 4)}


def compile_long_context(arch_path, train_ids, chars, overrides, seed=1337, order=4, free_blocks=2,
                         windows=(8, 32, 128), fit=True, device="cpu"):
    """Table in the first blocks, plus last blocks fitted on old-context features, all without backprop.

    Fitting uses a table blind to the last 10% of `train_ids` and residuals on that tail, because a table
    that has already seen the text would make the residuals look smaller than they are. The final model
    then gets the table built from all of `train_ids`.
    """
    if free_blocks < 1 or not windows:
        raise SynthesisError("long-context synthesis needs at least one free block and one window")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    classes = _class_matrix(chars)
    vocab = len(chars)
    layers = int(overrides["layers"])
    inner = int(len(train_ids) * 0.9)
    kwargs = dict(overrides=overrides, seed=seed, windows=windows, char_classes=classes, free_blocks=free_blocks)

    final = compile_ngram(arch_path, WittenBellTable(train_ids, vocab, order), **kwargs).eval()
    fits = []
    if fit:
        blind = compile_ngram(arch_path, WittenBellTable(train_ids[:inner], vocab, order), **kwargs).eval()
        with forbid_gradients():
            for layer in range(layers - free_blocks, layers):
                fits.append(fit_block(blind, train_ids[inner:], layer, seed, device=device))
                for name in ("c_fc", "c_proj"):
                    target = final.get_parameter(f"transformer.h.{layer}.mlp.{name}.weight")
                    target.copy_(blind.get_parameter(f"transformer.h.{layer}.mlp.{name}.weight").to(target.device))
    final.synthesis_info.update(provenance="closed_form_ridge", method="witten_bell_plus_ridge_blocks",
                                windows=list(windows), free_blocks=free_blocks, fits=fits,
                                fit_positions_end=len(train_ids))
    return final
