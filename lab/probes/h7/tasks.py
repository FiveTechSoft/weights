"""H7 task registry (h7_plan.md + addendum a): nine digit tasks, 3-shot prompts, deterministic disjoint splits.

Each task: gen(random.Random) -> (question ending in "=", answer string). Splits per task: 3 shots, train 8 000,
dev 200, test 500, no question shared. Grading: the generated text up to the first newline equals the answer.
"""
import random

import torch


def _nd(r, d):
    return r.randint(10 ** (d - 1), 10 ** d - 1)


def _bin(op, f):
    return lambda a, b: (f"{a}{op}{b}=", str(f(a, b)))


TASKS = {
    "mul3x3": lambda r: _bin("*", lambda a, b: a * b)(_nd(r, 3), _nd(r, 3)),
    "mul4x3": lambda r: _bin("*", lambda a, b: a * b)(_nd(r, 4), _nd(r, 3)),
    "sq4": lambda r: (lambda a: (f"{a}^2=", str(a * a)))(_nd(r, 4)),
    "mul2x2x1": lambda r: (lambda a, b, c: (f"{a}*{b}*{c}=", str(a * b * c)))(_nd(r, 2), _nd(r, 2), _nd(r, 1)),
    "muladd1": lambda r: (lambda a, b, c: (f"{a}*{b}+{c}=", str(a * b + c)))(_nd(r, 2), _nd(r, 1), _nd(r, 2)),
    "div6_2": lambda r: _bin("//", lambda a, b: a // b)(_nd(r, 6), _nd(r, 2)),
    "mod3_1": lambda r: _bin("%", lambda a, b: a % b)(_nd(r, 3), r.randint(3, 19)),
    "add3x3": lambda r: (lambda a, b, c: (f"{a}+{b}+{c}=", str(a + b + c)))(_nd(r, 5), _nd(r, 5), _nd(r, 5)),
    "rev4": lambda r: (lambda a: (f"reverse {a}=", str(a)[::-1]))(_nd(r, 4)),
}
ORDER = list(TASKS)
SIZES = {"train": 8000, "dev": 200, "test": 500}
_CACHE = {}


def make_splits(name):
    if name in _CACHE:
        return _CACHE[name]
    r = random.Random(2027 + ORDER.index(name))
    seen, items = set(), []
    need = 3 + sum(SIZES.values())
    while len(items) < need:
        q, a = TASKS[name](r)
        if q not in seen:
            seen.add(q)
            items.append((q, a))
    out, i = {"shots": items[:3]}, 3
    for k in ("train", "dev", "test"):
        out[k] = items[i:i + SIZES[k]]
        i += SIZES[k]
    _CACHE[name] = out
    return out


def prompt(name, q):
    return "".join(f"{sq}{sa}\n" for sq, sa in make_splits(name)["shots"]) + q


def target(answer):
    return f"{answer}\n"


def grade(text, answer):
    return text.split("\n", 1)[0] == answer


@torch.no_grad()
def eval_task(model, tok, name, items, batch=100, max_new=12):
    """Greedy 3-shot accuracy on items [(q, a)], grouped by prompt length (no padding)."""
    model.eval()
    oks = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = [tok.encode(prompt(name, q), add_special_tokens=False).ids for q, _ in chunk]
        groups, res = {}, [None] * len(chunk)
        for j, e in enumerate(enc):
            groups.setdefault(len(e), []).append(j)
        for L, idx in groups.items():
            x = torch.tensor([enc[j] for j in idx], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=0, use_cache=True)
            for j, row in zip(idx, y[:, L:].tolist()):
                res[j] = tok.decode(row)
        oks += [grade(t, a) for t, (_, a) in zip(res, chunk)]
    return {"n": len(oks), "acc": sum(oks) / len(oks), "ok_mask": oks}
