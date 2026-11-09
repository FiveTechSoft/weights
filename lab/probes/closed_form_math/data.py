"""Seeded problem sets for the closed-form write test. Operand pairs are disjoint across all sets of one skill."""
import random

SKILLS = {  # name: (digits_a, digits_b, op)
    "add2": (2, 2, "+"), "add3": (3, 3, "+"), "add4": (4, 4, "+"), "mul2x2": (2, 2, "*"),
}
SMOKE_PAIRS = {(47, 68), (12, 30)}   # observed in a 3-prompt smoke test before the sets were fixed
SEEDS = {"calibration": 1, "write": 2, "holdout": 3}
SIZES = {"calibration": 100, "write": 40, "holdout": 100}


def prompt(a, b, op):
    return "Q: %d %s %d = ?\nA:" % (a, op, b)


def answer(a, b, op):
    return str(a + b if op == "+" else a * b)


def make_sets(skill):
    da, db, op = SKILLS[skill]
    used, out = set(SMOKE_PAIRS), {}  # pairs seen in the smoke test are excluded from every set
    for name in ("calibration", "write", "holdout"):
        rng = random.Random(SEEDS[name] * 1000 + len(skill) * 7 + sum(map(ord, skill)))
        items = []
        while len(items) < SIZES[name]:
            a = rng.randrange(10 ** (da - 1), 10 ** da)
            b = rng.randrange(10 ** (db - 1), 10 ** db)
            if op == "+" and name != "calibration" and da == 2 and (a % 10 + b % 10) < 10 and rng.random() < 0.5:
                continue                      # keep roughly half the 2-digit sums carrying
            if (a, b) in used:
                continue
            used.add((a, b))
            items.append({"a": a, "b": b, "op": op, "prompt": prompt(a, b, op), "answer": answer(a, b, op)})
        out[name] = items
    return out


def is_hit(generated, ans):
    """Exact match: text up to the first newline, stripped, equals the answer string."""
    return generated.split("\n")[0].strip() == ans
