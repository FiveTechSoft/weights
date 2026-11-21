"""Extra evaluation sets for the behavior-extraction probe.
make_range_set: seeded 3-digit x 3-digit problems for range generalization (the mul2x2 sets stop at 2-digit
operands; pairs cannot overlap by construction). make_confirm_set: fresh-seed 2-digit x 2-digit problems
disjoint from every existing mul2x2 set.
usage: python -c "from range_data import make_range_set, make_confirm_set; print(len(make_range_set()), len(make_confirm_set()))"
"""
import random
import sys
sys.path.insert(0, "lab/probes/closed_form_math")
from data import prompt, answer, is_hit  # noqa: F401  (reuse the exact conventions of the main experiment)


def make_range_set(n=100, seed=13, digits=3):
    rng = random.Random(seed)
    items, used = [], set()
    while len(items) < n:
        a = rng.randrange(10 ** (digits - 1), 10 ** digits)
        b = rng.randrange(10 ** (digits - 1), 10 ** digits)
        if (a, b) in used:
            continue
        used.add((a, b))
        used.add((b, a))          # a*b and b*a count as the same problem, as in data.py
        items.append({"a": a, "b": b, "op": "*", "prompt": prompt(a, b, "*"), "answer": answer(a, b, "*")})
    return items


def make_confirm_set(n=100, seed=23):
    """Fresh 2-digit x 2-digit confirmation set, disjoint from EVERY existing mul2x2 set and the smoke pairs.
    Use as the final check if any decision is taken after the first holdout read (the main holdout's A-side
    numbers are already known from the teacher experiment, so it is not fully unseen for this probe)."""
    from data import make_sets, SMOKE_PAIRS
    sets = make_sets("mul2x2")
    used = set(SMOKE_PAIRS)
    for name in sets:
        for it in sets[name]:
            used.add((it["a"], it["b"]))
            used.add((it["b"], it["a"]))   # a*b and b*a count as the same problem, as in data.py
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        a = rng.randrange(10, 100)
        b = rng.randrange(10, 100)
        if (a, b) in used:
            continue
        used.add((a, b)); used.add((b, a))
        items.append({"a": a, "b": b, "op": "*", "prompt": prompt(a, b, "*"), "answer": answer(a, b, "*")})
    return items
