"""Discover a restricted modular affine law and lower it to WSP registers.

Hypothesis family: f(x,y)=(a*x+b*y) mod 16, a,b in 0..15.
No donor, no gradients, no inference-time symbolic interpreter.
Reject ambiguous or inconsistent observations rather than guessing.
"""
from numbers import Integral


def discover(examples):
    rows = list(examples)
    if not rows:
        raise ValueError("no observations")
    if any(len(row) != 3 or any(not isinstance(v, Integral) or isinstance(v, bool)
                                or not 0 <= v < 16 for v in row) for row in rows):
        raise ValueError("expected triples of integers in 0..15")
    candidates = [(a,b) for a in range(16) for b in range(16)
                  if all((a*x+b*y)%16 == z for x,y,z in rows)]
    if len(candidates) != 1:
        raise ValueError(f"expected unique law, found {len(candidates)}")
    return candidates[0]


def lower(law):
    """Shortest straight-line expression from a bounded constructive scheme.

    Shared register budget is eight assignments; signed binary expansions
    use additions/subtractions and fall back to rejection if too expensive.
    """
    a,b = law
    if any(not isinstance(v, Integral) or isinstance(v,bool) or not 0 <= v < 16 for v in (a,b)):
        raise ValueError("invalid coefficients")
    lines = []
    def emit(op, x, y):
        name = f"r{len(lines)}"
        lines.append(f"{name} = {op} {x} {y}")
        return name
    def multiple(k, x):
        if k == 0:
            return None
        if k == 1:
            return x
        # Positive binary expansion, at most six assignments per coefficient.
        acc = x
        for bit in bin(k)[3:]:
            acc = emit("ADD",acc,acc)
            if bit == "1":
                acc = emit("ADD",acc,x)
        return acc
    # Search both modular representatives and select the shortest lowering.
    variants = []
    for ca in (a,a-16) if a else (0,):
        for cb in (b,b-16) if b else (0,):
            lines.clear()
            x,y = multiple(abs(ca),"x"),multiple(abs(cb),"y")
            if x is None and y is None:
                ret = emit("SUB","x","x")
            elif x is None or y is None:
                ret = y if x is None else x
                if ca < 0 or cb < 0:
                    ret = emit("SUB",emit("SUB","x","x"),ret)
            elif ca >= 0 and cb >= 0:
                ret = emit("ADD",x,y)
            elif ca >= 0:
                ret = emit("SUB",x,y)
            elif cb >= 0:
                ret = emit("SUB",y,x)
            else:
                ret = emit("SUB",emit("SUB","x","x"),emit("ADD",x,y))
            variants.append((len(lines),"\n".join(lines+[f"RETURN {ret}"])+"\n"))
    count,source = min(variants)
    if count > 8:
        raise ValueError("law exceeds register budget")
    return source


def compile_examples(arch, examples, *, seed=1337):
    from engine.wsp_register import compile_program
    law = discover(examples)
    source = lower(law)
    return law, source, compile_program(arch, source, seed=seed)
