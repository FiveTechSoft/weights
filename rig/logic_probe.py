#!/usr/bin/env python3
"""Deterministic synthetic Horn-logic probe for a local OpenAI-compatible model.

Measures forward chaining / rule application, not general IQ or all forms of logic.
No external dataset or Python dependencies required.
"""
import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

PREDICATES = [
    ("is a glim", "glim"), ("is a nork", "nork"), ("is a vesh", "vesh"),
    ("is a tupa", "tupa"), ("is a zorp", "zorp"), ("is a pel", "pel"),
    ("is a dax", "dax"), ("is a wug", "wug"),
]
NAMES = ["Alda", "Borin", "Cessa", "Dorin", "Elva", "Fenn", "Gorin", "Hessa",
         "Ivar", "Jorin", "Kessa", "Lorin", "Mira", "Nerin", "Orla", "Perrin"]
RULE_FORMS = [
    "If someone {a}, then they {b}.",
    "Everyone who {a} also {b}.",
    "Being {a} implies being {b}.",
]
FACT_FORMS = [
    "{name} {pred}.",
    "It is known that {name} {pred}.",
]
QUESTION_FORMS = [
    "Can we conclude that {name} {pred}?",
    "Does it necessarily follow that {name} {pred}?",
    "Must {name} {pred} be true?",
]

def make_cases(n=120, seed=271828):
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        # Each case is an independent, consistent Horn theory over one person.
        name = NAMES[i % len(NAMES)]
        atoms = [p[0] for p in PREDICATES]
        rng.shuffle(atoms)
        edges = [(atoms[j], atoms[j+1]) for j in range(3)]
        # A fourth edge creates either a longer path or a distractor branch.
        if i % 3 == 0:
            edges.append((atoms[1], atoms[3]))
        facts = {atoms[0]}
        if i % 4 == 0:
            facts.add(atoms[4])
        closure = set(facts)
        changed = True
        while changed:
            changed = False
            for a, b in edges:
                if a in closure and b not in closure:
                    closure.add(b); changed = True
        # Alternate entailed and not-entailed conclusions. Negative queries are
        # selected from the same vocabulary but outside the least fixed point.
        positives = sorted(closure - facts)
        negatives = sorted(set(atoms) - closure)
        want_yes = (i % 2 == 0)
        if want_yes and positives:
            target = rng.choice(positives)
        elif negatives:
            target = rng.choice(negatives)
            want_yes = False
        else:
            target = next(a for a in atoms if a not in closure)
            want_yes = False
        rule_lines = []
        for a, b in edges:
            rule_lines.append(rng.choice(RULE_FORMS).format(a=a, b=b))
        fact_lines = [rng.choice(FACT_FORMS).format(name=name, pred=x) for x in sorted(facts)]
        rng.shuffle(rule_lines); rng.shuffle(fact_lines)
        q = rng.choice(QUESTION_FORMS).format(name=name, pred=target)
        prompt = (
            "Answer using only the stated facts and rules. Do not assume anything "
            "that is not stated or logically implied. A rule applies to every person.\n\n"
            "Rules:\n" + "\n".join("- " + x for x in rule_lines) +
            "\nFacts:\n" + "\n".join("- " + x for x in fact_lines) +
            "\n\n" + q + "\nReply with exactly YES or NO."
        )
        cases.append({"id": i, "expected": "YES" if want_yes else "NO", "prompt": prompt})
    return cases

def extract_answer(text):
    # Strict final-token extraction; explanations are allowed but answer must end in YES/NO.
    hits = re.findall(r"\b(YES|NO)\b", (text or "").upper())
    return hits[-1] if hits else ""

def ask(endpoint, model, case, temperature, seed, retries):
    payload = json.dumps({
        "model": model, "stream": False, "temperature": temperature,
        "top_p": 1.0, "seed": seed + case["id"], "max_tokens": 256,
        "messages": [{"role": "user", "content": case["prompt"]}]
    }).encode()
    error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(endpoint, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            choice = (data.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            answer = extract_answer(raw)
            return {"id": case["id"], "expected": case["expected"],
                    "predicted": answer, "correct": answer == case["expected"],
                    "finish": choice.get("finish_reason"), "response": raw[-500:],
                    "error": None}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError) as e:
            error = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
    return {"id": case["id"], "expected": case["expected"], "predicted": "",
            "correct": False, "finish": "error", "response": "", "error": error}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--model", default="mimo")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--generation-seed", type=int, default=20261103)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default="rig_runs/logic-mimo-orig.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        cases = make_cases()
        assert len(cases) == 120
        assert sum(c["expected"] == "YES" for c in cases) == 60
        assert sum(c["expected"] == "NO" for c in cases) == 60
        assert extract_answer("The answer is YES.") == "YES"
        assert extract_answer("NO") == "NO"
        assert extract_answer("The answer is YES.") == "YES"
        assert extract_answer("NO, because no rule proves it.") == ""
        assert extract_answer("YES, but actually NO") == ""
        assert extract_answer("uncertain") == ""
        print("selftest: OK (120 cases, 60 YES / 60 NO, answer parser)")
        return
    if not 1 <= a.n <= 120 or a.parallel < 1 or not 0 <= a.temperature <= 2:
        ap.error("--n debe estar entre 1 y 120; --parallel >= 1; temperature entre 0 y 2")
    cases = make_cases(seed=a.seed)[:a.n]
    started = time.time()
    if a.parallel == 1:
        results = [ask(a.endpoint, a.model, c, a.temperature, a.generation_seed, a.retries) for c in cases]
        for j, r in enumerate(results, 1):
            print(f"[{j}/{len(cases)}] id={r['id']} pred={r['predicted'] or 'ERROR'}")
    else:
        results = [None] * len(cases)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
            futs = {pool.submit(ask, a.endpoint, a.model, c, a.temperature, a.generation_seed, a.retries): k for k,c in enumerate(cases)}
            for j, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                results[futs[fut]] = fut.result()
                print(f"[{j}/{len(cases)}] id={results[futs[fut]]['id']} pred={results[futs[fut]]['predicted'] or 'ERROR'}")
    errors = sum(r["error"] is not None for r in results)
    scored = [r for r in results if r["error"] is None]
    correct = sum(r["correct"] for r in scored)
    meta = {"model": a.model, "endpoint": a.endpoint, "n": len(cases), "dataset_seed": a.seed,
            "generation_seed_base": a.generation_seed, "generation_seed_rule": "base + case_id",
            "temperature": a.temperature, "top_p": 1.0, "parallel": a.parallel,
            "dataset_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
            "metric": "exact YES/NO accuracy on synthetic Horn entailment",
            "counts": {"correct": correct, "scored": len(scored), "errors": errors, "total": len(results)},
            "accuracy": correct / len(scored) if scored else None, "seconds": round(time.time()-started, 1)}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "items": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] accuracy={correct}/{len(scored)} errors={errors} total={len(results)} -> {out}")

if __name__ == "__main__":
    main()
