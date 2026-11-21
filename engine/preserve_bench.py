"""engine/preserve_bench.py - capability-preservation battery from local data (lab/probes/preserve/PLAN.md).

Runs beside a sealed confirmation so an edit that helps GSM8K / MATH is also checked for what it may cost:
  humaneval    all 164 HumanEval tasks, greedy, pass@1 by the official check() in a sandboxed subprocess
               (lab/probes/close_probe/he_eval.py: build_program / run_program, imported, not copied);
  compliance   40 instruction-following prompts built here from GSM8K test questions (indices >= GSM_START, none in
               data/sealed_confirm*.json), four instruction kinds x 10, each checked by a regex-level rule. OUR OWN SET,
               NOT IFEval: it measures whether a simple surface instruction is still obeyed, nothing more;
  garbage      over every output of both components: repeated 16-gram (loop), a run of non-ASCII letters >= NONASCII_RUN
               characters (the fragment salad of a broken model), and cut at max_tokens.
  ifeval       (optional, ifeval_n > 0) IFEval prompts (data/bench/ifeval_input_data.jsonl, 541) run as plain chat;
               strict and loose scoring by engine/ifeval_checks.py (24 of 25 instruction types; prompts with the
               unsupported language check are counted but not scored); prompt- and instruction-level accuracies.
  triviaqa     (optional, trivia_n > 0) TriviaQA rc.nocontext validation, first 200 rows (data/bench/
               triviaqa_rc_val_200.jsonl); a short answer is asked for; exact match after SQuAD-style normalisation
               against the dataset's normalized_aliases (em) and the looser "an alias appears in the answer" (contains).
  The ifeval and triviaqa thresholds of the preservation rule are null until calibrated (compare() reports them, they
  do not block).
Same protocol as engine/bench_runner.py: temperature 0, seed 0, the model's own chat template, resume from
<out_dir>/<name>.partial.jsonl, one JSON report with summary + per-item rows (ids kept for paired comparisons).
"""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
from he_eval import build_program, run_program  # noqa: E402
from engine.bench_runner import endpoint_url, served_info, split_think  # noqa: E402
from engine import ifeval_checks  # noqa: E402

KINDS = ("end_done", "words_not_digits", "one_sentence", "all_caps")
TRIVIA_PATH = ROOT / "data" / "bench" / "triviaqa_rc_val_200.jsonl"
TRIVIA_PROMPT = "Answer the following trivia question with just the answer (a name, a number or a short phrase), no explanation.\nQuestion: {q}\nAnswer:"
THRESHOLDS = {"ifeval_strict": None, "ifeval_loose": None, "triviaqa": None}   # not calibrated yet (base twice + placebo)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
INSTRUCTIONS = {
    "end_done": "Answer the question. End your reply with the single word DONE as the last word.",
    "words_not_digits": "Answer the question. Write every number in your reply in words; your reply must contain no digits.",
    "one_sentence": "Answer the question in exactly one sentence.",
    "all_caps": "Answer the question. Write your entire reply in capital letters.",
}
GSM_START = 1000        # compliance prompts come from here on; sealed-set indices are excluded as well
NGRAM = 16
NGRAM_REPS = 3
NONASCII_RUN = 40
TAIL_CHARS = 2000


def load_humaneval(n=None):
    probs = [json.loads(l) for l in (ROOT / "data" / "humaneval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return probs[:n] if n else probs


def garbage_measures(text):
    """repeat_16gram: some NGRAM-word window occurs >= NGRAM_REPS times. nonascii_run: >= NONASCII_RUN consecutive
    non-ASCII letters/symbols with no ASCII letter, digit or space between them."""
    words = text.split()
    counts = {}
    rep = False
    for k in range(max(0, len(words) - NGRAM + 1)):
        g = " ".join(words[k:k + NGRAM])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= NGRAM_REPS:
            rep = True
            break
    # fragment salad: a run of >= NONASCII_RUN non-ASCII characters, or >= SALAD_SCRIPTS distinct non-ASCII scripts
    # inside one SALAD_WINDOW-character window (a broken model interleaves CJK, Cyrillic, Arabic and accented Latin
    # between ASCII fragments; a Spanish or French sentence is one script)
    runs = re.findall(r"(?:[^\x00-\x7f]|[ \t])+", text)
    longest = max((len(re.sub(r"\s", "", r)) for r in runs), default=0)
    salad = longest >= NONASCII_RUN
    if not salad:
        scripts = [_script(c) for c in text]
        for k in range(0, max(1, len(text) - SALAD_WINDOW + 1), SALAD_WINDOW // 4):
            if len({s for s in scripts[k:k + SALAD_WINDOW] if s}) >= SALAD_SCRIPTS:
                salad = True
                break
    return {"repeat_16gram": rep, "nonascii_run": salad}


SALAD_WINDOW = 60
SALAD_SCRIPTS = 3
_RANGES = (("latin_ext", 0x00C0, 0x024F), ("greek", 0x0370, 0x03FF), ("cyrillic", 0x0400, 0x04FF),
           ("hebrew", 0x0590, 0x05FF), ("arabic", 0x0600, 0x06FF), ("devanagari", 0x0900, 0x097F),
           ("thai", 0x0E00, 0x0E7F), ("hangul", 0xAC00, 0xD7AF), ("kana", 0x3040, 0x30FF), ("cjk", 0x4E00, 0x9FFF),
           ("vietnamese", 0x1EA0, 0x1EFF))


def _script(c):
    o = ord(c)
    if o < 0x80:
        return None
    for name, lo, hi in _RANGES:
        if lo <= o <= hi:
            return name
    return None  # punctuation, symbols, emoji: not a script


def _gsm_items():
    return [json.loads(l) for l in (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()]


def _sealed_indices():
    used = set()
    for p in sorted((ROOT / "data").glob("sealed_confirm*.json")):
        try:
            used |= set(json.loads(p.read_text(encoding="utf-8")).get("gsm8k", []))
        except Exception:
            pass
    return used


def compliance_set(n=40):
    """n prompts, n // len(KINDS) per kind, round-robin over eligible GSM8K indices (>= GSM_START, not sealed)."""
    gsm = _gsm_items()
    used = _sealed_indices()
    eligible = [i for i in range(GSM_START, len(gsm)) if i not in used][:n]
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} eligible GSM8K items for the compliance set")
    items = []
    for k, i in enumerate(eligible):
        kind = KINDS[k % len(KINDS)]
        items.append({"i": i, "kind": kind, "question": gsm[i]["question"], "instruction": INSTRUCTIONS[kind]})
    return items


def check_compliance(kind, content):
    text = (content or "").strip()
    if kind == "end_done":
        words = re.findall(r"[A-Za-z]+", text)
        return bool(words) and words[-1] == "DONE"
    if kind == "words_not_digits":
        return bool(text) and not re.search(r"\d", text)
    if kind == "one_sentence":
        if not text:
            return False
        body = re.sub(r"(?<=\d)\.(?=\d)", "", text)          # decimals are not sentence ends
        body = re.sub(r"\b(e\.g|i\.e|etc|Mr|Mrs|Dr|vs)\.", "", body)
        ends = re.findall(r"[.!?]+(?=\s|$)", body)
        return len(ends) == 1 and not re.search(r"\n\s*\n", text)
    if kind == "all_caps":
        letters = re.findall(r"[A-Za-z]", text)
        return len(letters) >= 10 and all(c.isupper() for c in letters)
    raise ValueError(f"unknown compliance kind {kind!r}")


def load_triviaqa(n=None, path=TRIVIA_PATH):
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return items[:n] if n else items


def normalize_answer(s):
    """SQuAD / TriviaQA normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _first_line(content):
    for line in (content or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _aliases(item):
    al = set(item.get("normalized_aliases") or [])
    al |= {normalize_answer(a) for a in (item.get("aliases") or [])}
    if item.get("value"):
        al.add(normalize_answer(item["value"]))
    return {a for a in al if a}


def trivia_match(content, item):
    """Exact match: the normalised first non-empty line of the answer equals one of the normalised aliases."""
    return normalize_answer(_first_line(content)) in _aliases(item)


def trivia_contains(content, item):
    """Looser reading: some normalised alias occurs as a whole-word phrase inside the normalised answer."""
    text = " " + normalize_answer(content) + " "
    return any((" " + a + " ") in text for a in _aliases(item))


def summarize(rows):
    he = [r for r in rows if r["bench"] == "humaneval"]
    co = [r for r in rows if r["bench"] == "compliance"]
    ie = [r for r in rows if r["bench"] == "ifeval"]
    tq = [r for r in rows if r["bench"] == "triviaqa"]
    out = {}
    if ie:
        sc = [r for r in ie if r.get("scorable")]
        ins_s = sum(sum(bool(x) for x in (r.get("per_instruction_strict") or [])) for r in sc)
        ins_l = sum(sum(bool(x) for x in (r.get("per_instruction_loose") or [])) for r in sc)
        ins_n = sum(len(r.get("instruction_ids") or []) for r in sc)
        out["ifeval"] = {"n": len(ie), "scorable": len(sc), "prompt_strict": sum(bool(r["strict"]) for r in sc),
                         "prompt_loose": sum(bool(r["loose"]) for r in sc), "instructions": ins_n,
                         "instruction_strict": ins_s, "instruction_loose": ins_l}
    if tq:
        out["triviaqa"] = {"n": len(tq), "em": sum(bool(r["ok"]) for r in tq), "contains": sum(bool(r.get("contains")) for r in tq),
                           "empty": sum(not (r.get("answer") or "").strip() for r in tq),
                           "cut": sum(r["finish"] == "length" for r in tq)}
    if he:
        toks = sorted(r["tokens"] for r in he)
        m = len(toks) // 2
        median = toks[m] if len(toks) % 2 else (toks[m - 1] + toks[m]) // 2
        out["humaneval"] = {"n": len(he), "ok": sum(bool(r["ok"]) for r in he),
                            "closed": sum(r["finish"] == "stop" for r in he), "median_tokens": median}
    if co:
        by = {}
        for r in co:
            a = by.setdefault(r["kind"], [0, 0])
            a[0] += int(bool(r["ok"]))
            a[1] += 1
        out["compliance"] = {"n": len(co), "ok": sum(bool(r["ok"]) for r in co), "by_kind": by}
    out["garbage"] = {"n": len(rows), "repeat_16gram": sum(bool(r.get("repeat_16gram")) for r in rows),
                      "nonascii_run": sum(bool(r.get("nonascii_run")) for r in rows),
                      "cut_at_max": sum(r["finish"] == "length" for r in rows)}
    return out


def _chat(url, text, max_tokens, lora=None):
    body = {"messages": [{"role": "user", "content": text}], "temperature": 0, "seed": 0, "max_tokens": max_tokens}
    if lora:
        body["lora"] = [{"id": int(i), "scale": float(s)} for i, s in lora]
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    ch = r["choices"][0]
    msg = ch["message"]
    content, reasoning = split_think(msg.get("content") or "", msg.get("reasoning_content") or "")
    return {"content": content, "reasoning": reasoning, "finish": ch["finish_reason"],
            "tokens": r["usage"]["completion_tokens"]}


def run(name, endpoint, out_dir, max_tokens=2048, parallel=2, he_n=164, comp_n=40, lora=None, log=print,
        ifeval_n=0, trivia_n=0, short_max_tokens=None):
    """ifeval_n / trivia_n: how many IFEval prompts / TriviaQA items to run (0 = component off). short_max_tokens caps
    the TriviaQA answers (default max_tokens; the thinking budget still applies to models that think)."""
    url = endpoint_url(endpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("humaneval", p["task_id"], p) for p in load_humaneval(he_n)]
    jobs += [("compliance", it["i"], it) for it in (compliance_set(comp_n) if comp_n else [])]
    jobs += [("ifeval", p["key"], p) for p in (ifeval_checks.load_ifeval(ifeval_n) if ifeval_n else [])]
    jobs += [("triviaqa", it["question_id"], it) for it in (load_triviaqa(trivia_n) if trivia_n else [])]
    short_max = short_max_tokens or max_tokens
    partial = out_dir / f"{name}.partial.jsonl"
    rows = [json.loads(l) for l in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["bench"], r["id"]) for r in rows}
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    log(f"[{name}] preserve: resuming with {len(rows)} done, {len(jobs)} to go")
    t0 = time.time()

    def one(job):
        bench, ident, p = job
        if bench == "humaneval":
            msg = ("Complete the following Python function. Return the full function in a ```python code block.\n\n"
                   f"```python\n{p['prompt']}```")
            r = _chat(url, msg, max_tokens, lora)
            ok = run_program(build_program(p["prompt"], r["content"], p["test"], p["entry_point"]))
            row = {"bench": bench, "id": ident, "ok": bool(ok)}
        elif bench == "compliance":
            r = _chat(url, f"{p['question']}\n{p['instruction']}", max_tokens, lora)
            row = {"bench": bench, "id": ident, "kind": p["kind"], "ok": check_compliance(p["kind"], r["content"])}
        elif bench == "ifeval":
            r = _chat(url, p["prompt"], max_tokens, lora)
            s = ifeval_checks.score_prompt(p, r["content"])
            row = {"bench": bench, "id": ident, "ok": bool(s["strict"]), **s}
        else:  # triviaqa
            r = _chat(url, TRIVIA_PROMPT.format(q=p["question"]), short_max, lora)
            row = {"bench": bench, "id": ident, "ok": trivia_match(r["content"], p),
                   "contains": trivia_contains(r["content"], p), "answer": _first_line(r["content"])[:200]}
        full = r["reasoning"] + r["content"]
        row.update({"finish": r["finish"], "tokens": r["tokens"], "reasoning_chars": len(r["reasoning"]),
                    "tail": full[-TAIL_CHARS:], **garbage_measures(full)})
        return row

    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            rows.append(row)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                log(f"[{name}] preserve {k}/{len(jobs)} {time.time() - t0:.0f}s")
    summary = summarize(rows)
    report = {"model": name, "endpoint": url, "max_tokens": max_tokens, "lora": lora, "served": served_info(url),
              "seconds": round(time.time() - t0), "summary": summary, "items": rows}
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{name} preserve {json.dumps(summary)}")
    return report


def compare(base_rows, arm_rows):
    """Paired reading (PLAN): per component, items base-ok->arm-not and the reverse; the preservation rule."""
    b = {(r["bench"], r["id"]): r for r in base_rows}
    a = {(r["bench"], r["id"]): r for r in arm_rows}
    out = {}
    for bench in ("humaneval", "compliance"):
        keys = [k for k in b if k[0] == bench and k in a]
        out[bench] = {"n": len(keys), "base": sum(bool(b[k]["ok"]) for k in keys), "arm": sum(bool(a[k]["ok"]) for k in keys),
                      "lost": sum(bool(b[k]["ok"]) and not a[k]["ok"] for k in keys),
                      "gained": sum((not b[k]["ok"]) and bool(a[k]["ok"]) for k in keys)}
    # optional components (ifeval strict + loose, triviaqa em); thresholds null until calibrated, so they report only
    ie = [k for k in b if k[0] == "ifeval" and k in a and b[k].get("scorable", True) and a[k].get("scorable", True)]
    if ie:
        out["ifeval"] = {"n": len(ie),
                         "base_strict": sum(bool(b[k].get("strict")) for k in ie), "arm_strict": sum(bool(a[k].get("strict")) for k in ie),
                         "base_loose": sum(bool(b[k].get("loose")) for k in ie), "arm_loose": sum(bool(a[k].get("loose")) for k in ie),
                         "lost_strict": sum(bool(b[k].get("strict")) and not a[k].get("strict") for k in ie),
                         "gained_strict": sum((not b[k].get("strict")) and bool(a[k].get("strict")) for k in ie),
                         "lost_loose": sum(bool(b[k].get("loose")) and not a[k].get("loose") for k in ie),
                         "gained_loose": sum((not b[k].get("loose")) and bool(a[k].get("loose")) for k in ie)}
    tq = [k for k in b if k[0] == "triviaqa" and k in a]
    if tq:
        out["triviaqa"] = {"n": len(tq), "base": sum(bool(b[k]["ok"]) for k in tq), "arm": sum(bool(a[k]["ok"]) for k in tq),
                           "lost": sum(bool(b[k]["ok"]) and not a[k]["ok"] for k in tq),
                           "gained": sum((not b[k]["ok"]) and bool(a[k]["ok"]) for k in tq)}
    out["thresholds"] = dict(THRESHOLDS)
    gb, ga = summarize(base_rows)["garbage"], summarize(arm_rows)["garbage"]
    rate = lambda g: (g["repeat_16gram"] + g["nonascii_run"]) / max(1, g["n"])  # noqa: E731
    out["garbage_rate"] = {"base": round(rate(gb), 4), "arm": round(rate(ga), 4)}
    ok = (out["humaneval"]["arm"] >= out["humaneval"]["base"] - 5
          and out["compliance"]["arm"] >= out["compliance"]["base"] - 3
          and rate(ga) <= rate(gb) + 0.02)
    notes = []
    if ie and THRESHOLDS["ifeval_strict"] is not None:
        ok = ok and out["ifeval"]["arm_strict"] >= out["ifeval"]["base_strict"] - THRESHOLDS["ifeval_strict"]
    if tq and THRESHOLDS["triviaqa"] is not None:
        ok = ok and out["triviaqa"]["arm"] >= out["triviaqa"]["base"] - THRESHOLDS["triviaqa"]
    if (ie or tq) and (THRESHOLDS["ifeval_strict"] is None or THRESHOLDS["triviaqa"] is None):
        notes.append("ifeval / triviaqa not calibrated: reported, not part of the preserves flag yet")
    out["preserves"] = ok
    out["notes"] = notes
    return out
