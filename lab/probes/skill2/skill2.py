"""Second behaviour (lab/probes/skill2/PLAN.md, roadmap gap 17): "reason in Spanish" on English problems, read from
Qwen3.5-9B's own answers (Spanish vs English content on the same discovery problems), applied as a llama.cpp control
vector, judged on the first 300 words of the thinking. Nothing here is "conclude": the measure is a language, taken
on the head of the reasoning, independent of length and of whether the trace is cut.

  python lab/probes/skill2/skill2.py split                     -> lab/runs/skill2/split.json
  python lab/probes/skill2/skill2.py gen PORT                  -> gens.jsonl (discovery problems, English and Spanish arms)
  python lab/probes/skill2/skill2.py rows                      -> pos.jsonl / neg.jsonl / rows.json (exit 1 if < 30 pairs)
  python lab/probes/skill2/skill2.py arm NAME PORT SET [es]    -> NAME.json (SET: screen | dev | sealed; "es" appends the
                                                                 Spanish instruction to every prompt)
  python lab/probes/skill2/skill2.py choose                    -> scale.json (screen rule)
  python lab/probes/skill2/skill2.py verdict PREFIX            -> PREFIX_verdict.json (C1-C3 on PREFIX_base_en /
                                                                 PREFIX_vec_en / PREFIX_placebo_en)
  python lab/probes/skill2/skill2.py make_sealed               -> data/sealed_skill2.json (refuses to overwrite)
  python lab/probes/skill2/skill2.py calib                     -> calib.jsonl (finished English traces, for the compile op)
"""
import concurrent.futures
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "lab" / "runs" / "skill2"
BASE_HF = ROOT / "hf_cache" / "qwen35-9b-hf"
SEALED = ROOT / "data" / "sealed_skill2.json"
SEALED_SEED = 20261118
SCALES = ["0.5", "1.0", "2.0", "4.0"]
INSTR_ES = " Piensa y responde en español."
HEAD_WORDS = 300
MIN_PAIRS = 30
GAIN_PP = 0.30        # screen and C1: Spanish thinking at least base + 30 pp
ACC_SLACK = 3         # screen and C3: correct answers at least base - 3 per bench

ES_WORDS = set("""de la que el en y los del se las por un para con una su al lo como más pero sus le ya o este sí
porque esta entre cuando muy sin sobre también hasta hay donde quien desde todo nos durante todos uno les ni contra
otros ese eso ante ellos esto antes algunos qué unos otro otras otra él tanto esa estos mucho quienes nada muchos cual
poco ella estas algunas algo nosotros entonces calculamos cuántas cuántos después luego tiene tienen ahora primero cada
son es número respuesta obtener sumamos restamos multiplicamos dividimos así tenemos debe queda quedan hacemos veamos
verificar comprobemos comprobar cálculo paso pasos problema dice pregunta cuánto cuánta restan quedó sería siguiente
mitad doble cuesta precio gasta gana compra vende minutos horas días semanas""".split())
EN_WORDS = set("""the an and of to in is are that for with as on by it this be we so then have has from or not at
which each if will can let first now total get how many much there their she they his her what need step answer check
compute calculate multiply divide subtract add equals means find gives remaining left after before because since
would should could does did were was been being into than them these those over per each every half twice costs price
spends earns buys sells minutes hours days weeks problem says asks next verify""".split())
_WORD = re.compile(r"[a-záéíóúñü]+")


# ---------------- judge ----------------
def head(text, n=HEAD_WORDS):
    return " ".join((text or "").split()[:n])


def lang(text):
    """'es' | 'en' | 'mixed' | 'unk' from stopword counts; the larger side must lead by 1.5x."""
    words = _WORD.findall((text or "").lower())
    if len(words) < 8:
        return "unk"
    es = sum(w in ES_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    if es + en < 4:
        return "unk"
    if es > 1.5 * en:
        return "es"
    if en > 1.5 * es:
        return "en"
    return "mixed"


# ---------------- rows ----------------
def build_rows(gens):
    """gens: [{i, prompt, en: {content}, es: {content}}] -> pos (Spanish answers), neg (English answers), stats.
    Same prompt for both rows: the direction is the answer's language, not the presence of an instruction."""
    pos, neg = [], []
    stats = {"n": len(gens), "kept": 0, "es_noncompliant": 0, "en_noncompliant": 0, "unknown": 0}
    for g in gens:
        le, ls = lang(g["en"]["content"]), lang(g["es"]["content"])
        if "unk" in (le, ls):
            stats["unknown"] += 1
        elif ls != "es":
            stats["es_noncompliant"] += 1
        elif le != "en":
            stats["en_noncompliant"] += 1
        else:
            stats["kept"] += 1
            pos.append({"prompt": g["prompt"], "response": g["es"]["content"], "think": False})
            neg.append({"prompt": g["prompt"], "response": g["en"]["content"], "think": False})
    return pos, neg, stats


# ---------------- data ----------------
def split(write=False):
    s = {"discovery_gsm_train": list(range(64, 128)), "screen_gsm_train": list(range(128, 160)),
         "note": "GSM8K train; cc used train 0-63 for its discovery; dev = bench_runner defaults; sealed = data/sealed_skill2.json"}
    if write:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "split.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


def _gsm(name):
    return [json.loads(x) for x in (ROOT / "data" / f"gsm8k_{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def sealed_ids(seed, gsm_n, math_n, gsm_total, math_total, dev_gsm, dev_math, existing):
    used_g = set(dev_gsm) | {i for s in existing for i in s["gsm8k"]}
    used_m = set(dev_math) | {i for s in existing for i in s["math500"]}
    r = random.Random(seed)
    g = sorted(r.sample(sorted(set(range(gsm_total)) - used_g), gsm_n))
    m = sorted(r.sample(sorted(set(range(math_total)) - used_m), math_n))
    return {"gsm8k": g, "math500": m}


def make_sealed(path=SEALED, gsm_n=100, math_n=50):
    from engine import bench_runner as b, sealed_set as ss
    if Path(path).exists():
        sys.exit(f"{path} exists: a sealed set is made once")
    m5 = b._m5()
    total_m = len(m5.load_items())
    existing = [ss.load(ROOT / "data" / f) for f in ("sealed_confirm.json", "sealed_confirm2.json", "sealed_confirm3.json")]
    ids = sealed_ids(SEALED_SEED, gsm_n, math_n, len(_gsm("test")), total_m, range(100), m5.select_indices(2026, 50, total_m), existing)
    s = {"seed": SEALED_SEED, "gsm8k": ids["gsm8k"], "math500": ids["math500"], "sha256": ss.digest(ids["gsm8k"], ids["math500"]),
         "excluded": "dev (GSM8K test 0-99, MATH-500 select_indices(2026, 50)) and data/sealed_confirm{,2,3}.json",
         "use": "single use: lab/probes/skill2/PLAN.md sealed step; do not run again"}
    Path(path).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({"path": str(path), "gsm8k": len(s["gsm8k"]), "math500": len(s["math500"]), "sha256": s["sha256"][:16]}))
    return s


def items_for(which):
    """-> [(bench, i, text, gold)] for screen (GSM8K train 128-159), dev (bench_runner defaults) or sealed."""
    from engine import bench_runner as b, sealed_set as ss
    m5 = b._m5()
    if which == "screen":
        tr = _gsm("train")
        return [("gsm8k", i, tr[i]["question"], b.gsm_gold(tr[i]["answer"])) for i in split()["screen_gsm_train"]]
    te, ma = _gsm("test"), m5.load_items()
    if which == "dev":
        g, m = list(range(100)), m5.select_indices(2026, 50, len(ma))
    elif which == "sealed":
        s = ss.load(SEALED)
        g, m = s["gsm8k"], s["math500"]
    else:
        raise ValueError(which)
    return ([("gsm8k", i, te[i]["question"], b.gsm_gold(te[i]["answer"])) for i in g]
            + [("math500", i, ma[i]["problem"], ma[i]["answer"]) for i in m])


# ---------------- generation ----------------
def gen(port, max_tokens=2048, parallel=2):
    """Discovery problems answered twice by the served base: plain, and with the Spanish instruction."""
    from engine import bench_runner as b
    RUNS.mkdir(parents=True, exist_ok=True)
    url = b.endpoint_url(port)
    tr = _gsm("train")
    out = RUNS / "gens.jsonl"
    done = {r["i"] for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())} if out.exists() else set()
    jobs = [i for i in split()["discovery_gsm_train"] if i not in done]

    def one(i):
        q = tr[i]["question"]
        en, es = b.ask(url, q, max_tokens), b.ask(url, q, max_tokens, extra=INSTR_ES)
        pack = lambda r: {"content": r["content"], "reasoning": r["reasoning"], "think_lang": lang(head(r["reasoning"])),
                          "finish": r["finish"], "tokens": r["tokens"]}
        return {"src": "gsm_train", "i": i, "prompt": q + b.SUFFIX, "en": pack(en), "es": pack(es)}
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 8 == 0:
                print(f"[gen] {k}/{len(jobs)}", flush=True)


def rows():
    gens = [json.loads(x) for x in (RUNS / "gens.jsonl").read_text(encoding="utf-8").splitlines()]
    pos, neg, stats = build_rows(gens)
    for n, rr in (("pos", pos), ("neg", neg)):
        (RUNS / f"{n}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rr), encoding="utf-8")
    stats["es_think_with_instruction"] = sum(g["es"]["think_lang"] == "es" for g in gens)
    stats["es_think_without"] = sum(g["en"]["think_lang"] == "es" for g in gens)
    stats["enough"] = stats["kept"] >= MIN_PAIRS
    (RUNS / "rows.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))
    return stats


# ---------------- arms ----------------
def run_arm(name, port, which, extra="", max_tokens=2048, parallel=2, out_dir=RUNS, log=print):
    """bench_runner's protocol (same ask, graders, tail, loop flag) plus the language of the reasoning head and of
    the answer. Resumable. extra = "" or INSTR_ES."""
    from engine import bench_runner as b
    m5 = b._m5()
    url = b.endpoint_url(port)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{name}.partial.jsonl"
    rows_ = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["bench"], r["i"]) for r in rows_}
    jobs = [j for j in items_for(which) if (j[0], j[1]) not in done]
    t0 = time.time()
    log(f"[{name}] resuming with {len(rows_)} done, {len(jobs)} to go")

    def one(job):
        bench, i, text, gold = job
        r = b.ask(url, text, max_tokens, extra=extra)
        pred = m5.extract_pred(r["content"])
        ok1 = bool(pred) and m5.equivalent(pred, gold)
        ok2 = b.grade_gsm_independent(r["content"], gold) if bench == "gsm8k" else (
            bool(pred) and b.simple_norm(pred) == b.simple_norm(gold))
        tail = (r["reasoning"] + r["content"])[-b.TAIL_CHARS:]
        tl, al = lang(head(r["reasoning"])), lang(r["content"])
        return {"bench": bench, "i": i, "gold": gold, "pred": pred, "ok": ok1, "ok_independent": ok2,
                "finish": r["finish"], "tokens": r["tokens"], "tok_s": r["tok_s"], "reasoning_chars": len(r["reasoning"]),
                "think_lang": tl, "answer_lang": al, "es_think": tl == "es", "es_answer": al == "es",
                "reasoning_head": head(r["reasoning"], 60), "tail": tail, "loop": b.loop_tail(tail)}
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for k, row in enumerate(ex.map(one, jobs), 1):
            rows_.append(row)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                log(f"[{name}] {k}/{len(jobs)} {time.time() - t0:.0f}s")
    report = {"model": name, "endpoint": url, "max_tokens": max_tokens, "extra": extra, "set": which,
              "served": b.served_info(url), "seconds": round(time.time() - t0), "summary": summarize(rows_), "items": rows_}
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{name} {json.dumps(report['summary'])}")
    return report


def summarize(rows_):
    out = {}
    for bench in ("gsm8k", "math500"):
        rs = [r for r in rows_ if r["bench"] == bench]
        if rs:
            out[bench] = {"n": len(rs), "es_think": sum(bool(r["es_think"]) for r in rs),
                          "es_answer": sum(bool(r["es_answer"]) for r in rs), "ok": sum(bool(r["ok"]) for r in rs),
                          "ok_independent": sum(bool(r["ok_independent"]) for r in rs),
                          "cut": sum(r["finish"] == "length" for r in rs), "loops": sum(bool(r.get("loop")) for r in rs),
                          "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs))}
    return out


# ---------------- rules ----------------
def choose_scale(base, arms, n):
    """Screen rule: among scales with correct >= base - ACC_SLACK, the most Spanish thinking (ties -> smaller scale);
    None unless it reaches base + GAIN_PP of n."""
    safe = [s for s in arms if arms[s]["ok"] >= base["ok"] - ACC_SLACK]
    if not safe:
        return None
    best = max(safe, key=lambda s: (arms[s]["es_think"], -float(s)))
    return best if arms[best]["es_think"] >= base["es_think"] + GAIN_PP * n else None


def binom_two_sided(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def _paired(base, arm, key):
    B = {(r["bench"], r["i"]): r for r in base}
    A = {(r["bench"], r["i"]): r for r in arm}
    ks = sorted(set(B) & set(A))
    win = sum(bool(A[k][key]) and not B[k][key] for k in ks)
    loss = sum(bool(B[k][key]) and not A[k][key] for k in ks)
    return {"n": len(ks), "base": sum(bool(B[k][key]) for k in ks), "arm": sum(bool(A[k][key]) for k in ks),
            "win": win, "loss": loss, "p": binom_two_sided(win, loss)}


def dev_verdict(base, vec, pla):
    """C1 vector: Spanish thinking >= base + 30 pp with p < 0.05 on discordant items; C2 placebo: p >= 0.05 and
    < base + 10 pp; C3 accuracy: per bench, vector correct >= base - 3. All items pooled for C1-C2."""
    v = _paired(base, vec, "es_think")
    p = _paired(base, pla, "es_think")
    n = v["n"]
    c1 = v["p"] < 0.05 and (v["arm"] - v["base"]) >= GAIN_PP * n
    c2 = p["p"] >= 0.05 and (p["arm"] - p["base"]) < 0.10 * n
    acc = {}
    for bench in ("gsm8k", "math500"):
        bb = [r for r in base if r["bench"] == bench]
        vv = [r for r in vec if r["bench"] == bench]
        if bb:
            acc[bench] = {"base": sum(bool(r["ok"]) for r in bb), "vec": sum(bool(r["ok"]) for r in vv),
                          "pairs": _paired(bb, vv, "ok")}
    c3 = all(a["vec"] >= a["base"] - ACC_SLACK for a in acc.values())
    return {"C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "verdict": "PASS" if c1 and c2 and c3 else "FAIL",
            "es_think_vec": v, "es_think_placebo": p, "accuracy": acc,
            "tokens": {"base": sum(r["tokens"] for r in base) / max(1, len(base)),
                       "vec": sum(r["tokens"] for r in vec) / max(1, len(vec))}}


def make_calib_rows(gens):
    """Calibration rows for compile_control_vector: finished English-arm traces, reasoning without a leading <think>
    (the Qwen template writes the tag; lab/probes/cc_llama/rows.py rule)."""
    out = []
    for g in gens:
        e = g["en"]
        if e.get("finish") != "stop" or not (e.get("reasoning") or "").strip() or not (e.get("content") or "").strip():
            continue
        text = e["reasoning"].lstrip("\n")
        if text.startswith("<think>"):
            text = text[len("<think>"):].lstrip("\n")
        out.append({"prompt": g["prompt"], "reasoning": text})
    return out


def calib():
    gens = [json.loads(x) for x in (RUNS / "gens.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_ = make_calib_rows(gens)
    (RUNS / "calib.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_), encoding="utf-8")
    print(json.dumps({"gens": len(gens), "calib": len(rows_)}))
    return rows_


def fidelity(vec, merged):
    """C4: the merged model equals the vector item by item on Spanish thinking (discordant pairs not significant)."""
    p = _paired(vec, merged, "es_think")
    return {"C4": p["p"] >= 0.05, "pairs": p}


def _load(name):
    return json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))["items"]


def choose():
    base = _load("screen_base")
    sb = summarize(base)["gsm8k"]
    arms = {s: summarize(_load(f"screen_s{s}"))["gsm8k"] for s in SCALES}
    sc = choose_scale(sb, arms, sb["n"])
    res = {"base": sb, "by_scale": arms, "scale": sc}
    (RUNS / "scale.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res


def verdict(prefix):
    """dev / sealed: C1-C3 on {prefix}_{base,vec,placebo}_en. cdev (compile): the same with merged / placebo_merged in the
    vec / placebo slots (the chain copies dev_base_en to cdev_base_en) plus C4 against dev_vec_en."""
    v = dev_verdict(_load(f"{prefix}_base_en"), _load(f"{prefix}_vec_en"), _load(f"{prefix}_placebo_en"))
    if prefix == "cdev":
        f = fidelity(_load("dev_vec_en"), _load("cdev_vec_en"))
        v.update(f)
        v["verdict"] = "PASS" if v["verdict"] == "PASS" and f["C4"] else "FAIL"
    for extra in ("base_es", "vec_es"):   # secondary arms: with the instruction (reported, not judged)
        if (RUNS / f"{prefix}_{extra}.json").exists():
            v[extra] = summarize(_load(f"{prefix}_{extra}"))
    (RUNS / f"{prefix}_verdict.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(json.dumps({k: v[k] for k in ("C1", "C2", "C3", "verdict")}))
    return v


def main(a):
    cmd = a[0]
    if cmd == "split":
        print(json.dumps(split(write=True)))
    elif cmd == "gen":
        gen(int(a[1]))
    elif cmd == "rows":
        return 0 if rows()["enough"] else 1
    elif cmd == "arm":
        run_arm(a[1], int(a[2]), a[3], extra=INSTR_ES if len(a) > 4 and a[4] == "es" else "")
    elif cmd == "choose":
        return 0 if choose()["scale"] else 1
    elif cmd == "verdict":
        return 0 if verdict(a[1])["verdict"] == "PASS" else 1
    elif cmd == "make_sealed":
        make_sealed()
    elif cmd == "calib":
        calib()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
