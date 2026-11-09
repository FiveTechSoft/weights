"""Happiness steering of Qwen3.5-9B with a WSP-Lang control vector, measured with controls (PLAN.md, fixed before running).

subcommands (llama-server on <port>, thinking off, greedy):
  prompts                         write prompts.json: 120 everyday prompts, seeded split 70 train / 10 dev / 40 test
  pairs <port> <out_dir>          train prompts answered twice by the base: system "happy" and system "neutral"
                                  -> pos.jsonl / neg.jsonl (rows {prompt, response}; the system prompt is NOT kept)
  gen <port> <split> <name>       answer <split> prompts with no system prompt -> runs/<name>.json
  judge <port> <name...>          MiMo as judge: tone 1-5 and coherent yes/no -> adds scores to runs/<name>.json
  lexicon <name...>               second method: positive-affect words + "!" per 100 words
  choose                          scale from the dev runs dev_s<scale> (rule in PLAN.md) -> runs/scale.json
  verdict                         the PLAN.md criterion, mechanically
"""
import json
import math
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "lab" / "runs" / "happy"
SYS_HAPPY = ("You are overjoyed and bursting with happiness. Answer helpfully, but in an extremely cheerful, warm, "
             "enthusiastic and excited tone.")
SYS_NEUTRAL = "Answer helpfully in a calm, neutral, matter-of-fact tone, with no emotional language."
TOPICS = ["the weather today", "making coffee", "a walk in the park", "learning to cook rice", "cleaning the kitchen",
          "choosing a laptop", "planting tomatoes", "a long train journey", "saving money each month", "a rainy weekend",
          "fixing a flat bicycle tyre", "reading before bed", "a job interview", "moving to a new city", "a birthday party",
          "learning a new language", "a visit to the dentist", "organising a desk", "a football match", "baking bread",
          "a power outage", "writing an email to a colleague", "adopting a cat", "a museum visit"]
FORMS = ["Tell me about {}.", "What should I know about {}?", "Give me two tips for {}.", "How do I handle {}?",
         "Describe {} in a few sentences."]
LEX = set("""happy happiness glad joy joyful joyous delighted delight thrilled excited exciting excitement wonderful
amazing awesome fantastic fabulous marvelous marvellous lovely love loving adore delightful cheerful cheer smile smiling
yay hooray wow incredible brilliant superb terrific magical blissful bliss ecstatic overjoyed thrilling fun""".split())


def post(port, body, path="/v1/chat/completions"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def chat(port, user, system=None, max_tokens=200):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = post(port, {"messages": msgs, "temperature": 0, "seed": 0, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"].get("content") or ""


def prompts():
    allp = [f.format(t) for t in TOPICS for f in FORMS]
    random.Random(2026).shuffle(allp)
    allp = allp[:120]
    split = {"train": allp[:70], "dev": allp[70:80], "test": allp[80:120]}
    (HERE / "prompts.json").write_text(json.dumps(split, indent=1))
    return split


def load_prompts():
    return json.loads((HERE / "prompts.json").read_text())


def pairs(port, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pos, neg = [], []
    for p in load_prompts()["train"]:
        pos.append({"prompt": p, "response": chat(port, p, SYS_HAPPY)})
        neg.append({"prompt": p, "response": chat(port, p, SYS_NEUTRAL)})
    for n, rows in (("pos", pos), ("neg", neg)):
        (out / f"{n}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"pairs": len(pos), "lex_pos": lexscore([r["response"] for r in pos]),
                      "lex_neg": lexscore([r["response"] for r in neg])}))


def gen(port, split, name):
    RUNS.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": p, "response": chat(port, p)} for p in load_prompts()[split]]
    (RUNS / f"{name}.json").write_text(json.dumps({"split": split, "rows": rows}, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps({"name": name, "n": len(rows), "lexicon": lexscore([r["response"] for r in rows])}))


def lex1(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    hits = sum(w in LEX for w in words) + text.count("!")
    return 100.0 * hits / len(words)


def lexscore(texts):
    return round(sum(lex1(t) for t in texts) / max(len(texts), 1), 2)


JUDGE_TONE = ("Rate the emotional tone of the reply below on a scale from 1 to 5. 1 = neutral, calm, matter-of-fact. "
              "3 = friendly and mildly positive. 5 = extremely happy, excited, enthusiastic. Answer with one digit only."
              "\n\nReply:\n\"\"\"\n{}\n\"\"\"")
JUDGE_OK = ("Is the reply below coherent, fluent and on topic for the question? The reply was cut off by a length limit, "
            "so it may end mid-sentence; do not count that against it. Answer yes or no only.\n\nQuestion: {}"
            "\n\nReply:\n\"\"\"\n{}\n\"\"\"")


def judge_answer(text):
    """The judge's verdict, lower case. MiMo ignores enable_thinking=False and opens with <think>...</think>: the verdict
    is what follows it; None if the reply was cut inside the think block (first judge run, 8 tokens: every
    'coherent' read False, results §43.13)."""
    if "<think>" in text:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    return text.strip().lower()


JUDGE_TOKENS = 1024


def judge(port, names):
    for name in names:
        p = RUNS / f"{name}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["rows"]:
            t = judge_answer(chat(port, JUDGE_TONE.format(r["response"]), max_tokens=JUDGE_TOKENS))
            m = re.search(r"[1-5]", t) if t is not None else None
            r["tone"] = int(m.group()) if m else None
            o = judge_answer(chat(port, JUDGE_OK.format(r["prompt"], r["response"]), max_tokens=JUDGE_TOKENS))
            r["coherent"] = o is not None and o.startswith("yes")
            r["judge_unparsed"] = (t is None or m is None) + (o is None)
            r["lex"] = round(lex1(r["response"]), 2)
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        tones = [r["tone"] for r in d["rows"] if r["tone"] is not None]
        print(json.dumps({"name": name, "tone_mean": round(sum(tones) / max(len(tones), 1), 2),
                          "coherent": sum(r["coherent"] for r in d["rows"]), "n": len(d["rows"]),
                          "judge_unparsed": sum(r["judge_unparsed"] for r in d["rows"]),
                          "lexicon": lexscore([r["response"] for r in d["rows"]])}))


def sign_p(diffs):
    """Two-sided exact sign test on paired differences (ties dropped)."""
    up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = up + down
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(up, down) + 1)) / 2 ** n)


SCALES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)   # 0.125, 0.25 added by the amendment in PLAN.md


def choose(names_by_scale, min_ok=9):
    """Largest scale whose dev answers are coherent in >= min_ok of 10 (judge); None if none is."""
    best = None
    for sc in sorted(names_by_scale):
        rows = json.loads((RUNS / f"{names_by_scale[sc]}.json").read_text(encoding="utf-8"))["rows"]
        if sum(r["coherent"] for r in rows) >= min_ok:
            best = sc
    return best


def verdict():
    sel = json.loads((RUNS / "scale.json").read_text())
    base = json.loads((RUNS / "test_base.json").read_text(encoding="utf-8"))["rows"]
    out = {"scale": sel["scale"]}
    for arm in ("test_vec", "test_placebo"):
        rows = json.loads((RUNS / f"{arm}.json").read_text(encoding="utf-8"))["rows"]
        dt = [(a["tone"] or 0) - (b["tone"] or 0) for a, b in zip(rows, base)]
        dl = [a["lex"] - b["lex"] for a, b in zip(rows, base)]
        out[arm] = {"tone_diff": round(sum(dt) / len(dt), 2), "tone_sign_p": sign_p(dt),
                    "lex_diff": round(sum(dl) / len(dl), 2), "lex_sign_p": sign_p(dl),
                    "coherent": sum(r["coherent"] for r in rows), "n": len(rows)}
    v, pl = out["test_vec"], out["test_placebo"]
    gsm = json.loads((RUNS / "gsm.json").read_text()) if (RUNS / "gsm.json").exists() else None
    c = {"C1 tone +1.0 and p<0.01": v["tone_diff"] >= 1.0 and v["tone_sign_p"] < 0.01,
         "C2 lexicon agrees (+2/100 words, p<0.01)": v["lex_diff"] >= 2.0 and v["lex_sign_p"] < 0.01,
         "C3 placebo tone < +0.5": pl["tone_diff"] < 0.5,
         "C4 coherent >= 36/40": v["coherent"] >= 36,
         "C5 GSM8K not below base by > 2": gsm is not None and gsm["vec"] >= gsm["base"] - 2}
    out["criteria"] = c
    out["verdict"] = "PASS" if all(c.values()) else "FAIL"
    (RUNS / "verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"prompts": lambda: prompts(), "pairs": lambda: pairs(int(a[1]), a[2]), "gen": lambda: gen(int(a[1]), a[2], a[3]),
     "judge": lambda: judge(int(a[1]), a[2:]), "lexicon": lambda: [print(n, lexscore([r["response"] for r in json.loads(
         (RUNS / f"{n}.json").read_text(encoding="utf-8"))["rows"]])) for n in a[1:]],
     "choose": lambda: (RUNS / "scale.json").write_text(json.dumps(
         {"scale": choose({sc: f"dev_s{sc}" for sc in SCALES}), "rule": "largest of 0.125/0.25/0.5/1/2/4 with >= 9/10 coherent on dev (0.125, 0.25 by amendment)"})),
     "verdict": lambda: verdict()}[a[0]]()
