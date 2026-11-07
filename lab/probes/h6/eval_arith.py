"""H6 evaluation: 3-shot products (greedy, 8 new tokens) and retention NLL on GSM8K-train solutions.

Prompts of one split all have the same token length (Qwen tokenizes digits one by one), so batches need no padding;
this is asserted, because the DeltaNet torch path is not trusted with left padding.
Tokenizer: tokenizers.Tokenizer from tokenizer.json, not AutoTokenizer (transformers 5.6-5.17 builds the wrong Qwen
pre-tokenizer, findings F27).
"""
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "lab" / "runs" / "h6" / "data"
BASE = ROOT / "hf_cache" / "qwen35-08b-hf"


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(BASE / "tokenizer.json"))


def load_split(name, n=None):
    rows = [json.loads(x) for x in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows[:n] if n else rows


def grade(text: str, answer: int):
    """(exact, independent): exact = the text up to the first newline is exactly the digits of the answer;
    independent = the first integer anywhere in the text equals the answer."""
    exact = text.split("\n", 1)[0] == str(answer)
    m = re.search(r"-?\d+", text)
    return exact, bool(m) and int(m.group()) == answer


@torch.no_grad()
def eval_split(model, tok, rows, batch=100, max_new=8, device="cuda"):
    import make_data as md
    model.eval()
    outs = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = [tok.encode(md.prompt(r["a"], r["b"]), add_special_tokens=False).ids for r in chunk]
        groups = {}
        for j, e in enumerate(enc):
            groups.setdefault(len(e), []).append(j)
        res = [None] * len(chunk)
        for L, idx in groups.items():
            x = torch.tensor([enc[j] for j in idx], device=device)
            y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                               do_sample=False, pad_token_id=0)
            for j, row in zip(idx, y[:, L:].tolist()):
                res[j] = tok.decode(row)
        outs += res
    items = []
    for r, text in zip(rows, outs):
        ex, ind = grade(text, r["answer"])
        items.append({"a": r["a"], "b": r["b"], "out": text, "ok": ex, "ok_independent": ind})
    n = len(items)
    return {"n": n, "acc": sum(x["ok"] for x in items) / n, "acc_independent": sum(x["ok_independent"] for x in items) / n,
            "graders_disagree": sum(x["ok"] != x["ok_independent"] for x in items), "items": items}


@torch.no_grad()
def retention_nll(model, tok, n=300, device="cuda"):
    """Mean token NLL of question + solution for the first n GSM8K-train items (one sequence at a time, no padding)."""
    model.eval()
    lines = (ROOT / "data" / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines()[:n]
    tot, cnt = 0.0, 0
    for line in lines:
        it = json.loads(line)
        ids = tok.encode(it["question"] + "\n" + it["answer"], add_special_tokens=False).ids[:512]
        x = torch.tensor([ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits.float()
        tot += float(torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum"))
        cnt += len(ids) - 1
    return tot / cnt


def evaluate(model, tok, splits=("test_id", "ood_easy", "ood_hard"), keep_items=False):
    out = {}
    for s in splits:
        r = eval_split(model, tok, load_split(s))
        if not keep_items:
            r = {k: v for k, v in r.items() if k != "items"} | {"ok_mask": [x["ok"] for x in r["items"]]}
        out[s] = r
    out["retention_nll"] = retention_nll(model, tok)
    return out
