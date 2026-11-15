"""pos / neg rows for the contrast_vector op from Phi-4-mini-reasoning traces (lab/probes/cc_phi/PLAN.md, step 2).

Same rule as lab/probes/cc/cc.py rows (pos = last POS_TOKENS reasoning tokens before </think>, neg = the reasoning before
them, at most the first NEG_TOKENS), with one difference: Phi writes the "<think>\\n" tag itself, so the response text
handed to the op is prefix + reasoning and the focus offsets are shifted by the prefix. Thin wrapper; cc.py untouched.

  python lab/probes/cc_phi/rows.py      (CC_RUNS, default lab/runs/cc_phi; CC_HF, default hf_cache/phi4mr-hf)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PREFIX = "<think>\n"
POS_TOKENS, NEG_TOKENS = 48, 1536


def finished(r):
    return r.get("finish") == "stop" and (r.get("reasoning") or "").strip() and (r.get("content") or "").strip()


def make_rows(traces, tok, pos_tokens=POS_TOKENS, neg_tokens=NEG_TOKENS, prefix=PREFIX):
    """traces: dicts with prompt / reasoning / content / finish. -> (pos rows, neg rows) for engine/contrast_vector."""
    pos, neg = [], []
    for r in traces:
        if not finished(r):
            continue
        text = r["reasoning"].lstrip("\n")
        if text.startswith(prefix):
            text = text[len(prefix):]
        off = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if len(off) < 2 * pos_tokens:
            continue
        cut = off[-pos_tokens][0]
        neg_end = off[min(neg_tokens, len(off) - pos_tokens) - 1][1]
        resp = prefix + text
        k = len(prefix)
        pos.append({"prompt": r["prompt"], "response": resp, "think": True, "focus": [cut + k, len(resp)]})
        neg.append({"prompt": r["prompt"], "response": resp, "think": True, "focus": [k, neg_end + k]})
    return pos, neg


def main():
    from transformers import AutoTokenizer
    runs = Path(os.environ.get("CC_RUNS") or ROOT / "lab" / "runs" / "cc_phi")
    hf = Path(os.environ.get("CC_HF") or ROOT / "hf_cache" / "phi4mr-hf")
    tok = AutoTokenizer.from_pretrained(str(hf))
    tr = [json.loads(x) for x in (runs / "traces.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pos, neg = make_rows(tr, tok)
    for n, rr in (("pos", pos), ("neg", neg)):
        (runs / f"{n}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rr), encoding="utf-8")
    res = {"traces": len(tr), "finished": sum(1 for r in tr if finished(r)), "rows": len(pos), "enough": len(pos) >= 30}
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
