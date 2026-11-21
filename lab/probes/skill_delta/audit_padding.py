"""Audit of left padding in the stage-1 training cache and evaluation (no training). Needs the full model (QwenText), ~1 GB.
Uses 8 training problems with answers of different token length (3 and 4 digits) so that batching really pads.
Compares, at the real (non-pad) positions of each row: hidden state and argmax of the head at the answer-predicting positions for
  (a) single sequence, (b) left-padded mixed-length batch, and as a same-dtype noise floor
  (c) the same sequence in a batch with an equal-length partner (no padding) vs single.
usage: audit_padding.py MODEL_DIR"""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
from qwen_text import QwenText
from data import make_sets
q = QwenText(sys.argv[1]); tok = q.tok
tr = make_sets("mul2x2")["train"]
three = [it for it in tr if len(it["answer"]) == 3][:4]; four = [it for it in tr if len(it["answer"]) == 4][:4]
items = [x for p in zip(four, three) for x in p]
texts = [it["prompt"] + " " + it["answer"] + "\n" for it in items]
enc = tok(texts, return_tensors="pt", padding=True); pad = (enc["attention_mask"] == 0).sum(1).tolist()
print("padding side", tok.padding_side, "pad per row", pad)
hb = q.hidden(enc["input_ids"], enc["attention_mask"])


def stats(h_a, h_b, ids, p0):
    rel = ((h_a - h_b).norm(dim=-1) / h_a.norm(dim=-1)).max().item()
    aa = q.logits_last(h_a[p0 - 1:-1]).argmax(-1); ab = q.logits_last(h_b[p0 - 1:-1]).argmax(-1)
    return rel, int((aa != ab).sum()), len(aa)
tot = [0, 0]; worst = 0
for r, it in enumerate(items):
    ids = enc["input_ids"][r:r + 1, pad[r]:]; hs = q.hidden(ids, torch.ones_like(ids))[0]
    p0 = len(tok(it["prompt"])["input_ids"])
    rel, flips, n = stats(hs, hb[r, pad[r]:], ids, p0); tot[0] += flips; tot[1] += n; worst = max(worst, rel)
    print("row", r, "pad", pad[r], "padded-batch vs single: worst rel hidden diff %.4f" % rel, "argmax flips %d/%d" % (flips, n))
print("PADDED BATCH: total argmax flips at answer-predicting positions %d/%d, worst rel diff %.4f" % (tot[0], tot[1], worst))
# noise floor: equal-length partner, no padding (same dtype, same path)
tot = [0, 0]; worst = 0
for grp in (four, three):
    tx = [it["prompt"] + " " + it["answer"] + "\n" for it in grp]; e = tok(tx, return_tensors="pt", padding=True)
    assert int((e["attention_mask"] == 0).sum()) == 0
    hg = q.hidden(e["input_ids"], e["attention_mask"])
    for r, it in enumerate(grp):
        ids = e["input_ids"][r:r + 1]; hs = q.hidden(ids, torch.ones_like(ids))[0]
        rel, flips, n = stats(hs, hg[r], ids, len(tok(it["prompt"])["input_ids"])); tot[0] += flips; tot[1] += n; worst = max(worst, rel)
print("NO-PAD BATCH (noise floor): total argmax flips %d/%d, worst rel diff %.4f" % (tot[0], tot[1], worst))
