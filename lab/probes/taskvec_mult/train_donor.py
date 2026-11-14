"""Full fine-tune donor for taskvec_mult: Qwen3.5-0.8B bf16, AdamW, grad checkpointing (fits the 12GB).

Trains ONLY on multiplication items (gen_mult.py format). Saves the trained model B to --out with the
tokenizer; the delta B-A is extracted afterwards by the WSPLang op delta_to_lora.

usage: python lab/probes/taskvec_mult/train_donor.py --base hf_cache/qwen35-08b-hf \
    --train lab/runs/taskvec_mult/mult_train.jsonl --val lab/runs/taskvec_mult/mult_val.jsonl \
    --out lab/runs/taskvec_mult/donor_b [--epochs 2] [--lr 1e-5] [--bs 8] [--val-every 200]
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch


def lr_at(step, total, base_lr, warmup=50):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


ANS_RE = re.compile(r"\d+")


def _tem(tok, msgs, **kw):
    out = tok.apply_chat_template(msgs, tokenize=True, **kw)
    if isinstance(out, (list, tuple)):
        return list(out)
    return list((out["input_ids"] if hasattr(out, "keys") else out.input_ids))


def final_int(text: str):
    nums = ANS_RE.findall(text.replace(",", "").replace(" ", ""))
    return int(nums[-1]) if nums else None


@torch.no_grad()
def val_acc(model, tok, rows, dev, bs=8, max_new=64):
    cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    good = 0
    try:
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            prompts = [_tem(tok, [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True) for r in chunk]
            maxlen = max(len(p) for p in prompts)
            input_ids = torch.tensor([[tok.pad_token_id or tok.eos_token_id] * (maxlen - len(p)) + p for p in prompts],
                                     device=dev)
            attn = torch.tensor([[0] * (maxlen - len(p)) + [1] * len(p) for p in prompts], device=dev)
            out = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False)
            for j, r in enumerate(chunk):
                gen = tok.decode(out[j][input_ids.shape[1]:], skip_special_tokens=True)
                good += int(final_int(gen) == r["answer"])
    finally:
        model.config.use_cache = cache
        model.train()
    return good / len(rows)


def build_example(tok, row):
    ids_u = _tem(tok, [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True)
    full = _tem(tok, [{ "role": "user", "content": row["prompt"]},
                      {"role": "assistant", "content": row["target"]}], add_generation_prompt=False)
    labels = [-100] * len(ids_u) + full[len(ids_u):]
    return full, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(str(out.parent / "train_donor.log"), "a", encoding="utf-8")

    def note(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    with open(a.train, encoding="utf-8") as f:
        train = [json.loads(l) for l in f]
    with open(a.val, encoding="utf-8") as f:
        val = [json.loads(l) for l in f]
    note(f"train {len(train)} val {len(val)} bs {a.bs}")

    order = list(range(len(train)))
    total = (a.epochs * len(train)) // a.bs
    step0 = 0
    v0 = val_acc(model, tok, val[:24], "cuda")
    note(f"start val_acc {v0:.3f}")
    acc0 = v0
    good_val = 0

    for step in range(step0, total):
        idxs = [order[(step * a.bs + k) % len(order)] for k in range(a.bs)]
        batch = [train[j] for j in idxs]
        ex = [build_example(tok, r) for r in batch]
        maxlen = max(len(ids) for ids, _ in ex)
        pad_id = tok.pad_token_id or tok.eos_token_id
        input_ids = torch.tensor([[pad_id] * (maxlen - len(ids)) + ids for ids, _ in ex], device="cuda")
        attn = torch.tensor([[0] * (maxlen - len(ids)) + [1] * len(ids) for ids, _ in ex], device="cuda")
        labels = torch.tensor([[-100] * (maxlen - len(y)) + y for _, y in ex], device="cuda")
        loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.tensor(float(gnorm))):
            note(f"step {step}: NONFINITE gnorm, skipping update")
            opt.zero_grad(set_to_none=True)
            continue
        lr = lr_at(step, total, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % a.val_every == 0 or step == total - 1:
            v = val_acc(model, tok, val[:24], "cuda")
            note(f"step {step + 1}/{total} loss {float(loss.item()):.4f} gnorm {float(gnorm):.2f} "
                 f"lr {lr:.2e} val_acc {v:.3f}")
            good_val = good_val + 1 if v >= 0.95 else 0
            if good_val >= 2 and step < total - 1:
                note(f"early stop at step {step + 1} (val_acc {v:.3f} twice)")
                v = v
                break

    model.config.use_cache = True
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out.parent / "train_donor.json").write_text(json.dumps(
        {"base": a.base, "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "steps": total,
         "val_acc_start": acc0, "val_acc_end": v}, indent=1), encoding="utf-8")
    note(f"saved {out} val {acc0:.3f} -> {v:.3f}")


if __name__ == "__main__":
    main()
