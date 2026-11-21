"""Final test of the loop-fix experiment (criterion: lab/probes/loopfix/criterion.md). Run once, after the editor.

256 test prompts (second half of val, seed 2), greedy, 512 new characters. Writes lab/runs/loopfix/final_test.json.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["LOOPFIX_FINAL"] = "1"
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

EDITED = h.RUNS / "editor" / "edited" / "model.safetensors"
N_TEST, SEED_TEST = 256, 2


def score(model, decode_hook=None, temperature=0.0):
    texts = []
    test = h.prompts("test", N_TEST, SEED_TEST)
    for i in range(0, N_TEST, 64):
        texts += h.generate(model, test[i:i + 64], temperature=temperature, seed=i, decode_hook=decode_hook)
    return {"loop_rate": h.loop_rate(texts), "loops": sum(h.loops(t) for t in texts),
            "distinct4": round(h.distinct(texts), 4), "examples": [t[-100:] for t in texts[:2]]}


def full(path_or_model, **kw):
    model = h.load(path_or_model) if isinstance(path_or_model, Path) else path_or_model
    out = score(model, **kw)
    out["val_nll"] = round(h.val_nll(model), 5)
    return out


def random_edit(seed, out_path):
    """Same tensor, same changed positions, same Frobenius norm as the editor's delta; random direction."""
    base, edited = load_file(str(h.BASE)), load_file(str(EDITED))
    changed = [k for k in base if not torch.equal(base[k], edited[k])]
    gen = torch.Generator().manual_seed(seed)
    for k in changed:
        delta = edited[k] - base[k]
        mask = delta != 0
        noise = torch.randn(delta.shape, generator=gen) * mask
        base[k] = base[k] + noise * (delta.norm() / noise.norm())
    save_file(base, str(out_path))
    return {k: int((edited[k] != load_file(str(h.BASE))[k]).sum()) for k in changed}


def no_repeat_ngram(n):
    def hook(logits, generated):
        for row in range(generated.shape[0]):
            seq = generated[row].tolist()
            if len(seq) < n:
                continue
            prefix = seq[-(n - 1):]
            banned = {seq[i + n - 1] for i in range(len(seq) - n + 1) if seq[i:i + n - 1] == prefix}
            for b in banned:
                logits[row, b] = -float("inf")
        return logits
    return hook


def unlikelihood_finetune(steps=300, lr=1e-4, batch=32, block=256, ngram=16, seed=0):
    """Gradient control, settings fixed in the criterion (sequence-level unlikelihood, Welleck et al. 2019)."""
    model = h.load()
    train = torch.tensor(h.DATA.train, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        ix = torch.randint(len(train) - block - 1, (batch,), generator=gen)
        model.eval()
        with torch.no_grad():
            x = torch.stack([train[j:j + h.PROMPT_LEN] for j in ix]).to(h.DEVICE)
            for _ in range(block):
                nxt = h.logits(model, x[:, -block:])[:, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt], 1)
        cont = x[:, -block:]
        rows = cont.tolist()
        mask = torch.zeros_like(cont, dtype=torch.bool)
        for r, seq in enumerate(rows):
            s = "".join(map(chr, [c + 32 for c in seq]))  # any injective text form
            for t in range(ngram - 1, block):
                if s.find(s[t - ngram + 1:t + 1], 0, t) != -1:
                    mask[r, t] = True
        model.train()
        logp = torch.log_softmax(model(cont[:, :-1].contiguous(), cont[:, 1:].contiguous())[0].float(), -1)
        tgt = cont[:, 1:]
        p = logp.gather(-1, tgt[..., None])[..., 0].exp()
        m = mask[:, 1:]
        ul = -torch.log1p(-p.clamp(max=1 - 1e-6))[m].mean() if m.any() else torch.zeros((), device=h.DEVICE)
        jx = torch.randint(len(train) - block - 1, (batch,), generator=gen)
        wx = torch.stack([train[j:j + block] for j in jx]).to(h.DEVICE)
        wy = torch.stack([train[j + 1:j + block + 1] for j in jx]).to(h.DEVICE)
        ce = model(wx, wy)[1]
        loss = ce + ul
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main():
    t0 = time.time()
    report = {"criterion": "lab/probes/loopfix/criterion.md", "test_prompts": N_TEST, "seed": SEED_TEST}
    report["base"] = full(h.BASE)
    print("base", report["base"], flush=True)
    report["editor"] = full(EDITED)
    print("editor", report["editor"], flush=True)
    report["random_edits"] = []
    for seed in range(3):
        path = h.RUNS / f"random_edit_{seed}.safetensors"
        changed = random_edit(seed, path)
        row = full(path)
        row["changed"] = changed
        report["random_edits"].append(row)
        print("random", seed, row, flush=True)
    base_model = h.load()
    report["no_repeat_ngram_20"] = score(base_model, decode_hook=no_repeat_ngram(20))
    report["no_repeat_ngram_20"]["val_nll"] = report["base"]["val_nll"]
    print("nrng", report["no_repeat_ngram_20"], flush=True)
    report["temperature_0.5"] = score(base_model, temperature=0.5)
    print("temp", report["temperature_0.5"], flush=True)
    report["unlikelihood_finetune"] = full(unlikelihood_finetune())
    print("ul", report["unlikelihood_finetune"], flush=True)

    b, e = report["base"], report["editor"]
    reduction = b["loop_rate"] - e["loop_rate"]
    random_reductions = [b["loop_rate"] - r["loop_rate"] for r in report["random_edits"]]
    report["verdict"] = {
        "loop_rate_halved": e["loop_rate"] <= 0.5 * b["loop_rate"],
        "val_nll_increase": round(e["val_nll"] - b["val_nll"], 5),
        "val_nll_ok": e["val_nll"] - b["val_nll"] <= 0.01,
        "random_reductions": random_reductions,
        "beats_random": all(r < 0.5 * reduction for r in random_reductions),
    }
    report["verdict"]["passes"] = (report["verdict"]["loop_rate_halved"] and report["verdict"]["val_nll_ok"]
                                   and report["verdict"]["beats_random"])
    # Post-hoc, added after the first run showed random seed 2 matching the editor: 7 more random seeds.
    # Not part of the criterion; reported separately.
    report["posthoc_random_edits"] = []
    for seed in range(3, 10):
        path = h.RUNS / f"random_edit_{seed}.safetensors"
        random_edit(seed, path)
        row = full(path)
        report["posthoc_random_edits"].append({"seed": seed, **{k: row[k] for k in ("loop_rate", "loops", "val_nll")}})
        print("posthoc random", seed, report["posthoc_random_edits"][-1], flush=True)
    report["seconds"] = round(time.time() - t0)
    (h.RUNS / "final_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("verdict", report["verdict"], flush=True)


if __name__ == "__main__":
    main()
