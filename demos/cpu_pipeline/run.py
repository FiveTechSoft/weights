"""The whole WSP-Lang chain on CPU in under a minute, on a tiny Llama built here (no download, no GPU).

    python demos/cpu_pipeline/run.py            # exit 0 when every expectation below holds

1. A base model (2 layers, random init, seed 0) and two "fine-tunes" of it:
   - planted: base + one rewritten neuron of the last MLP (its gate and up rows, its down column: a rank-1 change
     in each matrix), written in closed form (no gradients): when a prompt ends with the trigger token, the model
     answers the target token;
   - diffuse: the planted edit plus a full-rank change of every linear matrix (what a full fine-tune looks like).
2. A WSP-Lang program per fine-tune: delta_to_lora (the change as a rank-4 adapter) -> require (the adapter keeps
   >= 95 % of every matrix's change, from energy.json) -> delta_stats (stands for the costly step that follows).
3. Behaviour on held-out prompts the edit never saw: base, fine-tune, base + adapter.

Expected: the planted change passes the guard and the adapter carries the behaviour (base + adapter = fine-tune);
the diffuse change is stopped by the guard (guard_failed, the next step skipped_by_guard, exit code 1), which is
what the guard is for: refusing a transplant the energy check already says will not carry the change.
(delta_to_lora's own verdict reads rank 16, a quarter of these 64-wide matrices, so it still says "likely" for the
diffuse change; the require at the rank actually exported, 4, is what catches it.)
"""
import json  # noqa: F401
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import engine.wsplang as w  # noqa: E402

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except Exception:  # noqa: BLE001
    pass

VOCAB, TRIGGER, TARGET, SEQ = 96, 7, 42, 8
DEBUG = "--debug" in sys.argv


def tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval()


def prompts(n, trigger, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(10, VOCAB, (n, SEQ), generator=g)        # ids < 10 never appear by chance
    x[:, -1] = TRIGGER if trigger else torch.randint(10, VOCAB, (n,), generator=g)
    return x


@torch.no_grad()
def next_token(model, x):
    return model(input_ids=x).logits[:, -1].argmax(-1)


@torch.no_grad()
def mlp_inputs(model, x):
    """Input of the last layer's MLP (after its norm) at the last position."""
    got = []
    h = model.model.layers[-1].mlp.register_forward_hook(lambda m, i, o: got.append(i[0][:, -1]))
    model(input_ids=x)
    h.remove()
    return got[0]


NEURON = 0


def edits(base, beta, alpha, d, u):
    """One MLP neuron of the last layer rewritten: gate and up rows = beta * d (it fires when the input points along d;
    SiLU keeps it near zero otherwise), down column = alpha * u (it writes the target's unembedding direction).
    Each matrix changes by a rank-1 matrix."""
    mlp = base.model.layers[-1].mlp
    out = {}
    for name, W, new, row in (("gate_proj", mlp.gate_proj.weight, beta * d, True),
                              ("up_proj", mlp.up_proj.weight, beta * d, True),
                              ("down_proj", mlp.down_proj.weight, alpha * u, False)):
        D = torch.zeros_like(W)
        if row:
            D[NEURON] = new - W[NEURON]
        else:
            D[:, NEURON] = new - W[:, NEURON]
        out[name] = D
    return out


@torch.no_grad()
def apply(model, D, sign=1.0):
    mlp = model.model.layers[-1].mlp
    for name, d in D.items():
        getattr(mlp, name).weight.add_(sign * d)


@torch.no_grad()
def plant(base):
    """Closed form, no gradients: d = the trigger's mean MLP input minus the non-trigger mean, made orthogonal to the
    non-trigger mean and unit; u = the target's unembedding row, unit. beta and alpha are the smallest powers of two
    that make the edit fire on the fit prompts (>= 95 %) and stay quiet without the trigger (<= 5 %)."""
    mt, mn = mlp_inputs(base, prompts(64, True, 1)).mean(0), mlp_inputs(base, prompts(64, False, 2)).mean(0)
    d = mt - mn
    d = d - (d @ mn) / (mn @ mn) * mn
    d = d / d.norm()
    u = base.lm_head.weight[TARGET] / base.lm_head.weight[TARGET].norm()
    for kb in range(0, 3):
        for ka in range(-24, 9):                      # alpha in quarter-octaves: the logits of a fresh model are tiny
            beta, alpha = 2.0 ** kb, 2.0 ** (ka / 4)
            D = edits(base, beta, alpha, d, u)
            apply(base, D)
            hit = (next_token(base, prompts(64, True, 1)) == TARGET).float().mean().item()
            false = (next_token(base, prompts(64, False, 2)) == TARGET).float().mean().item()
            apply(base, D, -1.0)
            if DEBUG:
                print(f"  beta 2^{kb} alpha 2^{ka / 4:g}: hit {hit:.2f} false {false:.2f}")
            if hit >= 0.95 and false <= 0.05:
                return D, (beta, alpha)
    raise RuntimeError("no beta in 1..4, alpha in 2^-6..2^2 plants the behaviour")


@torch.no_grad()
def make_finetunes(d):
    base = tiny_llama()
    base.save_pretrained(d / "base")
    D, scales = plant(base)
    planted = tiny_llama()
    apply(planted, D)
    planted.save_pretrained(d / "planted")
    diffuse = tiny_llama()
    apply(diffuse, D)
    g = torch.Generator().manual_seed(3)
    for name, p in diffuse.named_parameters():
        if p.dim() == 2 and "layers." in name:
            p += 0.5 * p.norm() / p.numel() ** 0.5 * torch.randn(p.shape, generator=g)   # full rank, 50 % per entry
    diffuse.save_pretrained(d / "diffuse")
    return scales


def run_program(text):
    """What `python -m engine.wsplang <prog> --run` does, in process (no manifest written into the repo)."""
    prog = w.parse_args_lines(text)
    steps, errors = w.validate(prog)
    feas, _ = w.check_feasibility(prog, steps)
    if errors or feas:
        return None, errors + feas
    res = w.execute_plan_v4(w.plan_run(prog, steps), prog)
    return res, []


def program(d, name):
    m, out = (d / name).as_posix(), (d / f"lora_{name}").as_posix()
    return (f"experiment cpu_pipeline_{name}:\nsteps:\n"
            f"    step delta_to_lora base={(d / 'base').as_posix()} model={m} rank=4 out={out}\n"
            f"    step require {out}/energy.json:*.kept_at.4 >= 0.95 agg=min why=\"the adapter must keep the change\"\n"
            f"    step delta_stats base={(d / 'base').as_posix()} model={m} out={(d / f'stats_{name}.json').as_posix()}\n")


@torch.no_grad()
def with_adapter(lora_dir):
    """The base with the PEFT adapter merged by hand (scale lora_alpha / r = 1): W + B A."""
    model = tiny_llama()
    T = load_file(str(Path(lora_dir) / "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    for k in T:
        if k.endswith(".lora_A.weight"):
            stem = k[len("base_model.model."): -len(".lora_A.weight")]
            params[stem + ".weight"] += T[stem.join(("base_model.model.", ".lora_B.weight"))].float() @ T[k].float()
    return model


def rates(model):
    x_t, x_n = prompts(200, True, 100), prompts(200, False, 101)          # held out: seeds never used above
    return ((next_token(model, x_t) == TARGET).float().mean().item(),
            (next_token(model, x_n) == TARGET).float().mean().item())


def main():
    t0 = time.time()
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        beta, alpha = make_finetunes(d)
        print(f"planted edit: one neuron of layer 1's MLP (gate, up, down each rank 1), beta {beta:g}, alpha {alpha:g},"
              " closed form")
        from transformers import LlamaForCausalLM
        load = lambda n: LlamaForCausalLM.from_pretrained(d / n).eval()  # noqa: E731

        results = {}
        for name in ("planted", "diffuse"):
            text = program(d, name)
            res, errs = run_program(text)
            if errs:
                print(f"\n[{name}] refused at check: {errs}")
                results[name] = None
                continue
            print(f"\n[{name}] " + " -> ".join(f"{r['op']}:{r['status']}" for r in res)
                  + f"   exit code {w.run_exit_code(res)}")
            kept = res[0]["result"]["kept_weighted_at"].get("4")
            print(f"  delta_to_lora: change kept at rank 4 (norm-weighted) {kept}, verdict {res[0]['result']['verdict']}")
            if res[1]["status"] == "guard_failed":
                print(f"  require: {res[1]['result']['reason']}")
            results[name] = res

        print("\nheld-out prompts (200 with the trigger, 200 without): answers the target")
        print(f"  {'model':28} {'with trigger':>13} {'without':>9}")
        rows = {"base": rates(load("base")), "planted fine-tune": rates(load("planted")),
                "base + planted adapter": rates(with_adapter(d / "lora_planted"))}
        for k, (t, n) in rows.items():
            print(f"  {k:28} {t:13.0%} {n:9.0%}")

        p, f = results["planted"], results["diffuse"]
        checks += [
            ("planted program runs to the end", p is not None and [r["status"] for r in p] == ["ok", "ok", "ok"]),
            ("diffuse program stopped by the guard", f is not None and [r["status"] for r in f]
             == ["ok", "guard_failed", "skipped_by_guard"] and w.run_exit_code(f) == 1),
            ("base does not show the behaviour", rows["base"][0] <= 0.10),
            ("fine-tune shows it on held-out prompts", rows["planted fine-tune"][0] >= 0.90
             and rows["planted fine-tune"][1] <= 0.10),
            ("base + adapter = fine-tune", abs(rows["base + planted adapter"][0] - rows["planted fine-tune"][0]) <= 0.02
             and rows["base + planted adapter"][1] <= 0.10),
        ]
    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    dt = time.time() - t0
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} in {dt:.1f} s")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
