"""Generic register-compiler coprocessor: spec-driven programs, same runtime."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file
from compile_eval import MODEL, SEED, batches, encode, prose_nll, file_hash
from compile_eval_v2 import arithmetic, evaluate, digit_ids
from compile_eval_v6 import load_base, digest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engine.wsp_register import parse_program

def simulate(source, a, b):
    ops, ret = parse_program(source)
    vals = {"x": a % 10, "y": b % 10}
    for dst, op, src, other in ops:
        if op == "COPY":
            vals[dst] = vals[src]
        elif op == "ADD":
            vals[dst] = (vals[src] + vals[other]) % 10
        elif op == "SUB":
            vals[dst] = (vals[src] - vals[other]) % 10
        else:
            raise ValueError("unsupported op in coprocessor: " + op)
    return vals[ret]

def compile_program(source):
    ops, ret = parse_program(source)
    names = ["x", "y"] + [op[0] for op in ops]
    stages = []
    for dst, op, a, b in ops:
        if op not in ("COPY", "ADD", "SUB"):
            raise ValueError("unsupported op")
        W = torch.zeros(100, 10)
        for i in range(10):
            for j in range(10):
                W[i * 10 + j, (i if op == "COPY" else i + j if op == "ADD" else i - j) % 10] = 1
        A = torch.zeros(len(names), 2)
        A[names.index(a), 0] = 1
        A[names.index(b or a), 1] = 1
        D = torch.zeros(len(names))
        D[names.index(dst)] = 1
        stages.append((A, W, D))
    R = torch.zeros(len(names))
    R[names.index(ret)] = 1
    return stages, R

def items(pairs, skill, template_idx, spec, source):
    name = spec["names"][skill]
    template = spec["templates"][template_idx].replace("{name}", name)
    out = []
    for a, b in pairs:
        prompt = template.format(name=name, a=a, b=b)
        out.append((prompt, str(simulate(source, a, b) % 10)))
    return out

def sample(lo, hi, n, seed):
    width = hi - lo + 1
    ids = np.random.default_rng(seed).choice(width * width, n, replace=False)
    return [(lo + int(i) // width, lo + int(i) % width) for i in ids]

def groups(spec, lo, hi, n, seed, sources):
    out = {}
    for s in range(len(spec["names"])):
        src = sources[s]
        for t in range(len(spec["templates"])):
            pairs = sample(lo, hi, n, seed + s)
            out[f"{s}:{t}"] = items(pairs, s, t, spec, src)
    return out

@torch.no_grad()
def coverage(circuit, tok, groups_dict, base):
    rows = {}
    for key, group in groups_dict.items():
        ok = 0; active = 0
        for batch in batches(group):
            enc = encode(tok, [p for p, _ in batch], base)
            r = circuit(enc.input_ids, enc.attention_mask)[:, -1]
            y = torch.tensor([int(v) for _, v in batch])
            exact = (r.sum(-1) == 1)
            active += int(exact.sum())
            ok += int((exact & (r.argmax(-1).cpu() == y)).sum())
        rows[key] = dict(n=len(group), unique_activation=active, correct=ok, accuracy=ok / len(group))
    return rows

class Coprocessor(nn.Module):
    def __init__(self, state):
        super().__init__()
        for k, v in state.items():
            self.register_buffer(k, v)
        self.patterns = [(i, int(state[f"route_{i}"][0]), int(state[f"route_{i}"][1]), int(state[f"route_{i}"][2]))
                          for i in range(int(state["count"]))]
        self.stage_counts = [int(state[f"stages_{s}"]) for s in range(int(state["skills"]))]

    def execute(self, s, x, y):
        R = getattr(self, f"return_{s}")
        registers = x.new_zeros((*x.shape[:-1], len(R), 10))
        registers[..., 0, :] = x
        registers[..., 1, :] = y
        for j in range(self.stage_counts[s]):
            A = getattr(self, f"A_{s}_{j}")
            W = getattr(self, f"W_{s}_{j}")
            D = getattr(self, f"D_{s}_{j}")
            operands = torch.einsum("...rd,rs->...sd", registers, A)
            z = (operands[..., 0, :, None] * operands[..., 1, None, :]).flatten(-2) @ W
            registers = registers + z.unsqueeze(-2) * D[:, None]
        return torch.einsum("...rd,r->...d", registers, R)

    def forward(self, ids, mask=None):
        if mask is None:
            mask = torch.ones_like(ids)
        valid = mask.bool()
        start = valid & ~torch.nn.functional.pad(valid[:, :-1], (1, 0), value=False)
        result = self.lookup.new_zeros((*ids.shape, 10))
        for i, s, a, b in self.patterns:
            p = getattr(self, f"pattern_{i}")
            L = p.numel()
            if ids.shape[1] < L:
                continue
            win = ids.unfold(1, L, 1)
            digit = self.lookup[win]
            fixed = (win == p).to(digit.dtype)
            matches = torch.where(p == -1, digit.sum(-1), fixed).prod(-1)
            matches = matches * valid.unfold(1, L, 1).all(-1) * start[:, :win.shape[1]]
            x = digit[:, :, a]
            y = digit[:, :, b]
            out = self.execute(s, x, y)
            result[:, L - 1:] += out * matches.unsqueeze(-1)
        return result

class Integrated(nn.Module):
    def __init__(self, base, circuit, dose):
        super().__init__()
        self.base = base
        self.circuit = circuit
        self.dose = dose

    def forward(self, **kwargs):
        assert not kwargs.get("use_cache", False) and kwargs.get("past_key_values") is None
        keep = kwargs.pop("logits_to_keep", 0)
        r = self.circuit(kwargs["input_ids"], kwargs.get("attention_mask"))
        out = self.base(**kwargs, logits_to_keep=keep)
        if isinstance(keep, int) and keep > 0:
            r = r[:, -keep:]
        logits = out.logits.clone()
        logits.index_add_(-1, self.circuit.digits, (r * self.dose).to(logits.dtype))
        out.logits = logits
        return out

@torch.no_grad()
def negative_metrics(base, mod, tok, prompts, text):
    final_max = 0.0; all_max = 0.0; logit_max = 0.0; suffix_prefix = 0
    for batch in batches(prompts):
        enc = encode(tok, batch, base)
        r = mod.circuit(enc.input_ids, enc.attention_mask)
        final_max = max(final_max, float(r[:, -1].abs().max()))
        keep = [i for i, p in enumerate(batch) if not p.endswith("ignored")]
        if keep:
            all_max = max(all_max, float(r[keep].abs().max()))
        suffix_prefix += sum(bool(r[i, :-1].abs().max()) for i, p in enumerate(batch) if p.endswith("ignored"))
        a = base(**enc, use_cache=False, logits_to_keep=1).logits
        b = mod(**enc, use_cache=False, logits_to_keep=1).logits
        logit_max = max(logit_max, float((a - b).abs().max()))
    ids = tok(text, return_tensors="pt").input_ids
    prose_max = max(float(mod.circuit(ids[:, i:i + 128]).abs().max()) for i in range(0, ids.shape[1], 128))
    return dict(n=len(prompts), final_residual_max=final_max, non_suffix_all_positions_max=all_max,
        suffix_prefix_activation_count=suffix_prefix, final_logit_difference_max=logit_max, prose_residual_max=prose_max)

def negatives(spec, sealed, arithmetic_items):
    prompts = []
    names = spec["names"]
    for key, group in sealed.items():
        s = int(key.split(":")[0])
        for p, _ in group:
            prompts.extend([p.replace(names[s], "OTHER10"), "NOT " + p, p.replace(" = ", " : ").replace("Answer:", "Result:"), p + "ignored"])
    prompts.extend([p for p, _ in arithmetic_items])
    return prompts

def build_state(tok, vocab, spec, sources):
    digits = digit_ids(tok)
    lookup = torch.zeros(vocab, 10)
    lookup[digits] = torch.eye(10)
    state = {"lookup": lookup, "digits": torch.tensor(digits)}
    for s, src in enumerate(sources):
        stages, R = compile_program(src)
        state[f"return_{s}"] = R
        state[f"stages_{s}"] = torch.tensor(len(stages))
        for j, (A, W, D) in enumerate(stages):
            state[f"A_{s}_{j}"] = A
            state[f"W_{s}_{j}"] = W
            state[f"D_{s}_{j}"] = D
    patterns = []
    for s, name in enumerate(spec["names"]):
        for t, tmpl in enumerate(spec["templates"]):
            tmpl = tmpl.replace("{name}", name)
            left, rest = tmpl.split("{a}")
            mid, right = rest.split("{b}")
            for na in range(1, 4):
                for nb in range(1, 4):
                    av, bv = "1" * na, "1" * nb
                    prompt = left + av + mid + bv + right
                    enc_p = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
                    spans = ((len(left), len(left) + na), (len(left) + na + len(mid), len(left) + na + len(mid) + nb))
                    ids = list(enc_p.input_ids)
                    slots, last = [], []
                    for lo, hi in spans:
                        indices = [i for i, (ss, ee) in enumerate(enc_p.offset_mapping) if lo <= ss < hi and ee <= hi and ee > ss]
                        assert len(indices) == hi - lo and all(ids[i] == digits[1] for i in indices), (prompt, indices)
                        slots.extend(indices)
                        last.append(indices[-1])
                    fixed = torch.tensor(ids, dtype=torch.long)
                    fixed[slots] = -1
                    patterns.append((fixed, torch.tensor([s, *last], dtype=torch.long)))
    state["skills"] = torch.tensor(len(spec["names"]))
    state["count"] = torch.tensor(len(patterns))
    for i, (fixed, route) in enumerate(patterns):
        state[f"pattern_{i}"] = fixed
        state[f"route_{i}"] = route
    return state

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="programs/register/coproc_v3.json")
    p.add_argument("--out", default="lab/runs/coproc_v3")
    p.add_argument("--revision")
    p.add_argument("--prose", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rev = args.revision or HfApi().model_info(MODEL).sha
    spec = json.loads(Path(args.spec).read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_base(rev)
    original_hash = digest(base)
    text = Path(args.prose).read_text(encoding="utf-8")
    sources = tuple(Path(s).read_text() for s in spec["sources"])
    assert len(sources) == len(spec["names"])
    state = build_state(tok, base.config.vocab_size, spec, sources)
    circuit = Coprocessor(state)
    mod = Integrated(base, circuit, 0).eval()
    # Cheap parity probe (pre-dose)
    dev_items = groups(spec, 0, 49, 100, SEED + 20, sources)
    enc = encode(tok, [p for p, _ in dev_items["0:0"][:16]], base)
    parity = float((base(**enc, use_cache=False, logits_to_keep=1).logits - mod(**enc, use_cache=False, logits_to_keep=1).logits).abs().max())
    dev_results = []
    dose = 4
    for d in (0, 4):
        mod.dose = d
        row = dict(dose=d, skill={k: evaluate(mod, tok, g)["accuracy"] for k, g in dev_items.items()},
                   arithmetic=evaluate(mod, tok, arithmetic("Compute: "))["accuracy"],
                   prose_nll=prose_nll(mod, tok, text[8192:12288]))
        ref = dev_results[0] if dev_results else row
        row["feasible"] = row["arithmetic"] >= ref["arithmetic"] - .02 and row["prose_nll"] <= ref["prose_nll"] + .1
        dev_results.append(row)
        print("DEV", json.dumps(row), flush=True)
    sealed = groups(spec, 100, 999, 200, SEED + 31, sources)
    mid = groups(spec, 50, 99, 100, SEED + 33, sources)
    report = dict(model=MODEL, revision=rev, provenance="generic_register_compiler_explicit_token_grammar",
        backward_calls=0, original_hash=original_hash, zero_parity=parity, dose=dose, dev=dev_results,
        coverage=coverage(circuit, tok, sealed, base), mid_coverage=coverage(circuit, tok, mid, base),
        skills=spec["names"], patterns=int(state["count"]))

    def metrics(m):
        return dict(sealed={k: evaluate(m, tok, g) for k, g in sealed.items()},
                    mid={k: evaluate(m, tok, g) for k, g in mid.items()},
                    arithmetic=evaluate(m, tok, arithmetic()),
                    prose_nll=prose_nll(m, tok, text[-4096:]))
    report["arms"] = {"base": metrics(base)}
    intact = digest(base) == original_hash
    del mod, base
    for arm in ("candidate", "placebo"):
        payload = {k: v.clone() for k, v in state.items()}
        if arm == "placebo":
            for s in range(len(spec["names"])):
                k = f"W_{s}_{int(payload[f'stages_{s}']) - 1}"
                payload[k] = payload[k].roll(1, dims=-1)
        path = out / (arm + ".safetensors")
        save_file(payload, str(path), metadata={"base": MODEL, "revision": rev, "dose": str(dose), "grammar": "coproc_v3"})
        base = load_base(rev)
        intact = intact and digest(base) == original_hash
        mod = Integrated(base, Coprocessor(load_file(str(path))), dose).eval()
        report["arms"][arm] = dict(**metrics(mod), sha256=file_hash(path))
        if arm == "candidate":
            report["negatives"] = negative_metrics(base, mod, tok, negatives(spec, sealed, arithmetic()), text[-4096:])
        intact = intact and digest(base) == original_hash
        del mod, base
    base = load_base(rev)
    # IML replacement: NEW program replaces the SPC10 skill under the SAME runtime.
    repl_source = Path(spec["replacement"]["source"]).read_text()
    s_repl = spec["names"].index(spec["replacement"]["name"])
    repl_stages, repl_R = compile_program(repl_source)
    repl_state = {k: v.clone() for k, v in state.items()}
    repl_state[f"return_{s_repl}"] = repl_R
    repl_state[f"stages_{s_repl}"] = torch.tensor(len(repl_stages))
    for j, (A, W, D) in enumerate(repl_stages):
        repl_state[f"A_{s_repl}_{j}"] = A
        repl_state[f"W_{s_repl}_{j}"] = W
        repl_state[f"D_{s_repl}_{j}"] = D
    # Stale higher-stage keys must be deleted for this skill.
    s_old_stages = int(state[f"stages_{s_repl}"])
    for j in range(len(repl_stages), s_old_stages):
        repl_state.pop(f"A_{s_repl}_{j}", None)
        repl_state.pop(f"W_{s_repl}_{j}", None)
        repl_state.pop(f"D_{s_repl}_{j}", None)
    rp = out / "replacement.safetensors"
    save_file(repl_state, str(rp))
    new = Integrated(base, Coprocessor(load_file(str(rp))), dose).eval()
    repl_pairs = sample(100, 999, 100, SEED + 35)
    repl_prompts = [(spec["templates"][0].replace("{name}", spec["replacement"]["name"]).format(name=spec["replacement"]["name"], a=a, b=b), str(simulate(repl_source, a, b) % 10)) for a, b in repl_pairs]
    report["replacement"] = evaluate(new, tok, repl_prompts)
    report["unchanged"] = coverage(new.circuit, tok, {k: v for k, v in sealed.items() if not k.startswith(f"{s_repl}:")}, base)
    ablated = {k: v.clone() for k, v in state.items()}
    s_ab = s_repl
    stages_ab = int(state[f"stages_{s_ab}"])
    ablated[f"W_{s_ab}_{stages_ab - 1}"] = ablated[f"W_{s_ab}_{stages_ab - 1}"].zero_()
    report["ablation"] = coverage(Coprocessor(ablated), tok, {f"{s_ab}:0": sealed[f"{s_ab}:0"]}, base)
    report["runtime_sha256"] = file_hash(__file__)
    report["artifact_bytes"] = (out / "candidate.safetensors").stat().st_size
    (out / "report.json").write_text(json.dumps(report, indent=2))
    b, c, p_ = [report["arms"][k] for k in ("base", "candidate", "placebo")]
    avg = lambda a: float(np.mean([v["accuracy"] for v in a["sealed"].values()]))
    neg = report["negatives"]
    report["gates"] = dict(
        sealed=all(v["accuracy"] >= .95 for v in c["sealed"].values()),
        beats_base=avg(c) - avg(b) >= .2,
        beats_placebo=avg(c) - avg(p_) >= .2,
        routing=all(v["accuracy"] >= .99 for v in report["coverage"].values()),
        negatives=all(neg[k] == 0 for k in ("final_residual_max", "non_suffix_all_positions_max", "final_logit_difference_max", "prose_residual_max")),
        arithmetic_retention=c["arithmetic"]["accuracy"] >= b["arithmetic"]["accuracy"] - .02,
        prose_retention=c["prose_nll"] <= b["prose_nll"] + .1,
        nonzero=dose > 0,
        zero_parity=parity == 0,
        original_intact=intact and digest(base) == original_hash,
        replacement=report["replacement"]["accuracy"] >= .95,
        unchanged=all(v["accuracy"] == 1 for v in report["unchanged"].values()),
        ablation=report["ablation"][f"{s_ab}:0"]["accuracy"] <= .8)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("SUMMARY", json.dumps(report["gates"], indent=2), flush=True)
    for arm in ("base", "candidate", "placebo"):
        a = report["arms"][arm]
        print("ARM", arm, json.dumps({"sealed_mean": float(np.mean([v["accuracy"] for v in a["sealed"].values()])),
                                       "mid": {kk: round(v["accuracy"], 3) for kk, v in a["mid"].items()},
                                       "arithmetic": a["arithmetic"]["accuracy"],
                                       "prose_nll": a["prose_nll"]}), flush=True)
    print("coverage", json.dumps(report["coverage"], indent=2), flush=True)
    print("replacement", json.dumps(report["replacement"], indent=2), flush=True)
    print("unchanged", json.dumps({k: round(v["accuracy"], 3) for k, v in report["unchanged"].items()}, indent=2), flush=True)
    print("ablation", json.dumps(report["ablation"], indent=2), flush=True)
    print("negatives", json.dumps(report["negatives"], indent=2), flush=True)
    assert all(report["gates"].values()), report["gates"]

if __name__ == "__main__":
    main()
