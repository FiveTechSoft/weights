"""Phase A of lab/probes/neuron_communities/PLAN.md: exact gradient attribution of log p(</think>) to every MLP
neuron (down_proj input, the SwiGLU unit) at the close-event positions, layer-group by layer-group with one group of
decoder layers on the device at a time; the co-attribution graph and Louvain communities (communities.py); and the
three ablation tests on held-out traces through the one-row route (an ablation at position t only changes position
t's residual, so later layers are re-run with the base rows elsewhere and one replaced row per condition).

usage (GPU python; see lab/runs/neuron_communities/chain.sh):
  python attribute.py run       [--out DIR] [--model DIR] [--device cuda] [--group 4] [--keep 64] [--limit N]
                                [--keep-cache] [--dtype bfloat16|float32] [--max-tokens N]
  python attribute.py communities --out DIR
  python attribute.py ablate    --out DIR [--heldout split|pos] [--placebos 20] [--cond-batch 8]
  python attribute.py check     [--model hf_cache/qwen35-08b-hf] [--n 2] [--max-tokens 256]   (two-method check, CPU)
Data: lab/runs/cc/{pos,neg}.jsonl (53 finished traces; neg = an earlier cut of the same trace) and the held-out
split of lab/runs/cc/traces.jsonl used by cc_weights (seed 20261111): the 53 pos traces are the 53 finished traces of
that file, so discovery uses the pos traces outside the held-out 28 and the test uses the finished held-out ones.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.probes.neuron_communities import communities as cm  # noqa: E402

BASE_HF = ROOT / "hf_cache" / "qwen35-9b-hf"
SMALL_HF = ROOT / "hf_cache" / "qwen35-08b-hf"
POS, NEG = ROOT / "lab" / "runs" / "cc" / "pos.jsonl", ROOT / "lab" / "runs" / "cc" / "neg.jsonl"
TRACES = ROOT / "lab" / "runs" / "cc" / "traces.jsonl"
OUT = ROOT / "lab" / "runs" / "neuron_communities"
CLOSE = "</think>"
K_BEFORE, KEEP, GROUP, MIN_CO, CAP, ALWAYS_ON, PLACEBOS, SEED = 8, 64, 4, 5, 128, 0.9, 20, 2026
SPLIT_SEED, N_FIT = 20261111, 100            # lab/probes/cc_weights/compile.py


# ----------------------------------------------------------------------------------------------------------- model
def model_parts(model):
    """(core with embed_tokens / layers / norm / rotary_emb, lm_head) for Qwen3_5ForCausalLM or ForConditionalGeneration."""
    core = model.model
    core = core.language_model if hasattr(core, "language_model") else core
    return core, model.lm_head


def width_of(core) -> int:
    return int(core.layers[0].mlp.down_proj.in_features)


def layer_kwargs(core, B: int, T: int, device, dtype):
    """What Qwen3_5TextModel.forward computes once per sequence: rotary embeddings (4-row position ids, text row
    first), the causal mask for full-attention layers (None for linear attention: no padding here)."""
    from transformers.masking_utils import create_causal_mask
    pos = torch.arange(T, device=device).view(1, 1, -1).expand(4, B, -1)
    dummy = torch.zeros(B, T, core.config.hidden_size, device=device, dtype=dtype)
    core.rotary_emb.to(device)
    pe = core.rotary_emb(dummy, pos[1:])
    causal = create_causal_mask(config=core.config, inputs_embeds=dummy, attention_mask=None, past_key_values=None,
                                position_ids=pos[0])
    return {"pe": pe, "causal": causal, "text_pos": pos[0]}


def run_layer(core, i: int, x, kw):
    L = core.layers[i]
    mask = kw["causal"] if core.config.layer_types[i] == "full_attention" else None
    return L(x, position_embeddings=kw["pe"], attention_mask=mask, position_ids=kw["text_pos"], past_key_values=None,
             use_cache=False)


class OnDevice:
    """Moves modules to the device for a block and back to where they were (no-op when already there)."""

    def __init__(self, mods, device):
        self.mods, self.device = list(mods), torch.device(device)
        self.back = []

    def __enter__(self):
        for m in self.mods:
            p = next(m.parameters(), None)
            was = p.device if p is not None else self.device
            self.back.append(was)
            if was != self.device:
                m.to(self.device)
        return self

    def __exit__(self, *a):
        for m, was in zip(self.mods, self.back):
            if was != self.device:
                m.to(was)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()


def embed(core, ids, device):
    with OnDevice([core.embed_tokens], device):
        return core.embed_tokens(torch.tensor([ids], device=device))[0].detach()


def head_logprob(core, head, h_last, positions, token_id):
    """log p(token_id) at each position from the last layer's output (T, d): final norm + lm_head, (P,)."""
    y = core.norm(h_last[positions])
    logits = head(y).float()
    return logits[:, token_id] - torch.logsumexp(logits, -1)


# ------------------------------------------------------------------------------------------------- attribution
def _down_proj_hooks(core, layers, store):
    hs = []
    for i in layers:
        def pre(mod, inputs, i=i):
            store[i] = inputs[0]
        hs.append(core.layers[i].mlp.down_proj.register_forward_pre_hook(pre))
    return hs


def group_backward(core, head, layers: Sequence[int], x_in, incoming, positions: Sequence[int], token_id: int,
                   keep: int, device, want_full=False):
    """One group of consecutive layers: re-run them from x_in (T, d) with grad, then one vector-Jacobian product per
    position: incoming[p] (T, d) is d log p_p / d(group output), or None when the group is the top one (then
    log p(token_id) at positions is formed here with the final norm and the head). Returns
    {grad_in: (P, T, d) at the group input, same: {layer: {p: (idx, val)}}, total: ..., layer_sum: {layer: [P]},
    attr_full: {layer: (P, T, F)} when want_full}."""
    T = x_in.shape[0]
    x = x_in.detach().to(device).requires_grad_(True)
    kw = layer_kwargs(core, 1, T, device, x.dtype)
    acts: Dict[int, torch.Tensor] = {}
    hooks = _down_proj_hooks(core, layers, acts)
    try:
        h = x[None]
        for i in layers:
            h = run_layer(core, i, h, kw)
        top = incoming is None
        if top:
            logp = head_logprob(core, head, h[0], list(positions), token_id)
        P = len(positions)
        grad_in = torch.zeros(P, T, x.shape[1], dtype=torch.float32)
        same = {i: {} for i in layers}
        total = {i: {} for i in layers}
        layer_sum = {i: [0.0] * P for i in layers}
        full = {i: [] for i in layers} if want_full else None
        inputs = [x] + [acts[i] for i in layers]
        for p, t in enumerate(positions):
            if top:
                grads = torch.autograd.grad(logp[p], inputs, retain_graph=True)
            else:
                grads = torch.autograd.grad(h[0], inputs, grad_outputs=incoming[p].to(device, h.dtype),
                                            retain_graph=True)
            grad_in[p] = grads[0].detach().float().cpu()
            for k, i in enumerate(layers):
                a, ga = acts[i][0].detach().float(), grads[k + 1][0].detach().float()
                s = a[t] * ga[t]                                   # same-position attribution (PLAN's formula)
                tot = (a[: t + 1] * ga[: t + 1]).sum(0)            # the neuron's whole-context contribution to p
                layer_sum[i][p] = float(s.sum())
                for name, v, d in (("same", s, same), ("total", tot, total)):
                    kk = min(keep, v.numel())
                    tv, ti = torch.topk(v.abs(), kk)
                    d[i][p] = (ti.cpu().numpy().astype(np.int32), v[ti].cpu().numpy().astype(np.float32))
                if want_full:
                    full[i].append((a * ga).cpu())
    finally:
        for hk in hooks:
            hk.remove()
    out = {"grad_in": grad_in, "same": same, "total": total, "layer_sum": layer_sum}
    if want_full:
        out["attr_full"] = {i: torch.stack(full[i]) for i in layers}
    return out


def groups_of(n_layers: int, group: int) -> List[List[int]]:
    return [list(range(s, min(n_layers, s + group))) for s in range(0, n_layers, group)]


def attribute_sequence(core, head, ids, positions, token_id, group=GROUP, keep=KEEP, device="cpu",
                       return_grads=False):
    """Whole route for one sequence held in memory (the run streams the same steps per group over all traces):
    forward caching the group inputs, then group_backward from the top down."""
    n = len(core.layers)
    gs = groups_of(n, group)
    x = embed(core, ids, device)
    kw = layer_kwargs(core, 1, len(ids), device, x.dtype)
    cache = {}
    with torch.no_grad():
        for g in gs:
            cache[g[0]] = x.cpu()
            with OnDevice([core.layers[i] for i in g], device):
                h = x[None]
                for i in g:
                    h = run_layer(core, i, h, kw)
                x = h[0]
    incoming = None
    out = {"same": {}, "total": {}, "layer_sum": np.zeros((n, len(positions))), "grad_in": {}, "attr_full": {}}
    for g in reversed(gs):
        mods = [core.layers[i] for i in g] + ([core.norm, head] if incoming is None else [])
        with OnDevice(mods, device):
            r = group_backward(core, head, g, cache[g[0]], incoming, positions, token_id, keep, device,
                               want_full=return_grads)
        incoming = r["grad_in"]
        out["same"].update(r["same"])
        out["total"].update(r["total"])
        for i in g:
            out["layer_sum"][i] = r["layer_sum"][i]
        if return_grads:
            out["grad_in"][g[0]] = r["grad_in"]
            out["attr_full"].update(r["attr_full"])
    return out


def end_to_end_grads(core, head, ids, positions, token_id):
    """Reference for the check: plain autograd through the whole model (single device), gradients of log p at each
    position w.r.t. every layer's input and every down_proj input. {grad_in: {l: (P,T,d)}, a: {l: (T,F)},
    a_grad: {l: (P,T,F)}}."""
    xs, acts = {}, {}
    hooks = []
    for i, L in enumerate(core.layers):
        def pre(mod, inputs, i=i):
            inputs[0].retain_grad()
            xs[i] = inputs[0]
        hooks.append(L.register_forward_pre_hook(pre))
        def pre_d(mod, inputs, i=i):
            inputs[0].retain_grad()
            acts[i] = inputs[0]
        hooks.append(L.mlp.down_proj.register_forward_pre_hook(pre_d))
    try:
        emb = core.embed_tokens(torch.tensor([ids])).detach().requires_grad_(True)
        h = core(inputs_embeds=emb, use_cache=False).last_hidden_state   # already normed; recompute pre-norm via hook
    finally:
        for hk in hooks:
            hk.remove()
    # last_hidden_state is after the final norm; log p from it needs only the head
    logits = head(h[0, list(positions)]).float()
    logp = logits[:, token_id] - torch.logsumexp(logits, -1)
    grad_in = {i: [] for i in xs}
    a_grad = {i: [] for i in acts}
    for p in range(len(positions)):
        for t in list(xs.values()) + list(acts.values()):
            t.grad = None
        logp[p].backward(retain_graph=True)
        for i in xs:
            grad_in[i].append(xs[i].grad[0].detach().clone())
            a_grad[i].append(acts[i].grad[0].detach().clone())
    return {"grad_in": {i: torch.stack(v) for i, v in grad_in.items()},
            "a": {i: acts[i][0].detach() for i in acts},
            "a_grad": {i: torch.stack(v) for i, v in a_grad.items()}}


# ---------------------------------------------------------------------------------------------------- ablation
def per_layer(C: Set[int], width: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for g in C:
        l, j = cm.split_id(g, width)
        out.setdefault(l, []).append(j)
    return {l: sorted(v) for l, v in sorted(out.items())}


def zero_community(C: Set[int], width: int) -> Dict[int, List[int]]:
    return per_layer(C, width)


def zero_all_but(C: Set[int], S0: Set[int], width: int) -> Dict[int, List[int]]:
    """In C's layers, every neuron except C and the always-on set."""
    keep = per_layer(C, width)
    s0 = per_layer(S0, width)
    return {l: sorted(set(range(width)) - set(js) - set(s0.get(l, []))) for l, js in keep.items()}


def random_same_size(C: Set[int], width: int, n_sets: int, seed: int = SEED) -> List[Dict[int, List[int]]]:
    """n_sets random neuron sets with |C_l| neurons in each of C's layers (seeded; the A1 placebo)."""
    sizes = {l: len(js) for l, js in per_layer(C, width).items()}
    rng = random.Random(seed)
    out = []
    for _ in range(n_sets):
        out.append({l: sorted(rng.sample(range(width), k)) for l, k in sizes.items()})
    return out


def _mask_hook(mask_by_layer, t_e):
    """forward pre-hook factory: multiply the down_proj input at position t_e by a (B, F) keep mask."""
    def make(i):
        def pre(mod, inputs):
            a = inputs[0]
            m = mask_by_layer[i].to(a.device, a.dtype)
            a = a.clone()
            a[:, t_e, :] = a[:, t_e, :] * m
            return (a,)
        return pre
    return make


def ablate_probabilities(core, head, ids, t_e, token_id, conditions: Sequence[Dict[int, Sequence[int]]], device="cpu",
                         batch=8) -> List[float]:
    """p(token_id) at position t_e under each condition {layer: neurons to zero at t_e}. One base forward of
    ids[: t_e + 1] caches every layer's input; from the first touched layer on, each condition is one replaced row
    (the positions before t_e are unaffected by construction). Conditions run in batches."""
    ids = list(ids[: t_e + 1])
    T = len(ids)
    F = width_of(core)
    n = len(core.layers)
    x = embed(core, ids, device)
    kw1 = layer_kwargs(core, 1, T, device, x.dtype)
    X = []
    with torch.no_grad():
        for i in range(n):
            X.append(x)
            with OnDevice([core.layers[i]], device):
                x = run_layer(core, i, x[None], kw1)[0]
        base_last = x
    out = [None] * len(conditions)
    touched = [min(c) if c else None for c in conditions]
    with torch.no_grad(), OnDevice([core.norm, head], device):
        p_base = float(torch.softmax(head(core.norm(base_last[t_e][None])).float(), -1)[0, token_id])
        todo = [k for k, c in enumerate(conditions) if c]
        for k in range(len(conditions)):
            if not conditions[k]:
                out[k] = p_base
        for s in range(0, len(todo), batch):
            ks = todo[s:s + batch]
            B = len(ks)
            m = min(touched[k] for k in ks)
            kw = layer_kwargs(core, B, T, device, x.dtype)
            rows = X[m][t_e][None].expand(B, -1).clone()
            for i in range(m, n):
                masks = torch.ones(B, F)
                any_hit = False
                for b, k in enumerate(ks):
                    js = conditions[k].get(i)
                    if js:
                        masks[b, list(js)] = 0.0
                        any_hit = True
                Xb = X[i][None].expand(B, -1, -1).clone()
                Xb[:, t_e] = rows
                hk = None
                if any_hit:
                    hk = core.layers[i].mlp.down_proj.register_forward_pre_hook(_mask_hook({i: masks}, t_e)(i))
                try:
                    with OnDevice([core.layers[i]], device):
                        y = run_layer(core, i, Xb, kw)
                finally:
                    if hk is not None:
                        hk.remove()
                rows = y[:, t_e]
            pr = torch.softmax(head(core.norm(rows)).float(), -1)[:, token_id]
            for b, k in enumerate(ks):
                out[k] = float(pr[b])
    return out


def full_forward_prob(core, head, ids, t_e, token_id, condition: Dict[int, Sequence[int]]) -> float:
    """Reference: the whole model with hooks zeroing the given neurons at t_e (single device)."""
    ids = list(ids[: t_e + 1])
    F = width_of(core)
    hooks = []
    for i, js in condition.items():
        m = torch.ones(1, F)
        m[0, list(js)] = 0.0
        hooks.append(core.layers[i].mlp.down_proj.register_forward_pre_hook(_mask_hook({i: m}, t_e)(i)))
    try:
        with torch.no_grad():
            h = core(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
            return float(torch.softmax(head(h[0, t_e][None]).float(), -1)[0, token_id])
    finally:
        for hk in hooks:
            hk.remove()


# ------------------------------------------------------------------------------------------------------- data
def _rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def positions_for(tok, pos_row, neg_row, k_before=K_BEFORE):
    """The shared sequence (thinking template + the finished reasoning; its next token is </think>), the close event
    (last token) and its k_before predecessors, and the neg event: the last token of the neg row's cut of the same
    reasoning (the model keeps reasoning there) and its predecessors."""
    p = tok.apply_chat_template([{"role": "user", "content": pos_row["prompt"]}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    resp = pos_row["response"]
    if neg_row is not None and not resp.startswith(neg_row["response"]):
        raise ValueError("neg response is not a prefix of the pos response")
    enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
    ids = pi + enc["input_ids"]
    pos_event = len(ids) - 1
    out = {"ids": ids, "prompt_len": len(pi), "pos_event": pos_event,
           "pos_positions": [pos_event - i for i in range(k_before + 1)]}
    if neg_row is not None:
        cut = len(neg_row["response"])
        last = max(k for k, (s, e) in enumerate(enc["offset_mapping"]) if e <= cut)
        ne = len(pi) + last
        out.update(neg_event=ne, neg_positions=[ne - i for i in range(k_before + 1)])
    return out


def heldout_prompts(traces=TRACES, n_fit=N_FIT, split_seed=SPLIT_SEED):
    """The 28 held-out traces of cc_weights (same shuffle, same seed), as rows."""
    rows = _rows(traces)
    order = list(range(len(rows)))
    random.Random(split_seed).shuffle(order)
    return [rows[i] for i in order[n_fit:]]


def load_model(model_dir, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def _save_bf(path: Path, t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        np.save(path, t.view(torch.int16).numpy())
        return
    np.save(path, t.numpy())


def _load_bf(path: Path, dtype):
    a = np.load(path)
    t = torch.from_numpy(a)
    return t.view(torch.bfloat16) if dtype == torch.bfloat16 else t


# -------------------------------------------------------------------------------------------------------- run
def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "attributions.npz").exists():
        raise SystemExit(f"{out}/attributions.npz exists")
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok, model = load_model(args.model, dtype)
    core, head = model_parts(model)
    token_id = tok.convert_tokens_to_ids(CLOSE)
    pos, neg = _rows(POS), _rows(NEG)
    ho = {r["prompt"] for r in heldout_prompts()}
    pairs = [(a, b) for a, b in zip(pos, neg) if a["prompt"] not in ho]
    if args.limit:
        pairs = pairs[: args.limit]
    traces = []
    for a, b in pairs:
        d = positions_for(tok, a, b, K_BEFORE)
        if args.max_tokens and len(d["ids"]) > args.max_tokens:       # CPU checks only: cut the sequence
            d["ids"] = d["ids"][: args.max_tokens]
            d["pos_event"] = len(d["ids"]) - 1
            d["pos_positions"] = [d["pos_event"] - i for i in range(K_BEFORE + 1)]
            if d["neg_event"] >= d["pos_event"] - K_BEFORE:
                d["neg_event"] = max(K_BEFORE, d["pos_event"] - 2 * K_BEFORE - 1)
                d["neg_positions"] = [d["neg_event"] - i for i in range(K_BEFORE + 1)]
        d["positions"] = d["pos_positions"] + d["neg_positions"]
        traces.append(d)
    n_layers = len(core.layers)
    F = width_of(core)
    gs = groups_of(n_layers, args.group)
    meta = {"model": str(args.model), "dtype": args.dtype, "token": CLOSE, "token_id": token_id, "n_layers": n_layers,
            "width": F, "group": args.group, "keep": args.keep, "k_before": K_BEFORE, "discovery_traces": len(traces),
            "heldout_excluded": sum(a["prompt"] in ho for a in pos),
            "traces": [{"prompt_len": d["prompt_len"], "n_tokens": len(d["ids"]), "pos_event": d["pos_event"],
                        "neg_event": d["neg_event"], "positions": d["positions"]} for d in traces]}
    print(json.dumps({k: v for k, v in meta.items() if k != "traces"}), flush=True)
    print(json.dumps({"tokens_total": sum(len(d["ids"]) for d in traces),
                      "cache_gb_per_group": round(sum(len(d["ids"]) for d in traces) * core.config.hidden_size
                                                  * (2 if dtype == torch.bfloat16 else 4) / 1e9, 2)}), flush=True)
    dev = args.device
    # forward: cache every group's input per trace
    with torch.no_grad():
        for gi, g in enumerate(gs):
            with OnDevice([core.layers[i] for i in g] + ([core.embed_tokens] if gi == 0 else []), dev):
                for ti, d in enumerate(traces):
                    x = embed(core, d["ids"], dev) if gi == 0 else _load_bf(cache / f"x{gi}_{ti}.npy", dtype).to(dev)
                    _save_bf(cache / f"x{gi}_{ti}.npy", x) if gi == 0 else None
                    kw = layer_kwargs(core, 1, x.shape[0], dev, x.dtype)
                    h = x[None]
                    for i in g:
                        h = run_layer(core, i, h, kw)
                    if gi + 1 < len(gs):
                        _save_bf(cache / f"x{gi + 1}_{ti}.npy", h[0])
            print(f"[nc] forward group {gi + 1}/{len(gs)} {time.time() - t0:.0f}s", flush=True)
    # backward: top group first; incoming gradients per trace on disk between groups
    P = len(traces[0]["positions"])
    same = {i: [None] * len(traces) for i in range(n_layers)}
    total = {i: [None] * len(traces) for i in range(n_layers)}
    layer_sum = np.zeros((n_layers, len(traces), P), dtype=np.float32)
    for gi in range(len(gs) - 1, -1, -1):
        g = gs[gi]
        top = gi == len(gs) - 1
        with OnDevice([core.layers[i] for i in g] + ([core.norm, head] if top else []), dev):
            for ti, d in enumerate(traces):
                x_in = _load_bf(cache / f"x{gi}_{ti}.npy", dtype)
                incoming = None if top else torch.load(cache / f"g_{ti}.pt")
                r = group_backward(core, head, g, x_in, incoming, d["positions"], token_id, args.keep, dev)
                torch.save(r["grad_in"].to(torch.bfloat16) if dtype == torch.bfloat16 else r["grad_in"],
                           cache / f"g_{ti}.pt")
                for i in g:
                    same[i][ti] = r["same"][i]
                    total[i][ti] = r["total"][i]
                    layer_sum[i, ti] = r["layer_sum"][i]
                if ti % 10 == 0:
                    print(f"[nc] backward group {gi + 1}/{len(gs)} trace {ti}/{len(traces)} {time.time() - t0:.0f}s",
                          flush=True)
    # pack: arrays (layer, trace, position, keep)
    K = args.keep
    arrs = {}
    for name, store in (("same", same), ("total", total)):
        idx = np.full((n_layers, len(traces), P, K), -1, dtype=np.int32)
        val = np.zeros((n_layers, len(traces), P, K), dtype=np.float32)
        for i in range(n_layers):
            for ti in range(len(traces)):
                for p in range(P):
                    a, b = store[i][ti][p]
                    idx[i, ti, p, : len(a)] = a
                    val[i, ti, p, : len(b)] = b
        arrs[f"{name}_idx"], arrs[f"{name}_val"] = idx, val
    arrs["layer_sum"] = layer_sum
    np.savez_compressed(out / "attributions.npz", **arrs)
    meta["seconds_attribution"] = round(time.time() - t0)
    (out / "attributions.json").write_text(json.dumps(meta, indent=1))
    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
    print(json.dumps({"done": "attribution", "seconds": meta["seconds_attribution"]}), flush=True)


def selections_from(npz, meta, which="same", k=KEEP):
    """Per (trace, position): global ids of the top-k |attr| across layers, and their values."""
    idx, val = npz[f"{which}_idx"], npz[f"{which}_val"]
    n_layers, n_tr, P, K = idx.shape
    F = meta["width"]
    sel, vals = [], []
    for ti in range(n_tr):
        for p in range(P):
            per = {i: (idx[i, ti, p][idx[i, ti, p] >= 0], val[i, ti, p][idx[i, ti, p] >= 0]) for i in range(n_layers)}
            g, v = cm.merge_topk(per, k, F)
            sel.append(g.tolist())
            vals.append(v.tolist())
    return sel, vals


def communities_step(args):
    out = Path(args.out)
    meta = json.loads((out / "attributions.json").read_text())
    with np.load(out / "attributions.npz") as z:
        npz = {k: z[k] for k in z.files}
    F = meta["width"]
    P = len(meta["traces"][0]["positions"])
    kb = meta["k_before"]
    sel, vals = selections_from(npz, meta, "same", args.keep)
    G = cm.pmi_graph(sel, min_co=args.min_co)
    comms = cm.louvain(G, seed=SEED, cap=CAP)
    S0 = cm.always_on(sel, ALWAYS_ON)
    # masses at the event positions only: p = 0 (pos close event) and p = kb + 1 (neg event)
    mass_pos, mass_neg = {}, {}
    for ti in range(len(meta["traces"])):
        for p, mass in ((0, mass_pos), (kb + 1, mass_neg)):
            q = ti * P + p
            for g, v in zip(sel[q], vals[q]):
                mass[g] = mass.get(g, 0.0) + abs(v)
    C, scores = cm.candidate(comms, mass_pos, mass_neg)
    layers = per_layer(C, F)
    rep = {"n_positions": len(sel), "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
           "n_communities": len(comms), "sizes": [len(c) for c in comms][:50],
           "always_on": {"n": len(S0), "neurons": [list(cm.split_id(g, F)) for g in sorted(S0)]},
           "candidate": {"size": len(C), "layers": {str(l): js for l, js in layers.items()},
                         "neurons": [list(cm.split_id(g, F)) for g in sorted(C)],
                         "score": scores[0] if scores else None,
                         "layer26_members": layers.get(26, [])},
           "top_scores": [{k: v for k, v in s.items() if k != "neurons"} for s in scores[:10]],
           "layer_sum_mean_pos_event": npz["layer_sum"][:, :, 0].mean(1).tolist(),
           "layer_sum_mean_neg_event": npz["layer_sum"][:, :, kb + 1].mean(1).tolist()}
    means = ROOT / "lab" / "runs" / "cc_neurons" / "means.pt"
    if means.exists():
        try:
            m = torch.load(means)["mean"]
            da = {l: (m["pos"]["a"][l] - m["neg"]["a"][l]) for l in layers}
            rep["candidate"]["delta_a_43_23"] = {str(l): [float(da[l][j]) for j in js] for l, js in layers.items()}
        except Exception as e:      # descriptive only
            rep["candidate"]["delta_a_43_23"] = f"unavailable: {e}"
    (out / "communities.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("graph", "n_communities", "always_on")} |
                     {"candidate": {k: rep["candidate"][k] for k in ("size", "layers", "score")}}), flush=True)


def heldout_traces(tok, mode, k_before=K_BEFORE):
    """mode split: finished traces among cc_weights' 28 held-out (their close event is the last reasoning token);
    mode pos: the pos rows inside the held-out set (same traces, same positions by construction)."""
    rows = [r for r in heldout_prompts() if r["finish"] != "length"]
    if mode == "pos":
        ho = {r["prompt"] for r in rows}
        rows = [r for r in _rows(POS) if r["prompt"] in ho]
        return [positions_for(tok, r, None, k_before) for r in rows]
    out = []
    for r in rows:
        row = {"prompt": r["prompt"], "response": r["reasoning"].lstrip("\n")}
        out.append(positions_for(tok, row, None, k_before))
    return out


def ablate_step(args):
    out = Path(args.out)
    meta = json.loads((out / "attributions.json").read_text())
    rep = json.loads((out / "communities.json").read_text())
    F = meta["width"]
    dtype = getattr(torch, meta["dtype"])
    tok, model = load_model(args.model or meta["model"], dtype)
    core, head = model_parts(model)
    token_id = meta["token_id"]
    C = {cm.global_id(l, j, F) for l, j in rep["candidate"]["neurons"]}
    S0 = {cm.global_id(l, j, F) for l, j in rep["always_on"]["neurons"]}
    if not C:
        raise SystemExit("no candidate community")
    conds = [{}, zero_community(C, F)]
    plac = random_same_size(C, F, args.placebos, SEED)
    conds += plac
    conds.append(zero_all_but(C, S0, F))
    rand_keep = {cm.global_id(l, j, F) for l, js in plac[0].items() for j in js}
    conds.append(zero_all_but(rand_keep, S0, F))
    layers = sorted(per_layer(C, F))
    for l in layers:                                   # A3: one layer's part of C alone
        conds.append({l: per_layer(C, F)[l]})
    traces = heldout_traces(tok, args.heldout)
    if args.limit:
        traces = traces[: args.limit]
    if args.max_tokens:
        for d in traces:
            d["ids"] = d["ids"][: args.max_tokens]
            d["pos_event"] = len(d["ids"]) - 1
    t0 = time.time()
    ps = []
    for ti, d in enumerate(traces):
        p = ablate_probabilities(core, head, d["ids"], d["pos_event"], token_id, conds, device=args.device,
                                 batch=args.cond_batch)
        ps.append(p)
        print(json.dumps({"trace": ti, "n_tokens": len(d["ids"]), "p_base": p[0], "p_A1": p[1],
                          "p_A2": p[2 + args.placebos], "s": round(time.time() - t0)}), flush=True)
    ps = np.array(ps)                                  # (traces, conditions)
    base = ps[:, 0]
    drop = 1 - ps[:, 1] / base
    pl = 1 - ps[:, 2:2 + args.placebos] / base[:, None]
    pl_med_per_set = np.median(pl, 0)
    a1 = {"drop": float(np.median(drop)), "placebo": float(np.median(pl_med_per_set)),
          "placebo_max": float(pl_med_per_set.max()), "drop_per_trace": drop.tolist()}
    k2 = 2 + args.placebos
    ret = ps[:, k2] / base
    ret_r = ps[:, k2 + 1] / base
    a2 = {"retained": float(np.median(ret)), "random": float(np.median(ret_r)), "retained_per_trace": ret.tolist()}
    a3 = {str(l): float(np.median(1 - ps[:, k2 + 2 + k] / base)) for k, l in enumerate(layers)}
    summ = {"n_heldout": len(traces), "heldout_mode": args.heldout, "community": rep["candidate"],
            "p_base": base.tolist(), "A1": a1, "A2": a2, "A3_drop_by_layer": a3,
            "A1_pass": cm.a1_pass(a1), "A2_pass": cm.a2_pass(a2), "verdict": cm.verdict(a1, a2),
            "seconds": round(time.time() - t0), "conditions": len(conds)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("n_heldout", "A1", "A2", "A3_drop_by_layer", "verdict")}), flush=True)


def check(args):
    """Two-method check on a small model: the layered route against end-to-end autograd (float32 on CPU)."""
    dtype = getattr(torch, args.dtype)
    tok, model = load_model(args.model, dtype)
    core, head = model_parts(model)
    token_id = tok.convert_tokens_to_ids(CLOSE)
    pos, neg = _rows(POS), _rows(NEG)
    res = []
    for a, b in list(zip(pos, neg))[: args.n]:
        d = positions_for(tok, a, b, K_BEFORE)
        ids = d["ids"][: args.max_tokens]
        positions = [len(ids) - 1 - i for i in range(K_BEFORE + 1)] + [len(ids) // 2 - i for i in range(K_BEFORE + 1)]
        t0 = time.time()
        ref = end_to_end_grads(core, head, ids, positions, token_id)
        t1 = time.time()
        got = attribute_sequence(core, head, ids, positions, token_id, group=args.group, keep=KEEP, device=args.device,
                                 return_grads=True)
        t2 = time.time()
        errs_g = {l: float((got["grad_in"][l].double() - ref["grad_in"][l].double()).norm() / ref["grad_in"][l].double().norm())
                  for l in got["grad_in"]}
        errs_a = {}
        for l in range(len(core.layers)):
            r = (ref["a"][l] * ref["a_grad"][l]).double()
            errs_a[l] = float((got["attr_full"][l].double() - r).norm() / r.norm())
        res.append({"n_tokens": len(ids), "grad_in_rel_err": errs_g, "max_grad_in_rel_err": max(errs_g.values()),
                    "max_attr_rel_err": max(errs_a.values()), "seconds_end_to_end": round(t1 - t0, 1),
                    "seconds_layered": round(t2 - t1, 1)})
        print(json.dumps(res[-1]), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "communities", "ablate", "check"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(OUT))
        s.add_argument("--model", default=None)
        s.add_argument("--device", default="cuda" if name != "check" else "cpu")
        s.add_argument("--dtype", default="bfloat16" if name != "check" else "float32")
        s.add_argument("--group", type=int, default=GROUP)
        s.add_argument("--keep", type=int, default=KEEP)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--max-tokens", type=int, default=0)
        s.add_argument("--keep-cache", action="store_true")
        s.add_argument("--min-co", type=int, default=MIN_CO)
        s.add_argument("--heldout", default="split", choices=("split", "pos"))
        s.add_argument("--placebos", type=int, default=PLACEBOS)
        s.add_argument("--cond-batch", type=int, default=8)
        s.add_argument("--n", type=int, default=2)
    args = ap.parse_args(argv)
    if args.model is None:
        args.model = SMALL_HF if args.cmd == "check" else BASE_HF
    if args.cmd == "check" and args.max_tokens == 0:
        args.max_tokens = 256
    {"run": run, "communities": communities_step, "ablate": ablate_step, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
