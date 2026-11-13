"""A llama.cpp control vector from two sets of texts (WSP-Lang op contrast_vector).

pos / neg: jsonl rows {"prompt": ..., "response": ...}, usually the same prompts answered in two styles. Each row is
put through the model's chat template (user = prompt, assistant = response, no system prompt), and the residual stream
is averaged over the response tokens, per layer. D = mean(pos) - mean(neg).

llama.cpp adds direction.N (N >= 1) at the output of 0-based layer N and the residual stream carries it on, so the file
holds per-layer increments: direction.1 = D after layer 1 - D after layer 0, ..., and their running sum after layer N
is the difference itself (writing D directly summed it ~N times and broke the model, results §43.5). The vector is used
at run time (llama-server --control-vector-scaled file:scale); it cannot be stored in Qwen3.5's bias-free weights.

placebo(): same norm per layer, random direction; the control for "any push of this size does it".
"""
import json
from pathlib import Path

import numpy as np
import torch


def increments(D):
    """D: (n_layers + 1, d), row k = residual stream after layer k - 1 (row 0 = embeddings).
    -> {N: increment added at layer N's output}, N = 1 .. n_layers - 1 (layer 0 cannot be steered by llama.cpp)."""
    return {N: (D[N + 1] - D[N]).clone() for N in range(1, D.shape[0] - 1)}


def write_gguf(vecs, out, model_hint):
    import gguf
    w = gguf.GGUFWriter(str(out), "controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    w.add_uint32("controlvector.layer_count", len(vecs))
    for k in sorted(vecs):
        w.add_tensor(f"direction.{k}", np.asarray(vecs[k], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def placebo(vecs, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    for k, v in vecs.items():
        r = g.normal(size=np.shape(v)).astype(np.float32)
        out[k] = r * (np.linalg.norm(v) / max(np.linalg.norm(r), 1e-30))
    return out


def _rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _pairs(tok, rows, max_resp):
    """-> [(prompt ids, response ids, slice of response ids to average)].
    Optional row fields: "think": true uses the thinking-mode template (the response is then the reasoning, starting
    after "<think>\\n"); "focus": [a, b) character offsets in the response, averaged over the tokens that overlap it
    (the response is cut right after them: the model is causal, later tokens cannot change those states)."""
    out = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=bool(r.get("think")))   # thinking off = llama-server's template
        pi = tok(p, add_special_tokens=False)["input_ids"]
        if r.get("focus") is None:
            ri = tok(r["response"], add_special_tokens=False)["input_ids"][:max_resp]
            if ri:
                out.append((pi, ri, slice(0, len(ri))))
            continue
        a, b = r["focus"]
        enc = tok(r["response"], add_special_tokens=False, return_offsets_mapping=True)
        idx = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a]   # overlap: " So" starts at a-1
        if idx:
            ri = enc["input_ids"][:idx[-1] + 1]
            out.append((pi, ri, slice(idx[0], idx[-1] + 1)))
    return out


def _stream(layers, device):
    """Each decoder layer goes to the device only for its own forward (a 9B model in bf16 on the CPU, 12 GB GPU)."""
    def mv(x, dev):
        if torch.is_tensor(x):
            return x.to(dev)
        if isinstance(x, (list, tuple)):
            return type(x)(mv(v, dev) for v in x)
        if isinstance(x, dict):
            return {k: mv(v, dev) for k, v in x.items()}
        return x
    hs = []
    for L in layers:
        def pre(mod, args, kwargs):
            mod.to(device)
            return mv(args, device), mv(kwargs, device)

        def post(mod, args, kwargs, out):
            mod.to("cpu")
            return mv(out, "cpu")
        hs.append(L.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(L.register_forward_hook(post, with_kwargs=True))
    return hs


@torch.no_grad()
def hidden_states_pre_norm(core, input_ids):
    """transformers' hidden_states with the last row replaced by the last decoder layer's output BEFORE the final norm
    (transformers records hidden_states[-1] after it). Rows 0..n-1 unchanged. Every reader of the residual stream
    goes through here (layer_means, act_map capture, close_probe/cvec_extract; audit 2026-11-13)."""
    cap = {}
    hook = core.layers[-1].register_forward_hook(
        lambda mod, args, out: cap.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    try:
        hs = list(core(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states)
    finally:
        hook.remove()
    hs[-1] = cap.pop("h")
    return hs


@torch.no_grad()
def layer_means(model, pairs, batch=8):
    """Mean residual stream over each pair's selected response tokens, per hidden state: (n_layers + 1, d), float32.
    The last row is the last layer's output BEFORE the final norm: transformers records hidden_states[-1] after it,
    and llama.cpp adds direction.(n-1) to the residual before the norm (the post-norm row made the last increment the
    norm's rescaling: 0.8B 44.9 against <= 1 elsewhere, 9B 38.6 against <= 13; results 45.2, fixed 2026-11-11)."""
    core = model.model
    core = core.language_model if hasattr(core, "language_model") else core
    sums, count = None, 0
    if any(len(a) + len(b) > 1024 for a, b, _ in pairs):
        batch = 1                                   # long reasoning traces: hidden states of 8 x 4k tokens do not fit
    for k in range(0, len(pairs), batch):
        chunk = pairs[k:k + batch]
        n = max(len(a) + len(b) for a, b, _ in chunk)
        ids = torch.tensor([a + b + [0] * (n - len(a) - len(b)) for a, b, _ in chunk])   # right pad: causal
        hs = hidden_states_pre_norm(core, ids)
        for j, (a, b, s) in enumerate(chunk):
            sl = slice(len(a) + s.start, len(a) + s.stop)
            v = torch.stack([h[j, sl].float().cpu().sum(0) for h in hs])
            sums = v if sums is None else sums + v
            count += s.stop - s.start
    return sums / count


def contrast_vector(model_dir, pos, neg, out, model_hint="qwen35", max_resp=384, placebo_seed=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(out)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    P, N = _pairs(tok, _rows(pos), max_resp), _pairs(tok, _rows(neg), max_resp)
    if not P or not N:
        raise ValueError("contrast_vector: pos and neg need at least one row with a response")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    core = model.model
    layers = (core.language_model if hasattr(core, "language_model") else core).layers
    hooks = _stream(layers, device) if device != "cpu" else []
    try:
        D = layer_means(model, P) - layer_means(model, N)
    finally:
        for h in hooks:
            h.remove()
    del model
    vecs = {k: v.numpy() for k, v in increments(D).items()}
    write_gguf(vecs, out, model_hint)
    res = {"output": str(out), "pairs": min(len(P), len(N)), "pos": len(P), "neg": len(N), "layers": len(vecs),
           "last_layer_norm": float(D[-1].norm()), "increment_norms": {k: round(float(np.linalg.norm(v)), 4)
                                                                        for k, v in vecs.items()}}
    if placebo_seed is not None:
        pp = out.with_name(out.stem + f".placebo{placebo_seed}.gguf")
        write_gguf(placebo(vecs, placebo_seed), pp, model_hint)
        res["placebo"] = str(pp)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    return res
