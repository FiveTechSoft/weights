"""layer_probe.py - a local, functional probe of one layer: which parts of MiMo's delta reproduce its output change.

Scope and limits (read before quoting any number):
  * ONE layer at a time (layer 0 is a Gated DeltaNet layer) of 32, run in float32 on CPU with the transformers implementation.
    It says nothing about the full model or about behaviour (no generation, no benchmark).
  * Input: real token embeddings of the shared prompts, taken from the BASE model's embedding table, so the input
    that reaches layer 0 is exactly what the model sees there (layer 0 has no earlier layer). Later layers would
    need real activations from the layers before them; that is not done.
  * Variants add part of delta = W_mimo - W_base to the base layer. Metrics, defined before the first run:
      shift = out_mimo - out_base          (what MiMo's weights do to this layer's output on these tokens)
      captured = <out_v - out_base, shift> / <shift, shift>      (1 = reproduces, 0 = nothing, <0 = wrong way)
      resid = ||out_v - out_mimo|| / ||shift||                    (0 = exact; 1 = no better than the base layer)
    Every structured variant is paired with a RANDOM control: Gaussian noise with the same Frobenius norm per
    tensor, same tensors. A structured part is only interesting if it beats its random control.
  * Per-tensor streaming keeps memory under the 2 GB limit of this machine.

  python lab/probes/delta_patterns/layer_probe.py fetch   BASE@rev MODEL@rev CACHE_DIR LAYER
  python lab/probes/delta_patterns/layer_probe.py stack   CACHE_DIR out.json   (layers 0-3 composed; needs fetch for 0-3)
  python lab/probes/delta_patterns/layer_probe.py run     CACHE_DIR out.json LAYER   (layers in order: 0 first, each saves the next layer's input)
"""
import json
import sys
import time

import torch

from engine import weight_diff as wd

PROMPTS = [
    "The capital of France is Paris, and the capital of Spain is Madrid.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "<|im_start|>user\nWrite a function that reverses a string.<|im_end|>\n<|im_start|>assistant\n",
    "If 3x + 5 = 20, then x = 5. Let us check: 3 * 5 + 5 = 20.",
]
# Held out: written after the first layer-0 numbers on the four prompts above were seen, before any number on these was
# computed. The delta bases (rank k, entry thresholds) come from the weights only, never from any prompt.
HELD_OUT = [
    "El rey de España vive en Madrid y el río Ebro desemboca en el mar Mediterráneo.",
    '{"name": "get_weather", "arguments": {"city": "Tokyo", "unit": "celsius"}}',
    "A train leaves at 9:15 going 80 km/h. How far has it gone after 2 hours and 30 minutes?",
    "SELECT name, COUNT(*) FROM orders WHERE created_at > '2025-01-01' GROUP BY name ORDER BY 2 DESC;",
    "<|im_start|>system\nYou are a careful assistant.<|im_end|>\n<|im_start|>user\nSummarise the plot of Hamlet.<|im_end|>\n",
    "The committee met on Tuesday to review the budget, and after a long discussion it agreed to postpone the vote.",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233",
    "$ git log --oneline | head -5 && ls -la /var/log",
]
N_DEV = len(PROMPTS)
PROMPTS = PROMPTS + HELD_OUT
def is_matrix(s):
    return s.endswith(".weight") and not any(w in s for w in ("norm", "conv1d"))


def _reader(spec):
    repo, _, rev = spec.partition("@")
    rev = rev or "main"
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(repo, rev), wd.hf_fetch(repo, rev)), wd.text_model_name)


def fetch(base, model, cache, layer):
    from safetensors.torch import save_file
    import urllib.request, pathlib
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
    for tag, spec in (("base", base), ("mimo", model)):
        if pathlib.Path(f"{cache}/{tag}_layer{layer}.safetensors").exists():
            continue
        r = _reader(spec)
        keep = {k: r.get(k).contiguous() for k in r.keys() if k.startswith(f"model.layers.{layer}.")}
        keep["embed_rows_all"] = torch.zeros(1)  # placeholder so the file is never empty
        del keep["embed_rows_all"]
        save_file(keep, f"{cache}/{tag}_layer{layer}.safetensors")
        print(tag, len(keep), "tensors", sum(v.numel() * v.element_size() for v in keep.values()) // 2**20, "MiB", flush=True)
    if layer > 0:
        return  # tokenizer, config and the layer-0 input already exist
    repo, _, rev = base.partition("@")
    for f in ("tokenizer.json", "config.json"):
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/{rev or 'main'}/{f}", f"{cache}/{f}")
    # real embeddings of the shared prompts, from the base table (rows only)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{cache}/tokenizer.json")
    ids = [tok.encode(p).ids for p in PROMPTS]
    uniq = sorted({i for s in ids for i in s})
    r = _reader(base)
    emb = {i: r.get_rows("model.embed_tokens.weight", i, 1).reshape(-1) for i in uniq}
    save_file({"ids": torch.tensor([i for s in ids for i in s]), "lens": torch.tensor([len(s) for s in ids]),
               "table": torch.stack([emb[i] for i in uniq]), "uniq": torch.tensor(uniq)}, f"{cache}/inputs.safetensors")
    print("prompts", [len(s) for s in ids], "unique tokens", len(uniq))



def _chunks(n, step=2048):
    for i in range(0, n, step):
        yield i, min(n, i + step)


class Probe:
    BIG = 5_000_000  # tensors above this many entries are processed in row chunks (2 GB RAM limit)

    def __init__(self, cache, layer, seqs=None):
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        cfg = json.load(open(f"{cache}/config.json"))["text_config"]
        self.cfg = Qwen3_5TextConfig(**cfg)
        self.cfg._attn_implementation = "eager"
        self.fb = safe_open(f"{cache}/base_layer{layer}.safetensors", "pt")
        self.fm = safe_open(f"{cache}/mimo_layer{layer}.safetensors", "pt")
        with torch.device("meta"):
            self.layer = m.Qwen3_5DecoderLayer(self.cfg, layer)
        self.layer = self.layer.to_empty(device="cpu").float().eval()  # uninitialised; every tensor is overwritten
        self.params = dict(self.layer.named_parameters())
        self.params.update(dict(self.layer.named_buffers()))
        self.names = list(self.fb.keys())
        self.pre = f"model.layers.{layer}."
        self.full = self.cfg.layer_types[layer] == "full_attention"
        inp = load_file(f"{cache}/inputs.safetensors")
        lens = inp["lens"].tolist()
        self.seqs = []
        if seqs is not None:  # inputs given by the caller (a stack run)
            self.seqs = seqs
        elif layer == 0:  # real embeddings of the prompts, from the base table
            tbl = {int(u): inp["table"][i].float() for i, u in enumerate(inp["uniq"].tolist())}
            ids, o = inp["ids"].tolist(), 0
            for n in lens:
                self.seqs.append(torch.stack([tbl[i] for i in ids[o:o + n]])[None])
                o += n
        else:  # hidden states that the BASE model's earlier layers really produce (saved by the previous run)
            h = load_file(f"{cache}/hidden_in_{layer}.safetensors")
            self.seqs = [h[f"p{i}"][None].float() for i in range(len(lens))]
        self.rot = m.Qwen3_5TextRotaryEmbedding(self.cfg)
        self.cache, self.layer_id, self._norms = cache, layer, {}
        self.basis = {}
        missing = [n[len(self.pre):] for n in self.names if n[len(self.pre):] not in self.params]
        if missing:
            raise KeyError(f"layer tensors without a module parameter: {missing}")

    def _norm(self, n):
        if n not in self._norms:
            self._norms[n] = self._dnorm(n)
        return self._norms[n]

    def _pair(self, n):
        return wd.common_dtype(self.fb.get_tensor(n), self.fm.get_tensor(n))

    def _dnorm(self, n):
        a, b = self._pair(n)
        if a.numel() <= self.BIG:
            return float((b.float() - a.float()).norm())
        return sum(float((b[i:j].float() - a[i:j].float()).pow(2).sum()) for i, j in _chunks(a.shape[0])) ** 0.5

    def forward(self):
        outs = []
        with torch.no_grad():
            for x in self.seqs:
                pos = torch.arange(x.shape[1])[None]
                mask = None
                if self.full:  # eager attention needs an explicit causal mask
                    t = x.shape[1]
                    mask = torch.full((t, t), float("-inf")).triu(1)[None, None]
                outs.append(self.layer(x, position_embeddings=self.rot(x, pos), attention_mask=mask,
                                       position_ids=pos).reshape(-1))
        return torch.cat(outs)

    def delta_basis(self, n, kmax=1024):
        """Singular vectors of delta on its smaller side, from a chunked Gram matrix."""
        if n in self.basis:
            return self.basis[n]
        import os
        bf = f"{self.cache}/basis_{n}.pt"
        if os.path.exists(bf):
            self.basis[n] = torch.load(bf)
            return self.basis[n]
        a, b = self._pair(n)
        rows, cols = a.shape
        side = "right" if cols <= rows else "left"
        d = min(rows, cols)
        g = torch.zeros(d, d)
        if side == "right":
            for i, j in _chunks(rows, 1024):
                x = b[i:j].float() - a[i:j].float()
                g += x.T @ x
        else:
            for i, j in _chunks(cols, 1024):
                x = b[:, i:j].float() - a[:, i:j].float()
                g += x @ x.T
        w, v = torch.linalg.eigh(g)
        v = v[:, torch.argsort(w, descending=True)[:kmax]].contiguous()
        self.basis[n] = (side, v)
        torch.save(self.basis[n], bf)
        return self.basis[n]

    def write(self, n, mode, arg, seed):
        """Write base (+ part of delta) of tensor n into the module parameter, chunk by chunk."""
        param = self.params[n[len(self.pre):]]
        base = self.fb.get_tensor(n)
        a, b = self._pair(n)
        small = base.numel() <= self.BIG
        if mode == "none":
            param.data.copy_(base.float())
        elif mode == "full":
            if small:
                param.data.copy_(base.float() + (b.float() - a.float()))
            else:
                for i, j in _chunks(base.shape[0]):
                    param.data[i:j] = base[i:j].float() + (b[i:j].float() - a[i:j].float())
        elif mode == "random":
            g = torch.Generator().manual_seed(seed)
            scale = self._norm(n) / (base.numel() ** 0.5)  # E||noise||_F = scale * sqrt(numel) = ||delta||_F
            if small:
                param.data.copy_(base.float() + torch.randn(base.shape, generator=g) * scale)
            else:
                for i, j in _chunks(base.shape[0]):
                    param.data[i:j] = base[i:j].float() + torch.randn(j - i, base.shape[1], generator=g) * scale
        elif mode in ("rank", "rsub"):
            side, v = self.delta_basis(n)
            arg = min(arg, v.shape[1])
            if mode == "rank":
                v = v[:, :arg]
            else:  # control: a random orthonormal subspace of the same size on the same side
                gq = torch.Generator().manual_seed(seed)
                v = torch.linalg.qr(torch.randn(v.shape[0], arg, generator=gq)).Q.contiguous()
            if side == "right":
                for i, j in _chunks(base.shape[0]):
                    x = b[i:j].float() - a[i:j].float()
                    param.data[i:j] = base[i:j].float() + x @ v @ v.T
            else:
                proj = torch.zeros(arg, base.shape[1])
                for i, j in _chunks(base.shape[0]):
                    proj += v[i:j].T @ (b[i:j].float() - a[i:j].float())
                for i, j in _chunks(base.shape[0]):
                    param.data[i:j] = base[i:j].float() + v[i:j] @ proj
        elif mode == "entries":
            g = torch.Generator().manual_seed(0)
            samp = torch.cat([(b[i:j].float() - a[i:j].float()).reshape(-1)[
                torch.randint(0, (j - i) * base.shape[1], (20000,), generator=g)].abs() for i, j in _chunks(base.shape[0])])
            thr = torch.quantile(samp, 1 - arg)
            for i, j in _chunks(base.shape[0]):
                x = b[i:j].float() - a[i:j].float()
                param.data[i:j] = base[i:j].float() + x * (x.abs() >= thr)

    def run_variant(self, select, mode, arg=None, seed=0):
        for n in self.names:
            short = n[len(self.pre):]
            self.write(n, mode if select(short) else "none", arg, seed)
        return self.forward()


SETS = {
    "all_params": lambda s: True,
    "mlp_all": lambda s: s.startswith("mlp."),
    "mlp_gate_up": lambda s: s.startswith("mlp.gate") or s.startswith("mlp.up"),
    "mlp_down": lambda s: s.startswith("mlp.down"),
    "attn_all": lambda s: s.startswith("linear_attn.") or s.startswith("self_attn."),
    "norms": lambda s: s.endswith("layernorm.weight"),
}


def metrics(o, o_base, o_mimo):
    """All inner products and norms are over the flattened concatenation of every token and hidden dimension of the
    prompts in the group (one pooled number per group, not an average of per-prompt ratios).
      shift = o_mimo - o_base, d = o - o_base
      captured = <d, shift> / <shift, shift>   projection coefficient of the variant's change on MiMo's change.
                 1 = same size along it, 0 = none, <0 = opposite; it can exceed 1; it is NOT 1 - relative error.
      resid = ||o - o_mimo|| / ||shift||       relative error against MiMo's layer output (0 exact, 1 = the base layer)
      cos = <d, shift> / (||d|| ||shift||)     direction agreement, independent of size
      norm_ratio = ||d|| / ||shift||
    resid^2 = 1 - 2 captured + norm_ratio^2."""
    shift = o_mimo - o_base
    d = o - o_base
    ip = float((d * shift).sum())
    ns, nd = float(shift.norm()), float(d.norm())
    return {"captured": ip / ns ** 2, "resid": float((o - o_mimo).norm()) / ns,
            "cos": ip / (ns * nd) if nd > 0 else 0.0, "norm_ratio": nd / ns}


def run(cache, out, layer):
    t0 = time.time()
    pr = Probe(cache, layer)
    o_base = pr.run_variant(lambda s: False, "none")
    o_mimo = pr.run_variant(lambda s: True, "full")
    res = {"input": "real base embeddings" if layer == 0 else "hidden states of the BASE model at this depth (not MiMo's own trajectory)", "n_dev_prompts": N_DEV, "layer": layer, "layer_type": pr.cfg.layer_types[layer], "tokens": int(sum(x.shape[1] for x in pr.seqs)),
           "shift_rel_to_output": float((o_mimo - o_base).norm() / o_base.norm()), "variants": {}}
    print("shift/output", res["shift_rel_to_output"], flush=True)
    from safetensors.torch import save_file
    hid0 = pr.cfg.hidden_size
    nxt, i = {}, 0
    for k, x in enumerate(pr.seqs):  # input of the next layer = what the base model really produces here
        c = int(x.shape[1]) * hid0
        nxt[f"p{k}"] = o_base[i:i + c].reshape(x.shape[1], hid0).contiguous()
        i += c
    save_file(nxt, f"{cache}/hidden_in_{layer + 1}.safetensors")

    hid = pr.cfg.hidden_size
    cut = [int(x.shape[1]) * hid for x in pr.seqs]
    edge = [0]
    for c in cut:
        edge.append(edge[-1] + c)
    groups = {"dev": (0, edge[N_DEV]), "heldout": (edge[N_DEV], edge[-1])}

    def add(name, o):
        v = {}
        for g, (i, j) in groups.items():
            v[g] = metrics(o[i:j], o_base[i:j], o_mimo[i:j])
        v["captured_per_prompt"] = [round(metrics(o[edge[k]:edge[k + 1]], o_base[edge[k]:edge[k + 1]],
                                                  o_mimo[edge[k]:edge[k + 1]])["captured"], 3) for k in range(len(cut))]
        res["variants"][name] = v
        r = lambda d: {k: round(x, 3) for k, x in d.items()}
        print(f"{time.time() - t0:6.0f}s {name:34s} dev", r(v["dev"]), "heldout", r(v["heldout"]), flush=True)
        json.dump(res, open(out, "w"), indent=1)

    add("sanity_full_delta", o_mimo)
    add("sanity_base", o_base)
    for name, sel in SETS.items():
        if name == "all_params":
            continue
        add(f"{name}:true", pr.run_variant(sel, "full"))
        add(f"{name}:random", pr.run_variant(sel, "random", seed=1))
    mats = is_matrix
    for k in (16, 256, 1024):
        add(f"matrices_rank{k}:true", pr.run_variant(mats, "rank", k))
    add("matrices_rank256:random", pr.run_variant(mats, "random", seed=2))
    for k in (16, 256):  # energy-fair control: random subspace of the same size, same matrices
        add(f"matrices_rank{k}:random_subspace_s1", pr.run_variant(mats, "rsub", k, seed=11))
        add(f"matrices_rank{k}:random_subspace_s2", pr.run_variant(mats, "rsub", k, seed=12))
    for f in (0.1, 0.3):
        add(f"matrices_top{int(f * 100)}pct_entries:true", pr.run_variant(mats, "entries", f))
    add("matrices_all:random", pr.run_variant(mats, "random", seed=3))
    print("done", time.time() - t0, flush=True)


def stack(cache, out, layers=(0, 1, 2, 3)):
    """Composition test: the same rank-k budget in every layer of a short stack, run layer by layer so only one layer is in
    memory. Each layer receives the output of the PATCHED layer before it (the patched model's own trajectory); the
    embeddings are the base table's. Compared with the base stack, the full-MiMo-delta stack and same-size random controls."""
    from safetensors.torch import load_file
    t0 = time.time()
    seqs0 = Probe(cache, layers[0]).seqs
    hid = seqs0[0].shape[-1]
    lens = [int(x.shape[1]) for x in seqs0]

    def run_stack(select, mode, arg=None, seed=0):
        h = seqs0
        for L in layers:
            pr = Probe(cache, L, seqs=h)
            o = pr.run_variant(select, mode, arg, seed)
            h, i = [], 0
            for n in lens:
                h.append(o[i:i + n * hid].reshape(1, n, hid))
                i += n * hid
            del pr
        return o

    o_base = run_stack(lambda s: False, "none")
    o_mimo = run_stack(lambda s: True, "full")
    edge = [0]
    for n in lens:
        edge.append(edge[-1] + n * hid)
    groups = {"dev": (0, edge[N_DEV]), "heldout": (edge[N_DEV], edge[-1])}
    res = {"layers": list(layers), "input": "base embeddings; each layer fed by the patched stack's own previous layer",
           "shift_rel_to_output": float((o_mimo - o_base).norm() / o_base.norm()), "variants": {}}

    def add(name, o):
        v = {g: metrics(o[i:j], o_base[i:j], o_mimo[i:j]) for g, (i, j) in groups.items()}
        res["variants"][name] = v
        r = lambda d: {k: round(x, 3) for k, x in d.items()}
        print(f"{time.time() - t0:6.0f}s {name:34s} dev", r(v["dev"]), "heldout", r(v["heldout"]), flush=True)
        json.dump(res, open(out, "w"), indent=1)

    add("sanity_full_delta", o_mimo)
    add("sanity_base", o_base)
    for k in (16, 256):
        add(f"rank{k}:true", run_stack(is_matrix, "rank", k))
        for sd in (11, 12):
            add(f"rank{k}:random_subspace_s{sd - 10}", run_stack(is_matrix, "rsub", k, seed=sd))
    add("top10pct_entries:true", run_stack(is_matrix, "entries", 0.1))
    add("matrices_full:true", run_stack(is_matrix, "full"))
    add("mlp_full:true", run_stack(lambda s: s.startswith("mlp."), "full"))
    print("done", time.time() - t0, flush=True)


def check_causal(cache, layer):
    """Implementation check: changing the LAST token's input must not change any earlier output (causality), and must change the last one."""
    pr = Probe(cache, layer)
    pr.run_variant(lambda s: False, "none")
    x = pr.seqs[1].clone()
    y = x.clone()
    y[0, -1] += torch.randn(y.shape[-1], generator=torch.Generator().manual_seed(5)) * 3
    pr.seqs = [x]
    a = pr.forward().reshape(x.shape[1], -1)
    pr.seqs = [y]
    b = pr.forward().reshape(x.shape[1], -1)
    print({"layer": layer, "type": pr.cfg.layer_types[layer], "max_diff_earlier_tokens": float((a[:-1] - b[:-1]).abs().max()),
           "diff_last_token": float((a[-1] - b[-1]).abs().max())})


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "causal":
        check_causal(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "stack":
        stack(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
