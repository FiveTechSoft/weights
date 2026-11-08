"""layer_probe.py - a local, functional probe of one layer: which parts of MiMo's delta reproduce its output change.

Scope and limits (read before quoting any number):
  * ONE layer (layer 0, a Gated DeltaNet layer) of 32, run in float32 on CPU with the transformers implementation.
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

  python lab/probes/delta_patterns/layer_probe.py fetch   BASE@rev MODEL@rev CACHE_DIR
  python lab/probes/delta_patterns/layer_probe.py run     CACHE_DIR out.json
"""
import json
import sys
import time

import torch

from engine import weight_diff as wd

LAYER = 0
PROMPTS = [
    "The capital of France is Paris, and the capital of Spain is Madrid.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "<|im_start|>user\nWrite a function that reverses a string.<|im_end|>\n<|im_start|>assistant\n",
    "If 3x + 5 = 20, then x = 5. Let us check: 3 * 5 + 5 = 20.",
]
MATS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.out_proj", "linear_attn.in_proj_a", "linear_attn.in_proj_b")


def _reader(spec):
    repo, _, rev = spec.partition("@")
    rev = rev or "main"
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(repo, rev), wd.hf_fetch(repo, rev)), wd.text_model_name)


def fetch(base, model, cache):
    from safetensors.torch import save_file
    import urllib.request, pathlib
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
    for tag, spec in (("base", base), ("mimo", model)):
        r = _reader(spec)
        keep = {k: r.get(k).contiguous() for k in r.keys() if k.startswith(f"model.layers.{LAYER}.")}
        keep["embed_rows_all"] = torch.zeros(1)  # placeholder so the file is never empty
        del keep["embed_rows_all"]
        save_file(keep, f"{cache}/{tag}_layer{LAYER}.safetensors")
        print(tag, len(keep), "tensors", sum(v.numel() * v.element_size() for v in keep.values()) // 2**20, "MiB", flush=True)
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

    def __init__(self, cache):
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        cfg = json.load(open(f"{cache}/config.json"))["text_config"]
        self.cfg = Qwen3_5TextConfig(**cfg)
        self.cfg._attn_implementation = "eager"
        self.fb = safe_open(f"{cache}/base_layer{LAYER}.safetensors", "pt")
        self.fm = safe_open(f"{cache}/mimo_layer{LAYER}.safetensors", "pt")
        with torch.device("meta"):
            self.layer = m.Qwen3_5DecoderLayer(self.cfg, LAYER)
        self.layer = self.layer.to_empty(device="cpu").float().eval()  # uninitialised; every tensor is overwritten
        self.params = dict(self.layer.named_parameters())
        self.params.update(dict(self.layer.named_buffers()))
        self.names = list(self.fb.keys())
        self.pre = f"model.layers.{LAYER}."
        inp = load_file(f"{cache}/inputs.safetensors")
        tbl = {int(u): inp["table"][i].float() for i, u in enumerate(inp["uniq"].tolist())}
        ids = inp["ids"].tolist()
        self.seqs, o = [], 0
        for n in inp["lens"].tolist():
            self.seqs.append(torch.stack([tbl[i] for i in ids[o:o + n]])[None])
            o += n
        self.rot = m.Qwen3_5TextRotaryEmbedding(self.cfg)
        self.norm = {n: self._dnorm(n) for n in self.names}
        self.basis = {}
        missing = [n[len(self.pre):] for n in self.names if n[len(self.pre):] not in self.params]
        if missing:
            raise KeyError(f"layer tensors without a module parameter: {missing}")

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
                outs.append(self.layer(x, position_embeddings=self.rot(x, pos), position_ids=pos).reshape(-1))
        return torch.cat(outs)

    def delta_basis(self, n, kmax=1024):
        """Singular vectors of delta on its smaller side, from a chunked Gram matrix."""
        if n in self.basis:
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
            scale = self.norm[n] / (base.numel() ** 0.5)  # E||noise||_F = scale * sqrt(numel) = ||delta||_F
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
    "deltanet_all": lambda s: s.startswith("linear_attn."),
    "deltanet_qkv": lambda s: s.startswith("linear_attn.in_proj_qkv"),
    "deltanet_out": lambda s: s.startswith("linear_attn.out_proj"),
    "norms": lambda s: s.endswith("layernorm.weight"),
}


def metrics(o, o_base, o_mimo):
    shift = o_mimo - o_base
    d = o - o_base
    return {"captured": float((d * shift).sum() / (shift * shift).sum()),
            "resid": float((o - o_mimo).norm() / shift.norm()),
            "norm_ratio": float(d.norm() / shift.norm())}


def run(cache, out):
    t0 = time.time()
    pr = Probe(cache)
    o_base = pr.run_variant(lambda s: False, "none")
    o_mimo = pr.run_variant(lambda s: True, "full")
    res = {"layer": LAYER, "tokens": int(sum(x.shape[1] for x in pr.seqs)),
           "shift_rel_to_output": float((o_mimo - o_base).norm() / o_base.norm()), "variants": {}}
    print("shift/output", res["shift_rel_to_output"], flush=True)

    hid = pr.cfg.hidden_size
    cut = [int(x.shape[1]) * hid for x in pr.seqs]

    def per_prompt(o):
        out, i = [], 0
        for c in cut:
            out.append(round(metrics(o[i:i + c], o_base[i:i + c], o_mimo[i:i + c])["captured"], 3))
            i += c
        return out

    def add(name, o):
        res["variants"][name] = metrics(o, o_base, o_mimo)
        res["variants"][name]["captured_per_prompt"] = per_prompt(o)
        print(f"{time.time() - t0:6.0f}s {name:34s}", {k: (round(v, 3) if not isinstance(v, list) else v) for k, v in res["variants"][name].items()}, flush=True)
        json.dump(res, open(out, "w"), indent=1)

    add("sanity_full_delta", o_mimo)
    add("sanity_base", o_base)
    for name, sel in SETS.items():
        if name == "all_params":
            continue
        add(f"{name}:true", pr.run_variant(sel, "full"))
        add(f"{name}:random", pr.run_variant(sel, "random", seed=1))
    mats = lambda s: any(s.startswith(p) for p in MATS)
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


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
