"""Control vector from a base/improved pair: per layer, mean over token positions of h_improved - h_base on the SAME
texts (teacher forcing). Texts: for the 16 MATH items where the base has the answer but never closes (results §43.2),
chat prompt + the base's own last reasoning (its loop) and, separately, prompt + the improved model's last text.
Written as a llama.cpp control vector GGUF: direction.N (N >= 1) is added to the output of 0-based layer N, i.e.
transformers hidden_states[N + 1].

No accelerate: the model sits on the CPU in bf16 and each decoder layer is moved to the GPU only for its own forward
(pre/post hooks), 8 texts per forward (right padding). Run with llama-server stopped (RAM, F-note in REPRO).

usage: python lab/probes/close_probe/cvec_extract.py <base_hf> <improved_hf> <improved_runs_json> <out.gguf>
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_TAIL = 384   # tokens of continuation per text


def to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(to(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: to(v, dev) for k, v in x.items()}
    return x


def stream_layers(layers):
    for L in layers:
        def pre(mod, args, kwargs):
            mod.to("cuda")
            return to(args, "cuda"), to(kwargs, "cuda")

        def post(mod, args, kwargs, out):
            mod.to("cpu")
            return to(out, "cpu")
        L.register_forward_pre_hook(pre, with_kwargs=True)
        L.register_forward_hook(post, with_kwargs=True)


def texts(tok):
    m5dir = ROOT / "rig"
    sys.path.insert(0, str(m5dir))
    import math500 as m5
    items = m5.load_items()
    rev = json.loads((ROOT / "lab/runs/model_compare/base_cut_review.json").read_text())["classes"]
    base = {r["i"]: r for r in json.loads((ROOT / "lab/runs/model_compare/own_base.json").read_text(encoding="utf-8"))["items"]
            if r["bench"] == "math500"}
    imp = {r["i"]: r for r in json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["items"] if r["bench"] == "math500"}
    out = []
    for i in rev["answer_found_cannot_close"]:
        p = tok.apply_chat_template([{"role": "user", "content": items[i]["problem"] + SUFFIX}], tokenize=False,
                                    add_generation_prompt=True)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        for src in (base, imp):
            ti = tok(src[i]["tail"], add_special_tokens=False)["input_ids"][-MAX_TAIL:]
            out.append((pi, ti))
    return out


@torch.no_grad()
def layer_means(path, data):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model.eval()
    core = model.model
    layers = core.layers if hasattr(core, "layers") else core.language_model.layers
    stream_layers(layers)
    sums, count = None, 0
    B = int(__import__("os").environ.get("CVEC_B", 8))   # right padding: causal model, so real positions never see the pads after them
    for k in range(0, len(data), B):
        chunk = data[k:k + B]
        L = max(len(pi) + len(ti) for pi, ti in chunk)
        ids = torch.tensor([pi + ti + [0] * (L - len(pi) - len(ti)) for pi, ti in chunk])
        hs = core(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        for b, (pi, ti) in enumerate(chunk):
            sl = slice(len(pi), len(pi) + len(ti))
            v = torch.stack([h[b, sl].float().cpu().sum(0) for h in hs])     # (n_layers + 1, d)
            sums = v if sums is None else sums + v
            count += len(ti)
    del model
    return sums / count


def main():
    from transformers import AutoTokenizer
    import gguf
    import numpy as np
    base_p, imp_p, out = sys.argv[1], sys.argv[2], Path(sys.argv[4])
    tok = AutoTokenizer.from_pretrained(base_p)
    data = texts(tok)
    mb = layer_means(base_p, data)
    mi = layer_means(imp_p, data)
    D = mi - mb                                    # (n_layers + 1, d); row k = output of layer k - 1
    n_layers = D.shape[0] - 1
    w = gguf.GGUFWriter(str(out), "controlvector")
    w.add_string("controlvector.model_hint", "qwen35")
    w.add_uint32("controlvector.layer_count", n_layers - 1)
    stats = {}
    for N in range(1, n_layers):
        v = D[N + 1].numpy().astype(np.float32)
        w.add_tensor(f"direction.{N}", v)
        stats[N] = {"norm": float(np.linalg.norm(v)), "h_norm": float(mb[N + 1].norm())}
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out.with_suffix(".json").write_text(json.dumps({"texts": len(data), "per_layer": stats}, indent=1))
    print(json.dumps({"out": str(out), "texts": len(data), "layers": n_layers - 1,
                      "max_ratio": max(s["norm"] / s["h_norm"] for s in stats.values())}), flush=True)


if __name__ == "__main__":
    main()
