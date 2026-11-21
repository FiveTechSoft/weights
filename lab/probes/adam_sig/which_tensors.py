"""Exploratory (not in the plan's readings): which tensors of the donor differ from the base at all, by module kind.
Reads both safetensors sets shard by shard, compares raw bf16 bytes per tensor. -> out.json {kind: {same, diff, n}}.

  python lab/probes/adam_sig/which_tensors.py <base_hf_dir> <donor_hf_dir> <out.json>
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def kind(name):
    m = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|embed_tokens|lm_head|norm|[a-z_]+)\.weight$", name)
    if "norm" in name:
        return "norm"
    return m.group(1) if m else name


def main(base, donor, out):
    import torch
    from safetensors import safe_open
    ib = json.loads((Path(base) / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    idn = json.loads((Path(donor) / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    stats = defaultdict(lambda: {"same": 0, "diff": 0, "n": 0})
    diff_layers = defaultdict(set)
    only = {"base_only": sorted(set(ib) - set(idn)), "donor_only": sorted(set(idn) - set(ib))}
    handles = {}

    def h(path):
        if path not in handles:
            handles[path] = safe_open(path, framework="pt")
        return handles[path]
    for name in sorted(set(ib) & set(idn)):
        a = h(str(Path(base) / ib[name])).get_tensor(name)
        b = h(str(Path(donor) / idn[name])).get_tensor(name)
        same = a.shape == b.shape and torch.equal(a, b)
        k = kind(name)
        stats[k]["n"] += 1
        stats[k]["same" if same else "diff"] += 1
        if not same:
            m = re.search(r"layers\.(\d+)\.", name)
            diff_layers[k].add(int(m.group(1)) if m else -1)
    rep = {"kinds": dict(stats), "diff_layers": {k: sorted(v) for k, v in diff_layers.items()}, "only": only}
    Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep["kinds"].items()}))
    print({k: (min(v), max(v), len(v)) for k, v in rep["diff_layers"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
