"""Cut a PEFT LoRA made by delta_to_lora.py (singular directions in descending order) down to its first k ranks and/or
a subset of layers / matrix kinds. Reads only the adapter (small), so it is safe to run next to llama-server.

usage: python lab/probes/close_probe/slice_lora.py <in_dir> <out_dir> [--rank k] [--layers 0-15] [--kinds mlp,linear_attn]
"""
import argparse
import json
import re
from pathlib import Path


def keep(name, layers, kinds):
    m = re.search(r"layers\.(\d+)\.(\w+)\.", name)
    if not m:
        return False
    return (layers is None or int(m.group(1)) in layers) and (kinds is None or m.group(2) in kinds)


def parse_layers(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main():
    from safetensors.torch import load_file, save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--kinds", default=None)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    layers, kinds = parse_layers(a.layers), (set(a.kinds.split(",")) if a.kinds else None)
    T = load_file(str(src / "adapter_model.safetensors"))
    cfg = json.loads((src / "adapter_config.json").read_text())
    r = a.rank or cfg["r"]
    S = {}
    for n, t in T.items():
        if not keep(n, layers, kinds):
            continue
        S[n] = (t[:r] if n.endswith("lora_A.weight") else t[:, :r]).contiguous()
    if not S:
        raise SystemExit("nothing kept")
    out.mkdir(parents=True, exist_ok=False)
    save_file(S, str(out / "adapter_model.safetensors"))
    cfg.update(r=r, lora_alpha=r)
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({"out": str(out), "rank": r, "tensors": len(S)}))


if __name__ == "__main__":
    main()
