"""Build small test models for the inspection-op audit (outputs under lab/audit/inspect_ops/models)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file, save_file
from engine.wsp_arch import load_arch_file

OUT = Path(__file__).parent / "models"
OUT.mkdir(exist_ok=True)
ov = dict(block=64, layers=2, heads=4, d=64, ff=256, dropout=0.0, bias=False, vocab=65)
for seed in (1337, 7):
    m = load_arch_file(ROOT / "programs/arch/nanogpt.wsparch", overrides=ov, seed=seed)
    m.save(OUT / f"rand_{seed}.safetensors")
t = load_file(str(OUT / "rand_1337.safetensors"))
for k, v in t.items():
    print(k, tuple(v.shape), v.dtype)
# Known edit: modify exactly 37 entries of one tensor, + rank-1 delta in another
name = [k for k in t if k.endswith("mlp.c_fc.weight")][0]
t2 = {k: v.clone() for k, v in t.items()}
g = torch.Generator().manual_seed(0)
flat = t2[name].view(-1)
idx = torch.randperm(flat.numel(), generator=g)[:37]
flat[idx] += 0.5
print("edited", name, "37 entries")
save_file(t2, str(OUT / "rand_1337_edit37.safetensors"))
