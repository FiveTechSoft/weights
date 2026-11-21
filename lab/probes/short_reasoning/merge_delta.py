"""Task arithmetic between siblings: target' = target + alpha * (donor - base), tensor by tensor, never a whole model in RAM.

Use here: target = MiMo-V2.6-Distill-Qwen-9B, base = Qwen/Qwen3.5-9B (MiMo's SFT parent), donor = a Qwen3.5-9B fine-tune whose
only declared goal is to terminate reasoning (iAmBoosted/Qwen3.5-9B-OSS-Distilled: LoRA r16 merged, "spiral-out" 36.2 % -> 0.5 %).
Both target and donor descend from the same parent, so (donor - base) is defined on the same coordinates as MiMo.

--placebo SEED replaces each donor delta by a random one with the SAME Frobenius norm per tensor and, for matrices, the SAME rank
as measured on the donor delta (default cap 64): the control for "any perturbation of this size and shape does it".

usage:
  python merge_delta.py --target DIR --base DIR --donor DIR --out DIR [--alpha 1.0] [--placebo SEED] [--max-rank 64]
DIRs are local HF snapshots (*.safetensors + config/tokenizer). Output keeps the target's shard layout and copies its non-weight
files (config, tokenizer, chat template), so convert_hf_to_gguf.py sees a MiMo checkpoint. report.json lists every tensor:
changed / unchanged (zero delta) / skipped (missing or shape mismatch), delta norm, rank, and sha256 of the inputs' index files.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIXES = [r"^model\.language_model\.", r"^language_model\.model\.", r"^model\.text_model\.", r"^model\."]


def canon(name: str) -> str:
    for p in PREFIXES:
        if re.match(p, name):
            return re.sub(p, "", name)
    return name


class Snapshot:
    def __init__(self, d):
        self.dir = Path(d)
        self.where = {}
        for f in sorted(self.dir.glob("*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.where[k] = f
        if not self.where:
            raise SystemExit(f"no safetensors in {d}")
        self.by_canon = {canon(k): k for k in self.where}

    def get(self, key):
        with safe_open(self.where[key], "pt") as h:
            return h.get_tensor(key)

    def get_canon(self, c):
        k = self.by_canon.get(c)
        return None if k is None else self.get(k)


def numeric_rank(d: torch.Tensor) -> int:
    """Directions holding 90 % of the delta energy. A merged LoRA stored in bf16 is low rank plus full-rank rounding noise,
    so a relative singular-value threshold counts the noise; an energy fraction does not."""
    s2 = torch.linalg.svdvals(d.float()) ** 2
    if not s2.numel() or s2.sum() == 0:
        return 0
    return int((torch.cumsum(s2, 0) / s2.sum() < 0.90).sum().item()) + 1


def placebo_like(d: torch.Tensor, rank: int, gen: torch.Generator) -> torch.Tensor:
    if d.ndim == 2 and rank > 0:
        a = torch.randn(d.shape[0], rank, generator=gen)
        b = torch.randn(rank, d.shape[1], generator=gen)
        r = a @ b
    else:
        r = torch.randn(d.shape, generator=gen)
    n = r.norm()
    return r * (d.float().norm() / n) if n > 0 else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--placebo", type=int, default=None)
    ap.add_argument("--max-rank", type=int, default=64)
    a = ap.parse_args()

    T, B, D = Snapshot(a.target), Snapshot(a.base), Snapshot(a.donor)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.placebo) if a.placebo is not None else None
    report, totals = [], {"changed": 0, "unchanged": 0, "skipped": 0}

    shards = {}
    for k, f in T.where.items():
        shards.setdefault(f, []).append(k)
    for f, keys in shards.items():
        tensors = {}
        for k in keys:
            t = T.get(k)
            tensors[k] = t
            c = canon(k)
            b, d = B.get_canon(c), D.get_canon(c)
            e = {"name": k}
            if b is None or d is None:
                e.update(status="skipped", reason="missing in base" if b is None else "missing in donor")
            elif not (b.shape == d.shape == t.shape):
                e.update(status="skipped", reason=f"shapes {tuple(t.shape)} {tuple(b.shape)} {tuple(d.shape)}")
            else:
                if b.dtype != d.dtype:  # F26: an F32 vs BF16 storage cast is not a change; compare at the coarser dtype
                    low = torch.bfloat16 if torch.bfloat16 in (b.dtype, d.dtype) else (b.dtype if b.element_size() < d.element_size() else d.dtype)
                    b, d = b.to(low), d.to(low)
                    e["dtype_note"] = f"compared at {low}"
            if "status" in e:
                pass
            elif torch.equal(b, d):
                e.update(status="unchanged")
            else:
                delta = d.float() - b.float()
                rank = numeric_rank(delta) if (gen is not None and delta.ndim == 2) else None  # SVD only for the placebo
                e.update(delta_norm=round(delta.norm().item(), 6), rel_to_target=round((delta.norm() / t.float().norm()).item(), 6),
                         delta_rank=rank)
                if gen is not None:
                    delta = placebo_like(delta, min(rank or 0, a.max_rank), gen)
                tensors[k] = (t.float() + a.alpha * delta).to(t.dtype)
                e.update(status="changed")
                del delta
            totals[e["status"]] += 1
            report.append(e)
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})
        print(f"[merge] {f.name}: {len(keys)} tensors written; totals so far {totals}", flush=True)
        del tensors

    for p in T.dir.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p, out / p.name)
    ids = {n: hashlib.sha256("".join(sorted(s.where)).encode()).hexdigest()[:16] for n, s in (("target", T), ("base", B), ("donor", D))}
    meta = {"alpha": a.alpha, "placebo_seed": a.placebo, "max_rank": a.max_rank, "totals": totals, "name_sets_sha": ids,
            "target": a.target, "base": a.base, "donor": a.donor}
    (out / "merge_report.json").write_text(json.dumps({"meta": meta, "tensors": report}, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    if totals["changed"] == 0:
        raise SystemExit("nothing changed: check name mapping (see merge_report.json)")


if __name__ == "__main__":
    main()
