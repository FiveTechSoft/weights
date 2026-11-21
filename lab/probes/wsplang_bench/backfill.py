"""Backfill the WSP-Lang benchmark registry from existing bench_runner reports that match the dev-2048 suite exactly
(lab/probes/wsplang_bench/PLAN.md). CPU only, no new runs. Each imported run keeps the original report's `served`
field (older reports have none: identity unknown and marked so), records `imported_from`, and carries no preservation.

  python -I lab/probes/wsplang_bench/backfill.py            -> lab/runs/bench/imported/*.json, index.json, docs table
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine import wsplang_bench as wb  # noqa: E402

OUT = ROOT / "lab" / "runs" / "bench" / "imported"
# (report relative to lab/runs, arm label, model label)
CANDIDATES = [
    ("cc_v2/dev_base", "base", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_v2/dev_vec", "conclude vector 1.0", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_v2/dev_placebo", "same-norm placebo 1.0", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_v2/dev_no31", "conclude vector 1.0 without layer 31", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_v2/weights/dev_merged", "compiled conclude (merged Q6_K)", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_v2/weights/dev_placebo_merged", "compiled placebo (merged Q6_K)", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_weights/dev_lora", "compiled conclude as LoRA r1", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_weights/dev_merged", "compiled conclude (merged Q6_K, cc_weights)", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_weights/dev_placebo_merged", "compiled placebo (merged Q6_K, cc_weights)", "Qwen3.5-9B Q6_K (A0)"),
    ("cc_llama/dev_base", "base", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/dev_vec", "conclude vector 0.5", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/dev_placebo", "same-norm placebo 0.5", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/dev_vec_s0.25", "conclude vector 0.25", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/dev_vec_s0.35", "conclude vector 0.35", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/weights/dev_merged", "compiled conclude (merged Q6_K)", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_llama/weights/dev_placebo_merged", "compiled placebo (merged Q6_K)", "DeepSeek-R1-Distill-Llama-8B Q6_K"),
    ("cc_phi/dev_base", "base", "Phi-4-mini-reasoning Q6_K"),            # 1,024 tokens: expected not to match
    ("cc_phi/dev_vec", "conclude vector 0.5", "Phi-4-mini-reasoning Q6_K"),
    ("cc_phi/dev_placebo", "same-norm placebo 0.5", "Phi-4-mini-reasoning Q6_K"),
]
COMPARISONS = [  # (base, arm, placebo) by report name
    ("cc_v2/dev_base", "cc_v2/dev_vec", "cc_v2/dev_placebo"),
    ("cc_v2/dev_base", "cc_v2/weights/dev_merged", "cc_v2/weights/dev_placebo_merged"),
    ("cc_llama/dev_base", "cc_llama/dev_vec", "cc_llama/dev_placebo"),
    ("cc_llama/dev_base", "cc_llama/dev_vec_s0.35", None),
]


def matches(rep, suite):
    g = sorted(r["i"] for r in rep["items"] if r["bench"] == "gsm8k")
    m = sorted(r["i"] for r in rep["items"] if r["bench"] == "math500")
    why = []
    if g != suite["items"]["gsm8k"]:
        why.append("GSM8K items differ")
    if m != sorted(suite["items"]["math500"]):
        why.append("MATH-500 items differ")
    if int(rep.get("max_tokens", 0)) != suite["decoding"]["max_tokens"]:
        why.append(f"max_tokens {rep.get('max_tokens')} != {suite['decoding']['max_tokens']}")
    if rep.get("extra"):
        why.append("prompt changed (extra)")
    return why


# Older reports (before served_info existed) have no `served`. For the arms that served the base GGUF with or without a
# control vector, the chain scripts name the file (lab/runs/cc_v2/chain.sh M=hf_cache/h4/A0-Q6_K.gguf, cc_llama
# G=...); its sha is computed from today's file and recorded as declared by the backfill, not observed from /props.
CHAIN_BASE = {"cc_v2/dev_base": "hf_cache/h4/A0-Q6_K.gguf", "cc_v2/dev_vec": "hf_cache/h4/A0-Q6_K.gguf",
              "cc_v2/dev_placebo": "hf_cache/h4/A0-Q6_K.gguf", "cc_v2/dev_no31": "hf_cache/h4/A0-Q6_K.gguf",
              "cc_weights/dev_lora": "hf_cache/h4/A0-Q6_K.gguf"}
_sha_cache = {}


def to_run(rep, suite, name, arm):
    served = rep.get("served") or {}
    model_path, model_sha, declared = served.get("model_path"), served.get("sha256"), None
    if not model_path and name in CHAIN_BASE and (ROOT / CHAIN_BASE[name]).is_file():
        model_path = CHAIN_BASE[name]
        model_sha = _sha_cache.setdefault(model_path, wb.sha_file(ROOT / model_path))
        declared = "backfill: file named by the study's chain script, sha of today's file"
    identity = {"model_path": model_path, "model_sha": model_sha, "extras": {} if not declared else {"model_path_declared_by": declared},
                "binary": {"path": None, "sha256": None}, "suite_sha": suite["sha256"], "decoding": suite["decoding"],
                "unknown_model": not model_path,
                "note": "imported from a bench_runner report; control vectors / LoRAs of the arm are in the study's chain log, not here"}
    return {"wsplang_bench_version": wb.VERSION, "suite": suite["name"], "suite_sha": suite["sha256"],
            "model_class": suite["model_class"], "think": suite["think"], "arm": arm, "identity": identity, "provenance": None,
            "served": rep.get("served"), "binary": identity["binary"], "decoding": suite["decoding"], "extra_body": None,
            "lora": rep.get("lora"), "force_reuse": False, "sealed_use": None, "benchmarks": rep["summary"],
            "preservation": {}, "think_share": None, "items": rep["items"], "preservation_items": [],
            "seconds": rep.get("seconds"), "created": "", "imported_from": f"lab/runs/{name}.json"}


def main():
    suite = wb.resolve_suite("dev-2048")
    OUT.mkdir(parents=True, exist_ok=True)
    imported, skipped, paths = [], [], {}
    for name, arm, model in CANDIDATES:
        p = ROOT / "lab" / "runs" / f"{name}.json"
        if not p.exists():
            skipped.append((name, "missing"))
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        why = matches(rep, suite)
        if why:
            skipped.append((name, "; ".join(why)))
            continue
        run = to_run(rep, suite, name, arm)
        rp = OUT / (name.replace("/", "__") + ".json")
        rp.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
        wb.registry_add(wb.REGISTRY, rp, model=model)
        imported.append(name)
        paths[name] = rp
    for base, arm, placebo in COMPARISONS:
        if base not in paths or arm not in paths or (placebo and placebo not in paths):
            skipped.append((f"compare {base} vs {arm}", "a run was not imported"))
            continue
        load = lambda n: json.loads(paths[n].read_text(encoding="utf-8"))  # noqa: E731
        c = wb.bench_compare(load(base), load(arm), placebo=load(placebo) if placebo else None, band=None)
        c["reasons"].append("no base-twice noise band for this model yet (to be produced by the GPU queue)")
        cp = OUT / f"cmp__{base.replace('/', '__')}__vs__{arm.replace('/', '__')}.json"
        cp.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
        wb.registry_add_comparison(wb.REGISTRY, cp)
    wb.render_table(wb.REGISTRY, wb.TABLE)
    print(json.dumps({"imported": imported, "skipped": skipped, "table": str(wb.TABLE)}, indent=1))


if __name__ == "__main__":
    main()
