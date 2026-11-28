"""Fail-closed certification policy for behavioral edits.

This validates run records, not their authenticity or the physical norm of
a placebo. A passing record is not a sealed scientific experiment.
"""
import re

from engine import wsplang_bench as bench


def _identity(run):
    ident = run.get("identity") or {}
    sha = ident.get("model_sha")
    if not isinstance(sha,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",sha):
        raise ValueError("full model SHA256 required")
    if ident.get("unknown_model"):
        raise ValueError("unidentified model")
    return sha.lower()


def _items(run,key):
    rows = run.get(key)
    if not isinstance(rows,list) or not rows:
        raise ValueError(f"missing {key}")
    indexed = {}
    for r in rows:
        k = (r["bench"],r["id"])
        if k in indexed:
            raise ValueError(f"duplicate {key}: {k}")
        if type(r.get("ok")) is not bool:
            raise ValueError(f"missing boolean outcome: {k}")
        if key == "items" and type(r.get("ok_independent")) is not bool:
            raise ValueError(f"missing independent outcome: {k}")
        if key == "preservation_items":
            if any(type(r.get(flag)) is not bool for flag in ("repeat_16gram","nonascii_run")):
                raise ValueError(f"missing garbage measurements: {k}")
        indexed[k] = r
    return indexed


def certify(base, arm, placebo, base2, *, target):
    """Require paired full records and mandatory HumanEval/compliance gates.

    Noise and placebo controls are mandatory. Target is predeclared by the
    caller. No significant loss on any other target-suite benchmark.
    """
    runs = (base,arm,placebo,base2)
    if any(r.get("model_class") == "llm-logits" for r in runs):
        raise ValueError("behavioral certification requires scored task runs")
    sha = [_identity(r) for r in runs]
    suite = base.get("suite_sha")
    decoding = base.get("decoding")
    if not isinstance(suite,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",suite) or not decoding:
        raise ValueError("suite SHA256 and decoding required")
    if any(r.get("suite_sha")!=suite or r.get("decoding")!=decoding for r in runs):
        raise ValueError("suite or decoding mismatch")
    if sha[0]!=sha[3] or base.get("identity")!=base2.get("identity"):
        raise ValueError("base repeat identity mismatch")
    if base.get("identity")==arm.get("identity") or arm.get("identity")==placebo.get("identity") or base.get("identity")==placebo.get("identity"):
        raise ValueError("arms must be distinct")
    if any(bench.relation(base,r)!="edit vs its base" for r in (arm,placebo)):
        raise ValueError("missing base provenance")
    items=[_items(r,"items") for r in runs]
    pres=[_items(r,"preservation_items") for r in runs]
    if any(set(x)!=set(items[0]) for x in items[1:]) or any(set(x)!=set(pres[0]) for x in pres[1:]):
        raise ValueError("unpaired evaluation items")
    for name,n in (("humaneval",164),("compliance",40)):
        if sum(k[0]==name for k in pres[0])!=n:
            raise ValueError(f"preservation requires {n} {name} items")
    band=bench.noise_band(base,base2)
    cmp=bench.bench_compare(base,arm,placebo=placebo,band=band)
    direct=bench.bench_compare(placebo,arm)
    reasons=[]
    if cmp["verdict"]!="improves":
        reasons.append("benchmark comparison did not improve")
    if target not in cmp["significant_up"]:
        reasons.append("predeclared target did not improve")
    elif target not in band or abs(cmp["benchmarks"][target]["g1"]["arm"] -
                                   cmp["benchmarks"][target]["g1"]["base"]) < 2 * band[target]["abs_delta"]:
        reasons.append("predeclared target did not exceed repeat-base noise")
    if cmp["significant_down"]:
        reasons.append("significant benchmark regression")
    if target not in direct["significant_up"]:
        reasons.append("candidate did not beat placebo on target")
    if not cmp["preservation"] or not cmp["preservation"]["preserves"]:
        reasons.append("preservation failed")
    return {"schema_version":1,"certified":not reasons,"reasons":reasons,
            "target":target,"suite_sha":suite,"model_sha256":sha,
            "comparison":cmp,"candidate_vs_placebo":direct,
            "scope":"record-level screening; no sealed or same-norm authenticity claim"}
