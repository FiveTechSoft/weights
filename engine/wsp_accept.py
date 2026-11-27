"""Promote a standalone evaluated model by publishing an exclusive acceptance record.

No in-place edit or deletion. Only local single-file models are supported;
the record binds the exact evaluated bytes, not a directory or runtime overlay.
"""
import hashlib
import json
from pathlib import Path

from engine.wsp_preservation_gate import certify


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def accept(base_model, candidate_model, placebo_model, reports, *, target, out):
    models = [Path(p).resolve(strict=True) for p in
              (base_model, candidate_model, placebo_model)]
    sources = [Path(p).resolve(strict=True) for p in reports]
    out = Path(out)
    if len(sources) != 4 or len(set(sources)) != 4:
        raise ValueError("four distinct run records required")
    if len(set(models)) != 3 or any(not p.is_file() for p in models):
        raise ValueError("three distinct standalone model files required")
    if any(p.suffix != ".gguf" for p in models):
        raise ValueError("accept currently supports standalone GGUF only")
    if out.suffix != ".json" or out.exists() or out.is_symlink():
        raise ValueError("new JSON acceptance record required")
    hashes = [sha256(p) for p in models]
    if len(set(hashes)) != 3:
        raise ValueError("model contents must be distinct")
    raw = [p.read_bytes() for p in sources]
    runs = [json.loads(b) for b in raw]
    for run, expected in zip(runs, hashes + [hashes[0]]):
        ident = run.get("identity") or {}
        if ident.get("extras") != {}:
            raise ValueError("runtime overlays are not standalone models")
        if ident.get("model_sha") != expected:
            raise ValueError("evaluated model SHA256 mismatch")
    result = certify(*runs, target=target)
    if not result["certified"]:
        raise ValueError("acceptance rejected: " + "; ".join(result["reasons"]))
    if hashes != [sha256(p) for p in models]:
        raise ValueError("model changed during certification")
    record = {"schema_version":1, "state":"accepted",
              "models":[{"path":str(p),"sha256":h} for p,h in zip(models,hashes)],
              "runs":[{"path":str(p),"sha256":hashlib.sha256(b).hexdigest()}
                      for p,b in zip(sources,raw)],
              "target":target,"certification":result}
    payload = json.dumps(record,indent=2,allow_nan=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as f:
        f.write(payload)
    return record
