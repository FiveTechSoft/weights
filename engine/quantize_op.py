"""engine/quantize_op.py - extreme quantisation (2-bit and friends) as a WSP-Lang op.

Wraps llama.cpp's llama-quantize (and llama-imatrix for the importance matrix the IQ2 / IQ1 types need). Writes the
output to <out>.tmp and renames it only on success; writes <out>.manifest.json with the input and output sha256, the
type, bits per weight, the imatrix and calibration-text sha256, the exact command and caveats. engine.wsplang_bench
.read_provenance reads that manifest, so `bench_run provenance=` links a quantised model to its base.

Calibration: WikiText-2 *train* (wikitext_train), never the test split that the dist-shift suite measures on.
No behaviour claim: what 2 bits cost is measured by bench_run (dist-shift, rank-2048), with the base beside it.
"""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path.home() / "llamacpp" / "bin"
QUANTIZE_BIN = BIN / "llama-quantize.exe"
IMATRIX_BIN = BIN / "llama-imatrix.exe"
WIKITEXT_ZIP = ROOT / "data" / "bench" / "wikitext-2-raw-v1.zip"

TYPES = ("Q2_K", "Q2_K_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M", "TQ1_0", "TQ2_0",
         "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
NEEDS_IMATRIX = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M")
VALUE_FLAGS = ("--imatrix", "--tensor-type", "--output-tensor-type", "--token-embedding-type")


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def quantize_cmd(binary, src, out, qtype, imatrix=None, allow_requantize=False, tensor_types=None, leave_output=False,
                 nthreads=None):
    qtype = str(qtype).upper()
    if qtype not in TYPES:
        raise ValueError(f"quantize: type {qtype} not among {TYPES}")
    if qtype in NEEDS_IMATRIX and not imatrix:
        raise ValueError(f"quantize: {qtype} needs an importance matrix (imatrix=auto or a file)")
    cmd = [str(binary)]
    if allow_requantize:
        cmd.append("--allow-requantize")
    if leave_output:
        cmd.append("--leave-output-tensor")
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    for name, t in (tensor_types or {}).items():
        cmd += ["--tensor-type", f"{name}={t}"]
    cmd += [str(src), str(out), qtype]
    if nthreads:
        cmd.append(str(int(nthreads)))
    return cmd


def imatrix_cmd(binary, model, calib, out, chunks=100, ngl=99, ctx=512):
    return [str(binary), "-m", str(model), "-f", str(calib), "-o", str(out), "--chunks", str(int(chunks)),
            "-ngl", str(int(ngl)), "-c", str(int(ctx))]


def wikitext_train(zip_path=WIKITEXT_ZIP, out=None, max_chars=2_000_000):
    """WikiText-2 train split (never the test split) as a calibration text file."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("wiki.train.raw"))
        text = z.read(name).decode("utf-8")[:max_chars]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _run(cmd, log_path=None):
    if log_path:
        with open(log_path, "a", encoding="utf-8") as lf:
            return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def gguf_param_count(path):
    import gc
    import gguf
    r = gguf.GGUFReader(str(path))
    n = int(sum(int(t.n_elements) for t in r.tensors))
    del r
    gc.collect()
    return n


def quantize(model, qtype, out, imatrix=None, calib=None, tensor_types=None, allow_requantize=False, leave_output=False,
             chunks=100, runner=None, n_params=None, quantize_bin=QUANTIZE_BIN, imatrix_bin=IMATRIX_BIN, log_path=None):
    """Quantise `model` (a GGUF) to `qtype` at `out`. imatrix: None, "auto" (computed from `calib`, default WikiText-2
    train) or a file. Returns the report (also the manifest content plus its path)."""
    runner = runner or _run
    model, out = Path(model), Path(out)
    if out.resolve() == model.resolve():
        raise ValueError("quantize: out must differ from the input model")
    if out.exists():
        raise FileExistsError(f"quantize: {out} exists; outputs are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    caveats = []
    calib_sha = imatrix_sha = None
    im_path = None
    if imatrix == "auto":
        calib = Path(calib) if calib else wikitext_train(out=out.parent / "calib_wikitext2_train.txt")
        im_path = out.with_name(out.stem + ".imatrix.gguf")
        rc = runner(imatrix_cmd(imatrix_bin, model, calib, im_path, chunks=chunks), log_path)
        if rc != 0 or not im_path.exists():
            raise RuntimeError(f"quantize: llama-imatrix failed (exit {rc})")
        calib_sha, imatrix_sha = sha_file(calib), sha_file(im_path)
    elif imatrix:
        im_path = Path(imatrix)
        imatrix_sha = sha_file(im_path)
    tmp = out.with_name(out.name + ".tmp")
    cmd = quantize_cmd(quantize_bin, model, tmp, qtype, imatrix=im_path, allow_requantize=allow_requantize,
                       tensor_types=tensor_types, leave_output=leave_output)
    rc = runner(cmd, log_path)
    if rc != 0 or not tmp.exists():
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"quantize: llama-quantize failed (exit {rc})")
    tmp.rename(out)
    if allow_requantize:
        caveats.append("requantized from an already quantized source: the error adds to the source's own rounding; "
                       "quantising from BF16 would be better")
    n = n_params if n_params is not None else gguf_param_count(model)
    rep = {"op": "quantize", "type": str(qtype).upper(), "input": str(model), "input_sha256": sha_file(model),
           "output": str(out), "output_sha256": sha_file(out), "input_bytes": model.stat().st_size,
           "output_bytes": out.stat().st_size, "n_params": n, "bits_per_weight": out.stat().st_size * 8 / max(n, 1),
           "imatrix": str(im_path) if im_path else None, "imatrix_sha256": imatrix_sha,
           "calib": str(calib) if calib else None, "calib_sha256": calib_sha, "tensor_types": tensor_types or {},
           "allow_requantize": bool(allow_requantize), "leave_output": bool(leave_output),
           "command": [str(c) for c in cmd], "caveats": caveats, "behaviour_claim": "none"}
    mp = out.with_name(out.name + ".manifest.json")
    mp.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    rep["manifest"] = str(mp)
    return rep
