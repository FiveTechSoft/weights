"""Paridad del runtime frente al model.py de karpathy/nanoGPT fijado en lab/reference (commit 3adf61e).

Config pequeña (4L/d128/h4) con W0 generado por cada clase a partir de la misma semilla.
Sale con código 1 si el archivo de referencia cambió o si los logits divergen.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
REF = R / "lab" / "reference" / "karpathy_nanogpt"
sys.path[:0] = [str(R), str(REF)]

import model as upstream  # noqa: E402
from engine.nanogpt_runtime import GPTConfig, KarpathyGPT  # noqa: E402

UPSTREAM_MODEL_SHA256 = "63d46bccd28157de4ae3d4e714f3945370be708852d2f411e0d017a7da5f8d39"
CFG = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, vocab_size=65)
TOL = 2e-4  # margen fp32 por si un backend distinto cambia kernels; en una misma máquina se espera 0.0


def parity_verdict(diff):
    if diff <= TOL:
        return "PASSES_fp32"
    return "CLOSE_BUT_NOT_EXACT" if diff < 0.05 else "DIVERGES"


def interpretation(diff):
    if diff == 0.0:
        return "logits idénticos: esperado, el runtime es copia de model.py sobre los mismos kernels"
    if diff <= TOL:
        return f"logits dentro de la tolerancia fp32 ({TOL}); diferencia numérica, no de arquitectura"
    return f"divergencia real de {diff:.4g} frente a upstream: revisar arquitectura o inicialización"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    meta = json.loads((REF / "METADATA.json").read_text(encoding="utf-8"))
    res = {"upstream_commit": meta["commit_sha"], "upstream_model_sha256": sha256(REF / "model.py"), "cfg": CFG}
    res["upstream_file_matches_pin"] = res["upstream_model_sha256"] == UPSTREAM_MODEL_SHA256

    torch.manual_seed(1337)
    theirs = upstream.GPT(upstream.GPTConfig(**CFG)).eval()
    torch.manual_seed(1337)
    ours = KarpathyGPT(GPTConfig(**CFG)).eval()

    x = torch.randint(0, CFG["vocab_size"], (3, 47))
    y = torch.randint(0, CFG["vocab_size"], x.shape)
    x2 = x.clone()
    x2[:, 32] = (x2[:, 32] + 7) % CFG["vocab_size"]
    with torch.no_grad():
        l_theirs, loss_theirs = theirs(x, y)
        l_ours, loss_ours = ours(x, y)
        l2_theirs, _ = theirs(x2, y)
        l2_ours, _ = ours(x2, y)

    diff = float((l_theirs - l_ours).abs().max())
    res.update(max_logits_diff=diff,
               max_logits_diff_perturbed=float((l2_theirs - l2_ours).abs().max()),
               shapes_match=l_theirs.shape == l_ours.shape,
               causal_past_diff=float((l2_ours[:, :32] - l_ours[:, :32]).abs().max()),
               causal_future_diff=float((l2_ours[:, 32:] - l_ours[:, 32:]).abs().max()),
               loss_upstream=round(float(loss_theirs), 6), loss_runtime=round(float(loss_ours), 6),
               parity_verdict=parity_verdict(diff), interpretacion=interpretation(diff))

    print(json.dumps(res, indent=2, ensure_ascii=False))
    out = R / "rig_runs" / "karpathy_vendorized_parity.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado", out)
    ok = res["upstream_file_matches_pin"] and res["parity_verdict"] == "PASSES_fp32"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
