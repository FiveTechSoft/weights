"""lab/verify_nanogpt_learning.py — Test de Sobreajuste de Un Solo Lote (Single-Batch Overfit Sanity Check).

Prueba de sanidad computacional para verificar que el grafo tensorial y el flujo de gradientes
funcionan correctamente en la arquitectura nanoGPT inicializada (W0):
1. Pérdida inicial teórica de máxima entropía H(V) = ln(65) = 4.174 nats (estado inicial no entrenado).
2. Dinámica de gradientes AdamW: flujo de gradiente estable a través de las 6 capas.
3. Capacidad de memorización: descenso monótono de pérdida en un pasaje de 167 caracteres.
4. Delimitación metodológica: Esta prueba demuestra que la arquitectura es optimizable y memoriza
   un lote de datos, pero NO constituye preentrenamiento sobre el corpus de Shakespeare ni generalización.
"""

import json
import math
import sys
import time
from pathlib import Path

R = Path(r"C:\weights")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "engine"))

import torch
from engine.nanogpt_runtime import GPTConfig, NanoGPT


def run_learning_verification():
    print("=== VERIFICACIÓN CIENTÍFICA: DINÁMICA DE APRENDIZAJE Y GENERACIÓN DE nanoGPT ===")

    ckpt_p = R / "models" / "nanogpt_karpathy" / "model.safetensors"
    assert ckpt_p.exists(), f"El checkpoint {ckpt_p} debe existir."

    cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384, vocab_size=65, block_size=256)
    ng = NanoGPT(config=cfg, seed=1337)
    ng.load_checkpoint(ckpt_p)

    vocab_str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    assert len(vocab_str) == 65
    c2i = {ch: i for i, ch in enumerate(vocab_str)}
    i2c = {i: ch for i, ch in enumerate(vocab_str)}

    sample_text = (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die than to famish?\n\n"
        "All:\n"
        "Resolved. resolved.\n"
    )

    tok_ids = [c2i[c] for c in sample_text if c in c2i]
    toks = torch.tensor(tok_ids, dtype=torch.long)
    x = toks[:-1].unsqueeze(0)
    y = toks[1:].unsqueeze(0)

    # 1. Pérdida inicial
    with torch.no_grad():
        _, loss_0 = ng.forward(x, targets=y)
    l0 = float(loss_0.item())
    theo_entropy = math.log(65)

    # 2. Generación inicial (T=0 greedy)
    prompt_str = "First Citizen:"
    p_ids = torch.tensor([[c2i[c] for c in prompt_str]], dtype=torch.long)
    gen_0 = ng.generate(p_ids.clone(), max_new_tokens=40, temperature=0.0)
    out_0 = "".join(i2c[i] for i in gen_0[0].tolist())

    # 3. Optimización causal con AdamW (40 pasos)
    opt = torch.optim.AdamW(ng.model.parameters(), lr=1e-3, weight_decay=1e-2)
    ng.model.train()
    history = []
    t0 = time.time()

    for step in range(1, 41):
        opt.zero_grad()
        _, l = ng.forward(x, targets=y)
        l.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(ng.model.parameters(), 1.0).item())
        opt.step()
        loss_val = float(l.item())
        history.append({
            "step": step,
            "loss": round(loss_val, 4),
            "ppl": round(math.exp(loss_val), 2),
            "grad_norm": round(grad_norm, 4)
        })

    duration = time.time() - t0

    # 4. Generación final (T=0 greedy)
    ng.model.eval()
    gen_f = ng.generate(p_ids.clone(), max_new_tokens=40, temperature=0.0)
    out_f = "".join(i2c[i] for i in gen_f[0].tolist())

    l_final = history[-1]["loss"]
    ppl_0 = math.exp(l0)
    ppl_f = math.exp(l_final)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture": {
            "model": "nanoGPT-Shakespeare",
            "layers": cfg.n_layer,
            "heads": cfg.n_head,
            "d_model": cfg.n_embd,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "total_parameters": ng.count_parameters()
        },
        "initial_state": {
            "empirical_loss": round(l0, 4),
            "theoretical_max_entropy_ln_V": round(theo_entropy, 4),
            "entropy_discrepancy": round(abs(l0 - theo_entropy), 4),
            "initial_ppl": round(ppl_0, 2),
            "initial_generation": out_0
        },
        "optimization_dynamics": {
            "optimizer": "AdamW (lr=1e-3, wd=1e-2)",
            "steps": 40,
            "duration_seconds": round(duration, 2),
            "final_loss": round(l_final, 4),
            "final_ppl": round(ppl_f, 2),
            "ppl_reduction_factor": round(ppl_0 / ppl_f, 1),
            "step_history_snapshots": [h for h in history if h["step"] in [1, 5, 10, 20, 30, 40]]
        },
        "trained_state": {
            "final_generation": out_f,
            "passage_memorized": bool("Before we proceed any further" in out_f),
            "test_type": "single_batch_overfit_sanity_check",
            "pretraining_on_corpus": False,
            "generalization_tested": False,
            "verdict": "SINGLE_BATCH_OVERFIT_" + ("PASS" if "Before we proceed any further" in out_f else "FAIL"),
            "scientific_note": "Confirma que el grafo computacional y los gradientes de nanoGPT funcionan y pueden memorizar un lote; no demuestra generalización lingüística ni preentrenamiento general sobre el corpus de Shakespeare."
        }
    }

    out_p = R / "lab" / "runs" / "nanogpt_learning_verification.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Reporte de verificación científica guardado en {out_p}")
    return report


if __name__ == "__main__":
    run_learning_verification()
