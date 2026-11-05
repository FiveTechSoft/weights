"""lab/verify_official_karpathy_parity.py — Verificación Independiente y Rigurosa de Paridad con nanoGPT Oficial.

Compara de forma completamente independiente nuestro runtime WSPLang (`engine/nanogpt_runtime.py`)
frente al código oficial de Andrej Karpathy (`lab/reference/karpathy_nanogpt/model.py`, commit `3adf61e`).

Batería de pruebas:
  1. Identificación y fijación de versión upstream (commit SHA).
  2. Alineación exacta de claves tensoriales y formas (77 tensores con weight tying).
  3. Paridad de logits en múltiples longitudes (T=8, 32, 64, 128, 256) y batch sizes (B=1, 2, 4).
  4. Paridad de logits con y sin targets (inferencia vs entrenamiento).
  5. Paridad exacta de función de pérdida Cross-Entropy.
  6. Paridad de gradientes parámetro por parámetro tras backward().
  7. Paridad de pesos tras 1 paso completo de optimizador AdamW.
  8. Paridad token a token en generación autorregresiva greedy (T=0).
  9. Prueba independiente de invarianza causal (sin fuga al pasado).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "lab" / "reference" / "karpathy_nanogpt"))

from model import GPT as OfficialKarpathyGPT, GPTConfig as OfficialKarpathyConfig  # Official Karpathy code
from engine.nanogpt_runtime import NanoGPT, GPTConfig as WSPLangConfig             # Our WSPLang runtime


def build_verdict(tests, max_diff, commit):
    """Verdict derived only from the test statuses; never asserts more than what passed."""
    failed = [name for name, t in tests.items() if t["status"] != "PASS"]
    if failed:
        conclusion = f"Divergencia frente a upstream {commit[:7]} en: {', '.join(failed)}."
    else:
        conclusion = (f"Runtime nanoGPT equivalente a upstream {commit[:7]} en las pruebas listadas "
                      f"(W0 por semilla, logits, pérdida, gradientes, 1 paso AdamW, generación). "
                      f"Es copia de la arquitectura en W0; no implica pesos entrenados ni síntesis.")
    return {"independent_parity_certified": not failed, "failed_tests": failed,
            "max_numerical_drift": max_diff, "commit_sha": commit, "conclusion": conclusion}


def run_independent_parity_verification():
    print("==================================================================================")
    print("  VERIFICACIÓN INDEPENDIENTE: WSPLang nanoGPT vs nanoGPT OFICIAL (Andrej Karpathy)")
    print("==================================================================================")

    # 1. Metadatos de la versión upstream
    meta_p = R / "lab" / "reference" / "karpathy_nanogpt" / "METADATA.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    print(f"Repositorio Oficial: {meta['repository']}")
    print(f"Commit Upstream:     {meta['commit_sha']}")
    print(f"Autor Upstream:      {meta['author']}")

    ckpt_p = R / "models" / "nanogpt_karpathy" / "model.safetensors"
    assert ckpt_p.exists(), f"El checkpoint {ckpt_p} debe existir."

    # Configuración canónica Shakespeare
    cfg_off = OfficialKarpathyConfig(
        block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True
    )
    cfg_wsp = WSPLangConfig(
        block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, d_ff=1536, dropout=0.0, bias=True
    )

    # 2. Instanciación independiente
    print("\n[Paso 1] Instanciando modelos desde módulos independientes...")
    # W0 upstream generado aparte con la misma semilla: el artefacto WSPLang debe coincidir con él.
    torch.manual_seed(1337)
    off_model = OfficialKarpathyGPT(cfg_off)
    upstream_w0 = {k: v.clone() for k, v in off_model.named_parameters()}
    wsp_model = NanoGPT(config=cfg_wsp, seed=1337)

    # Cargar exactamente los mismos pesos desde SafeTensors
    raw_sd = load_file(str(ckpt_p))
    # Descartar buffers attn.bias (máscaras causales fijas que Flash Attention no almacena en state_dict)
    param_sd = {k: v for k, v in raw_sd.items() if not k.endswith(".attn.bias")}

    off_model.load_state_dict(param_sd, strict=True)
    off_model.eval()

    wsp_model.load_checkpoint(ckpt_p)
    wsp_model.model.eval()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream_reference": meta,
        "model_architecture": {
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "vocab_size": 65,
            "block_size": 256,
            "parameters": sum(p.numel() for p in off_model.parameters()),
        },
        "tests": {}
    }

    # 3. Comprobación de tensores y atado de pesos
    print("\n[Paso 2] Comprobación de claves, dimensiones y atado de pesos...")
    off_named_params = dict(off_model.named_parameters())
    wsp_named_params = dict(wsp_model.model.named_parameters())

    assert len(off_named_params) == len(wsp_named_params), f"Diferencia en número de parámetros: {len(off_named_params)} != {len(wsp_named_params)}"
    for name, p_off in off_named_params.items():
        assert name in wsp_named_params, f"Clave {name} ausente en WSPLang"
        p_wsp = wsp_named_params[name]
        assert p_off.shape == p_wsp.shape, f"Dimensión disconforme en {name}: {p_off.shape} vs {p_wsp.shape}"

    # Comparación real de W0: el artefacto WSPLang frente al W0 upstream de la misma semilla.
    # (Comparar tras cargar el mismo archivo en ambos sería una tautología.)
    w0_diff = max((upstream_w0[n] - p).abs().max().item() for n, p in wsp_named_params.items())

    # Verificar atado de pesos (wte == lm_head) en ambos
    off_tie_diff = (off_model.transformer.wte.weight - off_model.lm_head.weight).abs().max().item()
    wsp_tie_diff = (wsp_model.model.transformer.wte.weight - wsp_model.model.lm_head.weight).abs().max().item()
    tie_diff = max(off_tie_diff, wsp_tie_diff)

    print(f"  Total tensores de parámetros: {len(off_named_params)}")
    print(f"  Total parámetros escalares:   {sum(p.numel() for p in off_model.parameters()):,}")
    print(f"  Diferencia artefacto vs W0 upstream (seed=1337): {w0_diff:.2e}")
    print(f"  Atado de pesos (Weight Tying): {tie_diff:.2e}")
    report["tests"]["test1_weight_topology"] = {
        "num_parameter_tensors": len(off_named_params),
        "total_parameters": sum(p.numel() for p in off_model.parameters()),
        "w0_vs_upstream_same_seed_max_diff": w0_diff,
        "weight_tying_max_diff": tie_diff,
        "status": "PASS" if (w0_diff == 0.0 and tie_diff == 0.0) else "FAIL"
    }

    # 4. Paridad de Logits en múltiples longitudes y batches
    print("\n[Paso 3] Paridad de Logits en múltiples Batch Sizes y Longitudes...")
    test_configs = [
        (1, 8),
        (2, 32),
        (4, 64),
        (2, 128),
        (1, 256),
    ]

    logit_parity_results = []
    max_logit_diff_overall = 0.0

    for b, t in test_configs:
        torch.manual_seed(b * 1000 + t)
        x = torch.randint(0, 65, (b, t))
        targets = torch.randint(0, 65, (b, t))

        with torch.no_grad():
            # Inferencia (targets=None, evalúa último logit)
            off_last, _ = off_model(x)
            wsp_last, _ = wsp_model.forward(x)
            diff_last = (off_last - wsp_last).abs().max().item()

            # Entrenamiento (targets=targets, evalúa logits completos y pérdida)
            off_full, loss_off = off_model(x, targets=targets)
            wsp_full, loss_wsp = wsp_model.forward(x, targets=targets)
            diff_full = (off_full - wsp_full).abs().max().item()
            diff_loss = abs(loss_off.item() - loss_wsp.item())

        max_logit_diff_overall = max(max_logit_diff_overall, diff_last, diff_full)
        print(f"  Batch={b:2d}, SeqLen={t:3d} | Logit Diff Last: {diff_last:.2e} | Logit Diff Full: {diff_full:.2e} | Loss Diff: {diff_loss:.2e}")
        logit_parity_results.append({
            "batch_size": b,
            "seq_len": t,
            "diff_last_token": diff_last,
            "diff_full_sequence": diff_full,
            "diff_loss": diff_loss,
            "loss_official": loss_off.item(),
            "loss_wsp": loss_wsp.item()
        })

    report["tests"]["test2_multi_length_logits"] = {
        "cases": logit_parity_results,
        "max_diff_overall": max_logit_diff_overall,
        "status": "PASS" if max_logit_diff_overall == 0.0 else "FAIL"
    }

    # 5. Paridad de Gradientes y Paso de Optimizador
    print("\n[Paso 4] Paridad de Gradientes y Paso de Optimización (AdamW)...")
    off_train = OfficialKarpathyGPT(cfg_off)
    off_train.load_state_dict(param_sd, strict=True)
    off_train.train()

    wsp_train = NanoGPT(config=cfg_wsp, seed=1337)
    wsp_train.load_checkpoint(ckpt_p)
    wsp_train.model.train()

    opt_off = torch.optim.AdamW(off_train.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-1)
    opt_wsp = torch.optim.AdamW(wsp_train.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-1)

    torch.manual_seed(999)
    x_opt = torch.randint(0, 65, (4, 32))
    y_opt = torch.randint(0, 65, (4, 32))

    opt_off.zero_grad(set_to_none=True)
    _, l_off = off_train(x_opt, targets=y_opt)
    l_off.backward()

    opt_wsp.zero_grad(set_to_none=True)
    _, l_wsp = wsp_train.forward(x_opt, targets=y_opt)
    l_wsp.backward()

    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(off_train.named_parameters(), wsp_train.model.named_parameters()):
        if p1.grad is not None:
            g_diff = (p1.grad - p2.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, g_diff)

    opt_off.step()
    opt_wsp.step()

    max_weight_diff_after_step = 0.0
    for (n1, p1), (n2, p2) in zip(off_train.named_parameters(), wsp_train.model.named_parameters()):
        w_diff = (p1 - p2).abs().max().item()
        max_weight_diff_after_step = max(max_weight_diff_after_step, w_diff)

    print(f"  Pérdida backward:       Oficial = {l_off.item():.6f} | WSP = {l_wsp.item():.6f} (Diff: {abs(l_off.item() - l_wsp.item()):.2e})")
    print(f"  Diferencia de gradiente máxima:   {max_grad_diff:.2e}")
    print(f"  Diferencia tras paso AdamW:       {max_weight_diff_after_step:.2e}")

    report["tests"]["test3_optimization_dynamics"] = {
        "max_grad_diff": max_grad_diff,
        "max_weight_diff_after_adam_step": max_weight_diff_after_step,
        "loss_diff": abs(l_off.item() - l_wsp.item()),
        "status": "PASS" if (max_grad_diff == 0.0 and max_weight_diff_after_step == 0.0) else "FAIL"
    }

    # 6. Paridad en Generación Greedy (T=0)
    print("\n[Paso 5] Paridad en Generación Autorregresiva Greedy (T=0)...")
    prompts = [
        "First Citizen:\n",
        "To be, or not to ",
        "O Romeo, Romeo, ",
    ]
    vocab_str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    c2i = {ch: i for i, ch in enumerate(vocab_str)}
    i2c = {i: ch for i, ch in enumerate(vocab_str)}

    generation_cases = []
    all_gen_matches = True

    for p_str in prompts:
        p_toks = torch.tensor([[c2i[c] for c in p_str if c in c2i]], dtype=torch.long)
        
        # Generación con modelo WSPLang
        out_wsp = wsp_model.generate(p_toks.clone(), max_new_tokens=25, temperature=0.0)
        str_wsp = "".join(i2c[i] for i in out_wsp[0].tolist())

        # Generación con modelo Oficial Karpathy (greedy manual idéntico)
        curr_off = p_toks.clone()
        for _ in range(25):
            with torch.no_grad():
                logits, _ = off_model(curr_off)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_off = torch.cat((curr_off, next_tok), dim=1)
        str_off = "".join(i2c[i] for i in curr_off[0].tolist())

        match = (out_wsp == curr_off).all().item()
        if not match:
            all_gen_matches = False
        print(f"  Prompt: {p_str!r}")
        print(f"    Oficial: {str_off!r}")
        print(f"    WSPLang: {str_wsp!r}")
        print(f"    Coincidencia: {'EXACTA' if match else 'ERROR'}")

        generation_cases.append({
            "prompt": p_str,
            "generated_official": str_off,
            "generated_wsp": str_wsp,
            "exact_match": bool(match)
        })

    report["tests"]["test4_greedy_generation"] = {
        "cases": generation_cases,
        "all_match": all_gen_matches,
        "status": "PASS" if all_gen_matches else "FAIL"
    }

    # 6b. Paridad en generación muestreada, usando el generate() oficial
    print("\n[Paso 5b] Paridad en Generación Muestreada (T=0.8, top_k=10)...")
    sampled_cases = []
    for p_str in prompts:
        p_toks = torch.tensor([[c2i[c] for c in p_str if c in c2i]], dtype=torch.long)
        torch.manual_seed(4242)
        out_off = off_model.generate(p_toks.clone(), 25, temperature=0.8, top_k=10)
        torch.manual_seed(4242)
        out_wsp = wsp_model.generate(p_toks.clone(), max_new_tokens=25, temperature=0.8, top_k=10)
        sampled_cases.append(bool((out_off == out_wsp).all().item()))
    print(f"  Coincidencias: {sum(sampled_cases)}/{len(sampled_cases)}")
    report["tests"]["test4b_sampled_generation"] = {
        "cases_matching": sum(sampled_cases),
        "cases_total": len(sampled_cases),
        "status": "PASS" if all(sampled_cases) else "FAIL"
    }

    # 7. Prueba Independiente de Causalidad
    print("\n[Paso 6] Prueba Independiente de Invarianza Causal...")
    seq_len = 32
    torch.manual_seed(777)
    base_seq = torch.randint(0, 65, (1, seq_len))
    perturbed_seq = base_seq.clone()
    perturbed_seq[0, 20:] = torch.randint(0, 65, (1, 12))  # Perturbamos desde t=20 en adelante

    with torch.no_grad():
        logits_base, _ = wsp_model.forward(base_seq, targets=base_seq)
        logits_pert, _ = wsp_model.forward(perturbed_seq, targets=perturbed_seq)

    # Los logits para t < 20 deben ser IDÉNTICOS (fuga cero al pasado)
    past_leakage = (logits_base[:, :20, :] - logits_pert[:, :20, :]).abs().max().item()
    # Los logits para t >= 20 deben ser diferentes (sensibilidad causal al futuro)
    future_diff = (logits_base[:, 20:, :] - logits_pert[:, 20:, :]).abs().max().item()

    print(f"  Filtración hacia el pasado (t < 20): {past_leakage:.2e} (debe ser 0.0)")
    print(f"  Sensibilidad en el futuro  (t >= 20): {future_diff:.2e} (debe ser > 0.0)")

    report["tests"]["test5_causal_invariance"] = {
        "past_leakage": past_leakage,
        "future_sensitivity": future_diff,
        "status": "PASS" if (past_leakage == 0.0 and future_diff > 0.0) else "FAIL"
    }

    # Veredicto Global
    report["verdict"] = build_verdict(report["tests"], max_logit_diff_overall, meta["commit_sha"])
    overall_pass = report["verdict"]["independent_parity_certified"]

    out_file = R / "lab" / "runs" / "official_karpathy_parity_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n==================================================================================")
    print(f"  RESULTADO FINAL: {'PASS' if overall_pass else 'FALLO'}")
    print(f"  {report['verdict']['conclusion']}")
    print(f"  Informe guardado en: {out_file}")
    print(f"==================================================================================")
    return report


if __name__ == "__main__":
    final = run_independent_parity_verification()
    sys.exit(0 if final["verdict"]["independent_parity_certified"] else 1)
