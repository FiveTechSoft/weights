# LLM desde cero: plan honesto y resultados a 1500 pasos

**Fecha:** 24 de Noviembre de 2026
**Script:** [`lab/train_llm_scratch.py`](../lab/train_llm_scratch.py)
**Reporte:** [`lab/runs/llm_scratch/llm_scratch_train.json`](../lab/runs/llm_scratch/llm_scratch_train.json)
**Plan original:** [`lab/PLAN_LLM_FROM_SCRATCH.md`](../lab/PLAN_LLM_FROM_SCRATCH.md)
**Base de texto:** Shakespeare Shakespeare en caracteres (`lab/data/shakespeare_char/input.txt`, 1,115,394 caracteres)

---

## 1. Objetivo

Entrenar un LLM de verdad a partir de cero, sin donor weights ni cirugía,
con la misma disciplina de honestidad que las demás pruebas: pérdida descrita,
holdout sin tocar, objetivo declarado ANTES de entrenar.

## 2. Diseño (declarado antes de entrenar)

- Arquitectura: NanoWSP-3 desde cero (`d=256`, 6 capas, 8 cabezas, 512 ff,
  contexto 128, vocabulario de 256 caracteres).
- Sin donor. Sin WSPLang injection. Sin nada de ese tipo.
- Optimizer: AdamW, lr 8e-4, 1,500 pasos, batch 16, contexto 128.
- Corpus: Shakespeare en caracteres (100% de traducción de caracteres, no de tokens).
- Holdout = cola 10% (nunca visto en entrenamiento) — sin report de sample text,
  no se reporta nada que no se mide.
- Objetivo de éxito honesto: ppl < 10 en la parte de entrenamiento + una honest
  evaluación en holdout. **No se espera que pase el gate ZARP-10** — este es un
  LLM, no un skill compilado.

## 3. Resultados (verificados, 2026-11-24)

| Métrica | Valor |
|---|---|
| PPL inicial | **1537.70** (≈ e^7.31) |
| PPL final (1500 pasos) | **5.91** |
| Loss inicial → final | 7.31 → 1.34 |
| Segundos CPU (20 cores, aarch64) | 518.9 |
| Snapshots guardados | 5 (300, 600, 900, 1200, 1500) |
| Artefacto de pesos | `lab/runs/llm_scratch/llm_scratch_pretrained.safetensors` (gitignored, binario) |
| Trace | `lab/runs/llm_scratch/llm_scratch_trace.npz` (72 MB, no versionado) |

**Reproducción:** `python lab/train_llm_scratch.py` desde la raíz del repo.

## 4. Lo que es y lo que NO es

- **Sí es:** un Transformer de 256-dim / 6 capas / 8 cabezas
  entrenado de verdad (from scratch) en el corpus de Shakespeare con un report
  honesto. Perplexidad aprendida de extremo a extremo.
- **NO es:** un modelo que "redacte párrafos narrativos coherentes con la
  ironía de un LLM de producción". 1,500 pasos en CPU no lo alcanza.
  El output no es una muestra de texto: la ausencia de la muestra es honestidad,
  no sorpresa.
- Lo que aún falta para un LLM serio: entrenar de verdad con suficiente presupuesto
  (no 10 min en CPU), medir holdout, y que el output sea verificable.

## 5. Próximos pasos

1. Medir el holdout (cola 10%) y reportar ppl en él.
2. Fijar un presupuesto honesto (por ejemplo: 50k pasos o usar GPU si hay) y lanzarlo.
3. Recién después evaluar el LLM en las tareas ZARP-10... no antes.
