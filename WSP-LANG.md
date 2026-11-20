# WSP-Lang 0.8

**Parser, compilador y runtime para programar el espacio de pesos.** Implementación: `engine/wsplang.py`
(`__version__ = "0.8.4"`; CLI `python -m engine.wsplang <programa.wspl> [--check | --run]`).

**Spec canónica: [`docs/wsplang_specification.md`](docs/wsplang_specification.md)**, con la tabla de ops que el runtime
acepta y un test que la compara con el engine (`tests/test_wsplang_spec.py`: versión, cada op con su tipo, y ningún
`step` documentado que el runtime rechace). Este fichero resume y apunta ahí. Fundamento científico de lo que se hace
con el runtime: [`FOUNDATIONS.md`](FOUNDATIONS.md).

## 1. Modelo y programa

M = (A, W, t): arquitectura, pesos, intención. Programa P = (o1..on): W' = P(W).
El programa describe la intención experimental; el runtime resuelve rutas, valida, autoriza escrituras y ejecuta
con manifiesto. El checkpoint original es inmutable (write guard: nunca se sobreescribe el modelo fuente).

## 2. Sintaxis (dos formas, ambas reales)

A) metadata + pasos con argumentos (la de producción):
```
experiment transplant_jackrong_math:
steps:
    step delta_to_lora base=hf_cache/qwen35-9b-hf model=hf_cache/dp_keep_dsv4_distill rank=16 out=hf_cache/close_lora/demo_jack_r16 gguf=hf_cache/close_lora/demo_jack_r16.gguf
```
B) estilo E (primitivas semánticas, ver la spec): inspect / observe / compare / hypothesize / localize / intervene /
evaluate / confirm / record / select_candidate, resueltas por el runtime.

## 3. Comprobaciones (2026-11-10, repetibles)

1. `--check` sobre un programa real (`programs/taskvec_mult.wspl`, 4 pasos): `[PASS]` sintaxis, presupuestos y
   fronteras de seguridad.
2. Op inventado (`step teleport_weights`): `[FAIL] operacion de paso no permitida`.
3. Op de escritura sin `--model` safetensors: rechazado (los GGUF se editan con `wsp.py`).
4. Ops sin ejecutor: `distributed_memit` figuraba como implementado sin código que lo ejecutara; en HEAD un programa
   con él validaba y se "saltaba" en silencio. Retirado de las listas y de la spec; el validador lo rechaza
   (`tests/test_wsplang_phantom_ops.py`). `leace_erase` y `surprise_fwp` son STUB declarados (`STUB_OPS`): devuelven
   `stub_result`, sin afirmación causal.
5. Tests del runtime: `tests/test_wsplang*.py`, 69 en verde.
6. Uso real: `delta_to_lora` (trasplante con veredicto de energía) en 7 donantes de Qwen3.5-9B; el resultado
   principal está confirmado en un conjunto sellado (results §43.15). `contrast_vector` (vector de control a partir
   de dos conjuntos de textos, o por posición con filas `think`/`focus`) pasó su criterio en el estudio de felicidad
   (results §43.13). El veredicto devuelve `no_change` cuando el "fine-tune" es la base (Δ = 0), como OmniCoder-9B y
   OxCoder-9B en el hub.

## 4. Caso de uso completo: `compile_control_vector` (2026-11-11)

El primer pipeline WSPLang que va **de conducta a pesos confirmados en sellado**, todo en un programa declarativo
(`engine/cc_compile.py`; plan `lab/probes/cc_weights/OP_PLAN.md`):

```
experiment compile_conclude:
steps:
    step contrast_vector ... out=hf_cache/cc/cc.gguf          # vector "concluir" (o placebo= con placebo_seed=)
    step compile_control_vector model=hf_cache/qwen35-9b-hf vector=hf_cache/cc/cc.gguf \
         calib=lab/runs/cc/traces.jsonl out=lab/runs/cc_weights/vec base_gguf=hf_cache/h4/A0-Q6_K.gguf
```

- Escribe, en ficheros NUEVOS (nunca sobre la entrada): `fit.pt` (u por capa, ridge cerrado), `merged-Q6_K.gguf`
  (copia requantizada con `dW = v·uᵀ` en cada `ffn_down`), LoRA r1 PEFT, y el placebo compilado de misma norma.
- Gate **G0** incluido (mediana entre capas del error reservado |u·x−1| ≤ 0.5): sin benchmark, el gate evita
  promocionar el resultado; los benchmarks van aparte sobre conjunto sellado (`verdict.py` modo sellado, C1-C4).
- Procedencia del resultado que motivó el op: results §43.19 (dev GSM8K 50→87, MATH 6→34, placebo compilado 28/6);
  el sellado de la versión-.wspl era la aceptación declarada del plan.

## 5. Reglas de seguridad vivas (runtime, no prompt)

- `FORBIDDEN_TENSORS` / `FORBIDDEN_PATTERNS` (embeddings, lm_head, output_norm): protegidos salvo
  `budgets unlock_tensors` explícito.
- Presupuestos duros: `max_modified_weights`, `max_delta_norm`, `target_parameter_fraction`; formas y dtypes deben
  coincidir 1:1 (ni añadir, ni quitar, ni cambiar la forma de tensores).
- Chequeo de factibilidad antes de ejecutar: rechaza ops no implementadas (regla 2026-11-04), ops de escritura sobre
  GGUF (redirige a `wsp.py`) y referencias a archivos inexistentes.
- Cada intervención deja manifiesto con `wsplang_version` y resultados JSON inspeccionables.

## 6. Registro de experimentos

ExperimentRecord: hypothesis, evidence, target, intervention, model_hash, benchmark_version, baseline_result,
candidate_result, placebo_result, holdout_result, conclusion ∈ {SUPPORTED, REJECTED, INCONCLUSIVE}. Un experimento
fallido queda como conocimiento reutilizable (tabla de ideas retiradas en `FOUNDATIONS.md`).

## 7. Historia de versiones

- 0.1 (2026-11-03): spec original de Antonio (8 primitivas, ExperimentRecord, criterio de aceptación); el texto
  íntegro está en el historial de git de este fichero (`git show 4bd3603:WSP-LANG.md`).
- 0.2-0.6: `docs/wsplang_0.N_spec.md` (addenda por versión) y `docs/wsplang_specification.md`.
- 0.7: parser + compilador + runtime con chequeo de factibilidad; ops de producto `delta_to_lora` (con
  `transfer_verdict` sobre `energy.json`) y `contrast_vector`.
- 0.8 (2026-11-18, `docs/wsplang_0.8_spec.md`): localización y composición. `edit_column` / `edit_neurons` (una
  columna o pocas neuronas, placebo estructural de misma norma obligatorio para evaluar, `budget_params`, informe con
  parámetros tocados, manifiesto, programa para el GGUF), parches `.weightpatch.json` con `apply_patch` /
  `rollback_last` / `list_patches` / `commit_manifest` (pila en disco, hashes de base y delta verificados, nunca sobre
  la entrada), y `ternary_survival` (medida; sin escritor ternario, results §45.23). Reproduce la edición Exp 8 del MIT
  byte a byte a través del op nuevo. Sin resultado de conducta propio: cualquier afirmación sigue necesitando dev,
  placebo y sellado (§43.17).
- 0.8.1 (2026-11-18): **benchmarks como sección del lenguaje** (`docs/wsplang_0.8_bench_addendum.md`): suites estándar
  en `engine/bench_suites/` (dev-2048, dev-8192, sellados 1-3 con un solo uso, nanowsp-wspx) con hash, ops `bench_run`
  (identidad obligatoria: sha del modelo servido, del vector/LoRA, del binario y de la suite; `provenance=` verificada
  para modelos generados por WSP-Lang; clases de modelo con la NanoWSP en proceso) / `bench_compare` (pares, binomial en
  los dos correctores, placebo, banda de ruido con la base dos veces, preservación; veredicto solo para una edición
  frente a su base) / `bench_table` (registro y tabla `docs/wsplang_benchmarks.md`); `evaluate_benchmark` pasa por la
  misma ruta. Escritura ternaria para NanoWSP: `ternarize`, `imprint_ternary` (`docs/wsplang_0.8_ternary_addendum.md`;
  results §45.25: la capacidad ternaria es 1/18-1/4,5 de la float).
- 0.8.2 (2026-11-19): `model_report` lee GGUF directamente (`engine/gguf_reader.py`; SenseNova-Loop-9B §45.33).
- 0.8.3 (2026-11-20): op **`quantize`** — quantización extrema vía `llama-quantize` + `llama-imatrix` (calibración
  WikiText-2 train), manifiesto de procedencia (sha in/out, bits/weight, caveats de requant desde Q6_K) y
  `bench_run provenance=` que enlaza el modelo quantizado a su base. Op nativa sobre GGUF (`GGUF_WRITE_OPS`:
  escribe fichero NUEVO, nunca la entrada). 13 tests + smoke real 0.8B (Q6_K→Q2_K, 4.45 bits/weight por la
  embedding/head en más precisión). Fila 53 (lab/probes/quantize/PLAN.md): ¿la edición rank-1 "conclude"
  sobrevive a 2 bits? — puntuada ◐ (results §45.38): sobrevive como mecanismo de terminar, no como acierto.
- 0.8.4 (2026-11-21): op **`require`** — guarda sobre un artefacto JSON (`step require <f>.json:<ruta> >= 0.4
  [agg=all|any|min|max|mean] [why="..."]`; ruta con puntos, índices y `*`). Se juzga al validar si el fichero existe
  y ningún paso anterior lo escribe (el programa no arranca: nada de GPU); si no, al llegar. Si falla: `guard_failed`,
  los pasos siguientes `skipped_by_guard`, código de salida 1 (un chain con `set -e` también se para). Antes un paso
  fallido no paraba los siguientes. `engine/wsplang_require.py`, 20 tests primero (`tests/test_wsplang_require.py`);
  smoke real sobre `hf_cache/cc_op_v2/report.json` (`vector.merge.layers.*.cos_written_dW >= 0.8 agg=min` pasa;
  `>= 0.9` se rechaza al validar: 19 de 31 capas fallan). Idea de una revisión externa del repositorio (21 nov).
