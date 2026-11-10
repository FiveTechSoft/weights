# WSP-Lang 0.7

**Parser, compilador y runtime para programar el espacio de pesos.** Implementación: `engine/wsplang.py`
(`__version__ = "0.7.0"`; CLI `python -m engine.wsplang <programa.wspl> [--check | --run]`).

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

## 4. Reglas de seguridad vivas (runtime, no prompt)

- `FORBIDDEN_TENSORS` / `FORBIDDEN_PATTERNS` (embeddings, lm_head, output_norm): protegidos salvo
  `budgets unlock_tensors` explícito.
- Presupuestos duros: `max_modified_weights`, `max_delta_norm`, `target_parameter_fraction`; formas y dtypes deben
  coincidir 1:1 (ni añadir, ni quitar, ni cambiar la forma de tensores).
- Chequeo de factibilidad antes de ejecutar: rechaza ops no implementadas (regla 2026-11-04), ops de escritura sobre
  GGUF (redirige a `wsp.py`) y referencias a archivos inexistentes.
- Cada intervención deja manifiesto con `wsplang_version` y resultados JSON inspeccionables.

## 5. Registro de experimentos

ExperimentRecord: hypothesis, evidence, target, intervention, model_hash, benchmark_version, baseline_result,
candidate_result, placebo_result, holdout_result, conclusion ∈ {SUPPORTED, REJECTED, INCONCLUSIVE}. Un experimento
fallido queda como conocimiento reutilizable (tabla de ideas retiradas en `FOUNDATIONS.md`).

## 6. Historia de versiones

- 0.1 (2026-11-03): spec original de Antonio (8 primitivas, ExperimentRecord, criterio de aceptación); el texto
  íntegro está en el historial de git de este fichero (`git show 4bd3603:WSP-LANG.md`).
- 0.2-0.6: `docs/wsplang_0.N_spec.md` (addenda por versión) y `docs/wsplang_specification.md`.
- 0.7: parser + compilador + runtime con chequeo de factibilidad; ops de producto `delta_to_lora` (con
  `transfer_verdict` sobre `energy.json`) y `contrast_vector`.
