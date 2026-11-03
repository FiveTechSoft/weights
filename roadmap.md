# Roadmap WSP — Weight-Space Programming sobre MiMo-V2.6-Distill-Qwen-9B

Fecha: 2026-11-03. Estudio derivado del paper "Sparks of Weight-Space Programming" (MIT CSAIL, rikiyat.github.io/blog/weight-space-programming) — log Analysis: logs completos de los 8 experimentos en github.com/RikiyaT/lm_understand-logs; los editados en HF son de Qwen3.5-0.8B (exp 8: bucle de razonamiento).

## 1. Marco

El ajuste alcanzado corresponde a: **suficiente funcionalidad de edición** + **capacidad de mejora real del proceso** — objetivo separado para comprobarlos con datos:
- Suficiente funcionalidad técnica (edición costo HIPER): demostrado con 4 pases.
- Mejora de razonamiento/cajeta (project técnico final): par��: depende del destino.

Regla clave del paper (exp 8): WSP solo ayuda si existe una patología estructural; su control Gemma-270M (sin bucles) NO obtuvo mejora matemática.

## 2. Instanciado, verificado

### 2.1 wsp.py — toolkit de edición (skill weight-space-programming)
| Componente | Estado |
|---|---|
| info / tensors / stats / rows / delta / markers | OK verificado |
| edit in Q6_K (subtract-direction, scale rows/cols, blend-rows) | OK (copias en sitio + requant por bloques tocados) |
| requant Q6_K canónico (port de `quantize_row_q6_K_ref`) | OK: 87% byte-identidad sobre bloques reales no-editados; err decodificado 6e-6 |
| diff (-frac relativa) | OK verificado en smoke + E1 |
| serve (llama.cpp b11962, `-ngl 999 -c 16384 --jinja`) | OK: 3 cargas gestionadas (checin original/E1) |
| eval-harness agentic (port de mock_harness, format antml) | OK: baseline 6/10 loops |
| budget/journal (read≤200, write≤20, fwd≤6) | OK |
| manifiesto+fingerprint en cada edit | OK (idea del starter MCP) |

Maintenance: `C:\weights\{wsp.py, ediciones\, servidos\, rig_runs\, retos\, data\, logs\, journal\}` + skill global `~/.config/opencode/skills/weight-space-programming/SKILL.md`.

### 2.2 Diagnóstico del modelo (datos de este estudio)
- HF arch = `qwen3_5` multimodal: 32 capas texto; **atención en capas ≡3 (mod 4)**, resto GatedDeltaNet (`ssm_*`): estados clave para el diseño de intervenciones.
- Footprint de la destilación (base→distill): 99.999% de parámetros medianamente cambiar; mediana rel ~6.7%; **drift máx: ssm_beta profunda (blk.22 19%, blk.28 17.1%), ffn_gate medio (blk.17/19 ≈17%)**; **marcadores estructurales (72 tokens: `markers.json`): p95 23.3% en output.weight**.
- Baseline rig agentic (10 prompts rl_oss, seed 42, temp 0.3): **loop 6/10 (60%)** con patrón reproducible: re-llamada idéntica de comandos `ls /app/...`.
- R0 GSM8K diag (8 prompts): **8/8 (100%), 0 caps** — n.assert: nuestro modelo no HEREDA la patología matemática de Qwen3.5-0.8B (blog: 13.8% GSM8K, 93% caps). El cuello de nuestro modelo es el **comportamiento agéntico**, no math.
- E1 (subtract-direction SVD-drift, capas 17/19 ffn_down, target_frac 2%, mask top-64):
  - Pediada presa: filas con máximo drift repetidas entre capas 17 y 19 (843, 1047, 2479, 2610, 3321).
  - Resultado: **8/10 loops (peor que baseline 6/10) → DISCARDED** (semilla/idénticos prompts).
  - Conclusión experimental: la dirección del **drift de pesos** ≠ dirección del **comportamiento repetitivo**. blog: su `u` proviene de residuos observados (fwd4, salidas 16/20), NO del dif.
- Batería de páginas (MCP? no: run_battery.py P1/P5):
  - P1 (leer pesos → inferir qué calcula): **PASS** (el driver 9B razona la función sin fwd).
  - P5 (escribir pesos inline formato edits): **FAIL** técnico (el modelo no produce el bloque ```edits```: pendiente mejora prompt/parser de runtime batería).
- Serie causal E2–E5 (residuos reales + placebo) + P4: **CERRADA DISCARD** (ver results.md 2bis/2quinquies): placebo descartó el ruido de requant; la edición por activaciones solo "mejoró" en dev y EMPEORÓ en frescos (70% vs 55% loops).

### 2.3 Challenge Engine v0.2 (WSP-40)
- Catálogo formal: `retos\wsp40.json` (8 familias, 40 retos; etapa A=Qwen3.5-0.8B 01-22, B/C/D=MiMo).
- Jueces mecánicos implementados (computados SOLO al juzgar, nunca visibles antes): dims, emb_vocab, mlp_map, heads (metadata GQA), norms, outliers (Jaccard), delta_magnitud, similarity, emb_neighbors, estructura, markers, shared_rows, lang_rows + sellado de holdouts (LOCK tras 1 ejecución), historial y veredictos keep/discard.
- Retos aprobados actuales: **01–09** (9/40; 10–17 jueces implementados sin submit).
- MCP: `engine\wsp_mcp_server.py` (FastMCP stdio) + `C:\weights\opencode.json` con `mcpServers` — **requiere reinicio de opencode para activarse**; en formato CLI ya funciona.

### 2.4 Infra y datos
- datasets: `data\gsm8k_train.jsonl` (7473), `data\math500_test.jsonl` (500). Pendiente HumanEval/MBPP/IFEval.
- **Adaptador weightspace instalado**: `engine\ws_runner.py` + `.venv` (pip `weightspace`+`safetensors`, sin torch). Harness etapa A sobre F32/npz/safetensors: parches sparse (run/transcript/patch.npz) + replay de transcripciones sin tokens. API observada de leofmarciano/weightspace v0.2.0 (Result sin stop_reason). El pipeline GGUF/requant Q6_K sigue en wsp.py.
- HF checkpoint MiMo (18GB): descarga en background a `hf_cache\mimo-distill-hf` (necesario para captura de residuos).
- GPU 12GB: un llama-server Q6 a la vez (memoria 8.7GB + KV).
- Limpieza C:: liberados +121 GB (4 maestros F32 de sintetizaciones descartadas).

## 3. Conclusiones y aprendizajes de este estudio结
1. **Metodología alineada al paper**: leer (sandbox estadístico), editar (ops quirúrgicas), evaluar (budget A/B keep-if-better). Y alineado en restricciones (sin gradiente/entrenamiento/fwd-tool-only; prohibido tocar head/lm_head).
2. **WSP ≠ fórmula mágica**: sin patología estructural charofizada, no hay mejora (evidencia: blog Gemma-270M + nuestra E1).
3. **Medir el comportamiento primero**: nuestro target real no es math (100% baseline), es el bucle agéntico. Reordenar las hipótesis según patologías reales del modelo local.
4. **La dirección de edición debe deriver del comportamiento** (residuos/activaciones), no directamente del drift de pesos.
5. **Ruido de cuantización**: ±2% por bloque tocado; ediciones < 0.5% de magnitud son indistinguibles del ruido: target_frac mínimo 0.01 + exigir baseline/placebo.
6. **Placebo + no-degradación** (control del confundante cuant): necesarios por protocolo; implementados como sellado y `diff`.
7. **Expedientes acumulativos**: retos con jueces externos y memoria de hipótesis/tensores/intervenciones/resultado: capacidad para la gen-2+ del agente investigador (Knowledge Transfer Between Experiments, ref. hip 5 del marco).

## 4. Pendiente (priorizado)

### 4.0 Cierre de fase E2–E5 (2026-11-03, sesión linux)
- **P4 CONFIRMACIÓN ejecutada (commit 5c372a0) y NEGATIVA**: en 20 prompts frescos (10 general + 10 code) E5 da **14/20 loops vs 11/20 del original** (subset code: 8/10 vs 5/10) → E5 DISCARDED por keep-if-better. Detalle en results.md §2quinquies.
- Consecuencia: **la serie E1–E5 de edición agéntica queda CERRADA sin keep**. Lecciones: (1) targeting por activaciones ayuda en dev pero no generaliza; (2) evaluar frescos ANTES de declarar señal; (3) placebo (E4) queda como estándar obligatorio del PROTOCOL.md.
- **NUEVO OBJETIVO (dirección del driver): aumentar la puntuación MATH de MiMo** (R0 había dado 8/8 GSM8K diag; el margen está en MATH-500/harder). Se reactiva PROTOCOL.md con el rigor E4 (placebo) y frescos.

### 4.1 Infra linux (nueva sesión 2026-11-03-b)
| # | Tarea | Depende de |
|---|---|---|
| 1 | Inferencia MiMo en linux: copiar GGUF Q6_K de la máquina windows O eval HF (transformers/torch bf16) — GPUs A5000 ocupadas por vLLM: pedir liberación o CPU lenta | nada |
| 2 | Download MiMo HF 18GB reanudado (`.venv-hf`, `data/download_mimo_hf.py` linux) | nada |
| 3 | R0 math500-subset baseline (sample fija sellada) actualizar 100% GSM8K diag ya registrado | #1 |
| 4 | weightspace 0.3.0 upgrade (+openai extra) + ws_runner paridad de stop_reason | nada |

## 4.1bis CHECKPOINT sesión 2026-11-03-c (linux, opencode por SSH — RESUMIR AQUÍ si se corta)

Estado al cutoff:
1. **Auth opencode→vLLM RESUELTO**: el 401 era porque vLLM (:8001, de charles) exige `--api-key` y `{env:VLLM_API_KEY}` de opencode solo se resuelve en shell interactivo. Fix: key puesta literal en `~/.config/opencode/opencode.json` (provider vllm, chmod 600). Verificado end-to-end con `opencode run --model vllm/qwen3.8-27b`.
2. **Runner math500 LISTO**: `rig/math500.py` (stdlib-only). CLI: `--endpoint :8090/v1/chat/completions --model mimo --n --seed 42 --sell --temperature 0.3 --max-tokens 8192 --tag --out`. Selftest 44 asserts OK (boxed balanceado con `\{`, `\frac` sin llaves tipo `\frac43`, miles `10,\!080`, subscripts, `^\circ`, `\text`, equivalencia numérica Fractions, reducción gcd). Sellado en `rig/sellados/` (lock global-estricto: mismo seed+n reusa índices y verifica hash del dataset; otro seed/n → error, nunca sobrescribe). Determinismo via sha256 propio estable.
3. **Descargas en curso** (background, reanudables): GGUF Q6_K bartowski en `hf_cache/mimo-gguf` (~7.3GB al cutoff, target ~7.7GB); HF distill en `hf_cache/mimo-distill-hf` (~4.7GB de 18GB).
4. **Servir MiMo**: plan A = `/home/antonio/.local/lib/ollama/llama-server -m <gguf>` (verificar soporte arch `qwen3_5` multimodal, Windows usó llama.cpp b11962); plan B = compilar llama.cpp actual (10–15 min). Port sugerido :8090.
5. **Decisión GPU tomada (opus.2 del roadmap §5)**: pedir liberación de 1 A5000 → GGUF Q6 en llama.cpp full-GPU (~30 min para 50 prompts, ~5–6 h math500 completo) o HF bf16 vLLM batched (~1 h el 500). Fallback si niegan: subset-50 en CPU (~4–5 h, noche).
6. **Próximos pasos inmediatos**: (a) al terminar download GGUF → smoke serve + rig `--selftest` + rig smoke `--n 3` (sin sell) sobre MiMo; (b) pedir GPU a charles; (c) R0: `rig/math500.py --seed 42 --n 50 --sell --tag orig` (baseline sellado); (d) experimento principal PROTOCOL.md con placebo E4 + frescos.

## 4.1ter Referencia externa: github.com/csinva/weight-space-programming-autoresearch (review 2026-11-03)

Repo autónomo de escritura de pesos en transformers pequeños (tareas aritmética/NLP); modelo NUNCA entrenado, pesos escritos en forma cerrada. NO es edición de LLM reales (es stage-A puro), pero ideas útiles a importar para la gen-2 (WSP-Researcher):
- `evolve/program.md`: instrucción "LOOP FOREVER, no pausar a humano, ≥20 iteraciones aunque dé 100%" + ideas guiadas por papers de mech-interp. Importable como sección del skill.
- Ledger plano `results/overall_results.csv` (task,accuracy,status,n_params,description) tras CADA intento: mejor memoria cross-run que nuestro journal JSON (añadir export CSV al journal).
- Snapshot lib (`interpretable_transformers_lib/<nombre>.py` por intento): equivalente a nuestro `ediciones/` — ya lo cumplimos.
- Run-folder aislado con src/ de solo-lectura (symlinks): pattern para rig y retos (evita self-grading).

| # | Tarea (histórico, pendiente) | Depende de |
|---|---|---|
| 5 | Fix P5 batería blog (prompt más directo + parser tolerante del bloque edits) | nada |
| 6 | WSP-40: jueces 10-17 + bajar HumanEval/MBPP/IFEval + runners | nada |
| 7 | Etapa A (Qwen3.5-0.8B): download HF (~1.6GB) + parque de retos 01-22 con `ws_runner` | #6 parcial |
| 8 | Experimento principal PROTOCOL.md (math) con placebo, triple holdout y seed multiset | #3 + #4.0 |
| 9 | Extractores de "principio" (knowledge transfer) desde expedientes → gen-2 WSP-Researcher | #8/#6 |
| 10 | Editar opencode MCP para la máquina linux (config + reinicio) | reinicio |

## 5. Riesgos / trampas activos
- **linux**: 4× A5000 OCUPADAS por vllm Qwen3.8-27b (22.4/24.5GB c/u) — no lanzar MiMo en GPU sin revisar memoria.
- Git: si el remoto se adelanta (la otra máquina también empuja), `git pull --rebase` con stash antes de push.
- millones de GB: un solo server a la vez en 12GB (stop antes de arrancar otro) [regla de la máquina windows].
- antml formato del agentic harness |tool-call投机: si cambia el modelo, revisar parser.
- no borramos gold-holdouts hasta la primera evaluación sellada final (protocolo).
- Q6_K requant detail: siempre regenerar play党和国家前 a concluyentes.
- opencode MCP exige reinicio después de editar OPENCEN.

## 6. Referencias de archivos del estudio
- Paper + logs WSP método (paper baseline): rikiyat.github.io/blog/weight-space-programming/; github.com/RikiyaT/lm_understand-logs (exp8_math/code/apply_edit.py + manifest).
- Protocolo del 1er experimento matemático: `PROTOCOL.md`.
- Pension: `ediciones\e1.json` (descartado) + `ediciones\e1-smoke.json`; manifiestos en `servidos\*.manifest.json`.
- Toolkit (motor WSP): `wsp.py`; engine retos: `engine\challenge.py` + `engine\wsp_mcp_server.py` + `retos\wsp40.json`.
- Skill: `~/.config/opencode/skills/weight-space-programming/SKILL.md`.

# WSP-Lang 0.1 � lenguaje experimental para programar el espacio de pesos (propuesta Antonio, 2026-11-03)

Hip�tesis central: un LLM contiene un programa computacional impl�cito en sus pesos; descubrimos primitivas y las combinamos para producir cambios verificables. Especificaci�n completa en WSP-Lang 0.1 (texto de Antonio; resumen operativo en `retos/gramatica_wsp.json`).

Modelo: M=(A,W,t); programa P=(o1..on); W'=P(W). 8 primitivas ya IMPLEMENTADAS en el runtime:
inspect (wsp info/stats), observe (capture_residual/heads/activaciones), compare (delta vs rig), hypothesize (expedientes, agente), localize (delta/markers/activation score), intervene (subtract-direction/scale/cols_mask: **project** de la gram�tica), evaluate (rig+eval_math+sellar: **test**), rollback (servidos)}

Sistemas de tipos: Model/Tensor/Activation/Direction/Behavior/Hypothesis/Intervention/Benchmark/Experiment/Result.
Matem�tica: escala W' = W+ad; proyecci�n W' = W - a�W�ddT (normalizada); intervenci�n de ACTIVACIONES h' = h - a�ddT�h (comprobar hip�tesis sin editar); composici�n P2�P1 NO conmutativa (registrar orden).
Seguridad (runtime, no prompt): identidad inmutable (sha256), target whitelist, dimensiones/tipos/limites de magnitud, budget, reversibilidad, confirmaci�n ciega (P4).
Criterio de aceptaci�n: ?B>0 Y ?R_i = -e_i para cada regresi�n i Y modified_parameters = B_max Y evaluaci�n independiente separada (diagn�stico valida, confirmaci�n sella).

Registro de experimentos: ExperimentRecord{hypothesis, evidence, target, intervention, model_hash, benchmark_version, baseline_result, candidate_result, placebo_result, holdout_result, conclusion?{SUPPORTED, REJECTED, INCONCLUSIVE}} � un experimento fallido ES conocimiento (ya 5 REJECTED con lecciones).

WSP-Lang 0.2 (futura): descubrimiento de primitivas por el agente, composici�n autom�tica de ops compatibles, meta-programaci�n del proceso investigador (RSI-shaped).

SIGUIENTE PIEZA A CONSTRUIR: parser + ejecutor m�nimo de WSP-Lang en el MCP: interprete peque�o de programas tipo E2 que validan cada op, generan manifiesto y resultados autom�ticamente.

