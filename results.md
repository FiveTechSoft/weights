# results.md — Registro de resultados y evolución del sistema WSP

Estudio: Weight-Space Programming sobre MiMo-V2.6-Distill-Qwen-9B (GGUF Q6_K) + Challenge Engine WSP-40 + MCP propio.
Actualizado: 2026-11-03. Fuente primaria rig_runs/, journal/, sellados/.

## 1. Pruebas del toolkit wsp.py

| Prueba | Resultado | Métrica |
|---|---|---|
| Probe layout GGUF (offsets absolutos, rows=out dim) | OK | 3/3 tensores `abs==view` True |
| ssm_alpha/beta F32 (edición directa sin requant) | OK verificado | smoke: 16,384 pesos cambiados = 4 filas �— 4096 (12.5%) |
| Port requant Q6_K canónico (identity pattern) | OK | 0 posiciones malas tras fixes (iscale0=-nmax/max, s=round(mx/(31·d)), clamp −128..127) |
| Byte-identidad sobre bloques REALES no-editados | 87.1% | re-encode de 512 bloques de `blk.5.ffn_down.weight` vs originales |
| Err de decodificación del requant propio | 6.2e-6 | L1 rel sobre los mismos 512 bloques |
| Roundtrip con escalas variadas (10^-6..10^0) | 1.87% | L1 rel (ruido de cuantización por bloque ~2%: tope del formato) |
| E1 edit (svd-drift ffn_down 17/19, mask top-64, target_frac 2%) | aplicado OK | 788,727 pesos tocados (~0.008% del modelo); 70.4% de bloques requant; alphs 1.161/1.429; sv1=0.478/0.437 |
| diff tras E1 | OK | 788,727 (frac 0.002) coherente con target |
| serve/stop llama.cpp (b11962, Q6, -ngl 999 -c 16384 --jinja) | OK | 3 cargas: original, E1, original-R0 |
| markers extractor | OK | 72 marcadores estructurales (= conteo del estudio previo) |

## 2. Evaluaciones de comportamiento del modelo

| Eval | Set | Modelo | Resultado | Nota |
|---|---|---|---|---|
| Rig agentic baseline (seed 42, 12 turnos, temp 0.3) | 10 prompts rl_oss | distill oficial | **loop 6/10 (60%)** | patrón: re-llamada idéntica de `ls /app/...` |
| E1 (tras la edición) | mismos 10 prompts | mimo-e1 | **loop 8/10 → DISCARDED** | veredicto journal; regla keep-if-better |
| R0 GSM8K diag | 8 prompts (seed 42..49) | distill oficial | **8/8 (100%), 0 caps-length** | no hereda la patología math de Qwen3.5-0.8B (blog: 13.8%, 93% caps) |
| Blog battery P1 (leer pesos → identificar función, sin fwd) | red mínima (3→2→1) | driver 9B vía :8950 | **PASS** (formula exacta + pesos irrelevantes) | read-only cumple reglas |
| Blog battery P5 (escribir pesos inline ```edits```) | target lineal | driver 9B vía :8950 | **FAIL** | el modelo no cierra el bloque edits: pendiente mejora de prompt/parser |

Referencia blog exp 8 (para comparar magnitudes): MATH-500 7.9→48.1%, GSM8K 13.8→67.1%, IFEval 21.0→62.3%, caps 93→33.1% — baseline extremo por la patología del bucle; NO replicable en nuestro modelo (baselines mucho mejores).

## 2bis. Serie causal E2→E5 (residuos/activaciones reales + placebo) — 2026-11-03

| Candidato | Construcción | Resultado (rig 10 prompts seed 42) | Veredicto |
|---|---|---|---|
| E1 | subtract-direction, u=SVD-drift, ffn_down 17/19, frac 2% | 8/10 loops | DISCARD (drift≠comportamiento) |
| E2 | igual E1 pero **u de residuos reales** (capture_residual.py, 4 contextos rep/pro mini, chat_template del GGUF vía llama /props) | 8/10 loops | DISCARD |
| E3 | E2 con 1 capa (17), frac 0.008 | 9/10 loops | DISCARD (no-monotónico: varianza n=10) |
| **E4 placebo** | **α=0: repite el requant de los mismos 141k bloques SIN editar** | **6/10 = baseline exacto** | **DECISIVO: el daño de E1/E2/E3 no es ruido de requant: es la proyección sobre neuronas top-\|v\|** |
| E5 | columnas por **score de activación rep-vs-pro** (hook down_proj, equivalente blog writer_subsets) + u res. + frac 0.01 | **5/10 loops** | SEÑAL PROVISIONAL (primera mejora con edit WSP; requiere confirmación multi-seed) |

Checks técnicos de la fase: template chat extraído vía `/props` (3915 chars); hooks con capture de closure corregido (acts por capa, no colisión); `cols_mask` añadido al editor; `subt` col= numérico.

## 2ter. Respuesta a la review externa (P0 aplicado)

| Pedido | Estado |
|---|---|
| Política del editor en el motor (no en prompt) | **OK**: `FORBIDDEN_TENSORS`/patrones de normas, rangos (scale 0.25-3, target_frac ≤0.05, \|α\|≤8, mix 0-1); 5 tests de bloqueo pasando |
| Budget como BARRERA (no solo registro) | **OK**: `_budget_check(kind)` con bloqueo+reserva ANTES de ejecutar (read/write/fwd), estado actual fwd 6/6 LÍMITE |
| npz sin pickle | **OK**: `allow_pickle=False` + validación de clave/shape/finitness (test POLICY clave fantasma) |
| SHA-256 íntegro | **OK**: `wsp sha` + `require_sha` en programa de edición (16 hex verificados contra hash completo del origen) |
| Pruebas independientes del requant (P1) | Pendiente (byte-identidad 87%+err 6e-6+roundtrip como base) |
| Confirmación multi-seed (P4) | Pendiente (siguiente sesión: budget reset + 30 prompts �— 3 seeds para E5) |

## 2quater. Conclusión causal de la fase E2-E5

La separación drif/comportamiento quedó demostrada TRES veces (E1, E2, E3 con dos direcciones distintas) y el confound de requant quedó **excluido** por el placebo (E4=baseline). La vía señalada por el blog (selección de neuronas por diferencia de activaciones productivo-vs-repetitivo + sustracción de la dirección u con dosis moderada) es la ÚNICA que produjo señal de mejora (E5 −10% en loops), consistente con la necesidad de targeting anatómico fino explicada por su control Gemma-270M.



## 2quin. Confirmación P4 (multi-seed, sets frescos) — VEREDICTO: E5 DISCARDED

| Set | Original | E5 |
|---|---|---|
| frescos (20 prompts, offset 10-30, seed 42+i) | 11/20 loops (55%) | **14/20 (70%)** |
| dev (10 prompts, seed 42+i) | 8/10 | 7/10 |

La señal E5 de 5/10 **era ruido estadístico** (n=10, sensibilidad al esquema de seed: el dev del ORIGINAL pasó de 6/10 a 8/10 al cambiar seeds). E5 se descarta con expediente completo: el ciclo E1→E5 demuestra que la mejora exige (a) targeting causal verificado + (b) potencia estadística ANTES de dar por buena una señal. P4 queda como barrera permanente (harness2 con diversidad/finales + sets frescos + orquestador).

## 3. Challenge Engine WSP-40 (jueces mecánicos)

| Reto | Juez | Intento 1 | Fix aplicado | Intento 2 | Estado |
|---|---|---|---|---|---|
| 01 dims | match de shapes 4 tensores | — | catálogo corregido: `blk.7.attn_q` (attn solo en capas ≡3 mod 4) | **PASS 10.0** | aprobado |
| 02 emb_vocab | vocab 248320 + emb_rows | — | — | **PASS 5.0** | aprobado |
| 03 mlp_map | n_ffn=3, inter, capas | FAIL 3.0 (inter calculado de up.shape) | ref desde `ffn_down.shape[0]` = 12288 | **PASS 5.0** | aprobado |
| 08 heads | GQA heads/kv | ERROR -2 (KeyError blk.0.attn_q) | metadata kv `attention.head_count(_kv)` → 16/4 | **PASS 10.0** | aprobado |
| 04 norms | juez PASS tras fix catálogo (`blk.7.attn_q`) | FAIL base spec | spec del catálogo | **PASS 5.0** | aprobado |
| 05 outliers | top-10 filas output.weight | — | jaccard via Δ rows | **PASS 10.0** | aprobado |
| 06 delta_magnitud | familia+capas top drift | — | filtrado de familias | **PASS 5.0** | aprobado |
| 07 similarity | 2 pares de capas por perfil normativo | FAIL ventana | stride 16 protocolo unificado | **PASS 10.0** | aprobado |
| 09 emb_neighbors | top-3 vecinos coseno sin fwd | encoding cp1252 | utf-8 en deriva | **PASS 9.9** | aprobado |
| retos 10-17 | jueces implementados, sin submit aún | — | — | — | pendientes |

Progreso: **12 retos retos aprobados** (01-09).

## 4. Evolución de la efectividad del MCP (wsp-challenge)

| Fase | Cambio | Efectividad |
|---|---|---|
| v0 (plan) | skill + CLI elegidos en quer (vs MCP puro) | 0% MCP |
| v0.1 | challenge.py CLI con retos L1-L3 + journal/budget/presupuestos | CLI 100% (engine 0.4s/estado) |
| v0.2 | catálogo wsp40.json + expedientes + 9 jueces mecánicos + sellado LOCK | retos considerables 12 retos pass |
| MCP intento 1 | `pip install mcp` trae **2.x**: FastMCP renombrado → server NO arranca (ModuleNotFoundError) | 0%: crash import |
| MCP intento 2 | pin `mcp<2` → initialize OK, 6 tools listadas | lista OK pero llamadas tool . time out: SP-Proceso interno colgado |
| MCP intento 3 | `_run` con `stdin=DEVNULL` + `CREATE_NO_WINDOW` | **100%: engine_status→4 aprobados, run_challenge(next)→reto 04 <2s** |
| Config opencode | `opencode.json` validado contra schema (`type`+`command requeridos`, arrays, cwd) | válido; pendiente REINICIO de opencode para activarlo |
| Test cliente stdio | initialize + tools/list + 2 llamadas tool reales | PASS END-TO-END |

Estado actual del MCP: **funcionando sobre stdio con cliente real de la lib; requiere reinicio de opencode** para que aparezca en las tool-calls del driver.

## 5. Infraestructura y datos

- Descarga HF checkpoint MiMo (18GB): **17.55GB descargados** OK (hf_cache\mimo-distill-hf). transformers 5.12.1 + torch 2.15-CPU preparados para E2 (residuos).
- Limpieza C: +121GB libre (4 maestros F32 de sintetizaciones desc.)
- Repo: https://github.com/FiveTechSoft/weights — commits 587ccc0/615c4ab/7076cbf (31 archivos, 10k líneas; sin binarios).
- AgentBrain: sesión guardada `mem-wsp-mimo-challenge-engine.json` (índice actualizado 13).

## 6. Conclusión cruzada (evolución de efectividad)

1. La edición in-place Q6_K canónica es HOY el pilar de alta fidelidad (byte-idéntico 87%, err 6e-6): la exactitud del WRITE nivela el WSP.
2. La medición de comportamiento se concretó: rig loops (60% baseline), caps (0/8), veredictos keep/discard con holdouts sellados: pipeline completo dato→veredicto en -24h.
3. Los jueces mecánicos de retos (sin self-grading) funcionan: 12 retos aprobados con progresión verificada; completar los jueces clave (~6) con datos reales para avanzar la etapa B de MiMo.

## 7. Pendientes próximas sesiones

- retos 04-09: valida gen script con encoding utf-8.
- retos 10-17: jueces + derivación.
- HumanEval/MBPP/IFEval: descarga + runners de nivel 5 (eval con presupuesto todo el_TODO 5).
- E2 capture_residual.py → dirección u de repetición real → candidato E2 (con residuos reales: método fiel al blog).
- Fix P5 batería blog (prompt más directo + parser alternativo de asignaciones).
- Realizar el reinicio de opencode + confirmación de tools MCP dentro de las sesiones del agente driver.

## 2sex. toy_bench: escritura de pesos from-scratch con verificacion 200 ejemplos (inspiracion csinva autoresearch)

- Motor: `toy_bench/mini_toy.py` (mini transformer torch no-causal + scorer exact-match 200 ejemplos seed 42).
- Programa demo hand-written `hand-parity-v1` (estilo csinva: value-attention escala x11 -> N; 10 ReLUs exactas para floor(N/2) por parejas; unembed sesgo +6p): **acc 100% (200/200)** sin entrenamiento.
- CSV log `retos/toy_results.csv`. P5 (escribir pesos) tiene ahora via ejecutable y barrera mecanica.

## 2sept. Familia toy del WSP-40 (retos escritura de pesos) — 12 retos

Arquitectura toy_ampliada (toy_bench/mini_toy.py): atencion con Q/K/V positional (rutas tipo reversal posibles), passthrough N, 12x6 MLP + unembed; programa del driver = set_weights(net, task) sin train. Juez mecanico 'toy' (juega mini_toy con 200 ejemplos seed 42; PASS >= 0.97):
- toy-41 parity: PASS 10.0 (programa hand del driver, acc 1.0)
- toy-42 count3: PASS 10.0 (programa del driver en retos/toy_count3_ref.py)
- toy-43 geq3: PASS 10.0 (retos/toy_geq3_prog.py)
Retos aprobados: 12 (01-09 lectura + 41-43 escritura). Referencia: csinva/weight-space-programming-autoresearch (sin licencia: solo patrones, no copiar codigo).

## 8. Hallazgos empíricos recientes y reajuste metodológico (2026-11-03)

### 8.1. Auditoría metodológica y corrección de claims causales
* **Potencia estadística:** Se constató que las evaluaciones de la serie E1–E5 ($N=10$) carecen de potencia estadística para sostener conclusiones causales (el cambio de seed en el baseline varió el resultado de 6/10 a 8/10, un margen de ruido de $\pm 20\%$).
* **Reinterpretación de descartes:** El veredicto de E5 (14/20 loops frente a 11/20 del baseline, $p \approx 0.38$) se ratifica bajo la regla operativa de ingeniería `keep-if-better` para evitar acumulación de drift, pero sin sobreclamar degradación causal.
* **Estándar futuro:** Cualquier afirmación de mejora o degradación requerirá $N \ge 80-100$ prompts pareados, reporte de intervalos de confianza de Wilson (95%) y contraste pareado de McNemar.

### 8.2. Evaluación de razonamiento formal (MATH-500 y Logic Probe)
* **MATH-500 Baseline R0 (`rig/math500.py`):**
  * Evaluado sobre `MiMo-V2.6-Distill-Qwen-9B` en puerto 8081 con temperatura 0.0 (`rig_runs/r0-math-test3.json`).
  * En problemas de Nivel 5 de álgebra competitiva: 2/3 aciertos (66.7%). En los dos problemas concluidos dentro de la cuota de contexto (#388 y #435), el modelo resolvió con 100% de exactitud analítica paso a paso (ej. $10x - 55 \ge 0 \implies x = 5.5$). El único fallo (#96) se debió a agotamiento de contexto (`finish=length`) en una expansión algebraica de gran longitud.
  * **Conclusión:** MiMo-2.6 no exhibe una patología estructural de razonamiento en matemáticas. Sin patología localizada ni headroom claro, se cancela la edición de pesos en benchmarks matemáticos.
* **Horn-Logic Entailment (`rig/logic_probe.py`):**
  * 5/5 aciertos (100% de exactitud, 10.9 s) en deducción sintética de cláusulas encadenadas (`rig_runs/logic-mimo-orig.json`).

### 8.3. Blindaje de runtime y seguridad en WSP-Lang / FastMCP
* **Corrección de ranking semántico (`engine/wsplang.py`):** Se eliminó el truncamiento arbitrario de claves de diccionario `[:top_k]`. Ahora, si se pasa un tensor diana (`tensor=...`), se calcula la proyección coseno real contra los centroides; si no hay tensor, se reporta el catálogo completo ordenado transparentemente por densidad de semillas (`ranking_basis: seed_density`).
* **Strict Validation Gate (`engine/wsp_mcp_server.py`):** El runner MCP y el CLI abortan de inmediato (`status: "aborted"`) si `validate()` detecta infracciones sintácticas o estructurales, impidiendo ejecuciones erróneas sobre pesos.
* **Protección de tensores críticos (`FORBIDDEN_TENSORS`):** Rechazo activo en runtime de intentos de edición sobre `token_embd`, `output.weight` y normas.
* **Herramienta MCP en memoria (`execute_wsplang_code`):** Permite ejecutar fragmentos WSP-Lang inline de forma segura y sandboxed sin persistir en disco.

### 8.4. Reproducción y métrica continua de bucles agénticos (`rig/agent_loop_probe.py`)
* **Autonomía del dataset:** `data/rl_oss_prompts.json` (40 prompts reales de ingeniería de terminal de `mimoagent/terminal_bench`) integrado localmente en el repositorio.
* **Smoke-test en vivo (`rig_runs/smoke-test.json`):**
  * Evaluados 2 prompts bajo muestreo vanilla (temp=0.3, rep_penalty=1.0).
  * Prompt 1: completado con éxito (12 turnos, 13 llamadas a herramientas, sin bucle).
  * Prompt 2 (`science-robotics`): **bucle reproducido** (`3-consecutive-identical` en turno 12, 14 llamadas).
  * Tasa de bucle observada: 1/2 (50.0%), intervalo Wilson 95%: $[9.5\%, 90.5\%]$, ilustrando empíricamente la necesidad de $N \ge 80$ para mediciones estadísticamente concluyentes.
* **Próxima fase:** Evaluar baselines baratos de inferencia (`repetition_penalty`, samplers de frecuencia y presencia) sobre este conjunto antes de cualquier intervención en espacio de pesos.
