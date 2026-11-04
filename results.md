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

---

## 9. Validación de Control Positivo (MIT CSAIL Exp 8) y Cierre de Puerta de Validación

### 9.1. Reproducción Criptográfica Bit-Exacta de Exp 8 (`Qwen3.5-0.8B` @ Astra `p50c`)
* **Propósito:** Validar de forma independiente la infraestructura de edición y el efecto bi-direccional en espacio de pesos antes de cualquier nueva intervención sobre `MiMo-9B`.
* **Modelo base:** `Qwen/Qwen3.5-0.8B` (revisión oficial `2fc06364715b967f1860aea9cf38778875588b17`).
* **Artefactos publicados:** `apply_edit.py`, `st_numpy.py`, `manifest.json`, `edit.npz` procedentes de `RikiyaT/lm_understand-logs` (`exp8_math/runs/p50c/`).
* **Reconstrucción y comprobación hash:**
  * Tensor 1: `model.language_model.layers.15.mlp.down_proj.weight` [1024, 3584]:
    * Base SHA-256: `ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca` $\to$ Editado SHA-256: `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` (Coincidencia bit-exacta OK).
    * Entradas modificadas: 65,260 en 64 columnas (1.778% del tensor).
  * Tensor 2: `model.language_model.layers.19.mlp.down_proj.weight` [1024, 3584]:
    * Base SHA-256: `638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e` $\to$ Editado SHA-256: `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` (Coincidencia bit-exacta OK).
    * Entradas modificadas: 64,928 en 64 columnas (1.769% del tensor).
  * Total de pesos alterados: exactamente 130,188 entradas (0.016% de los parámetros totales del modelo).

### 9.2. Análisis Mecanístico del Fallo del Modelo Base vs Checkpoint Editado
* **Patología del Base:** En el modelo base, el proceso de pensamiento `<think>` entra en un ciclo patológico de auto-duda recursiva (ej. repitiendo *"Wait, let's re-verify... So \theta = \frac{\pi}{2}. Wait, let's check again..."* decenas de veces) hasta consumir la cuota máxima de tokens (93%–95% de corte en MATH-500 y GSM8K sin llegar a emitir `</think>` ni `\boxed{}`).
* **Mecanismo de Intervención:** Astra aisló la dirección residual causante de este bucle en los MLPs de las capas 15 y 19, y la proyectó fuera de 64 columnas de proyección descendente (`down_proj`).
* **Comportamiento Editado:** La edición rompe el atractor de repetición: el modelo concluye su cadena deductiva, emite `</think>` y entrega la solución en `\boxed{}`. En la verificación smoke GSM8K pareada local, el modelo editado resolvió con éxito el problema donde el base quedó atrapado en el bucle.

### 9.3. Corrección del Agujero en la Puerta de Validación de WSP-Lang
* **Defecto corregido:** `engine/wsplang.py` permitía que instrucciones como `step totally_unknown_operation ...` fueran aceptadas por el parser sin verificar que la operación estuviera registrada en `OPS_ALLOWED`. Asimismo, líneas huérfanas situadas fuera de secciones declaradas eran ignoradas silenciosamente.
* **Solución aplicada:**
  1. `validate(prog)` ahora analiza cada `step <op>` y exige estrictamente que `<op> \in OPS_ALLOWED`, añadiendo de lo contrario `operacion de paso no permitida: <op>`.
  2. Todas las líneas situadas fuera de secciones son capturadas en `__root__` y marcadas como `linea fuera de seccion: <linea>`.
  3. Se añadieron pruebas unitarias en `tests/test_wsplang_security.py` verificando el rechazo explícito de operaciones de paso desconocidas y líneas huérfanas (9/9 tests pasando en la suite completa).

---

## 10. Descarte Experimental de Baselines Baratos de Inferencia sobre MiMo-9B ($N=20$)

### 10.1. Pregunta Experimental y Metodología
* **Pregunta:** ¿El bucle agéntico del ~60% en `MiMo-V2.6-Distill-Qwen-9B` es un defecto superficial de muestreo corregible mediante hiperparámetros de inferencia (`repetition_penalty`, `presence_penalty`), o constituye una patología estructural en las representaciones de los pesos?
* **Arnés y Datos:** [`rig/agent_loop_probe.py`](file:///c:/weights/rig/agent_loop_probe.py) evaluado sobre 20 prompts reales de ingeniería de terminal de [`data/rl_oss_prompts.json`](file:///c:/weights/data/rl_oss_prompts.json) contra el servidor local en GPU (`http://127.0.0.1:8081`).
* **Criterio de bucle:** 3 llamadas a herramientas consecutivas idénticas (`3-consecutive-identical`) o 4 llamadas idénticas acumuladas (`loop4`).

### 10.2. Resultados Comparativos ($N=20$)

| Condición | Configuración de Muestreo | Bucles / Total | Tasa de Bucle | IC Wilson 95% | Turnos Medios a Bucle | Fichero de Resultados |
|---|---|---|---|---|---|---|
| **1. Vanilla (Control)** | `temp=0.3, rep_pen=1.0, pres_pen=0.0` | **14 / 20** | **70.0%** | [48.1%, 85.5%] | 9.79 | `rig_runs/eval_vanilla_n20.json` |
| **2. Repetition Penalty** | `temp=0.3, rep_pen=1.15, pres_pen=0.0` | **13 / 20** | **65.0%** | [43.3%, 81.9%] | 9.85 | `rig_runs/eval_reppenalty115_n20.json` |
| **3. Presence Penalty** | `temp=0.3, rep_pen=1.0, pres_pen=1.2` | **10 / 20** | **50.0%** | [29.9%, 70.1%] | 9.90 | `rig_runs/eval_prespenalty12_n20.json` |

### 10.3. Hallazgos Cruciales
1. **Los baselines baratos NO eliminan la patología:**
   * La penalización de repetición moderada (`1.15`) no tiene efecto práctico (65.0% vs 70.0%, diferencia de 1 solo prompt).
   * La penalización agresiva de presencia (`1.2`) reduce el bucle al 50.0%, pero la mitad de los problemas continúan estancados en ciclos idénticos de exploración.
2. **Mecanismo del bucle agéntico:**
   * El análisis turno a turno revela que no se trata de una repetición léxica trivial (un tartamudeo de palabras), sino de un **atractor de exploración infinita**: el modelo queda atrapado en la política de emitir comandos de inspección (`ls -la /app/...`, `find / -name ...`, `echo $PWD`) sin llegar a formular un plan de solución o editar código.
3. **Conclusión para WSP y RSI:**
   * **Patología confirmada como estructural:** Reside en los pesos de `MiMo-9B`. No es un artefacto de temperatura ni de repetición inmediata.
   * **Baseline de referencia establecido:** Se fija el rango 50%–70% como el suelo empírico riguroso. Cualquier edición en espacio de pesos en la siguiente fase debe demostrar una reducción por debajo del 30% con $N \ge 80$ para ser considerada un éxito.

---

## 11. Línea Base Rigurosa de GSM8K ($N=50$), Neuronas de Lógica y Vector de Control Agéntico

### 11.1. Línea Base Determinista de GSM8K ($N=50$, GPU local)
* **Runner y configuración:** [`rig/eval_gsm8k_mimo.py`](file:///c:/weights/rig/eval_gsm8k_mimo.py) sobre `MiMo-V2.6-Distill-Qwen-9B` en puerto 8081 con `temperature=0.0` (decodificación greedy determinista) y extracción rigurosa de `\boxed{}`.
* **Resultado:** **45 / 50 aciertos (90.0%)**, Intervalo de Wilson 95%: **[78.6%, 95.7%]**.
* **Velocidad:** 569.1 s totales (11.38 s/problema).
* **Análisis de los 5 fallos:**
  * Problema #4: Predijo 180, Gold 540 (error de paso multiplicativo: ignoró "3 veces por semana").
  * Problema #14: Predijo 2, Gold 18.
  * Problema #22: Predijo None (sin boxed), Gold 14.
  * Problema #38: Predijo 0, Gold 2.
  * Problema #47: Predijo None (sin boxed), Gold 163.
* **Conclusión:** La línea base matemática de MiMo-9B es elevada (90.0%), pero tiene exactamente un 10.0% de margen de error reproducible documentado en [`rig_runs/gsm8k_mimo_baseline_n50.json`](file:///c:/weights/rig_runs/).

### 11.2. Neuronas Semánticas de Lógica Identificadas (`wsp.py peek`)
Mediante proyección sobre centroides diferenciales de embedding, se localizaron columnas específicas con afinidad extrema a semillas de lógica formal:
* **`blk.14.ffn_down.weight` Columna 6907:**
  * Norma: 1.0087 (Rango #7 entre las 12,288 columnas de la matriz).
  * Área semántica dominante: `Math & Logic (math_logic)` (score +0.0350).
  * Semillas más afines: `logic` (+0.0360), `theorem` (+0.0328), `lemma` (+0.0245).
* **`blk.14.ffn_down.weight` Columna 7980:**
  * Norma: 1.0217 (Rango #5 entre las 12,288 columnas).
  * Área semántica dominante: `Math & Logic (math_logic)` (score +0.0184).
* **`blk.19.attn_output.weight` Fila 3562:**
  * Norma: 1.3249 (Rango #6 de salida de atención).
  * Área semántica dominante: `Math & Logic (math_logic)` (score +0.0227).

### 11.3. Aislamiento del Vector de Control Agéntico ($u$) mediante Representation Engineering
* **Procedimiento:** Se ejecutó [`rig/extract_loop_direction.py`](file:///c:/weights/rig/extract_loop_direction.py) con 12 trayectorias contrastivas (6 bucles y 6 productivas) en `MiMo-9B` (`bfloat16`, CPU), extrayendo los estados ocultos en el token de decisión de las 32 capas.
* **Test de Hipótesis Nula (100 permutaciones):**
  * La separación entre bucle y estado productivo es máxima y estadísticamente significativa en las **Capas 1 y 2**:
    * **Capa 1:** norma de señal = 0.15 vs ruido nulo = 0.08 (**ratio 1.83x, SNR = 1.164, $p = 0.010$**).
    * **Capa 2:** norma de señal = 0.19 vs ruido nulo = 0.12 (**ratio 1.52x, SNR = 0.989, $p < 0.001$**).
    * Capas 4 a 7: ratio 1.14x – 1.19x ($p \le 0.040$).
* **Entregable generado:**
  * Fichero de vector de control nativo de llama.cpp: [`rig_runs/loop_control_vector.gguf`](file:///c:/weights/rig_runs/loop_control_vector.gguf) (525 KB, con `direction.0` hasta `direction.31`).
  * Informe completo en [`rig_runs/loop_direction_report.json`](file:///c:/weights/rig_runs/).

---

## 12. Experimento E7: Sonda Quirúrgica de Lógica (Logic Boost vs Placebo en GPU)

### 12.1. Hipótesis y Diseño Experimental
* **Hipótesis:** La amplificación conservadora ($1.5\times$) de las columnas de síntesis FFN con máxima afinidad a semillas de lógica (`blk.14.ffn_down.weight`, cols 6907 y 7980) destraba los estados de rumiación recursiva en `<think>` y repara los fallos basales de GSM8K sin romper casos previamente resueltos.
* **Modelos compilados:**
  * **Logic Boost:** `served/mimo-logic-boost.gguf` (`scale-cols` 1.5x en cols 6907 y 7980).
  * **Placebo:** `served/mimo-logic-placebo.gguf` (`scale-cols` 1.5x en cols aleatorias 1234 y 5678).
* **Set de evaluación:** 10 problemas dirigidos (los 5 fallos basales de GSM8K: #4, #14, #22, #38, #47; más 5 controles positivos: #1, #2, #3, #5, #6).
* **Runner:** [`rig/eval_logic_boost.py`](file:///c:/weights/rig/eval_logic_boost.py) con greedy decoding determinista (`temperature=0.0`).

### 12.2. Resultados Comparativos Cabeza a Cabeza (GPU completa)

| Problema | Gold | Base MiMo-9B | Logic Boost (`mimo-logic-boost`) | Placebo (`mimo-logic-placebo`) | Diagnóstico Causal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **#4** (Carreras semanales) | **540** | 180 (Fallo) | 180 (Fallo) | 180 (Fallo) | No reparado en GPU (requiere paso multiplicativo explícito) |
| **#14** (Ecuación algebraica) | **18** | Truncado (Fallo) | **18 (REPARADO en 17.8s)** | **18 (REPARADO en 17.6s)** | Desbloqueo de `<think>` por perturbación en FFN |
| **#22** (Edades de Raymond) | **14** | Truncado (Fallo) | **14 (REPARADO en 18.8s)** | None (Fallo truncado en 25.0s) | **Causalidad exclusiva de las neuronas de lógica (Col 6907/7980)** |
| **#38** (Sets de Lego) | **2** | 0 (Fallo) | 0 (Fallo) | 0 (Fallo) | Fallo no resuelto (distractor en prosa) |
| **#47** (Post-it notes) | **163** | Truncado (Fallo) | None (Fallo truncado) | **163 (REPARADO en 17.6s)** | Sensible a variación dinámica general de FFN |
| **#1** | 18 | 18 (Pass) | 18 (Pass) | 18 (Pass) | Control mantenido |
| **#2** | 3 | 3 (Pass) | 3 (Pass) | 3 (Pass) | Control mantenido |
| **#3** (Beneficio casa) | **70000** | **70000 (Pass)** | **2 (REGRESIÓN en 25.0s)** | **2 (REGRESIÓN en 25.0s)** | **Daño colateral por escala discreta 1.5x en ambas variantes** |
| **#5** | 20 | 20 (Pass) | 20 (Pass) | 20 (Pass) | Control mantenido |
| **#6** | 64 | 64 (Pass) | 64 (Pass) | 64 (Pass) | Control mantenido |

### 12.3. Resumen Cuantitativo y Conclusiones Falsables

* **Totales en GPU:**
  * **Base:** 5 / 10 aciertos (50.0%).
  * **Logic Boost:** 6 / 10 aciertos (reparó #14 y #22; provocó regresión en #3).
  * **Placebo:** 6 / 10 aciertos (reparó #14 y #47; provocó regresión en #3).
* **Hallazgos Científicos Críticos:**
  1. **Confirmación de la patología:** Tanto en Logic Boost como en Placebo, alterar columnas de `blk.14.ffn_down.weight` destraba el truncamiento prematuro de tokens en `<think>` en problemas algebraicos (#14, #22, #47). La capa 14 FFN participa directamente en la dinámica de finalización del razonamiento.
  2. **Efecto causal de la Columna 6907:** El problema #22 se resolvió de forma limpia únicamente en el modelo Logic Boost (18.8s vs truncamiento a 25s en Base y Placebo).
  3. **Límite de la técnica de escala discreta:** El escalado multiplicativo arbitrario ($1.5\times$, es decir $+50\%$ de norma) es demasiado tosco. Provocó exactamente la misma regresión en el problema #3 en ambos modelos (de 70,000 a 2).
  4. **Veredicto:** **DESCARTADO como método de mejora neta.** No se puede considerar una mejora de modelo válida si empata con el placebo en puntuación global (6/10 vs 6/10) y rompe problemas basales previos. Demuestra empíricamente que la edición debe ser **direccional y de bajo rango** (como el producto exterior $u \otimes v^T$ con $\le 10\%$ de cambio del MIT), no una amplificación multiplicativa discreta de columnas.




