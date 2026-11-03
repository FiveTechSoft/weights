# results.md — Registro de resultados y evolución del sistema WSP

Estudio: Weight-Space Programming sobre MiMo-V2.6-Distill-Qwen-9B (GGUF Q6_K) + Challenge Engine WSP-40 + MCP propio.
Actualizado: 2026-11-03. Fuente primaria rig_runs/, journal/, sellados/.

## 1. Pruebas del toolkit wsp.py

| Prueba | Resultado | Métrica |
|---|---|---|
| Probe layout GGUF (offsets absolutos, rows=out dim) | OK | 3/3 tensores `abs==view` True |
| ssm_alpha/beta F32 (edición directa sin requant) | OK verificado | smoke: 16,384 pesos cambiados = 4 filas × 4096 (12.5%) |
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
| Confirmación multi-set de E5 (P4) | **DISCARDED**: rigs `conf-dev-*` (10 prompts del estudio) y `conf-fresca-*` (20 prompts NUEVOS, 10 general + 10 code): orig 11/20 loops (55%) vs E5 14/20 (70%); por subset: general 6/10=6/10, **code 8/10 vs 5/10 (peor)**. La señal 5/10 de E5 no se sostiene fuera de los prompts de diseño |

## 2quater. Conclusión causal de la fase E2-E5

La separación drif/comportamiento quedó demostrada TRES veces (E1, E2, E3 con dos direcciones distintas) y el confound de requant quedó **excluido** por el placebo (E4=baseline). La vía señalada por el blog (selección de neuronas por diferencia de activaciones productivo-vs-repetitivo + sustracción de la dirección u con dosis moderada) produjo la única señal provisional (E5 5/10 en dev), pero **la confirmación multi-set la NEGÓ: en 20 prompts frescos E5 es PEOR que baseline (70% vs 55% loops)** → conclusión: ninguna edición WSP de la serie E1–E5 pasa keep-if-better; queda la lección metodológica (targeting anatómico fino, placebo obligatorio, evaluación en set fresco ANTES de declarar señal).

## 2quinter. Conclusión P4 (2026-11-03)

1. E5 en conf-fresca: general-50 6/10 vs 6/10 (igual); code-0 **8/10 vs 5/10 (empeora en code)**.



2. Ligera mejora de loops en dev (4/6 vs 8/10 de orig en subset común paralizado 4 prompts; incomparable directo).
3. **Verdicto P4: la vía evaluate-cols-score + u (E5) queda DESCARTADA como mejora general del bucle; regla keep-if-better no se cumple en frescos.**

### 2quinquies. Fase P4 — confirmación multi-set de E5 (transcripciones commit 5c372a0; conteo ejecutado en linux)

| Rig | Prompts | orig | E5 |
|---|---|---|---|
| conf-dev | 10 "general" del estudio | 8/10 loops | 4/6 |
| conf-fresca | 20 NUEVOS (10 general id≥2500 + 10 code) | **11/20 (55%)** | **14/20 (70%)** |
| conf-fresca general-50 | 10 | 6/10 | 6/10 |
| conf-fresca code-0 | 10 | **5/10** | **8/10** |

| Reto | Juez | Intento 1 | Fix aplicado | Intento 2 | Estado |
|---|---|---|---|---|---|
| 01 dims | match de shapes 4 tensores | — | catálogo corregido: `blk.7.attn_q` (attn solo en capas ≡3 mod 4) | **PASS 10.0** | aprobado |
| 02 emb_vocab | vocab 248320 + emb_rows | — | — | **PASS 5.0** | aprobado |
| 03 mlp_map | n_ffn=3, inter, capas | FAIL 3.0 (inter calculado de up.shape) | ref desde `ffn_down.shape[0]` = 12288 | **PASS 5.0** | aprobado |
| 04 norms | orden L2 6 tensores | pendiente (bug en script de derivación) | — | **PASS 5.0** | aprobado |
| 05 outliers | top-10 filas output.weight (Jaccard≥0.3) | pendiente | — | **PASS 10.0** | aprobado |
| 06 delta_magnitud | familia+capas top drift | script derivó (blk.10 NO tiene attn_q: filas del tgt nuevas incluidas) | — | **PASS 5.0** | aprobado |
| 07 similarity | 2 pares de capas por perfil normativo | 3 submits fallidos | — | **PASS 10.0** | aprobado |
| 08 heads | GQA heads/kv | ERROR -2 (KeyError blk.0.attn_q) | metadata kv `attention.head_count(_kv)` → 16/4 | **PASS 10.0** | aprobado |
| 09 emb_neighbors | top-3 vecinos coseno sin fwd | bug de encoding (cp1252 en tokens): usar encoding utf-8 | — | **PASS 9.9** | aprobado |
| 10..17 | (resto L1-L3) | implementados en código; sin submit aún | — | — | pendientes |

Progreso: **9/40 retos aprobados** mecánicamente (01–09); jueces implementados (dims, emb_vocab, mlp_map, heads, norms, outliers, delta_magnitud, similarity, emb_neighbors, estructura, markers, shared_rows, lang_rows) de 40 citas del catálogo.

## 4. Evolución de la efectividad del MCP (wsp-challenge)

| Fase | Cambio | Efectividad |
|---|---|---|
| v0 (plan) | skill + CLI elegidos en quer (vs MCP puro) | 0% MCP |
| v0.1 | challenge.py CLI con retos L1-L3 + journal/budget/presupuestos | CLI 100% (engine 0.4s/estado) |
| v0.2 | catálogo wsp40.json + expedientes + 9 jueces mecánicos + sellado LOCK | retos considerables 4/40 pass |
| MCP intento 1 | `pip install mcp` trae **2.x**: FastMCP renombrado → server NO arranca (ModuleNotFoundError) | 0%: crash import |
| MCP intento 2 | pin `mcp<2` → initialize OK, 6 tools listadas | lista OK pero llamadas tool . time out: SP-Proceso interno colgado |
| MCP intento 3 | `_run` con `stdin=DEVNULL` + `CREATE_NO_WINDOW` | **100%: engine_status→4 aprobados, run_challenge(next)→reto 04 <2s** |
| Config opencode | `opencode.json` validado contra schema (`type`+`command requeridos`, arrays, cwd) | válido; pendiente REINICIO de opencode para activarlo |
| Test cliente stdio | initialize + tools/list + 2 llamadas tool reales | PASS END-TO-END |

Estado actual del MCP: **funcionando sobre stdio con cliente real de la lib; requiere reinicio de opencode** para que aparezca en las tool-calls del driver.

## 5. Infraestructura y datos

- **Nueva máquina: servidor linux** (antes Windows C:\weights). GPU 4× RTX A5000 24GB — ocupadas por `vllm serve Qwen3.8-27b` (no tocar sin permiso); ollama `flint:40b`; router nginx con TLS en 8080/8950. 533GB libres.
- Peso workspace migrado en git; **rig_runs no-commiteados se PERDIeron en la migración** (e1–e5.json solo quedan como resumen en este doc; las ediciones de la serie E quedan en `ediciones\`).
- Descarga HF reanudada: `data/download_mimo_hf.py` adaptado a linux (`hf_cache/mimo-distill-hf`; venv `.venv-hf` con huggingface_hub 2.4.0).
- `engine/ws_runner.py` verificado en linux smoke: info/describe, edit con diff tracked, run --replay (ScriptedLLM) → TODO OK.
- weightspace instalado **0.2.0** (upstream 0.3.0 publicado hoy: stop_reason, token/cost/time_budget, eventos/OTel, fix apply_patch en arrays no-contiguos, diff/patch 6x): pendiente upgrade `[openai]` p/· run con LLM real.
- MCP de la otra máquina OK por stdio; en linux pendiente de configurar/activar en opencode.

## 6. Conclusión cruzada (evolución de efectividad)

1. La edición in-place Q6_K canónica es HOY el pilar de alta fidelidad (byte-idéntico 87%, err 6e-6): la exactitud del WRITE nivela el WSP.
2. La medición de comportamiento se concretó: rig loops (55-60% baseline), caps (0/8), veredictos keep/discard con holdouts sellados: pipeline completo dato→veredicto en -24h.
3. Los jueces mecánicos de retos (sin self-grading) funcionan: **9/40 aprobados** (01–09) con progresión verificada; completar los jueces clave (~6) con datos reales para avanzar la etapa B de MiMo.
4. **Serie E1–E5 cerrada en DISCARD** (ningún edit sobrevive a frescos): para el experimento principal (PROTOCOL.md) se requiere re-diseño — la dirección u/activaciones de "repetición" no generaliza.

## 7. Pendientes próximas sesiones

- retos 10-17: jueces + derivación (04-09 CERRADOS aprobados).
- HumanEval/MBPP/IFEval: descarga + runners (holdouts code/IF del PROTOCOL.md).
- Fijar infra linux de inferencia MiMo: GGUF Q6 (copy desde la otra máquina) + llama.cpp, o HF bf16 en GPU libre (A5000 24GB si vLLM libera) — de esto depende TODO eval de math.
- Experiemento principal PROTOCOL.md con math: R0 baseline math500-subset + diag GSM8K; hipótesis desde hitos (no desde loops, ya descartados).
- Fix P5 batería blog (prompt más directo + parser alternativo de asignaciones).
- Upgrade weightspace 0.3.0 en .venv (+extra openai) en ws_runner p/ stop_reason/budgets.
