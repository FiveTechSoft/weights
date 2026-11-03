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

## 3. Challenge Engine WSP-40 (jueces mecánicos)

| Reto | Juez | Intento 1 | Fix aplicado | Intento 2 | Estado |
|---|---|---|---|---|---|
| 01 dims | match de shapes 4 tensores | — | catálogo corregido: `blk.7.attn_q` (attn solo en capas ≡3 mod 4) | **PASS 10.0** | aprobado |
| 02 emb_vocab | vocab 248320 + emb_rows | — | — | **PASS 5.0** | aprobado |
| 03 mlp_map | n_ffn=3, inter, capas | FAIL 3.0 (inter calculado de up.shape) | ref desde `ffn_down.shape[0]` = 12288 | **PASS 5.0** | aprobado |
| 08 heads | GQA heads/kv | ERROR -2 (KeyError blk.0.attn_q) | metadata kv `attention.head_count(_kv)` → 16/4 | **PASS 10.0** | aprobado |
| 04 norms | orden L2 6 tensores | pendiente (bug en script de derivación) | — | — | en_curso |
| 05 outliers | top-10 filas output.weight (Jaccard≥0.3) | pendiente | — | — | en_curso |
| 06 delta_magnitud | familia+capas top drift | script derivó (blk.10 NO tiene attn_q: filas del tgt nuevas incluidas) | — | — | en_curso |
| 07 similarity | 2 pares de capas por perfil normativo | pendiente | — | — | en_curso |
| 09 emb_neighbors | top-3 vecinos coseno sin fwd | bug de encoding (cp1252 en tokens): usar encoding utf-8 | — | — | en_curso |
| 04..17 | (resto L1-L3) | implementados en código; sin submit aún | — | — | pendientes |

Progreso: 4/40 retos aprobados mecánicamente; 9 jueces implementados (dims, emb_vocab, mlp_map, heads, norms, outliers, delta_magnitud, similarity, emb_neighbors, estructura, markers, shared_rows, lang_rows) de 40 citas del catálogo.

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

- Descarga HF checkpoint MiMo (18GB): **17.55GB descargados** OK (hf_cache\mimo-distill-hf). transformers 5.12.1 + torch 2.15-CPU preparados para E2 (residuos).
- Limpieza C: +121GB libre (4 maestros F32 de sintetizaciones desc.)
- Repo: https://github.com/FiveTechSoft/weights — commits 587ccc0/615c4ab/7076cbf (31 archivos, 10k líneas; sin binarios).
- AgentBrain: sesión guardada `mem-wsp-mimo-challenge-engine.json` (índice actualizado 13).

## 6. Conclusión cruzada (evolución de efectividad)

1. La edición in-place Q6_K canónica es HOY el pilar de alta fidelidad (byte-idéntico 87%, err 6e-6): la exactitud del WRITE nivela el WSP.
2. La medición de comportamiento se concretó: rig loops (60% baseline), caps (0/8), veredictos keep/discard con holdouts sellados: pipeline completo dato→veredicto en -24h.
3. Los jueces mecánicos de retos (sin self-grading) funcionan: 4/40 aprobados con progresión verificada; completar los jueces clave (~6) con datos reales para avanzar la etapa B de MiMo.

## 7. Pendientes próximas sesiones

- retos 04-09: valida gen script con encoding utf-8.
- retos 10-17: jueces + derivación.
- HumanEval/MBPP/IFEval: descarga + runners de nivel 5 (eval con presupuesto todo el_TODO 5).
- E2 capture_residual.py → dirección u de repetición real → candidato E2 (con residuos reales: método fiel al blog).
- Fix P5 batería blog (prompt más directo + parser alternativo de asignaciones).
- Realizar el reinicio de opencode + confirmación de tools MCP dentro de las sesiones del agente driver.
