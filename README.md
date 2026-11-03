# weights

Weight-Space Programming (método del paper MIT CSAIL *Sparks of WSP* — rikiyat.github.io/blog/weight-space-programming) aplicado a **MiMo-V2.6-Distill-Qwen-9B** (GGUF Q6_K) con driver local 9B.

## Componentes

| Carpeta | Contenido |
|---|---|
| `wsp.py` | Toolkit WSP: leer pesos GGUF (info/stats/delta/markers), editar (subtract-direction rank-1 con máscara top-k, scale, blend), requant Q6_K canónico (port de `quantize_row_q6_K_ref`, 87% byte-identidad, err 6e-6), diff, serve llama.cpp, eval A/B agentic, journal+budget |
| `engine/` | WSP Challenge Engine (wsp40.json: 8 niveles/40 retos + jueces mecánicos sin self-grading + expedientes acumulativos) + servidor MCP FastMCP (`wsp_mcp_server.py`) |
| `blog_prompts/` | Batería de prompts del blog (P1 lector PASS / P5 escritor) + `eval_math.py` (GSM8K/MATH-500, sampling tipo blog) |
| `ediciones/` | Programas de edición JSON (e1 descartado documentado) |
| `retos/` | Catálogo WSP-40 + expedientes por reto |
| `data/` | GSM8K train, MATH-500, marcadores estructurales (72 tokens) |
| `rig_runs/` | Resultados de evaluaciones (baseline 6/10 loops, E1 8/10 → discard) |
| `PROTOCOL.md` | Protocolo del experimento de matemáticas (holdout ciego, placebo, no-degradación) |
| `roadmap.md` | Estado del estudio: conclusiones, análisis, pendientes |

## Reglas fundamentales
- Sin gradientes ni训练: solo ediciones directas computadas de pesos (o residuos con forward budget ≤4).
- Prohibido editar `token_embd`/`output.weight`/normas (head/lm_head).
- Presupuestos journalados: read≤200 / write≤20 / fwd-eval≤6 por sesión.
- Los holdouts (MATH-500, code, IF) NO alimentan el diseño: sellado 1-ejecución por candidato (`retos/sellados/`).
- El peso de este repo NO incluye binarios de modelos (GGUF/HF) ni caches: solo código, programas, expedientes y resultados.

## Referencias (paper y URLs que el blog referencia)

- Blog del paper: https://rikiyat.github.io/blog/weight-space-programming/
- Logs completos de los 8 experimentos (prompts, código de Astra, diffs de pesos, evaluaciones): https://github.com/RikiyaT/lm_understand-logs
  - código exacto del exp 8 (apply_edit.py + manifest): https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/code
  - dataset QA del exp 7 (sycophancy): https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy/dataset
- Modelo editado por Astra (harness núcleo): https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Demo live (zerogpu): https://huggingface.co/spaces/akhaliq/weight-space-programming-demo
