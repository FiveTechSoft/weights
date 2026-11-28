# WSP-40: retos clase PC (17-34, 39, 40)

Estos retos exigen edicion de pesos + evals de inferencia (diag, rig, holdouts
sellados). No se puntuan desde el engine en sandbox: `submit` devuelve
`clase: pc`. Se ejecutan en la maquina local (PC/GPU).

## Prerequisitos

- `llama.cpp` server en `:8950` sirviendo el GGUF candidato (los sellados
  llaman `blog_prompts/eval_math.py <alias> <set> -p <port>`).
- GGUFs: `hf_cache/mimo-gguf/MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf` (modelo) y
  `hf_cache/qwen35base/Qwen3.5_9B_Base.Q6_K.gguf` (base).
- Sets ya en repo: `data/math500_test.jsonl`, `data/gsm8k_{train,test}.jsonl`,
  `data/humaneval.jsonl`, `data/rl_oss_prompts.json`.
- Sellado 1 ejecucion por candidato+set (regla global); veredictos via
  `python engine/challenge.py verdicto <cand> keep|discard <notas>`.

## Secuencia por reto

| reto | trabajo | criterio (catalogo) |
|------|---------|---------------------|
| 17 | `wsp edit scale-rows` sobre una anomalia local + smoke set | diff>0, smoke >=90% baseline |
| 18 | `wsp edit subtract-direction` en 1 capa (target_frac 0.01-0.02) + rig | n pesos editados + diff sano + diag eval |
| 19 | `wsp edit scale-cols` 16-64 cols + diag + 1 holdout smoke | >=1 metrica impactada sin romper |
| 20 | `wsp edit blend-rows` sobre variantes synth q6 degradadas | mejora sobre variante sin tocar subej |
| 21 | 2 edits con mismo budget, comparacion ORMS | decision con argumentacion cuantitativa |
| 22 | mismo efecto con menos pesos | <50% de pesos del edit original, efecto igual |
| 23 | mejorar 1 tarea, verificar otra no cae | sellado dual no-regresivo |
| 24 | diag vs holdout gap | identificar gap |
| 25 | mejorar math sin tocar tokens multilingue (ver defecto del reto 38: 457 tokens tailandes/CJK/devanagari con cabeza movida y embedding intacta — holdout linguistico candidato) | keep-if-better + latin no-degrad |
| 26 | mismo edit, 3 seeds | std/mean del delta <= 0.4 |
| 27 | minimizar area tocada con efecto >= | minimo touched |
| 28 | mejorar math (PROTOCOL.md) | sellado triple PASS |
| 29 | mejorar code sin degradar math | sellado code PASS + math no-deg |
| 30 | IFEval >= +5 sin degradar | IFEval + no-deg |
| 31 | reducir loops del harness agentic (baseline sandbox: 90/141 fallos = 64% loops; corridas con aciertos tienen 0 loops) | loop-rate < baseline-15pts + math no-deg |
| 32 | regression no visible en 1 set | reportar regression en holdout no analizado |
| 33 | robustez adversarial de un edit vs otro | delta adversarial menor |
| 34 | 3 edits apilados | apilado == sumatorio sin regresiones |
| 39 | mejora abierta con budget propio | candidato + sellado |
| 40 | campaña diag > hip > edit > sellada > next | >=2 candidatos, keep >=1 |

## Hallazgos sandbox ya listos para reutilizar

- Drift por fila de `token_embd.weight` y `output.weight` (248320 filas):
  correlacion Pearson 0.7775 entre ambos (reto 35); 457 tokens asimetricos
  (reto 38) como set de riesgo linguistico para 25/33.
- Familia dominante de fallo en rig_runs: loops (64%), con 0 loops en corridas
  con aciertos (reto 15) — baseline para 31.
- Explicaciones marker-embedding y u_L descartadas como causa dominante de
  loops (reto 36); siguiente discriminante: drift por capa completo.
