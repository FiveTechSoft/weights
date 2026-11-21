# PREREG taskvec_mult — vectores de tarea + SVD selectivo sobre Qwen3.5-0.8B

Driver (Antonio), decisión 2026-11-09 ~09:30. Único experimento en marcha; no se abren otras líneas hasta veredicto.
Precedente aceptado: M-0 (cirugía ridge ≈ Adam en NanoWSP-tiny, 82% vs 86% holdout) y Jackrong r16 (trasplante entre
modelos reales completos). Lo NUEVO aquí: donante entrenado por nosotros (verdad conocida del ΔW completo) sobre el
0.8B, y selección espectral del ΔW en vez del rank uniforme.

## Hipótesis central

Existe un subconjunto escogido por SVD del ΔW = W_entrenado − W_base que recupera ≥90% de la mejora de la
capacidad objetivo (multiplicación) usando ≤10% de los parámetros modificados, sin destruir las capacidades
generales (GSM8K/MATH s10), y un placebo de igual norma no lo logra.

Si falla: la curva recovery(params_frac) mediana será el resultado, y el diagnóstico del espectro (¿capacidad
difusa como MiMo, o concentrada como Jackrong?) dicta si la cirugía selectiva sirve o hace falta entrenar.

## Fases

| # | Fase | Criterio (preregistrado) |
|---|---|---|
| 1 | Entrenar B (fine-tune completo bf16, AdamW, datos mult) en `hf_cache/qwen35-08b-hf` | Donante supera claramente la base en mult-test 200 (D: +0.40 abs sobre la base) |
| 2 | ΔW = B − A; espectro de energía por tensor y global (`delta_to_lora` guarda energy.json) | demostrativo |
| 3 | Adaptadores de menor rango por rank uniforme r ∈ {8,16,64,256} + lectura de frac. de parámetros | demostrativo |
| 4 | Aplicar cada adaptador a la base y evaluar | R1: algún adaptador ≤10% de parámetros cambia recupera ≥90% de la mejora del donante |
| 5 | Batería general por candidato: gsm8k s10 + math s10 mismo protocolo | R2: retención ≥0.90 en los ítems que la base resolvía; neto no-negativo |
| 6 | Placebo norm-matched del mejor adaptador (`placebo_lora.py`, seed fija) | R3: recovery del placebo < 30% del donante |
| 7 | `analyze.py` → veredicto PASS/FAIL de R1-R3 + curva recovery(params_frac) + energy.json citado | informar, sin editar criterios a posteriori |

## Reglas del protocolo

- Greedy (temperature 0), prompts idénticos al formato de entrenamiento para mult; s10 sellados existentes para
  gsm8k/math500 (protocolo base_guard equivalente pero bajo el 0.8B: bases recién almacenadas en
  `lab/runs/taskvec_mult/fast/` y apareadas contra LA MISMA semilla).
- Un solo server a la vez en la 12GB (regla de la casa). La cadena espera a que la cadena donor2 en vuelo escriba
  CHAIN DONE: no se retracta el trabajo ya en marcha.
- Filtrado anti-otra-cosa: el donante se entrena SOLO con multiplicaciones; si Donante golpea fuerte gsm8k/math
  con recovery bajo, se interpreta como coste no justificado (R2).
- No se acepta "capacidad" sin exact match del resultado final; ensayos de conteo no valen.
- Placebo antes de proclamar any R1 (ruido de cuantización en Q6_K descartado con placebo).

## Entorno

- Datos: `gen_mult.py` (rng fijo seed 20261109; sin dígito 0; 2-3 dígitos × 2-3 dígitos; parcial-dígito CoT).
- Donante: `train_donor.py` full-FT bf16 (base congelada = fuente; ΔW total; 12GB suficiente con grad ckpt).
- Adaptadores: op WSPLang `delta_to_lora` (programa `programs/taskvec_mult.wspl`) sobre base 0.8B HF.
- Evaluación: `eval_mult.py` (exact match último entero) + `fast_eval.py --bench gsm8k/math500` del close_probe.
- Conversión FM: `convert_hf_to_gguf.py` + `llama-quantize` Q6_K (mismo pipeline de la casa).
- Todo en `lab/runs/taskvec_mult/`; logs de cadena `chain.log`; resultados `fast/*.json`.
