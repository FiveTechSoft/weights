---
name: weight-space-programming
description: "Weight-Space Programming (método del blog post de MIT CSAIL 'Sparks of WSP', no revisado por pares): el LLM lee pesos crudos de GGUF, razona bloques, y escribe ediciones sin gradientes ni entrenamiento, evaluando A/B. Use ONLY when: editar pesos de un modelo (MiMo/Qwen GGUF) sin reentrenar, corregir un bucle/patología vía ediciones de pesos, experimento de weight programming, wsp.py, subtract-direction. NOT for: fine-tuning con datos, entrenamiento, cuantización general de GGUF."
---

# Weight-Space Programming (wsp.py)

Toolkit: `C:\weights\wsp.py`. Datos: `C:\weights\data\`, ediciones: `C:\weights\ediciones\`, modelos parcheados: `C:\weights\servidos\`, rig: `C:\weights\rig_runs\`, journal: `C:\weights\journal\journal.jsonl`. Infra llama.cpp: `C:\Users\Anto\llamacpp` (Qwen3.5 base Q6_K, MiMo-V2.6-Distill-Qwen-9B Q6_K, abliterado).

## WSP Challenge Engine (retos progresivos, activo)

Currículo completo: `C:\weights\challenges\wsp40.json` (el nombre del fichero es histórico: contiene **43 retos en 8 niveles**; etapa B = MiMo 9B retos 01-22). Cada reto tiene expediente acumulativo en `challenges\records\<id>.json`. Motor: `python C:\weights\engine\challenge.py <cmd>`; como MCP (tras reiniciar opencode): tools `run_challenge`, `inspect_feedback`, `experiment_history`, `sell_holdout`, `record_verdict`, `engine_status`.

Regla clave: los jueces calculan referencias SOLO al juzgar (sin self-grading) y los holdouts quedan sellados 1 ejecución (`sell_holdout` → lock). El agente no recibe golds mientras investiga.

Flujo por reto: `challenge.py reto` (brief) → investigar con wsp (READ commands, presupuestos) → escribir respuesta JSON en `challenges\resp_<id>.json` → `challenge.py submit <id> challenges\resp_<id>.json` → registrar hipótesis/intervención/resultado en `expediente` → al validar intervenciones: `sell_holdout` + `record_verdict keep|discard`. Niveles 6-8 requieren 2 verdictos keep en nivel 5.

Retos aprobados hasta ahora: 01 (dims), 02 (emb_vocab), 03 (mlp_map), 08 (heads).

## Verbatim rules (del blog post, exp 8)

> RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool (harness eval); no simular el cómputo de la red en el kernel; (c) internet access. Allowed: any direct edit of the weights that you compute.
> No editar: `token_embd`, `output.weight`, RMS norm ni decaimiento por multiplicación constante (head/lm_head prohibidos).

## Presupuestos (auto-contados en journal)

`python C:\weights\wsp.py budget` — read ≤200, write ≤20, fwd (evals) ≤6 por sesión. Los setters del blog post: sandbox ≤200–300, forward-passes ≤50.

## Workflow

1. **Read**: `wsp info <gguf>`, `tensors <gguf> <pat>`, `stats <gguf> <tensor> [rows|cols]`, `rows <gguf> <tensor> r0:r1`, `delta <base> <dist> <tensor> -k 10` (filas/tensor con mayor drift; filas repetidas entre capas = señal fuerte). `markers <gguf>` → 72 ids token estructurales en `data\markers.json`. No decodifiques tensores gigantes completos al contexto: usa stats/delta y top-k.
2. **Hypothesize**: qué bloque codifica el comportamiento a corregir (p.ej. bucle de re-llamada = dirección de repetición en MLPs del medio; filas con drift alto compartidas entre capas). El blog exp 8 usó `W' = W0 − α·outer(u, mask·(u@W0))` en `down_proj` de 2 capas, 64 columnas/neuronas, 131072 pesos (0.016%). Reproducido aquí en Qwen3.5-0.8B: MATH-500 holdout n=20 base 1/20, edit 14/20, placebo 0/20. La misma idea en un nanoGPT entrenado no supera el criterio 4 preregistrado, y el loop fix en MiMo-9B se descartó (17/40 vs 20/40 con placebo, p = 0.654).
3. **Write** el programa en `ediciones\<id>.json` (¡JSON puro, sin comentarios!):
   ops: `subtract-direction` {tensor, u_from: "svd-drift"|u, axis: rows|cols, mask: top-k, k, target_frac|alpha, touch_tol} | `scale-rows` {rows, scale} | `scale-cols` {cols, scale} | `blend-rows-toward` {rows, mix} (requiere "base"). `"u_from":"svd-drift"` usa la 1ª componente sing. del delta base→modelo (dirección pedagógica, sin datos). target_frac = ||Δ||/||W||: empieza 0.01–0.02.
4. **Apply**: `wsp edit <gguf> ediciones\<id>.json` — copia en sitio + requant Q6_K SOLO de bloques tocados (los intactos quedan byte-idénticos; port canónico verificado: 87% byte-identidad, dequant err 6e-6). Reporta n pesos editados (compara con ~0.02% del modelo).
5. **Verify**: `wsp diff <orig> <edit> <tensor> -frac 0.002` (cuenta pesos cambiados; compara contra la magnitud de tu op). Nunca diff=0 si la op debía tocar.
6. **Eval A/B**: serve baseline o editado en :8950 (`wsp serve <gguf> -p 8950 -a alias -c 16384 -n 999` — detach; log `logs\llama-<port>.log`; `--stop` para parar). `wsp eval <alias> -p 8950 -n 10` (10 prompts agentic de `C:\temp\opencode\rl_oss_prompts.json`, seed 42, hasta 12 turnos; LOOP = misma llamada normalizada ≥3 seguidas o ≥4 total). `wsp vs <baseline> <edit>` compara. **Keep-if-better**: conserva la edición solo si baja el loop-rate; si no, registra en el journal y prueba hiper (target_frac, capas, k) como NUEVO archivo editado (nunca sobreescribir en servidos).
7. **Fit de resultado**: registra en journal/sesión: los pesos editados, los números A/B y la hipótesis. Guarda el resultado duradero en AgentBrain.

## Trampas conocidas

- JSON de programa sin comentarios ni trailing commas; `wsp edit` no sobreescribe archivos existentes (borra antes manualmente si regeneras).
- GPU 12GB: un solo llama-server Q6 a la vez (-ngl 999, -c 16384); si liberas, `--stop` primero.
- El modelo emite tool calls formato antml `<function=NAME><parameter=A>v</parameter></function>` (verificado).
- `Start-Process` hijo muere si abortas ese comando del shell: lanzar serve/eval largos en comando aparte y ver resultados con logs/rig_runs luego.
- ruido de requant ≈2% por bloque tocado; edits de scale < 0.5% sobre magnitud del bloque son inmediatamente indistinguibles del ruido → target_frac mínimo 0.01 o sube k/target.
- gguf-py: `gguf.dequantize` devuelve la shape INVERTIDA (orden C de gguf): restaura con `reshape(int(t.shape[0]), int(t.shape[1]))` del ReaderTensor (ver `decomp\mimo_head.py` `_tens`).
- Anatomía de head de atención qwen35/MiMo (verificada vs llama.cpp `qwen35.cpp`): attn_q POR HEAD = 512d = **Q(256) + GATE(256) sigmoid multiplicativo** (`out *= sigmoid(x@Wgate)` antes de `wo`); GQA 16→4; k/v 256d con RMS-norm per-head; RoPE solo 64/256 dims (MRoPE multimodal, base 1e7). Decompiler mini con juez: `C:\weights\decomp\mimo_head.py` (dims/scan/decompile/judge).
- Hallazgo 2026-11-04: scan offset-aware (M_δ=Wqᵀ R_δ Wk, δ∈{1..32}) de los 128 heads de MiMo: máx c1=0.04 (~2.5× azar) — NINGÚN matcher rank-1; los heads son multi-eje, el análisis "match prev-token" del blog NO aplica a la atención destilada.
- Referencias: blog post (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026; no revisado por pares) https://rikiyat.github.io/blog/weight-space-programming/ logs con el código exacto: github.com/RikiyaT/lm_understand-logs (exp8_math/code/apply_edit.py).
