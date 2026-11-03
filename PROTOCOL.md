# Protocolo WSP-MiMo: mejora de razonamiento matematico (v1)

Objetivo: producir UNA mejora en matematicas de MiMo-V2.6-Distill-Qwen-9B via WSP que sea
verificable, reproducible y explicable, SIN degradar programacion ni seguimiento de instrucciones.
(driver humano: Antonio; ejecutor cientifico: agente opencode con wsp.py)

## Reglas de oro
1. HOLDOUT CIEGO: math500 y holdout-code/instrucciones NO se usan para disenar la modificacion.
   Solo se evaluan al final (una vez por candidato o al cerrar la iteracion).
2. DIAGNOSTICO: los errores se analizan SOLO sobre el set diag (GSM8K train muestra fija.
   8 primeros task ids con seed 42+i).
3.KEEP-IF-BETTER + HOLD-OUT: se conserva la modificacion solo si diag mejora Y los 3 holdouts
   no caen (math >= baseline, code/IF >= baseline-2pts).
4. PLACEBO para discriminar ruido de cuantizacion: misma forma de edicion (mismo target_frac,
   mismas capas) sobre un tensor CONTROL (p.ej. una capa ffn_down no implicada: blk.8) ->
   si el placebo "mejora" igual, el efecto era ruido/requant y se descarta la hipotesis.
5. Presupuestos (journal): read<=200, write<=20, fwd-evals<=6/sesion. Sampling blog:
   temp 0.6 top_p 0.95 top_k 20 max_tokens 2048 seed fija por problema.
6. Cada candidato => archivo nuevo en servidos (nunca sobreescribir) + manifest con fingerprint.

## Sets
- diag: data\gsm8k_train.jsonl primeros 8 (seed 42..49)
- holdout math: data\math500_test.jsonl muestra fija (los primeros n segun eval, nunca para disenar)
- holdout code/IF: C:\temp\opencode\rl_oss_prompts.json subsets code/general/cyber

## Metrica
GSM8K: match numerico (####); MATH-500: \boxed{} y fallback numerico; exact match normalizado.

## Procedimiento
R0. Baseline: eval diag + holdouts del modelo ORIGINAL (servidos none). Registrar.
R1. Diagnostico: clasificar fallos del set diag (aritmetica/logica/formato/bucle/formato
    respuesta). Identificacion mecanica: filas/tensores con drift compartido entre capas
    (wsp delta -k), outliers p95 (wsp stats), tokens de bucle (markers).
R2. Hipotesis: formular 1 hipotesis mecanica (componente X codifica comportamiento Y).
R3. Candidato E# : spot edicion en ediciones\ (subtract-direction/scale/blend),
    wsp edit -> diff -> placebo -> eval diag. Solo si diag mejora: eval holdouts.
R4. Veredicto: keep/discard + registro (journal + AgentBrain). Siguiente hipotesis.

## Estado
- R0 baseline pendiente (tras cerrar E1 bucle agentic).
- E1 bucle agentic: en curso (tecnico: valida pipeline complete).
