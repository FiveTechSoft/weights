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
  - Patrón observado: filas con máximo drift repetidas entre capas 17 y 19 (843, 1047, 2479, 2610, 3321).
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

## 3. Conclusiones y aprendizajes de este estudio
> **Desde 2026-11-10 el fundamento consolidado está en [`FOUNDATIONS.md`](FOUNDATIONS.md)** (tesis T1-T8 con estado,
> evidencia, qué las refutaría, e ideas retiradas). Los puntos de abajo son históricos (2026-11-03).

1. **Metodología alineada al paper**: leer (sandbox estadístico), editar (ops quirúrgicas), evaluar (budget A/B keep-if-better). Y alineado en restricciones (sin gradiente/entrenamiento/fwd-tool-only; prohibido tocar head/lm_head).
2. **WSP ≠ fórmula mágica**: sin patología estructural charofizada, no hay mejora (evidencia: blog Gemma-270M + nuestra E1).
3. **Medir el comportamiento primero**: nuestro target real no es math (100% baseline), es el bucle agéntico. Reordenar las hipótesis según patologías reales del modelo local.
4. **La dirección de edición debe deriver del comportamiento** (residuos/activaciones), no directamente del drift de pesos.
5. **Ruido de cuantización**: ±2% por bloque tocado; ediciones < 0.5% de magnitud son indistinguibles del ruido: target_frac mínimo 0.01 + exigir baseline/placebo.
6. **Placebo + no-degradación** (control del confundante cuant): necesarios por protocolo; implementados como sellado y `diff`.
7. **Expedientes acumulativos**: retos con jueces externos y memoria de hipótesis/tensores/intervenciones/resultado: capacidad para la gen-2+ del agente investigador (Knowledge Transfer Between Experiments, ref. hip 5 del marco).

## 4. Pendiente (priorizado)

### 4.0sexies PRIORIDAD (decisión del driver 2026-11-09): trasplante de mejoras como producto (runtime WSP-Lang 0.7.0)

**Lo único demostrado:** WSPLang traslada la mejora de un fine-tune a la base como un adaptador pequeño, sin
entrenar (Jackrong → MATH 6→30/36, GSM8K 9→37/48), y `energy.json` predice antes de evaluar si se podrá (rango bajo sí,
difuso como MiMo no). Crear capacidades sin donante: 5 intentos, todos fallan; **no se sigue por ahí**.

**Objetivo:** un op que, dado base + fine-tune (local o en Hugging Face, sin copiarlo al disco):
1. mide con `energy.json` si el cambio cabe en un adaptador (veredicto antes de evaluar);
2. si cabe, escribe el adaptador (+ GGUF);
3. lo valida solo con la reward gain × retención (fast_eval + `reward.py`).

**Criterio de éxito:** un **segundo donante** distinto de Jackrong (otro autor u otro método) pasa por el op y su
veredicto acierta: si dice "cabe", el adaptador recupera la mayor parte de la mejora del donante completo con reward
> 0; si dice "no cabe", el adaptador no funciona. Si solo vale para Jackrong, se dice.

**Conjunto sellado de confirmación (2026-11-09):** `data/sealed_confirm.json` (GSM8K 100 + MATH-500 50, disjunto del
set de desarrollo, sha256 comprobado por `engine/sealed_set.py`). Un trasplante que pasa en desarrollo se confirma ahí
una sola vez antes de anunciarse; regla en `lab/probes/sealed/README.md`.

**Estado al cierre del 2026-11-09 (results §43.11-43.14):**
- Veredicto de energía coherente en 4 donantes: Jackrong 58.6 % → transfiere (MATH 6→30/36); OSS-Distilled 98.3 % →
  transfiere la conducta (pensar ≈ 3× menos, también sobre MiMo); LeoZotos 7.5 % y MiMo 3.8 % → el adaptador no
  transfiere de forma fiable.
- OSS sobre MiMo FALLA su criterio (acorta, pero −9 GSM8K, p = 0.035): transferir una conducta no garantiza que sea buena.
- `contrast_vector` (felicidad) PASA C1-C5 tras dos arreglos del juez: segundo uso del runtime, dirigir sin tocar pesos.
- Heretic frente a MiMo: empate estadístico en matemáticas; heretic piensa ≈ 2.5× menos.
- **2026-11-10: confirmación sellada del resultado principal CONFIRMA** (results §43.15): adaptador Jackrong ×0.5
  GSM8K 43 → 91/100, MATH 4 → 35/50 a 2 048 tokens, placebo 46 / 5; McNemar p < 1e-7 con dos correctores.
- Siguiente: un donante con mejora útil (no solo conducta) por el op.
- **2026-11-10, búsqueda de donantes (cadena `chain_donors4.sh`, veredicto por hub):** OmniCoder-9B (Tesslate, 697 likes)
  publica **los pesos de Qwen3.5-9B sin cambios** (Δ = 0 en las 248 matrices con el op; sha256 de 6 tensores idéntico al
  repo oficial `Qwen/Qwen3.5-9B` por HTTP range; pesos subidos una vez el 2026-03-12). Resto (results §43.16): OxCoder-9B
  **también es la base**; JackOD-9B-Coder 3.6 % unlikely (mezcla); HuatuoGPT-3-9B 25.3 % uncertain. Ningún donante de
  código útil en esta tanda. Bug del runtime corregido: Δ = 0 daba `likely`, ahora `no_change`.
- **2026-11-10, réplica "cc" sin donante — CONFIRMADA en el sellado (results §43.17-43.18):** dev GSM8K 50 → 84,
  MATH 6 → 32; sellado 43 → 87/100, 4 → 25/50; placebo de misma norma por debajo de la base (23 / 2). **Compilada en `down_proj`
  (rango 1 por capa en forma cerrada, fusionada en Q6_K, sin flags): PASA en dev (§43.19)** — GSM8K 50 → 87, MATH 6 → 34,
  igual que el vector (p 0.45 / 0.73), placebo compilado 28 / 6. **Sellado 2 CONFIRMA (§43.21):** GSM8K 46 → 89, MATH 7 → 26,
  placebo fusionado 30 / 4; HumanEval-20 19/20 (vector 15/20). Re-derivar en 0.8B falla: 1/256 trazas cierran (§43.22). Plan original `lab/probes/cc/PLAN.md`: dirección "concluir"
  sacada de las propias trazas de la base (últimos 48 tokens antes de `</think>` menos el resto del pensamiento), op
  `contrast_vector` ampliado con filas `think`/`focus` (posición), aplicada como vector de control; criba de escala en
  ítems no usados, dev con placebo de misma norma; si pasa, conjunto sellado y luego compilar en `down_proj`.
  Pregunta: ¿la base arregla "no saber cerrar" **sin donante**? Referencia:
  - Concept Country, *Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B* (working paper v1, 2026-11-08):
    https://concept.country/research/weight-space-programming · post: https://x.com/varien/status/2119880941711827025 ·
    código: github.com/conceptcountry/weight-space-programming · pesos: huggingface.co/ConceptCountry/qwen-concept-27b.
    Rango 1 en `down_proj` de 9 capas (32-48 pares), α 0.1; GPQA 81.8 → 86.1, LCB v6 58.0 → 71.0, −27-54 % tokens; control
    aleatorio de misma norma nulo. Límites: T = 1 con 1-2 muestras, cortes a 32k en todos los brazos (la prueba a 128k
    va por 5/48), un solo control aleatorio.
  - Relacionado: nathanrs.rs/posts/weight-space-lp (LP para la edición anti-bucle de MIT Exp 8; un empuje constante da
    casi lo mismo que la edición).

**Reglas:** nada de líneas nuevas mientras tanto; posts solo si cambian este plan; las cadenas ya en marcha (ronda 3,
cf6, placebo) terminan y se anotan, sin seguirlas con más variantes (el placebo sí cuenta: valida las cifras).

### 4.0septies ESTADO 2026-11-16 (sesión windows, Claude Code) — qué hay y qué falta

**Hecho desde el 2026-11-10** (detalle en results §45.x, README tabla de estado, FOUNDATIONS T1-T8, predictions filas 20-35):
- Dirección "concluir" sin donante, op corregido (cc_v2): dev y sellado confirman otra vez (§45.4); compilada en `down_proj`
  y recompilada: sellado 2 confirma (§45.8); placebo con 8 semillas: ninguna gana (§45.10); control de presupuesto a 8 192:
  el vector sigue +13 sobre la base y el fusionado +18 (§45.11, §45.13) → no es solo "acabar dentro del presupuesto".
- Mapas de activación 9B → 0.8B (puertas 1-3b, §45.12, §45.14): lo que cruza el mapa es "deja de pensar", no "termina
  tras razonar"; la curva de dosis lo muestra. Comprobación de neurona implantada (§43.24) hecha; Lazarus anotado en T8.
- **Segunda familia:** Phi-4-mini-reasoning 3,8B. Vector: dev pasa, sellado confirma (§45.15). **Compilado con la misma op:
  dev C1-C4 y sellado 2 confirman, igual que el vector (§45.16).** Cambia la dosis (×0,5), no la op.
- Infra: `served_info` (cada JSON guarda ruta y sha256 del modelo servido), `chain_guard` (config por cadena), smoke por
  configuración, sha256 en el log; `split_think` para plantillas que no separan el pensamiento.
- **Otra sesión (16-nov, 11:57, commits 1b09e9a y 46b620d):** `docs/phi_findings_2026-11-16.md`, síntesis de §45.15-45.17
  (qué establece y qué no), enlazada desde el README. Verificado aquí: sus cuentas (59/6 → 85/19 → 81/19; sellado 2
  62/7 → 86/14; 2 048: 81/26, 85/35, 91/29; 12/2 p 0,01294; 7/4 p 0,549; 25 de 42; 2 de 12 victorias sobre acabados
  mal; cuenta 6/14/12/4) coinciden con results y con el recuento propio; WeightBench (arXiv 2611.18553: Opus 5.5 0,535
  en 42, GPT-6.1-sol 0,568 vs 0,546 en 38) comprobado en el abstract y añadido a la lista de lectura; enlace awsp y
  directorios citados existen. Sin experimento nuevo. Su propuesta coincide con el hueco 2: sellado a 2 048 y brazo
  4 096 / 8 192 para MATH.

**Qué falta (del roadmap y de los límites declarados), por orden de lo que cambiaría una afirmación:**
1. **Muestreo.** Todo es voraz (FOUNDATIONS, límites). Un brazo base/fusionado a T 0,6 con varias semillas diría si el
   efecto de "concluir" sobrevive al muestreo (AHA y el MIT trabajan muestreando). Sin hacer.
2. **Control de presupuesto en Phi** (lo que §45.11/45.13 hicieron en Qwen): base, vector y fusionado a 2 048. Decide si en
   la segunda familia el efecto también excede "acabar dentro del presupuesto". **Hecho (§45.17, fila 36 ◐):** base
   81 / 26, fusionado 91 / 29 (+10 GSM con p 0,013; MATH +3, ruido), vector 85 / 35; la ganancia sale de los cortes de la
   base (techo AHA). En Phi el presupuesto explica la mitad de GSM y casi todo MATH a 1 024. Queda: brazo a 4 096-8 192
   para MATH (sigue cortando 24/50) y el sellado a 2 048.
3. **Segundo donante con mejora útil por el op** (criterio de 4.0sexies): no encontrado (OmniCoder/OxCoder = base, JackOD
   mezcla, Huatuo incierto, §43.16). Sigue abierto; depende de que exista en el hub.
4. **Intervención condicional** (v por LP en vez de ridge): hecha fuera por AHA con sus límites medidos (más condicional =
   peor). Comparar su u (pares rebobinar-remuestrear) con la nuestra (posición) en el 9B: experimento corto, no lanzado.
5. **Fila E** (réplica Exp 8 en el 0.8B con controles): en pausa; parcialmente cubierta por §9.1 (bit-exacta), §45.12 (el
   edit de Astra como referencia, 13/13) y por AHA/Barry desde fuera.
6. **Adaptadores de arquitectura para `model_report`** (J1: nombres mapeados, no calibrados); Phi da ya un segundo caso.
7. **Lazarus / `implant_detect` en la última capa** (T8, anotado por decisión del driver; sin hacer).
8. **Reconstrucción desde descripción** (T4, el test de legibilidad de iWSP): sin hacer.
9. Pendientes viejos congelados: P5 de la batería blog, retos WSP-40 10-17, auditoría punto 3, `comparative_drift` →
   `weight_diff`, mojibake de este archivo, purga de `.npz`.
10. Divulgación (no investigación): PR al awesome-list del survey con el hueco "Model Editing" en LLMs; solo con OK del
    driver.
11. **Firma de Adam en los embeddings de nuestros donantes** (revisión Chaudhuri del 16-nov, más abajo): filas a cero,
    filas de |Δ| constante, agrupación en pasos, en Jackrong y MiMo menos la base. CPU, minutos. Diría si el relato
    del hilo se sostiene en artefactos que ya tenemos, y daría a `implant_detect` una referencia sin base.
    **Hecho en Jackrong (§45.18, fila 39 ❌): embeddings y `lm_head` idénticos a la base; solo difieren las
    proyecciones lineales → LoRA fusionada, sin huella posible.** MiMo no está en HF en disco. Queda: un donante de
    ajuste completo con base HF sin puerta (p. ej. DeepSeek-R1-Distill-Qwen-7B sobre Qwen2.5-Math-7B, ≈ 30 GB); no lanzado.
12. **Tercera familia (decisión del driver 2026-11-16: "verificación en tres").** `DeepSeek-R1-Distill-Llama-8B`
    (Llama, 32 capas, d 4 096; plantilla escribe `<think>\n`, regla de filas de Qwen). Plan `lab/probes/cc_llama/PLAN.md`
    fijado 2026-11-17 antes de correr, filas 37 (vector) y 38 (compilado) en predictions; misma op, mismos criterios
    C1-C3 / C1-C4, sellados 1 y 2 en primer uso con este modelo; una cadena (`lab/runs/cc_llama/chain.sh`) con las dos
    partes, la B solo si la A pasa dev. Gemma descartada por su norma tras `down_proj` (lección Muse/Kocabay). Descarga
    ≈ 23 GB aprobada. ≈ 12-14 GPU h. **Hecho 2026-11-17 (cadena 00:36-12:19, 11,7 GPU h; §45.19-45.20; filas 37 y 38
    ❌).** Dev PASS en las dos partes (vector 76 / 32 sobre 74 / 22, MATH p 0,021; fusionado 74 / 35), **sellados no
    confirman**: sellado 1 base 83 / 28 → vector 76 / 26, placebo 82 / 23; sellado 2 base 75 / 23 → fusionado 77 / 27,
    vector 77 / 28 (fusionado = vector ítem a ítem). El vector hace lo mecánico (cortes 13 → 2, −41 % tokens, placebo
    inerte, op fiel: G0 0,030, cos 0,74) pero pierde tantos ítems como recupera, acabándolos mal sin cortar. Lectura:
    ahorro de tokens a precisión igual en esta familia; ganancia de precisión mostrada en dos familias de tres. Abierto
    (no lanzado): otro presupuesto, muestreo T 0,6 (lo que pide la tarjeta). **Parte C (dosis, decisión del driver
    2026-11-17 16:1x): vector a 0,25 y 0,35 en dev, puerta prefijada (`dose_gate.py`), sellado 3 solo si pasa; fila 40;
    cadena `chain_c.sh` 16:58-21:54 (4,9 GPU h). Hecho (§45.21): dev 0,25 → 84 / 31, 0,35 → 81 / 37 (MATH 15/0); puerta
    elige 0,35; sellado 3 (200 + 100, primer uso) CONFIRMA: base 158 / 40 → vector 156 / 58 (14/16 p 0,86; 20/2 p 0,0001,
    dos correctores), placebo 152 / 38, tokens −31 % / −18 %, C3 en el límite. 19 de 20 MATH ganados son cortes de la base
    recuperados; 16 GSM perdidos acabados mal (10 % de los que la base acierta, frente a 17-20 % a 0,5): la dosis reduce
    a la mitad el cierre prematuro, no lo quita. Fila 40 ❌ (apostamos "no confirma"). Queda: compilado a 0,35, otro
    presupuesto, muestreo.**
13. **Poda D4k de alphaXiv sobre la réplica Exp 8 (0.8B)** (decisión del driver 2026-11-18): plan
    `lab/probes/exp8_prune/PLAN.md`, fila 41, cadena `lab/runs/exp8_prune/chain.sh` lanzada 04:09. Brazos `full` (réplica
    otra vez: ruido), `d4k` (4 096 entradas mayores por capa; conserva 0,76-0,79 del empuje a lo largo de u) y `d4k_comp`
    (las otras 61 440). Lecturas R1-R3 fijadas. ≈ 40 GPU min. **Hecho (§45.22, fila 41 ◐): `full` 8 (14-nov: 13, 11
    discordantes de 100 con el mismo archivo), `d4k` 2, `d4k_comp` 5; todas las diferencias dentro del ruido del voraz
    por lotes (± 5). La poda no mejora; si pierde no se decide a n = 100. Queda: 3 corridas por brazo o 300 ítems o
    `-np 1`, y D4k reescalado a 1 / 0,77. No lanzado.**
14. **Control de sesgo constante para el vector "concluir"** (lo que le falta a DAWE y a nosotros, 2026-11-18): un
    `logit_bias` sobre `</think>` (llama-server lo admite por petición) elegido en dev para igualar el ahorro de tokens
    del vector, contra el vector a la misma dosis en los mismos 150 ítems. Si el sesgo iguala la precisión, la parte
    "dependiente del estado" del vector no está demostrada (§45.12 ya vio que el puerto solo lleva "deja de pensar").
    Barato (2 brazos × 25 min en la 9B o la 0.8B); no lanzado.
15. **Qwen3.8-27B ternario (ttf, 6,74 GB) como cuarta familia por tamaño**, solo en espacio de activaciones (vector
    "concluir" por el protocolo §43.17): cabe en la 3060; hace falta compilar su fork de llama.cpp (parches CUDA) y
    comprobar `--control-vector`. No sirve para leer o escribir pesos (ternario + 72M tokens de recuperación). Descarga
    y compilación requieren OK del driver; ≈ 2,5 h por brazo de 150 ítems a ~35 tok/s. No lanzado.
16. **Backend ternario de WSPLang** (pedido del driver 2026-11-18): **medido y cerrado (§45.23, fila 42 ◐).** Op
    `ternary_survival` (`engine/ternary.py`, 11 tests, recuento independiente): a dosis 1 la edición Exp 8 sobrevive con
    cos 0,42-0,47 y el compilado con 0,08-0,10 bajo TQ2_0 (alpha 0,6-0,9, ruido 2-12×); Exp 8 necesita 10-20× dosis,
    el compilado no llega; `absmean` pierde dirección al subir k. La 0.8B a TQ2_0 sin recuperación: 0/10, ruido. No se
    construye escritor ternario; las ediciones van en escalas/adaptadores float (lo que ya existe). Queda, solo con un
    modelo ternario entrenado (ttf) y su runtime: medir conducta de la versión ruidosa (T7). No lanzado.
17. **Segunda conducta distinta de "concluir", mismo protocolo** (decisión del driver 2026-11-18 "ve a por todo"):
    dirección leída de las trazas de la propia base (`contrast_vector`), placebo de misma norma, dev y sellado nuevo,
    compilado solo si el vector pasa. Candidatas con margen medible en local: formato / idioma de respuesta, repetición
    de cómputo, supresión de bucles en la 0.8B. Plan `lab/probes/skill2/PLAN.md`, fila 44. Agente asignado.
    **Elegida "razonar en español" (formato y repetición sin margen). Parada por la regla en el descubrimiento (§45.28,
    fila 44 ❌): con la instrucción la 9B sigue pensando en inglés 64/64 y corta más; 11 pares de 30. Sin filas positivas
    no hay dirección. Queda: conducta con filas positivas en la base (respuesta en español medida en la respuesta con
    presupuesto mayor, o un formato que la base cumpla a veces), criba de 64 antes de prometer.**
18. **Control de sesgo constante** (`logit_bias` sobre `</think>`, elegido en dev para igualar el ahorro de tokens del
    vector) frente al vector "concluir" a la misma dosis en los mismos ítems de dev de la 9B. Si iguala la precisión,
    la parte dependiente del estado del vector no está demostrada. Plan `lab/probes/cc_bias/PLAN.md`, fila 43. ≈ 1 GPU h.
    Agente asignado. **Hecho (§45.27, fila 43 ◐): +2/+4/+6/+8 → GSM 53/54/59/60, MATH 6/9/7/9, cortes 96 → 80, tokens
    −7 %; el elegido (+8) pierde contra el vector 1/24 y 0/26. R2: el vector no es un empuje constante. Queda (no
    lanzado): sesgos +12/+16 para ver a qué precio igualarían los tokens.**
19. **Batería de preservación** para los sellados: HumanEval completo (164) y tasa de bucles/basura por brazo con datos
    locales; IFEval y TriviaQA requieren descarga (pedir OK al driver). Código `engine/preserve_bench.py` con tests.
    Agente asignado. **Hecho el código; primera fila (§45.29): base 9B HumanEval 144/164, cumplimiento 15/40, colas
    repetidas 38/204. Banda y comparaciones con la corrida doble de `bench_run` (cola 2).** IFEval (541, comprobador
    propio de los tipos deterministas) y TriviaQA (200 del rc validation, vía la API de filas de HF) aprobados por el
    driver ("como tú veas", 18-nov 12:30): componentes `ifeval` y `triviaqa` en `preserve_bench`, suite nueva
    `dev-2048-full` (la `dev-2048` no cambia de hash). Agente asignado; GPU tras `dist-shift`. **Corrida 18-nov
    (§45.32, 6,3 GPU h):** base 9B IFEval estricto 129 / 510 (25 %), laxo 134; TriviaQA em 77 / 200 con 78 cortes a
    2 048; HumanEval 148, cumplimiento 14; GSM 46 / 47 frente a 50 / 49 de la banda → 24 ítems discordantes con la
    misma base (los lotes de `-np 2` cambian con el conjunto): la banda de la medida es GSM |Δ| 4, MATH 1, HumanEval
    4-5, no la de la repetición idéntica (§45.30 enmendada). Queda: brazos en la suite completa solo si pasan la
    barata (≈ 6 GPU h cada uno); una corrida de la base con `-np 1` para separar lote de suite; que `bench_compare`
    acepte una banda de otra suite cuando los ítems de cada banco coinciden por índice.
20. **Escritor discreto por fila (mecanismo nuevo, prerrequisito de cualquier intento de crear capacidad sin
    donante):** nuestro objetivo de compilado (ridge) resuelto como problema discreto por fila de salida en el alfabeto
    bf16 (vuelcos de signo, desplazamientos de exponente; Tree-Ising 2611.14210) en vez de redondear (§45.23 muestra
    que redondear falla). Prototipo CPU con tests; dev en GPU solo si el prototipo reproduce el dW compilado con
    cos ≥ 0,7. Plan `lab/probes/discrete_writer/PLAN.md`, fila 45. Agente asignado.
21. **Op `edit_column` / `edit_neurons` con placebo estructural y parches con rollback** (del plan externo del 18-nov):
    columna o lista corta de neuronas, dosis, dirección; manifiesto con parámetros tocados, ‖Δ‖, cos; placebo de misma
    norma en la misma columna con dirección aleatoria; `.wsplpatch` (sha del tensor origen + índices + delta compacto),
    `apply_patch` / `rollback_last` con pila por sesión y reconstrucción exacta. CPU, TDD, sin GPU. Agente asignado.
    No se adopta "reproducir el +29 de Hu": es otro fenómeno (bucles de la 0.8B) y nuestro arnés de la 0.8B tiene
    ruido ± 5 (§45.22).
    **Cola de GPU (una a la vez, la coordina esta sesión): 18 → 19 (base) → 17 → 20.**

22. **Revisión externa del README (18-nov) triada:** ya en curso: batería de colaterales (19), ruido de la 0.8B
    (§45.22), ediciones condicionales (objetivo acoplado de §45.24 y AHA, hueco 4), dosis por métrica interna (regla
    de tokens de la fila 43). Cerrado: huella de Adam (§45.18, hace falta donante de ajuste completo); centroides de
    `peek` (heurística declarada; sondas = lectura, T7). Nuevo y barato: **supervivencia de ediciones a Q4_K_M /
    Q5_K_M** generalizando `engine/ternary.py` a redondeo de k bits por bloque (CPU, minutos) antes de escribir
    soporte Q4 en `wsp.py`. Pospuesto: conversor LoRA→GGUF nativo, `model_report` calibrado para Phi/Llama (J1, solo
    si el driver lo pide), panel visual (no mientras los resultados cambien a diario).

23. **Soporte ternario de escritura en WSP-Lang + prueba de compatibilidad NanoWSP** (pedido del driver 2026-11-18,
    tras cerrar §45.23 como "medir, no escribir"): ops `ternarize` (copia ternaria W = diag(s)·T con manifiesto) e
    `imprint_ternary` (asociación escrita en float y proyectada a {−1,0,+1}·s **resolviendo por fila**, con
    reimpresión del residuo R rondas, al estilo del acumulador de TTF). Banco: las mismas asociaciones escritas en un
    NanoWSP float y en su versión ternaria (una pasada frente a resuelta), capacidad n a recall ≥ 95 % por anchura,
    falsos positivos, y matriz de compatibilidad de las ops existentes sobre el modelo ternario. Plan
    `lab/probes/ternary_nanowsp/PLAN.md`, fila 46, CPU. **Hecho (§45.25, fila 46 ❌):** ops registradas con 9 tests;
    capacidad float 144 / 144 (d 64 / 128) frente a ternario 8 / 32, igual convertido que escrito en ternario; la
    dirección sobrevive (cos 0,88), la precisión de cancelación entre claves colineales no; la reimpresión diverge;
    `peek` lee el ternario, `project_out`, imprint float encima y delta pequeño fallan. Queda (no lanzado): tarea con
    claves menos colineales, reimpresión amortiguada prerregistrada, TQ1_P/ttf.

24. **Benchmarks como sección de WSP-Lang** (pedido del driver 2026-11-18; corrección: "parte integrante, no un
    WSPBench aparte"): suites estándar dentro del motor (`engine/bench_suites/dev-2048.json`, `dev-8192`, envoltorios
    de los sellados; versionadas con hash), ops `bench_run` / `bench_compare` / `bench_table` (y `evaluate_benchmark`
    pasa por la misma ruta), regla de veredicto (p < 0,05 en los dos correctores, placebo no significativo, |Δ| ≥ 2 ×
    banda de ruido de la base corrida dos veces, preservación no fallida), registro `lab/runs/bench/index.json`, tabla
    `docs/wsplang_benchmarks.md`, sección "Benchmarks" en la spec; relleno con las corridas de dev existentes que
    coincidan exactamente. Agente asignado. Primera corrida GPU a encolar: base 9B dos veces (banda).
    **Requisito del driver (18-nov): "valorar objetivamente cualquier modelo, incluidos los generados por WSPLang".**
    Añadido al diseño: (a) identidad obligatoria (sha del modelo servido vía `/props`, del vector/LoRA, del binario y
    de la suite; sin identidad no se registra); (b) procedencia `provenance=` (manifiesto de `wsp.py`, informe del
    compilado, bundle de `commit_manifest`, manifiesto de `edit_column`), verificada contra el sha servido, fila
    "base@sha + N parches"; "edición frente a su base" solo si la base del manifiesto es la base medida, si no "dos
    modelos"; (c) clase de modelo en la suite (`llm-chat-think`, `llm-chat`, `nanowsp`) con rechazo de desajustes y
    suite `nanowsp-wspx` (recall, parity8, mod13, fsm4, noecho, en CPU) para los micro-modelos sintetizados, incluidos
    los ternarios del hueco 23; (d) correctores válidos sin `<think>`; (e) `bench_compare` entre dos modelos cualesquiera
    (misma suite, pares y banda, sin veredicto de mejora, que exige placebo y procedencia). Objetividad = misma suite
    con hash + misma decodificación + dos correctores + banda de ruido medida + placebo si hay edición + identidad y
    procedencia en cada fila. **Hecho (0.8.1, 2026-11-18):** `engine/wsplang_bench.py`, seis suites, tres ops +
    `evaluate_benchmark` por la misma ruta, 20 tests, §5.9 de la spec, addendum; relleno de 16 corridas de dev
    (9B y Llama; Phi excluida por presupuesto 1 024) y 4 comparaciones (vector 9B mejora; vector Llama 0,5 y 0,35
    mejoran por MATH; fusionado 9B "dos modelos" por falta de sha). Regla (b) ajustada tras el relleno: el placebo
    bloquea si **gana** significativamente, no si daña (el placebo de la 9B está por debajo de la base). **Banda 9B
    medida (§45.30, 18-nov):** base dos veces por `bench_run` en `dev-2048`: GSM 0 / 100 discordantes (tokens
    idénticos), MATH 1 / 50, HumanEval 15 / 164 (saldo +1), cumplimiento 2 / 40; una tercera corrida (la base
    importada de §43.17) coincide 0 / 1. Aplicada a las dos comparaciones 9B (`gsm8k ±0; math500 ±1`). Queda: banda
    de Llama y Phi (sus filas dicen "none"), banda a 8 192 antes de volver a D4k, y pasar los brazos a `bench_run`
    propio.
25. **¿Banco desde los pesos?** (pregunta del driver 2026-11-18; T7 con número): ≈ 20 brazos con pesos y resultado
    (vectores/compilados 9B, Phi, Llama con placebos y dosis; Exp 8, D4k, complemento). Rasgos solo de pesos: magnitud
    (‖ΔW‖/‖W‖, rango efectivo, energía, G0) y dirección (proyección sobre la fila `</think>` del unembedding, sobre la
    dirección de contraste de la base, sobre las direcciones del placebo). Lecturas: Spearman con Δ aciertos y Δ tokens
    por familia y global; **AUC real frente a placebo** por rasgo. Predicción antes de calcular: magnitud AUC ≈ 0,5
    (por construcción), Spearman con tokens 0,6-0,8 dentro de familia; direccional AUC > 0,8; con aciertos entre
    familias nada > 0,5. Plan `lab/probes/weight2bench/PLAN.md`, fila 47, CPU. **Hecho (§45.26, fila 47 ◐):** 78
    brazos; magnitud AUC 0,21 (nada); alineación con la fila `</think>` AUC 0,93 real/placebo y ρ −0,6/−0,7 con el
    ahorro de tokens por familia; con aciertos |ρ| ≤ 0,42. Los pesos dicen qué empuja y cuánto recorta, no si acierta.
    Queda: encontrar la fila de lectura de una segunda conducta sin saberla de antemano (hueco 17).

26. **Suite `dist-shift` sobre las herramientas de llama.cpp** (OK del driver 2026-11-18 a descargas y GPU): KL frente a
    los logits guardados de la base (`llama-perplexity --kl-divergence`), perplejidad en WikiText-2, HellaSwag /
    Winogrande / MMLU en el formato de llama.cpp; clase de modelo `llm-logits` (GGUF directo, sin servidor) dentro de
    `bench_run`; regla "preserva distribución" con umbrales calibrados con el placebo de misma norma (nulos hasta
    medir); datos en `data/bench/` con sha y licencia. Primera cadena GPU: base 9B, compilado y placebo compilado
    (y vectores si `llama-perplexity` admite `--control-vector`), en cola tras la banda. Plan
    `lab/probes/dist_shift/PLAN.md`, fila 48. **Construido 2026-11-18:** `engine/dist_shift.py` (constructores de
    comandos y parsers con fixtures reales de la 0.8B en CPU), clase `llm-logits` en `bench_run` (`model=<gguf>
    base=<gguf>`), suite con los cuatro ficheros fijados por sha (WikiText-2 test vía el zip de `ggml-org/ci` que usa
    `scripts/get-wikitext-2.sh`; el parquet de HF no se pudo leer sin pyarrow), 13 tests, cadena
    `lab/runs/bench/chain_dist9b.sh` (base, fusionado, placebo fusionado, vector 1,0, placebo vector; ≈ 30 GPU min;
    logits de la base ≈ 2 GB, ignorados por git) en la cola 3 tras la banda. Umbrales nulos hasta calibrar con el
    placebo. MMLU en binario exige `-bf` (con `-f` Windows trunca el archivo; reproducido y arreglado). **Corrido
    18-nov (§45.31, fila 48 ❌):** 4 de 5 brazos; base ppl 8,03 / HS 0,78 / WG 0,715 / MMLU 0,41; vector 1,0 ppl
    ×1,156, KL 0,175, top-1 80,4 %; placebo vector KL 0,056; placebo fusionado KL 0,024. La primaria (placebo
    desplaza más que la edición real) falla al revés por 3× en el par del vector. El fusionado cayó por un bug del
    lector de procedencia (report con `files` en lista; arreglado con test) y va en la cola 6
    (`lab/runs/bench/chain_dist9b_merged.sh`) tras dt_bias. Queda: fijar `kl_mean` / `same_top_p` de la suite con el
    par compilado completo; vectores a 0,5 y 0,35 (¿el coste en ppl escala con la dosis?).

27. **Sesgo de paso temporal +0,5 (Fan et al. 2611.21885) con placebo, en nuestra 9B** (18-nov): su edición de mayor
    ganancia (+6,6 en 490 ítems, sin control) es subir `dt_bias` de todas las capas de atención lineal en 0,5. En
    nuestro GGUF son `blk.N.ssm_dt.bias` (F32, 32 valores, 168 tensores ssm). Brazos en `dev-2048` por `bench_run`:
    +0,5, −0,5 y ±0,5 con signo aleatorio por capa (placebo), contra la base; lectura por tokens, cortes y aciertos;
    predicción antes (fila 49). Si +0,5 iguala al vector "concluir" en ahorro, es un cierre por otra vía (decaimiento de
    la memoria lineal), no por dirección. ≈ 3 GPU h. Agente asignado; cola 5 tras la base en `dev-2048-full`.

28. **Comunidades de neuronas (Goodfire, arXiv 2611.23112) como unidad de edición en WSP-Lang** (18-nov, pedido por el
    driver: "veamos cómo aplicarlo desde WSPLang"). Su unidad no es una dirección sino un *evento de token*: pocas
    neuronas (a menudo < 10, en varias capas) atribuidas juntas a la siguiente predicción, necesarias (quitarlas
    cambia la predicción) y suficientes en sus capas (quitando todo lo demás salvo unas "siempre activas" la
    predicción se conserva). Nuestro §43.23 midió la *dirección* "concluir" y la encontró repartida (~100 neuronas por
    capa, salvo la 26 con 4); eso no dice nada del evento "emitir `</think>` ahora". Plan
    `lab/probes/neuron_communities/PLAN.md` (fijado antes de código): fase A, atribución exacta por gradiente del
    logit de `</think>` (frente al token que siguió) en las posiciones de cierre de las 53 trazas `pos` y en
    posiciones emparejadas de las 53 `neg`, capa a capa con activaciones en disco; grafo de co-atribución (top-64
    por posición, PMI), comunidades Louvain (`networkx`), las tres pruebas de Goodfire con placebo de neuronas al
    azar del mismo tamaño por capa; fase B solo si A1-A2: `edit_neurons` (escala de las columnas de `down_proj` de la
    comunidad, placebo estructural del op) en `dev-2048` por `bench_run` frente a la banda de §45.32. Fila 50. ≈ 2 GPU h
    (A) + 2 × 1,7 GPU h (B). **OK del driver 18-nov 22:40.** Código de la fase A (agente, 11 tests primero,
    `lab/probes/neuron_communities/{attribute,communities}.py`): ruta por capas = autograd extremo a extremo en la
    0.8B con error 0,0; ablación de una fila = pase completo a 5 decimales. [rev] en PLAN: pos y neg son las mismas
    53 trazas y 9 caen en el held-out de cc_weights → descubrimiento con 44, prueba con 9. Cola 7
    (`lab/runs/queue7_20261118.sh`, única espera de `QUEUE6 DONE`) lanzada 23:07; corre tras la cola 6 (≈ 03:15),
    1,5-2 GPU h; resultado para la mañana del 19-nov.

29. **Verificación rápida de un modelo publicado desde WSP-Lang** (18-nov, pedido por el driver sobre
    `harness-hub/Qwen3.8-27B-pi-adapted` y `SenseNova-Loop-9B`): qué se puede afirmar en minutos sin servir el modelo.
    Hoy `model_report` cubre "modelo completo frente a su base" por HTTP (27B validado). Faltan dos casos: (a)
    **repositorio solo-adaptador** (PEFT `adapter_model.safetensors`): informe del adaptador (r, alpha, módulos y
    capas tocadas, ‖B·A‖ por módulo frente a las filas de la base leídas por rango, consistencia de dirección entre
    capas, embeddings / cabeza intactos) sin descargar la base; (b) **linaje de una arquitectura nueva con formas de
    Qwen** (SenseNova: ¿parte de Qwen3.5-9B?): coseno por tensor y rango de `delta_to_lora` sobre los tensores de
    forma común, desde GGUF BF16 (18,4 GB, descarga pendiente de permiso) o safetensors (gated). Lo que ningún
    informe de pesos da: las cifras de banco del anuncio; eso exige servir (27B fuera de 12 GB; SenseNova exige un
    fork de llama.cpp compilado). **(a) hecho 18-nov 23:00 (0.8.2):** op `adapter_report` (`engine/adapter_report.py`,
    15 tests primero; `programs/adapter_report_pi.wspl`): pi-adapted = r 32, 256 módulos, 159 M parámetros, MLP en las
    64 capas + q/k/v/o solo en las 16 de atención completa, ‖ΔW‖/‖W‖ 0,2 % en todas partes, rango efectivo 25/32 en
    MLP y 1-4 en `v_proj` de las capas 3-15, sin dirección global. **(b) en curso:** GGUF BF16 de SenseNova
    descargado (18,4 GB, sha en `hf_cache/sensenova-loop-gguf/`); lectura por bloques de filas
    (`lab/runs/model_report/sensenova_rowblock_check.json`): embeddings cos 0,98, cabeza 0,96, matrices de atención
    y MLP cos 0,95-0,99 con cambio relativo 10-33 %, `ssm_out` igual salvo permutación de columnas (espectro cos
    1,000), normas reescaladas; pesos nuevos solo mHC (`hc_attn_fn` / `hc_ffn_fn`, 8192×8 + 8 por capa 4-27, F32).
    **(b) hecho 23:30 (0.8.2):** `engine/gguf_reader.py` (agente, 16 tests primero; deshace normas w + 1, −exp(A_log),
    entrelazado de cabezas de valor, bloques no contiguos, tensores fusionados; validado sobre A0: 427 / 427 formas,
    F32 exactos, Q6_K cos 0,9998) y `model_report model=<.gguf>`; `programs/sensenova_report.wspl` →
    `full_fine_tune`, 427 tensores, 0 desajustes, matrices 14-32 %, `dt_bias` 0,017 %: parte de Qwen3.5-9B (§45.33).
    Queda: el consejo "pierde la cabeza MTP" es falso para vistas GGUF (bloque 56 = MTP, agrupado 4 + 11); el lector
    no lee adaptadores GGUF; una cuantización frente a su propia base lee como cambio difuso del 1,8 % (suelo Q6_K).

### Versiones de WSP-Lang (anotado 2026-11-18 a petición del driver; textos externos contrastados con el repo)
- **0.8 "localización + composición" — hecho 2026-11-18** (`docs/wsplang_0.8_spec.md`): `edit_column` / `edit_neurons`
  con placebo estructural, `budget_params` e informe; parches `.weightpatch.json` con `apply_patch` / `rollback_last` /
  `list_patches` / `commit_manifest` (pila en disco, reconstrucción byte a byte); `ternary_survival`. Criterio de salida
  del texto externo ("edición < 2 000 parámetros con placebo, sellada y reconstruible") **cumplido en la parte mecánica**
  (Exp 8 reproducido byte a byte por el op; A→B→rollback = A por hash; base + parches = final) y **no en la de
  conducta**: no hay sellado de una edición de columna propia. Eso es un experimento, no una versión (hueco 17 / 21).
- **0.9 "descubrimiento + fiabilidad" (propuesta externa, no empezada):** `probe_columns` / `probe_neurons` /
  `rank_candidates` con placebo y held-out automáticos; `discrete_edit` (solo con objetivo acoplado, §45.24); selección
  de tensores con comodines; batería de preservación en todo sellado (hueco 19); composición de 10-30 parches;
  protocolo multi-familia (ya: Qwen, Phi, Llama). Corrección nuestra al texto: "concluir" **no** depende de un humano o
  frontier LLM (la dirección se lee de la base con un procedimiento fijo); el descubrimiento de locus solo tiene sentido
  para fenómenos con locus disperso (bucles de la 0.8B), no para "concluir" (§43.23, densa).
- **1.0 "programar el espacio de pesos" (propuesta externa):** las tres condiciones del texto: localización +
  descubrimiento automático con placebo y sellado; composición fiable; **al menos un caso prerregistrado de creación de
  capacidad nueva** (no activación ni transferencia) que pase todos los controles. Nuestro registro dice que la tercera
  es la que nadie tiene (H6/H7/H9, Naik, Ok & Lee); hasta entonces, 0.9x. De acuerdo con el texto en eso.
  Nota 18-nov: el mapa "descripción → ΔW" que los textos de 0.9/1.0 dejaban a un LLM externo existe ya como modelo
  entrenado (Cohen & Shalev, Qwen3.5-2B, 59/75 bancos arriba frente a la base con el mismo párrafo en el prompt, lista
  de lectura del README); lo que escribe son hábitos (sesgo de posición / etiqueta), no conocimiento, y daña con el
  razonamiento activado (MATH-500 22,8 → 10,5). No cambia la condición tercera: sigue sin haber creación.

### 4.0quinquies ESTADO 2026-11-09 (sesión windows, Claude Code) — RESUMIR AQUÍ si se corta

Dirección del driver (2026-11-08): partir del par base → mejorado, buscar diferencias reutilizables, solo pruebas de
minutos, y llegar a algo **concreto y demostrable** (comparado con `areu01or00/qwen0.8B-wsp`, que sí lo tiene).
Detalle y cifras: results.md §43, findings F49-F51. Métrica única nueva: **reward IWIBench = gain × retention**
(`lab/probes/close_probe/reward.py --all`; no-op = 0).

**Lo que funciona (resultado concreto):**
- Jackrong (fine-tune de rango bajo) → `delta_to_lora` r16 ×0.5, 87 MB, sobre la base sin tocar: MATH atascados 1→14/16
  (reward 0.867, más que MiMo completo 0.733), GSM8K 3→18/28 (0.600), HumanEval 20→19/20. **No borrar
  `hf_cache/dp_keep_dsv4_distill`: es el donante de la demo.**
- Demo `demos/math_transplant/` (README, `demo.py`, `programs/transplant_jackrong_math.wspl`): cifras del adaptador de
  laboratorio; confirmado con la salida exacta del op (30/36, 37/48, reward 0.800 / 0.560).

**Lo que no:** MiMo (fine-tune completo, Δ difuso: r512 guarda 45 % de energía) no cabe en ningún LoRA (≤4/16).
Destilación de conducta de MiMo r16 (1 pasada, 93 respuestas): reward 0.213 MATH / 0.467 GSM8K.

**En cola (un trabajo de GPU a la vez; logs en `lab/runs/close_probe/distill/`):**

| # | Cadena | Qué mide | Estado |
|---|---|---|---|
| 1 | `chain3.sh` | Destilación MiMo ronda 2: 171 respuestas (78 nuevas), r16 MLP, 1 pasada | HECHO: MATH 24/36, GSM8K 41/48, reward 0.400 / 0.720 (F52) |
| 2 | `chain_cf5.sh` | Ajuste en forma cerrada (idea WSP/MEMIT, sin gradientes): `down_proj` de cada capa sigue el residual de MiMo → LoRA r256 (`cf_distill.py`) | HECHO (5.º intento): MATH 17/36, GSM8K 25/48, reward 0.267 / 0.233 (F53) |
| 3 | `chain_demo.sh` + `chain_demo2.sh` | Op `delta_to_lora` exacto sobre Jackrong → demo + eval completa ×0.5 | HECHO: 30/36, 37/48, reward 0.800 / 0.560; lado a lado base 0/3, adaptador 3/3. README confirmado |
| 4 | `chain_self.sh` | **Sin donante** (STaR, 1 ronda voraz) | HECHO, NO FUNCIONA: 8/36, 15/48, no distinguible de la base (F52) |
| 5 | `chain_r3.sh` | Destilación MiMo ronda 3: últimos 150 ítems de MATH-500 (se agota) → r16 MLP sobre rondas 1+2+3. Lanzada 06:35 | HECHO: 27/36, 41/48 (r2: 24, 41); vs r2 MATH 4/1 p = 0.375, GSM8K igual → **se aplana**, no se sigue (§43.10) |
| 6 | `chain_cf6.sh` | Forma cerrada en dos escrituras por capa: salida de atención (`o_proj` / `linear_attn.out_proj`) y luego `down_proj`; mismos datos que cf5 → aísla la escritura extra (`cf_distill.py --targets attn,down`) | HECHO, NO MEJORA: 15/36, 26/48 (cf5: 17, 25) → techo del ajuste lineal capa a capa; **vía sin gradientes para conductas cerrada** (§43.10) |
| 7 | `chain_placebo.sh` | Placebo de cf5: mismo rango y norma de Frobenius por `down_proj`, direcciones aleatorias (`placebo_lora.py`, test) → solo evaluación. Descarta "cualquier cambio de ese tamaño hace cerrar" | HECHO: 10/36, 14/48; vs base no significativo (p 0.13 / 0.23) pero algo inespecífico; cf5 le gana (p 0.039 / 0.003) → leer cf5 frente al placebo (§43.10) |

Resumen 2026-11-09: tres vías transfieren conducta con medida significativa (pesos de rango bajo de Jackrong; destilar
respuestas de MiMo; forma cerrada sin gradientes). Sin donante, una ronda voraz no.

**Veredicto 2026-11-09 ~09:10 (cadenas 5–7 cerradas):** destilación sigue subiendo (r3: MATH 0.533, GSM8K 0.720 — solo
gana en MATH; la curva de datos manda → siguiente fuente de datos con muestreo por ítem). La forma cerrada extra de
atención (cf6) no aporta sobre cf5 → la vía de gradientes queda como la buena para subir el techo. El placebo de cf5
queda por debajo (3/16 vs 6/16) → señal cf5 confirmada, ya queda descartada la explicación "cualquier LoRA de esa
norma hace cerrar". Con esto cierra también la pregunta abierta de las decisiones del bloque anterior por ambos lados.
Donante descargado y verificado (`get_leo.done`), cadena donor2 lista para arrancar en cuanto el placebo cierre (~10 min, GSM8K en curso). Todo está en rieles. Disco: 75 GB libres.
**Cerrado 2026-11-09 10:33 (donor2, criterio de éxito del op de trasplante): CUMPLE.** Segundo donante
`LeoZotos/Qwen3.5-9B_math_grpo@bc9e0186`: `energy.json` dio veredicto **"unlikely"** (kept_weighted_at_16 = 0.0752, Δ
difuso) y el resultado confirmó la predicción: el adaptador r16 no hace nada (MATH 2/16 reward 0.040, GSM8K 6/28
0.060) y ni siquiera el donante COMPLETO en Q6_K suma (2/16, 8/28: 0.053/0.133 — el donante es débil, su delta no
representa una capacidad transferible). Second donor evaluado, el veredicto de energía ACIERTA por el lado negativo.
**Corrección (2026-11-09, results §43.11):** esas cifras son de un fast_eval de 16/28 ítems. Con la medida de §43.8
(36 + 48 ítems) el donante completo SÍ mejora GSM8K (9 → 21/48, McNemar 16/4, p = 0.012); el adaptador r16 (15/48) no
se distingue ni de la base (p = 0.21) ni del donante (p = 0.15). El veredicto "unlikely" es coherente, pero la prueba
es débil con este n: no separa "transfiere la mitad" de "no transfiere".
El criterio de éxito queda demostrado en ambos lados (Jackrong: verdict funciona; Leo: improbable→no hace nada) y ya
confirma el dato de fondo: las capacidades transferibles por este op son las de rango bajo. Este op es el producto.
Se anota y cierra; viene el experimento único 4.0sexto.
**mimo_verify (2026-11-11 07:52, re-evaluación fresca sin tocar nada):** `distill3_mimo_r16` (trasplante de
conducta MiMo→base) reproduce en servidor limpio: GSM8K 22/28 reward 0.760, MATH 10/16 reward 0.600, retención
1.00 — y el op nativo sobre el delta COMPLETO de MiMo (`mimo_r64`, delta_to_lora r64) confirma "no cabe": MATH
1/16 (0.000), GSM8K 0.213. Ambos veredictos del op siguen siendo ciertos dos días después
(`lab/runs/close_probe/distill/mimo_verify.log`); cadena endurecida con gate GPU por memoria+utilidad
(tras el crash de contención del 9-nov).
### 4.0sexto PRIORIDAD (decisión del driver, 2026-11-09 ~09:40): UN SOLO EXPERIMENTO — vector de tarea + SVD selectivo en el 0.8B

Cambio de dirección: no se abren ni retoman más líneas hasta veredicto (dspine, subespacios, Behavior Extraction y
demás quedan congelados; las cadenas en vuelo terminan y se anotan). Hipótesis y criterios preregistrados en
`lab/probes/taskvec_mult/PREREG.md`: entrenar el propio donante (multiplicaciones, full-FT bf16 sobre
`hf_cache/qwen35-08b-hf`), ΔW = B − A, y comprobar si UNA SELECCIÓN ESPECTRAL del ΔW recupera ≥90% de la mejora con
≤10% de los parámetros modificados (r ∈ {8,16,64,256} vía op `delta_to_lora` del runtime WSPLang 0.7.0
(`engine/wsplang.py` __version__ = 0.7.0; el doc WSP-LANG.md conserva la spec 0.1 como referencia)), sin romper general (gsm8k/math s10,
retención ≥0.90) y con placebo norm-matched fracasando (R3). Cadena automática reanudable:
`bash lab/probes/taskvec_mult/chain_taskvec.sh` (espera a que donor2 libere la GPU; un server a la vez).
Si falla, el resultado es la curva recovery(params_frac) + el espectro de energía: eso decide si la cirugía selectiva
sirve o la generación con gradientes es la única vía para capacidades.

**Escribir una capacidad vs escribir un dato (2026-11-09, results §43.9):** la propuesta externa "usar la escritura
de WSP para una mejora matemática y comprobarla en preguntas no usadas" ya está hecha (cf5: 6→17/36, 9→25/48,
significativo). Diferencias con el WSP original: el objetivo R sale de un profesor (MiMo), no de una respuesta
escrita; la conducta ocupa las 32 capas, no 5 tensores; down_proj solo tiene techo (≤ 20 % explicado desde la capa 9).

**Decisiones según resultado:** si la ronda 3 sube otra vez → la curva de datos manda; siguiente fuente de datos sería
generar con muestreo (varias respuestas por ítem) porque MATH-500 se agota. Si cf6 sube claramente sobre cf5 → probar
la forma cerrada en todas las matrices lineales (q/k/v, gate/up) con el mismo esquema secuencial; si no sube → el
techo no es de qué matriz se escribe sino del ajuste lineal capa a capa, y la vía de gradientes queda como la buena.

**Otras propuestas (no lanzadas):** exporte de mayor rango de cf; autodestilación con muestreo (varias respuestas por
ítem, quedarse con las cortas). Opcional: un editor en IWIBench (licencia de Gemma 3 1B y API key del driver).

**Disco (2026-11-09):** C: llegó a 157 MB libres; con OK del driver se borraron `own_q6k/mimo_a05|a025-Q6_K.gguf` y
los LoRA fallidos `close_lora/mimox_r512`, `mimox_mlp_r512`, `mimox_mlp_r256` (+ .gguf): 22 GB libres.

**Revisado 2026-11-09 (no se adopta):** `khordoo/dusty-wsp` (8M: nombre por edición de rango 2 sí, actitud por
contraste rango 1 no, 0/48; rigor bueno; aporta la idea del placebo y del término de protección); Jerry Liu / ActAdd
(ya probado: vector de control MiMo 6→15/36, por debajo de cf5); Cadence (`muellerberndt/cadence`, red recurrente por
equilibrio + recompensa, sin LLM: otro paradigma, sin relación con editar pesos de un LLM).

**Revisado 2026-11-09 (steering/control vectors — NO se adopta, fuera de la prioridad 4.0sexto):** la familia
EvilQwen (lyramakesmusic/neuron-explorer, dahara1/EvilQwen-control-vectors, metrovoc/neuron-steering,
occcat/qwen35-evil-neuron, shmuelamar/evil-qwen) y generadores genéricos (vgel/repeng + cvector-generator nativo de
llama.cpp con --control-vector-scaled, IBM/activation-steering, jukofyork/creative-writing-v3, aneoncore/gemma-cv).
Ya medido aquí: control vectors sobre MiMo (cvec_inc 0.107-0.267, cvec_mimo 0.000) por debajo o igual que cf5 y muy
por debajo del trasplante Jackrong (0.800); el steering por neuronas sueltas es "push" lineal por activación, no une
pesos, y su señal es de conducta/estilo (catalogables como los cvec). Si algún día se hace un reto de localización
fina de neuronas (tipo reto WSP-40 de "dosis-neurona con dose-response"), neuron-explorer + repeng son las piezas
RFC que ahorrarían scripts propios. Hoy úsalos solo como referencia; no se abren líneas.

**Revisado y descartado (2026-11-08):** `areu01or00/qwen0.8B-wsp` (MEMIT de hechos; útil solo la resolución en forma
cerrada; su resultado cae a 0/5 al comprobar con pesos bf16 guardados); thePM_001/AEP (promocional, sin código de
pesos); "MIT 90 % de pesos sobra" (lottery ticket 2018, no aplica a LLMs a ese nivel); AbridgeKV (comprime caché, no
acorta el razonamiento); Terminal-RSI de Tencent (iteración experta → origen de la cadena 4).

**Cerrado por el driver (2026-11-09 ~17:05): taskvec_mult SIN entrenamiento.** La producción queda cancelada: la GPU
12GB está ocupada el día entero por otra sesión de agente (chain_happy/chain_heretic, llama-server :8095, 7.5-9GB
VRAM, 93% util) — el full-FT iba a 45 min por 100 pasos (50× lento) y murió a las 16:53; y la regla nueva es
"pruebas ≤10 min". Datos ya conseguidos (quedan en `lab/runs/taskvec_mult/`): base 0.8B 0/200 en mult-test
(capacidad ausente; `fast/base_mult.json`), data sellada generada (7000/64/200, rng fijo). Lo que sobrevive si se
retoma: chain_taskvec.sh es reanudable por marcadores y el protocolo en dos fases (criba mult barata → batería solo
para el ganador) ya reduce cada prueba a ≤10 min; el bloqueo real es la GPU compartida, no el diseño. Sin entrenar
quedan los donantes EXISTENTES como vía del op (Jackrong 0.800 demostrado; Leo "unlikely" acertado).

**Forense 27B verificado con el driver (2026-11-09 ~20:15): concept27b blind test PASADO en estructura, FAILED en semántica.**
Sobre los pesos públicos de ConceptCountry/qwen-concept-27b (la edición sin gradiente de 9 down_prof alternadas
32-48 de Qwen3.8-27B: GPQA +4.3, LCB +13.0, tokens −27-54%, placebo norm-matched = 0 — paper + código Apache-2.0 +
pesos públicos): `step model_report` (rows=128, rangos HTTP, CPU) instinto, con predicción sellada previa
(`lab/runs/concept27b_forensic/sealed_prediction.md`), y el informe lo clavo: layers 32-48 (9 touched),
mlp.down_proj.weight único, rank1/tensor, method=directional_ablation → Jaccard 1.0 vs receta publicada. Falso
positivo declarado: el "usable" lo va de decensoring (la firma estructural de abliteración coincide; la dirección
aquí es de conducta) → lección J4: modelo_report orienta la estructura de la verdad, no las intenciones. Herramienta
reparada en la correrida: `hf_files` descubre checkpoints en subfolder-variante (bf16/) y `model_report` admite
`base=` sin `hf:` varado (401). Veredicto y tabla en `lab/runs/concept27b_forensic/verdict.md`. Este caso pasa a ser
el caso J4 de segunda familia (Qwen3.8-27B) con habrá znaka conocida.

**Revisado 2026-11-09 (mapa del campo): survey "A Survey of Weight Space Learning" (arXiv 2603.10090,
Han/Wang/.../Borth/Maron, mar-2026; awesome-list Zehong-Wang/Awesome-Weight-Space-Learning).** Taxonomía
WSU (geometría/simetrías) / WSR (embeddings de pesos) / WSG (generación). Nuestro programa ya vive en las
tres vías sin gap crítico: WSU = spectral delta/symmetries (ops `permute_align`, `rosetta_align`,
`vicreg_preserve`, `jepa_steer` ya llevan los nombres de la escuela Borth/Maron), WSR = `sig_*.pt` +
`model_report`/atlas (= su "model retrieval"), WSG = delta_to_lora/producto trasplante + imprint/NanoWSP
(= su "conditional generation"/"data-free reconstruction"). HUECO CONFIRMADO: su sección "Model Editing"
solo tiene hypernets de imagen (StyleGAN); el WSP de LLMs (rikiyat blog, Gribben, ConceptCountry 27B,
nathanrs LP) NO está indexado — oportunidad de PR al awesome-list cuando haya veredicto del experimento
único (outreach, no línea de investigación). Útil también como fuente de evaluaciones estándar
("Model Zoo" benchmark section) si algún día se compara con el resto del campo.
Outreach iniciado: respuesta de una línea al post del survey (2026-11-09) señalando el hueco "Model Editing"
para LLMs reales — pendiente de veredicto del experimento único para el PR con datos objetivos.

**Revisado 2026-11-09 (sin ejecutar; candidato a primitiva): nathanrs/weight-space-lp + blog (nathan.rs,
9-nov-2026).** Sobre la receta de Astra del paper CSAIL (rank-1 ΔW=u·vᵀ sobre down_proj L15/L19 del 0.8B,
v elegida por ~28 pases de prueba en ~8 h): demostrar que ELEGIR v ES UN PROGRAMA LINEAL (v·h_t ≤ −δ en
tokens de bucle, |v·h_t| ≤ ε en productivos, min ‖v‖₁ con holguras; solver PDHG ~1 min vs LP-HiGHS 20 min)
→ 8 columnas por capa, 8× menos pesos que Astra, resultados dentro del ruido con su u (GSM8K 69.6 vs 68.2)
y ~93% con u heurística sin agente. Dos hallazgos que CONFIRMAN lo nuestro de forma independiente:
(1) "un steering vector hace el mismo trabajo" — el empuje es casi constante por token (89% tokens activos,
ventana 0.5-2× estrecha): enlaza con la línea cvec/loop del repo; (2) "la edición quita obstáculos, no añade
habilidades" (Gemma 270M no mejora). Huecos: sin holdout/OOD explícito, no publica pesos, todo del 0.8B.
VERIFICACIÓN BARATA SUGERIDA (si se retoma): su PDHG contra nuestro harness exp8-replica 0.8B (30-45 min).
Candidato directo a primitiva WSPLang: `lp_select_v` (u + etiquetas de conducta → columnas/escala por LP,
determinista, sustituye el buscar-y-escalar) — familia forma-cerrada (cf_distill). Anotado; no adoptado.

**Revisado 2026-11-16 (sin ejecutar): AHA, Khamis / Tutunov / Bou Ammar (haithamb.github.io/aha, código
github.com/Eppy-EpHard/AHA, MIT, prerregistros en docs/).** Cierra el hueco de Barry: la u ya no viene del agente
sino de pares emparejados del propio modelo (rebobinar una traza en bucle 8 tokens antes del inicio del bucle,
muestrear 8 continuaciones, emparejar la que antes entra en bucle con la productiva más larga, diferencia de medias
del residual en tokens 64-224; 228 pares de 300 prefijos), v por el LP de Barry sin cambios (8 columnas por capa),
capas por tasa de acierto del LP en reserva (resultado: da igual qué capas, < 1,6 puntos). 0.8B a 32k: GSM8K 14,0 →
71,2, MATH-500 9,8 → 51,8, IFEval 19,6 → 62,7 (nunca visto, sin matemáticas); 2B: 75,8 → 88,4, 69,6 → 79,6, 61,4 →
77,1. Construcción 22 min en una 4090. Confirma de nuevo lo nuestro: (1) el empuje casi constante gana — hacerlo más
condicional (objetivos por token, ε 0,25, 10× el salto) empeora; (2) mejor separabilidad (Fisher, AUC 0,953) → peor
edición (21,9 vs 64,1): el oráculo es la conducta, no la métrica intermedia; (3) techo = tasa de corte de la base,
"quita un obstáculo, no añade nada". Huecos: sin placebo de misma norma propio (cita el de Barry), una familia, los
números del 0.8B vienen de la primera construcción (dirección mixta, capas del agente), no determinismo entre
máquinas (base 9,8 aquí / 7,0 en la de Barry). Relación con lo nuestro: misma primitiva que `compile_control_vector`
(rango 1 en `down_proj`, sin gradientes); nuestra v es ridge (empuje constante, G0) y la suya LP (condicional):
es la "intervención condicional" de la revisión externa del 15-nov, ya hecha fuera y con sus límites medidos. Su
receta de pares (rebobinar y remuestrear) es un segundo modo de leer la dirección, distinto de nuestro contraste por
posición dentro de la traza (últimos 48 tokens antes de `</think>`); comparar ambas u en el 9B sería un experimento
corto si se retoma la fila E. Anotado; no adoptado.

**Revisado 2026-11-16 (sin ejecutar): Alapan Chaudhuri, hilo en X del 14-nov ("A model's weights keep a record of its
own training"; sin paper ni código, "los problemas —embeddings atados, LoRA solo en atención— para otra vez").** Lectura,
no escritura: la huella de AdamW en la matriz de embeddings de entrada. Fila de token nunca visto: no se mueve (bit a
bit sin weight decay); token visto en un solo paso: la fila se mueve ≈ 31,8 × lr de ese paso en casi todas las
coordenadas (Adam sin historia normaliza el gradiente a su signo) → sello de tiempo; fila nunca tocada con weight
decay: encoge (1 − lr·wd) por paso, exacto en fp32 → la planificación se reproduce bit a bit. Afirma: lista de tokens
exacta (39 663 / 39 663; Qwen2.5-7B 50 180 / 50 180), tipo de planificación, pasos y calentamiento en 15/15
fine-tunes, lr pico al 0,01 %, 3 121 de 4 000 documentos en su paso exacto, 38/40 documentos intercambiados
detectados, un fine-tune público ajeno reproducido bit a bit en 25 463 296 pesos (un paso de desfase: solo coinciden
los 187 ceros), pertenencia 64 % con cero falsas alarmas (test estándar 0,5 %). Casos públicos: OLMo 2 SFT con 20 341
filas sin mover (el desajuste de tokenizador que AI2 corrigió), Qwen2.5-Instruct toca 4 de 22 tokens de control en
todos los tamaños, Pythia-1B entrenado a 2,5e-4 y no al 3,0e-4 de la tarjeta. **Comprobado por nosotros** (dos
fuentes): `pythia-1b.yml` dice `lr 0.00025, min_lr 0.000025, cosine`; la tabla de la tarjeta HF dice 3,0 × 10⁻⁴. El
resto no es comprobable sin código. Relación con lo nuestro: (1) es T7 por el otro lado — lo que se lee exacto de los
pesos es la aritmética del optimizador, no la conducta; nuestras escrituras en forma cerrada no dejan ninguna de esas
marcas, lo que las distingue de un fine-tune; (2) las filas que no pertenecen a ningún token son una referencia sin
base dentro del propio checkpoint, justo lo que le falta a `implant_detect` sin base (T8); (3) nuestro h5_plan.md
archivó las filas de embedding más cambiadas de MiMo como "tokens raros: ruido"; por este mecanismo son los sellos de
tiempo. Sonda barata posible (CPU, minutos, sin GPU): embeddings de Jackrong y MiMo menos la base — cuántas filas
exactamente cero, cuántas con |Δ| casi constante por fila (histograma de la desviación relativa por fila), y si las
constantes se agrupan en pocos valores (pasos). Predicción antes de correr. Anotado; no adoptado.

**Revisado 2026-11-16 (sin ejecutar): Kocabay (Poindexter Labs), "Weight-Space Reading", alphaXiv 13-nov, código
github.com/suayptalha/wsr (Apache 2.0), arXiv pendiente.** La mitad "leer" de WSP medida como debe: lectores LLM en
sandbox ven los pesos de un objetivo y una edición quirúrgica (escalar bloque, filas de vocabulario, quitar dirección,
ruido de norma igualada, ganancias de norma, controles no-op) y predicen arriba / abajo / igual en cinco conductas
(longitud de pensamiento, matemáticas, opción múltiple, adulación, rechazo); la verdad es el modelo editado frente a
cinco semillas de la base (Welch p < 0,01 + efecto mínimo). Tres rondas prerregistradas en Qwen3.8-27B, Qwen3.5-9B,
Muse-Glimmer-30B. Resultado: en ediciones sutiles, 27B leyendo sus propios pesos 0,730 de exactitud balanceada frente a
0,578 con descripción textual (+0,152, IC [+0,056, +0,256]; 10 ediciones con cambio, un par de cuatro significativo);
recall 0,278 → 0,667, especificidad 0,878 → 0,793. Lo que los autores ponen delante: (1) las ediciones destructivas
miden detección de daño (Muse lee las de 27B igual que 27B, 0,594 vs 0,593); (2) sin ventaja de autolectura (el
contraste cruzado incluye cero en las dos rondas); (3) en Muse-30B ninguna de 40 ediciones (6 no-op) cambió nada
(su RMSNorm tras el sub-bloque cancela el escalado uniforme) y los lectores predijeron cambio en 53-88 %, Muse sobre
sí mismo 85 %. Una corrida por celda, sin corrección múltiple, intervalos "optimistas" según ellos. Relación con lo
nuestro: es T6/T7 dicho desde fuera y con números — leer pesos añade algo medible solo frente a adivinar, ediciones
inertes y arquitectura; la conducta sigue siendo el oráculo (su protocolo la necesita para puntuar). Los falsos
positivos sobre ediciones inertes son el placebo de misma norma del lado del lector. El caso Muse es nuestra lección
de §45.3 (la fila de norma) a escala de arquitectura: antes de compilar en una familia nueva, comprobar dónde está la
norma respecto a `down_proj`. Nuestro forense ciego (§43.20: sitio exacto, rango y efecto no) encaja con su cuadro.
Su harness sirve para una cosa concreta si se retoma la lectura: pasar nuestras ediciones (vector compilado, placebo
compilado, trasplante) por un lector y ver si distingue el placebo del real sin correr nada. Anotado; no adoptado.

**Revisado y VERIFICADO 2026-11-09 (referencia externa, no se adopta por la prioridad única): HypaNet
(github.com/Hypabolic/HypaNet + 2 posts matthewgribben.com, mismo autor matt-gribben, repo creado 11-05,
último commit 11-09).** Fable(claude-fable-5.1) escribe pesos desde spec sin entrenar: detector de bucles de
agentes 2.1M par v0.3 macro-F1 0.957 sealed / 0.908 OOD vs el MISMO arqu entrenado con 223,930 labels
0.923/0.488 (34.5% falsas alarmas OOD vs 6.8%); v0.4 edición rank-1 de neuronas SwiGLU "silenciosas"
sobre Qwen3.5-0.8B (10.240/752M entradas): 0.419 (+0.196 sobre base), techo bajo = mismo veredicto que
nuestra serie E1-E5/cf6 (ganchos locales no traguen capacidades); v0.5 esc das 0.750 en 4 familias OOD
no vistas. Comprobación de procedencia: sin commits future-dated (alarma de fechas 10/11-nov descartada;
README sin fechas). Importable sin abrir líneas (salvaguardas de evaluación): (1) generador OOD escrito
por un agente INDEPENDIENTE como juez; (2) splits sellados con hash y evaluador que se niega a puntuar
dos veces; (3) CIRCUITS.md — documentación por parte del modelo escrito (nuestro decomp usa manifiestos
afines). Su ruta "C transferencia entre modelos" la cubre el nuestro `delta_to_lora` (0.867 Jackrong).

### 4.0quater ESTADO 2026-11-07 (sesión windows, Claude Code) — RESUMIR AQUÍ si se corta

Rama `feat/wsp-synthesis` = `main` = 8649b8e, todo subido (GitHub Pages publica desde `main:/docs`). Detalle y cifras en results.md §36-§38 y `lab/probes/delta_patterns/findings.md` (F1-F26).

**Punto de reanudación (2026-11-07, actualizado tras terminar el benchmark):**
1. **Nada en marcha.** El benchmark de la base Qwen3.5-9B Q6_K terminó: GSM8K 60/100 y MATH 24/50, frente a MiMo
   93 y 34.
2. **Informe MiMo vs base: hecho** (results.md §39). La ventaja de MiMo es sobre todo terminar dentro del tope de
   8192 tokens: cuando los dos terminan, 58 frente a 60 y 20 frente a 20. Siguiente prueba: los ítems cortados de
   la base con tope de 32k.
3. **Hecho 2026-11-06/07:**
   - lector local sin mmap (a719b01);
   - firma dsv4_heretic;
   - corrección del cast F32→BF16 (F26, 783b14a);
   - auditoría del README (e7e178a) y de docs/portal (8649b8e).
4. **Abierto:**
   - por qué cambia `A_log` en las capas 6 y 14 de mimo, fara, neohorse y tmax (F26);
   - 27B TURBO a ciegas — HECHO y verificado (ver §4.0quater, abajo);   - H4 causal;
   - borrar a mano `hf_cache/dp_keep_dsv4_distill`.

| # | Tarea | Estado |
|---|---|---|
| A | **Comparar TwIL-8B (Q4_K_M) con MiMo-9B (Q6_K)**: `lab/probes/model_compare/eval.py <name> <port>`, GSM8K 100 + MATH-500 50, greedy, tope 8192 tokens, llama-server CUDA (`C:\Users\Anto\llamacpp\bin`) sobre los blobs de ollama, `-c 24576 -np 2 -fa on --jinja` | HECHO: TwIL GSM8K 96 %, MATH 88 %; MiMo 93 %, 68 % (19 cortes por razonar largo, no bucles). results.md §37 |
| B | **`engine/weight_diff.py`**: diff tensor a tensor sin cargar modelos enteros; lector local (todos los shards) y lector por rangos HTTP (HF, sin disco); rel_delta, coseno, espectro de ΔW (energy_top1, stable_rank), filas del embedding, totales por familia y capa. Op `diff` de WSPLang usa esto (antes miraba solo el primer shard y cargaba todo en RAM). Tests `tests/test_weight_diff.py` (14); suite 291 OK | HECHO, sin commit |
| C | **Qué cambia un post-entrenamiento (informe riguroso).** Plan prerregistrado: `lab/probes/delta_patterns/plan.md` (H1 dónde, H2 forma, H3 dirección, H4 trasplante; controles: abliteración = control positivo del método, Nyx-RP = control negativo, nulo entre capas distintas, alineación con W). Modelos sobre Qwen3.5-9B: MiMo, NeoHorse, abliterated, tmax (RL), Fara, Nyx-RP; y Base→9B oficial. Pipeline `run.py pipeline` (log `lab/runs/delta_patterns_pipeline.log`): descarga a `hf_cache/_dp_*`, stats, check numpy, firma `sig_<name>.pt` (16 direcciones + count-sketch), borra la descarga. Reanudable (`sig_<name>.done`). Después: análisis de pares desde firmas + informe. Controles añadidos tras F15 (findings F16, marcados como posteriores al plan): `tencent/WeMM-Embedding-9B` (fine-tune completo, otro objetivo) y `axiomofmind/Angry-Claudius-9B` (degradación deliberada, LoRA en todas las lineales). Orden de cola: official_post → wemm_embed → angry_claudius → dsv4_distill (Jackrong, LoRA-like, se conserva en `hf_cache/dp_keep_dsv4_distill`, borrar a mano al final) → dsv4_heretic (ARA capas 9-24 documentadas: verdad conocida para el forense, F17) → 27B TURBO (espera `sig_dsv4_heretic.done`). Hallazgos F1-F16 en `lab/probes/delta_patterns/findings.md`. | EN CURSO (MiMo §38 hecho) |
| D | De diff a causa: trasplantes parciales (base + Δ de una familia/capas) y medir qué comportamiento trae cada parte | tras C |
| E | Réplica exp. 8 en Qwen3.5-0.8B con controles (edición aleatoria, decodificación). Diseño: GSM8K con tope 8192 (en los textos de Astra: base 12.5 %, p50c 31 %); MATH-500 no sirve con ese tope (p50c mediana 24k tokens). Bloqueo: HF transformers sin `fla` va <70 tok/s y llena los 12 GB; probar llama-server (presence penalty nativo) o batch menor | PAUSADO, sin criterio escrito aún |
| F | Auditoría punto 3: permute_align, jepa_steer, mimo_head.py | pendiente |
| G | `engine/comparative_drift.py`: `compute_architectural_drift` devuelve un JSON cacheado de otro par de modelos sin mirar los argumentos; `compute_spectral_delta` ignora el target si no es GGUF; `decode_tensor_weights` carga el shard entero por tensor. Pasarlo a `weight_diff` | pendiente |
| I | **`step model_report`**: informe rápido de qué cambia un modelo frente a su base y qué es aprovechable. Huella por **bloques de filas** (64 filas de cada matriz y capa por rangos HTTP, ~200-350 MB por modelo 9B) → método (completo / LoRA y módulos / abliteración-ARA y capas), perfil, familia del atlas; ficha HF y cabeceras. Código `engine/model_report.py`, tests `tests/test_model_report.py` con modelos sintéticos de receta exacta; validación con casos reales de receta documentada (abliterated, heretic ARA 9-24) y concordancia con las estadísticas completas (segundo método). | HECHO v1: op `step model_report`, 6/6 casos conocidos, 21-48 s por modelo 9B remoto (F18-F19). Siguiente: probar en modelos nuevos, añadir dirección compartida del atlas (H3) por tensor al apartado "aprovechable" |
| H | **Op `subspace_transfer` (WSPLang)**: W' = W + α·P_S(W_fuente − W_base) por tensor, con S = base de una firma (`sig:<ruta>`), el subespacio compartido de dos firmas (`shared:<a>,<b>`) o aleatorio (`random`, el control). Proyección por el lado pequeño (flujo residual). Write guard, presupuestos, tensores protegidos, procedencia `synthetic_transfer` con fuentes. Código `engine/subspace_ops.py`, tests `tests/test_subspace_transfer.py`. Uso previsto: H4 (base + ΔMiMo en el subespacio compartido con NeoHorse, frente a subespacio aleatorio de igual dimensión, frente a ΔMiMo completo). Falta para evaluar el 9B: conversión a GGUF (`convert_hf_to_gguf.py` de llama.cpp) | EN CURSO (tests escritos) |
| J | **Análisis de cualquier LLM, no solo Qwen** (pedido 2026-11-06). Hoy: las matemáticas del delta son genéricas, pero los nombres (`q_proj`/`o_proj`/`down_proj`/`layers.N`), `RESIDUAL_WRITERS`, `text_model_name` (multimodal de Qwen3.5) y los umbrales (H3 ×3, `classify`, atlas) son de la familia Llama/Qwen y están calibrados solo en Qwen3.5-9B. Pasos por valor: **J1** nombres canónicos por arquitectura (GPT-NeoX, Falcon, Phi-2/3 fusionados, Mixtral `w1/w2/w3`, multimodales `language_model.*` y torres de visión), para que `family`, `tensor_kind`, `layer_of` y `RESIDUAL_WRITERS` funcionen sin cambios. **J2** lectura FP8 (`F8_E4M3` con escalas por bloque) y GGUF. **J3** MoE con expertos fusionados en 3D (por experto, muestreado). **J4** calibrar en una segunda familia con verdad conocida (Llama-3 base frente a abliterado y frente a fine-tune) antes de fiarse de `model_report` fuera de Qwen | J1 HECHO (079bb52: nombres canónicos NeoX, Falcon, Phi-2, Mixtral, multimodales; tests en test_weight_diff/test_model_report). Siguiente J2 |

Ideas del driver (2026-11-06), por evaluar:
- **Atlas de Δ (dataset de mejoras):** muchos derivados de un mismo base (Qwen3.5-9B) con metadatos, comportamiento medido con un solo protocolo (`lab/probes/model_compare/eval.py`) y firma de pesos (`sig_<name>.pt`). Usos para WSPLang: priors de dónde/cuánto editar por tipo de mejora; base de direcciones de mejora repetidas (SVD de subespacios apilados, no medias de Δ: los Δ son casi ortogonales) → op tipo `apply_improvement_basis` con procedencia `synthetic_transfer`; relación rasgos del Δ ↔ ganancia. Condición: el piloto de 7 modelos (C) debe mostrar que las direcciones compartidas no las comparte también el control Nyx-RP; luego 15-20 modelos y trasplante H4 contra edición aleatoria de la misma norma.

- **Qwen3.8-27B TURBO — prueba forense ciega: CERRADA 2026-11-11 (P1 pass / P2 fail / P3 pass), verificado por segundo método con `turbo27b_verify2.py`: Jaccard 0.7895, energy_top1 0.338, k90 mediana 13 con 24/60>16 (exacto), Spearman por tipo 0.9286 — todo concuerda.** Predicción ciega commiteada en 98b3dfc ANTES de abrir sealed/ (predicción: rank-1 extra en escritores residuales desde capa 26, rango 26-63; rango medido 26-55 → pass con el final errado como se declaró; P2 fail: ARA reparte 90% de la energía en ~13 direcciones, no en 1). Plan y predicciones: `lab/probes/delta_patterns/plan_turbo27b.md`; verificación: `turbo27b/sealed/verify2.{md,json}`.
  - Modelos (BF16, 851 tensores de texto, mismas formas): base `Qwen/Qwen3.8-27B`; etapa 1 `trohrbaugh/Qwen3.8-27B-heretic-ara` (Heretic ARA, su ficha documenta capas 26-56 de 64, KL 0.0535, rechazos 99→0/100); final `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` (fusión + Heretic + curación + razonamiento; receta no publicada; el GGUF popular `...-NEO-CODER-MAX-MTP-GGUF` es su cuantización).
  - Mejoras declaradas por el autor: ARC-C 0.591→0.735 (Nightmedia, modo instruct), "pensamiento 1/2-1/10 de tokens, mediana ~2/3 menos" (sin medición publicada).
  - Ejecución: `turbo27b.py` (una pasada por streaming, ~162 GB de red, sin disco; arranca solo cuando existe `lab/runs/delta_patterns/sig_dsv4_heretic.done`; log `lab/runs/delta_patterns_turbo27b.log`; reanudable).
  - Regla ciega: `lab/runs/delta_patterns/turbo27b/sealed/` (Δ heretic y Δ resto) NO se abre hasta escribir y subir la predicción hecha solo con `total/` (capas de la descensura, su rango, patrón del resto). Puntuación fijada en el plan (Jaccard ≥ 0.7 con 26-56, rango ±×2, Spearman ≥ 0.7).
  - Opcional, a decidir: prueba de tokens de pensamiento con los GGUF Q4_K_M (27B no cabe en 12 GB; 4-6 h).

- **Búsqueda con presupuesto para H4 (idea tomada de HKUDS/AgentPilot, 2026-11-06):** AgentPilot (MIT) usa un 9B como estratega que elige padre, dirección y cuándo parar, con KL-UCB o memoria de políticas, y mide en tareas no vistas (RSI Arena; declaran +22.2 pts AIME 24+25 y +24.5 pts Terminal-Bench 2.1 en 10 intentos, autoinformado; mejora programas/harness, no pesos; su 9B no es público). Para WSPLang: un `step search` que reparta las evaluaciones dev entre candidatos de `subspace_transfer` (familia, subespacio, k, α) con KL-UCB, y evalúe al final en test no visto, en lugar de una rejilla fija.

- **`step model_report` (petición del driver, 2026-11-06): WSPLang debe analizar un modelo muy rápido y dar un informe completo de lo usable y cómo.** Etapas: (1) ficha HF: base, licencia, método y mejoras declaradas; (2) compatibilidad por cabeceras con su base (segundos, sin descarga); (3) huella por muestreo HTTP de un tensor por tipo en pocas capas (1-3 min): método (fine-tune completo / LoRA y sus módulos y rango / abliteración-ARA y sus capas / merge), perfil dónde cambia, familia del atlas (agente / razonamiento / control); (4) qué es aprovechable con `subspace_transfer` (componentes con dirección compartida) y qué no (p. ej. β de DeltaNet), riesgos (descensura, licencia); (5) opcional, estadística y firma completas. Reglas de clasificación derivadas de F5, F9, F11, F15, F17 y **validadas con los modelos de receta conocida como casos etiquetados** (abliterated, Nyx-RP, MiMo, tmax, dsv4 distill/heretic); si clasifica mal un caso conocido, no se usa.

- **H4 causal: "identificar la cualidad de un subespacio, transferirlo y verificar que la rige" (driver, 2026-11-06).** Diseño: (1) cualidad nombrada antes de medir; (2) suficiencia (añadir P_S(Δ) a un modelo sin la cualidad → aparece) y necesidad (`project_out` del subespacio en el modelo que la tiene → desaparece); (3) controles: subespacio aleatorio de igual dimensión y norma, Δ completo como techo; (4) especificidad: doble disociación (subespacio A mueve la cualidad A y no la B, y viceversa); (5) dosis-respuesta α ∈ {0.25, 0.5, 1, 1.5}, monótona; (6) prerregistro y test no visto. **Primer caso propuesto: longitud de razonamiento con TwIL balanced vs verbose** (difieren en una sola etapa de RL que premia trazas cortas; cualidad benigna y medible: tokens de razonamiento, sin perder accuracy GSM8K/MATH). No usar la descensura como cualidad de prueba (produciría modelos sin salvaguardas). **Cofactores e interacciones (driver: "un subespacio puede estar relacionado con cosas fuera de él"):** diseño factorial 2×2 (base, +S, +C = resto del Δ fuera de S, +S+C; interacción si efecto(S+C) > efecto(S)+efecto(C)); lo mismo por capas (S en todas a la vez vs capa a capa); necesidad con redundancia (si la cualidad persiste al quitar S, quitar también S en capas vecinas antes de concluir); batería amplia de efectos secundarios (perplejidad general, otra tarea, formato/longitud, peticiones normales), todo lo que se mueva se registra; comprobación en activaciones (la proyección sobre S durante la tarea debe variar con la cualidad); conclusiones graduadas: suficiente / necesario / contribuye con C / sin efecto, nunca "S rige la cualidad" sin el factorial completo. Conversión a GGUF RESUELTA (2026-11-06, con permiso del driver): `C:\Users\Anto\llamacpp\convert-src` = checkout parcial de ggml-org/llama.cpp en el commit del build instalado (27b7a8a00: `convert_hf_to_gguf.py`, `conversion/`, `gguf-py/`); en Git Bash usar `MSYS_NO_PATHCONV=1` y rutas Windows. Prueba 0.8B → bf16: 28 s, tensores bit-exactos a safetensors (3 comprobados), llama-server responde. Pendiente para el 9B: bajar `config.json`/tokenizer a `hf_cache/qwen35-9b-hf` (la descarga solo trajo safetensors).

Notas de entorno (2026-11-06):
- **ollama no usa la GPU** desde al menos el 2026-11-02 (log: `inference compute ... library=cpu`), aunque existe `lib\ollama\cuda_v12`. Con ollama TwIL va a 7.5 tok/s; con llama-server CUDA a 47-55.
- **Disco C:** se borraron duplicados exactos (MiMo Q6_K de `llamacpp`, TwIL de `hf_cache/twil8b`) y los smoke tests `smoke_f32*`; después, con permiso, los 11 GGUF de `served/` (quedan sus `.manifest.json`), cachés HF de Qwen3.5-4B/TinyLlama/ggml-org, cachés pip/npm, instaladores y drivers de Descargas: 13 → 139.6 GB libres. `C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf` (ruta por defecto en `engine/wsplang.py` y `engine/comparative_drift.py`) es ahora un **hardlink** al blob de ollama `sha256-ef96…`.
- No cargar dos modelos grandes a la vez: con ~32 GB de RAM Claude Code mata los trabajos en segundo plano.

### 4.0ter ESTADO CONSOLIDADO 2026-11-04 (sesión windows, opencode) — lo que cambió hoy

**Resultados soportados (commits bf42ecd, 32bc917, 5712402...):**
1. **exp8-replica (0.8B, Q6_K)**: control positivo replicado: dev 0/20→10/20, paired holdout 5%→70%, placebo norm-matched null. El edit del paper (down_proj δ = 64 vectores-per-neurona únicos, full-rank, 131,072 pesos) sobrevive la requant Q6_K. Op nueva `add-cols-vectors` en wsp.py.
2. **loop-fix del 9B (A/B con GPU RTX 3060)**: base vanilla 70% loops (14/20) → editado L01/L02 (**u honesto, p≤0.01**) 35% → placebo 65%. Señal causal; `keep` pendiente de holdout fresco + no-regresión.
3. **Auditoría anti-alucinación**: 5 claims sin soporte corregidos (L14 nulo → señal real L01/L02; u_critic_L16 es embedding-proxy sin activaciones; §15.3 sin run-artifact). WSPLang corregido: feasibility gate (ops no-implementadas RECHAZO antes "skipped" silencioso; backend GGUF guard para escritura; closure de artefactos; STUB_OPS marcados) + suite v8 (7 tests).
4. **Anatomía destilación v2**: la destilación retoca **GATES** (ssm_beta TODAS las GDN 14-19% + ffn_gate medios 15-17%; normas intactas 0.14%) y el drift es **ortogonal a los ejes semánticos** (share=baseline aleatoria en 12 tensores): la razón de ser no es direccional-estática sino comportamental — la ruta causal A/B es la válida (refuta E1).
5. **LAB M-0 (Training Trace → Surgery Compiler)**: cirugía ridge closed-form ≈ entrenamiento Adam (82% vs 86% holdout) con 1/8 de la norma de delta; placebo falla. Primer dato: tratamiento de capacidades lineales-changeable y compila sin gradiente.

**Objetivo del periodo (nuevo marco): "Training Trace → Weight Surgery Compiler"** — entrenar el paciente (NanoWSP-3 real re-iniciado, char-level TinyStories ~30-90min CPU), grabar traces (snapshots + exp_avg + loss), y compilar cirugías que reproduzcan el comportamiento sin gradientes (lab M-0→M-3). El pretraining es `pretrain_infra` (fuera de los experimentos, guard v7). Meta falsable de la semana: M-2 (mini-transformer + MEMIT: el hueco honesto entrenamiento-vs-cirugía).

**Pendiente inmediato (reordenado):**
| # | Tarea | Estado |
|---|---|---|
| A | holdout fresco (seed 1337) del loopfix-9B + no-regresión gsm8k-n20 | PENDIENTE (para keep formal) |
| B | `lab/train_nanowsp.py` (pretrain_infra char-level, trace completo) | EN PREPARACIÓN (hoy) |
| C | M-1 lab: 2-capas aprendibles + cirugía por influencia top-k, protocolo 3-brazos | HOY |
| D | WSPLang 0.7: op `compile_from_trace`, incorporar `add-cols-vectors`, provenance obligatorio, spec al 0.6/0.7 | después de C |
| E | M-2: mini-transformer + SENTIDO MEMIT-imprint → hueco entrenamiento-vs-cirugía | tras B |
| F | Retos WSP-40 10-17 + P5 fix + ExperimentRecord v1 | congelados hasta D/M-2 salvo demanda |

### 4.0 Cierre de fase E2–E5 (2026-11-03, sesión linux)
- **P4 CONFIRMACIÓN ejecutada (commit 5c372a0) y NEGATIVA**: en 20 prompts frescos (10 general + 10 code) E5 da **14/20 loops vs 11/20 del original** (subset code: 8/10 vs 5/10) → E5 DISCARDED por keep-if-better. Detalle en results.md §2quinquies.
- Consecuencia: **la serie E1–E5 de edición agéntica queda CERRADA sin keep**. Lecciones: (1) targeting por activaciones ayuda en dev pero no generaliza; (2) evaluar frescos ANTES de declarar señal; (3) placebo (E4) queda como estándar obligatorio del PROTOCOL.md.
- **NUEVO OBJETIVO (dirección del driver): aumentar la puntuación MATH de MiMo** (R0 había dado 8/8 GSM8K diag; el margen está en MATH-500/harder). Se reactiva PROTOCOL.md con el rigor E4 (placebo) y frescos.

### 4.1 Infra linux (nueva sesión 2026-11-03-b)
| # | Tarea | Estado (2026-11-03-d, linux) |
|---|---|---|
| 1 | Inferencia MiMo en linux: GGUF Q6_K + llama-server | **HECHO**: GGUF Q6_K bartowski descargado a `hf_cache/mimo-gguf/` (7.8GB); `/usr/local/lib/ollama/llama-server` (sin GPU: binario sin CUDA) sirviendo en `:8090` CPU (`-ngl 0 -c 8192 --jinja`); smoke "Say OK" OK (~0.33 tok/s — CPU compartida con sglang 27B: baseline viable en batches nocturnos, no en caliente) |
| 2 | Download MiMo HF 18GB | **BLOQUEADO/INNECESARIO**: el GGUF cubre la inferencia; HF solo si se necesita captura de residuos con torch |
| 3 | R0 math500-subset baseline (sample fija sellada) | **RE-LANZADA 2026-11-04 (nohup, cadena)**: `rig/math500.py --seed 42 --n 50 --sell --tag orig-linux --max-tokens 4000` tras terminar logic-120. Causa del fallo del nohup previo: nunca arrancó (sin logs). Ajustes CPU: REQ_TIMEOUT 600→1800s, logic timeout 180→900s, cap 4000 tok (2.3 tok/s medidos ×2) |
| 3b | Baseline logic_probe 120 Horn | **RUNNING 2026-11-04 (nohup pid 631009)**: `rig/logic_probe.py` (defaults 120, temp 0) → `rig_runs/nohup-logic.log` (buffered: sin output hasta flush/final; vigilar con ps y JSON final) → `rig_runs/logic-mimo-orig.json`; luego math500 (cadena). ETA total ~20-24h @ 2.3 tok/s CPU (4×A5000 ocupadas) |
| 4 | weightspace 0.3.0 upgrade (+openai extra) + ws_runner paridad de stop_reason | pendiente (no crítico: solo etapa A 0.8B) |
| 5 | MCP wsp-challenge en linux | **HECHO**: rutas portables (`__file__`), `creationflags` condicional, auto-detección de GGUF (`_buscar_gguf`: hf_cache→servidos→env→default); `mcp<2`+`gguf` vía `pip --user`; smoke stdio OK (10 tools; engine_status 12/43); `opencode.json` portable (`python3` + ruta relativa) — requiere reinicio de opencode |

## 4.1bis CHECKPOINT sesión 2026-11-03-c (linux, opencode por SSH — RESUMIR AQUÍ si se corta)

Estado al cutoff:
1. **Auth opencode→vLLM RESUELTO**: el 401 era porque vLLM (:8001, de charles) exige `--api-key` y `{env:VLLM_API_KEY}` de opencode solo se resuelve en shell interactivo. Fix: key puesta literal en `~/.config/opencode/opencode.json` (provider vllm, chmod 600). Verificado end-to-end con `opencode run --model vllm/qwen3.8-27b`.
2. **Runner math500 LISTO**: `rig/math500.py` (stdlib-only). CLI: `--endpoint :8090/v1/chat/completions --model mimo --n --seed 42 --sell --temperature 0.3 --max-tokens 8192 --tag --out`. Selftest 44 asserts OK (boxed balanceado con `\{`, `\frac` sin llaves tipo `\frac43`, miles `10,\!080`, subscripts, `^\circ`, `\text`, equivalencia numérica Fractions, reducción gcd). Sellado en `rig/sellados/` (lock global-estricto: mismo seed+n reusa índices y verifica hash del dataset; otro seed/n → error, nunca sobrescribe). Determinismo via sha256 propio estable.
3. **Descargas en curso** (background, reanudables): GGUF Q6_K bartowski en `hf_cache/mimo-gguf` (~7.3GB, parece estancado / verify al retomar: `du -sh hf_cache/mimo-gguf` y `pgrep -af 'hf download'`); HF distill en `hf_cache/mimo-distill-hf` (~4.7GB de 18GB).
4. **Servir MiMo**: plan A = `/opt/local/lib/ollama/llama-server -m <gguf>` (verificar soporte arch `qwen3_5` multimodal, Windows usó llama.cpp b11962); plan B = compilar llama.cpp actual (10–15 min). Port sugerido :8090.
5. **Decisión GPU tomada (opus.2 del roadmap §5)**: pedir liberación de 1 A5000 → GGUF Q6 en llama.cpp full-GPU (~30 min para 50 prompts, ~5–6 h math500 completo) o HF bf16 vLLM batched (~1 h el 500). Fallback si niegan: subset-50 en CPU (~4–5 h, noche).
6. **Próximos pasos inmediatos**: (a) al terminar download GGUF → smoke serve + rig `--selftest` + rig smoke `--n 3` (sin sell) sobre MiMo; (b) asignar GPU dedicada; (c) R0: `rig/math500.py --seed 42 --n 50 --sell --tag orig` (baseline sellado); (d) experimento principal PROTOCOL.md con placebo E4 + frescos.

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
| 11 | **wsp decompile** → traductor semántico de piezas de MiMo a C/pseudo-C. Idea (derivación 2026-11-03): neurona ReLU → cláusula de matching (`if x>θ: W·x`), head de atención → matcher estilo copying (`match prev-token; copy C[t-1]`), MLP entera → lookup de 12288 entradas; juez tipo P1: tests automáticos contra el fragmento traducido. Usos: (a) retos WSP-40 nuevos de "traduce y verifica"; (b) hacer legibles los efectos de una edición (qué circuito/cambio de programa provoca el `subt`); (c) vía de interpretability para la gen-2. Límite conocido (exp 1–3 del paper): viable por PIEZAS (17k–1.5M pesos), no para la red de 9B entera — los pesos son millones de programas superpuestos, sin módulos separables | nada (paralelo; útil para gen-2) |
| 12 | **WSP-Lang 0.1: parser + ejecutor mínimo en el MCP**. Spec completa en `WSP-LANG.md` (2026-11-03): lenguaje declarativo de experimentos sobre pesos (modelo M=(A,W,τ), 10 tipos, 8 primitivas inspect/observe/compare/hypothesize/localize/intervene/evaluate/rollback, ops matemáticas proyección/escalado/activaciones/composición-ordenada, validación de seguridad en runtime, criterio de aceptación ΔB>0 + sin regresiones + budget + set fresco, ExperimentRecord SUPPORTED/REJECTED/INCONCLUSIVE con aprendizaje de fallos). Mapeo actual: inspect/observe/compare/intervene/evaluate OK en wsp.py/rig; hypothesize/localize formalización; rollback/compose ops nuevas. 0.2 previsto: descubrimiento de primitivas, composición automática, meta-programación (RSI con mejora validada externamente, no self-reportada) | retos (jueces), rig y PROTOCOL.md |
| 12bis | **ExperimentRecord v1 (JSON-schema validado)** — decisión 2026-11-03: ADELANTE ahora; parser WSP-Lang completo DIFERIDO post-PROTOCOL. Rationale: (a) el driver no necesita DSL para operar (wsp.py+CLI con budget-barrera ya funciona); (b) el valor real del lenguaje es control/reproducibilidad, no capacidad — el ExperimentRecord validado captura el 80% (memoria ejecutable para gen-2 #9) con el 5% del costo; (c) el parser se diseña data-driven contra los formats reales que PROTOCOL.md genere (lección P5: no construir infra contra formatos hipotéticos). Alcance v1: schema de `{hypothesis, evidence, target, intervention, model_hash, benchmark_version, baseline, candidate, placebo, holdout, conclusion∈{SUPPORTED,REJECTED,INCONCLUSIVE}}` + validador en `engine/record.py` + migración de journal/ existente + entrada automática al final de cada rig | schema configurado y PROTOCOL.md en ejecución |

## 4.1quater WSP-Lang a modo de capa de programación (nota de diseño)

DECISIÓN (2026-11-03, ver #12bis): WSP-Lang queda como **spec de referencia** (`WSP-LANG.md`) + implementación parcial: ExperimentRecord v1 validado AHORA, parser de programas dif ERIDO hasta post-PROTOCOL. El concepto queda: el agente propone intención experimental; el runtime valida y ejecuta (el agente nunca opera pesos directamente ni se auto-evalúa). Las salvaguardas en producción (FORBIDDEN_TENSORS, budget como barrera, sellado de holdouts) son hoy el "sistema de tipos" de facto; el parser formal los unificaria cuando haya evidencia empírica de qué formatos de programa hacen falta.

## 4.2 Ejecutado: wsp decompile (mini #11) — 2026-11-04, sesión windows

Header: `decomp\mimo_head.py` (dims / scan / decompile / judge). Arquitectura de head qwen35 verificada contra llama.cpp `src/models/qwen35.cpp` (master):
- `attn_q` [4096, 8192] = 16 heads × **512d = Q(256) + GATE(256, sigmoid multiplicativo sobre la salida de atención)** → `out *= sigmoid(x@Wgate)` antes de `wo`. GQA 16→4, k/v 256d con RMS-norm per-head + RoPE (64/256 dims, MRoPE multimodal, base 1e7).
- `gguf.dequantize` de gguf-py devuelve shape invertida (orden C): `_tens` con reshape restaurado — trampa documentada.
- **scan offset-aware de los 128 heads (8 capas × 16)**: c1 estática máx 0.044, y con RoPE por offsets δ∈{1..32} máx **c1=0.04 (L23 q8, δ2)** → ~2.5× azar (0.0156 en N=256) pero **NINGÚN matcher rank-1 singleton**: los heads del MiMo son multi-eje. El análisis máximo-rank (paper exp 8, blog P1) NO describe la atención destilada de MiMo como "match prev-token" — consistente con la serie E1–E5 (el valor WSP de este modelo no está en estructuras de atención triviales). Datos: `decomp\scan.json`.
- **Head decompilado L23 q8**: `decomp\head_L23_q8.{npz,c,manifest.json}` con IR ejecutable + juez **PASS** (T1 shapes, T2 IR==ref numpy cos 0.9999999, T3 null-test 1.7e-5 con edición de columna -10%, T4 diag: c1 0.0506, gate_ratio 0.996 — el gate es tan denso como el Q). Sirope a juez con column-scaled en vez de elemento único (1 flip = 1e-6, bajo el ruido).
- Pendiente de la línea de compile: (a) interpretar gate (¿selector de canal residual? probar align de filas Wgate con regions semánticas de semantic-map), (b) capturar activaciones reales por fwd (rig) para verificación funcional del pseudo-C, (c) reto WSP-40 nuevo tipo "traduces una pieza y el juez la veriﬁca".

## 5. Riesgos / trampas activos
- **linux**: 4× A5000 OCUPADAS por vllm Qwen3.8-27b (22.4/24.5GB c/u) — no lanzar MiMo en GPU sin revisar memoria.
- Git: si el remoto se adelanta (la otra máquina también empuja), `git pull --rebase` con stash antes de push.
- millones de GB: un solo server a la vez en 12GB (stop antes de arrancar otro) [regla de la máquina windows].
- antml formato del agentic harness |tool-call: si cambia el modelo, revisar parser.
- no borramos gold-holdouts hasta la primera evaluación sellada final (protocolo).
- Q6_K requant detail: siempre regenerar play a concluyentes.
- opencode MCP exige reinicio después de editar configuración MCP.

## 6. Referencias de archivos del estudio
- Decompiler mini (#11): `decomp\mimo_head.py` + `decomp\head_L23_q8.{npz,c,manifest.json}` + `decomp\scan.json` (semántica verificada vs llama.cpp qwen35.cpp).
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



---

## 7. Proyecto NanoWSP: Pequeño LLM Sintetizado y Programado desde Cero vía WSPLang

**Fecha de inicio:** 2026-11-04  
**Motivación:** En lugar de limitarnos a la cirugía sobre checkpoints preexistentes entrenados con SGD masivo (MiMo-9B, Qwen3.5-0.8B), ¿es posible construir, compilar e imprimir un modelo autoregresivo pequeño (=64$, =2$, =96$) **directamente en el espacio de pesos** usando primitivas de WSPLang y Fast Weight Programming (Schmidhuber & Irie, ICML 2021) sin backpropagation?

### 7.1 Preguntas Fundamentales
1. **Expresividad Gramatical:** ¿Permite la gramática declarativa de WSPLang especificar no solo intervenciones post-hoc, sino arquitecturas completas, contratos de inicialización espectral y presupuestos tensoriales?
2. **Capacidad Asociativa en Espacio de Pesos:** ¿Cuántas reglas factuales o transiciones de n-gramas pueden imprimirse en matrices MLP mediante productos exteriores de rango 1 ($\\Delta W = \\eta \\cdot v \\otimes k^T$) antes de que el ruido de interferencia (cross-talk de Hopfield  \\approx 0.14 d$) destruya las memorias previas?
3. **Desacoplo de la Atención Bilineal:** En capas de atención (Q K^T / \\sqrt{d}) V$, la interacción es bilineal. Modificar $ sin coordinar $ dispersa la atención. ¿Qué operadores de orden superior necesita WSPLang para manipular circuitos de atención?
4. **Propagación Jerárquica sin Gradiente:** Cuando se modifica la Capa 0, la distribución de entrada a la Capa 1 cambia ('representation drift'). ¿Cómo compensa WSPLang esta distorsión sin retropropagación?

### 7.2 Arquitectura y Componentes
- **Configuración de NanoWSP:**
  -  = 96$ (tokens ASCII imprimibles,  \\dots 126$, <unk>, <pad>, \\n).
  - {model} = 64$, {max} = 32$.
  -  = 2$ capas Transformer (RMSNorm + Multi-Head Attention =2, d_k=32$ + MLP GELU {ff}=128$).
  - LM Head  \\times 64$ desenrollado.
  - Parámetros totales: $\\approx 77.000$ floats ($\\approx 300$ KB en SafeTensors).
- **Entregables:**
  - programs/nanowsp_architecture.wspl: Declaración de arquitectura y programa de programación de pesos.
  - engine/nanowsp_runtime.py: Runtime puro en Python/NumPy/PyTorch para inferencia, introspección e impresión asociativa.
  - 	ests/test_nanowsp.py: Suite de verificación de inicialización, inferencia determinista, impresión de memoria y desaprendizaje ortogonal.
  - docs/nanowsp_findings.md: Informe científico formal de limitaciones y propuestas para WSPLang 0.5.
