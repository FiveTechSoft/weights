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
  MATH 6 → 32; sellado 43 → 87/100, 4 → 25/50; placebo de misma norma por debajo de la base (23 / 2). Siguiente:
  compilar la dirección en `down_proj` (plan propio, mismo placebo, mismo sellado). Plan original `lab/probes/cc/PLAN.md`: dirección "concluir"
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
   - 27B TURBO a ciegas (espera el OK del usuario);
   - H4 causal;
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

- **Qwen3.8-27B TURBO — prueba forense ciega (anexo exploratorio, EN COLA tras los 9B).** Plan y predicciones: `lab/probes/delta_patterns/plan_turbo27b.md`.
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
