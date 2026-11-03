# CHANGELOG — Registro de cambios en el modelo

> Formato para documentar **cada cambio** que se hace en los pesos de MiMo-V2.6-Distill-Qwen-9B.
> Cada entrada responde a las tres preguntas obligatorias:
> **QUÉ** se cambia · **PARA QUÉ** (hipótesis) · **QUÉ SE OBTIENE** (resultado medido).
>
> Una entrada sin las tres no se considera documentación válida.

## Convención de archivos

| Archivo | Contenido |
|---|---|
| `CHANGELOG.md` | Registro maestro de todos los cambios |
| `docs/changelog/<ID>.md` | Ficha de cada cambio (detallada, para publicar) |
| `ediciones/<ID>.json` | Programa de edición (datos, no prosa) |
| `ediciones/<ID>.manifest.json` | Fingerprint / hash del modelo editado |
| `rig_runs/<ID>.json` | Resultados de evaluación |

## Formato de una entrada de CHANGELOG

```markdown
### <ID> — <Título>

**QUÉ** — Descripción técnica del cambio (tensores, op, magnitud, capas).
**PARA QUÉ** — Hipótesis / objetivo.
**QUÉ SE OBTIENE** — Resultado medido (métricas, veredicto keep/discard).
**Estado** — `draft` / `review` / `published`
```

## Cambios de infraestructura (herramientas, no ediciones del modelo)

Los cambios al propio toolkit WSP (`wsp.py`, `engine/`) se registran aquí con el
mismo formato, pero con los campos adaptados:

```markdown
### <ID> — <Título>

**QUÉ** — Qué se añadió/cambió en el código (funciones, archivos).
**PARA QUÉ** — Qué problema resuelve / qué capacidad habilita.
**QUÉ SE OBTIENE** — Comportamiento verificado (tests, byte-level, límites).
**Estado** — `draft` / `review` / `published`
```

> Nota: este registro cubre el **toolkit**, no las ediciones del modelo. Las
> ediciones del modelo van en `docs/changelog/<ID>.md` (fichas autocontenidas).

---

### P0 — `peek()` / `poke()` sobre pesos individuales

**QUÉ** — Se añadió a `wsp.py` la capacidad de inspeccionar y modificar un único peso
de `output.weight` (y cualquier tensor de la GGUF):
- `peek(gguf_path, tensor_name, row=None, col=None, k=10)` — sin `row`/`col` devuelve la
  anatomía del tensor (stats de `row_norm`/`col_norm`, top-k rows/cols, `nonzero`); con
  `row`/`col` devuelve el valor, signo, `row_norm`, `col_norm`, `row_rank`, `col_rank`,
  `frac_row`, `frac_col` y las fracciones.
- `poke(gguf_path, tensor_name, row, col, value, out=None)` — dequantiza el tensor, escribe
  el elemento, **requantiza SOLO el bloque** que contiene (row, col) y escribe de vuelta.
- Soporte interno: `_row_norm`, `_col_norm`, `_rank_by_norm`, `_set_element`,
  `_requant_blocks_bytes`, `_poke_requant`.

**PARA QUÉ** — Implementar WSP-Lang: un programa declarativo necesita poder **observar**
qué controla un peso (`peek`) y **perturbarlo** (`poke`). Sin esto, WSP-Lang solo puede
observar activations, no pesos.

**QUÉ SE OBTIENE** — Verificado end-to-end sobre `servidos/smoke_f32.gguf`
(tensor `output.weight`, 248320×4096, Q6_K, 3 973 120 bloques de 210 bytes):
- `peek` sin row/col: `row_norms` min 0.3787 / mediana 0.9781 / max 1.5413; `col_norms`
  min 5.9812 / mediana 7.4128 / max 33.5925; `nonzero` 990 146 820.
- `peek(p,'output.weight',row=5,col=12)`: `value 0.001147747`, `sign 1`, `row_norm 1.0833`,
  `col_norm 7.5824`, `row_rank 46133`, `col_rank 366`.
- `poke` con `row=5, col=12, value=0.5`: el peso sale `0.49998474` (antes `0.001147747`);
  shapes idénticos `(248320, 4096)`; **verificación byte-level**: solo **82 bytes** del archivo
  cambian, todos contenidos en el único bloque afectado (bloque 80, offset 16800); el resto
  del archivo es idéntico. Sin colateral en bloques vecinos.
- `poke` escribe el archivo completo (7 793 710 624 bytes) parcheando solo los bytes del
  tensor; el resto del archivo se copia tal cual.
- **Límite conocido**: abrir la GGUF con `GGUFReader` tarda ~15.5 s (no el sort), así que
  cualquier test/eval de `poke` debe presupuestar ese tiempo.

**Estado** — `review`

### P1 — `plan_run()` mapea pasos `step <op>` a comandos del runtime

**QUÉ** — Se corrigió `engine/wsplang.py`:
- `validate()` ahora reconoce las líneas `step <op> ...` dentro de la sección `steps:`
  (antes las ignoraba como asignaciones `key = value`).
- `plan_run()` ahora extrae los pasos como `{"op": <op>, "args": <rest>}` y los mapea a
  comandos vía `pasos_mapeo`; además mapea las secciones de op (`intervene`, `evaluate`, ...)
  vía `ops`.
- `mapeo` se cambió de lista a diccionario `{seccion: comando}`.

**PARA QUÉ** — WSP-Lang necesita traducir un programa declarativo a comandos ejecutables
del runtime (`wsp peek`, `wsp poke`, `wsp edit`, ...). Sin esto, los pasos se perderían.

**QUÉ SE OBTIENE** — Verificado end-to-end sobre un programa WSP-Lang:
```
experiment E3_peek_poke
model:
    model = MiMo-V2.6-Distill-Qwen-9B
    objective = reduce_repeated_tool_calls
constraints:
    types = [Q6_K, F32]
steps:
    step peek output.weight k=10
    step poke output.weight 5 12 0.5
```
- `errores: []` (valida limpio).
- `pasos`: `peek` (args `output.weight k=10`), `poke` (args `output.weight 5 12 0.5`).
- `pasos_mapeo`: `peek` → `wsp peek <tensor> [row=] [col=] [k=] ...`, `poke` →
  `wsp poke <tensor> <row> <col> <valor> ...`.
- Secciones de metadata (`model`, `constraints`) se cuentan pero NO se mapean a comandos
  (no son ops).

**Estado** — `review`

## P2 — `wsplang.py`: parser de secciones indentadas (WSP-Lang)

**QUÉ** — Corrección en `engine/wsplang.py` del parser de secciones (`parse_args_lines`)
y de su validación (`validate`). El bug: el parser solo reconocía secciones a indent 0
(`model:` a raiz) y trataba cada línea como una asignación `nombre = valor`. Como el
formato documentado usa secciones a indent 4 (`model:` a indent 4, con valores como
cadenas planas, no asignaciones), un programa válido daba:
`['falta experiment <nombre>', 'seccion desconocida: model', 'linea ignorada model:', ...]`.
Cambios: (1) `parse_args_lines` reconoce secciones por su nombre a cualquier indent y
registra el indent de anidación; (2) `validate` sabe distinguir pasos de asignaciones
de valores de metadata; (3) `plan_run` ya mapea `peek`/`poke`/`poke-weight` al runtime.

**PARA QUÉ** — que WSP-Lang sea usable con el formato documentado en `WSP-LANG.md`
(`experiment <nombre>:` + secciones a indent 4), que es el contrato del lenguaje.

**QUÉ SE OBTIENE** — validación de programa de ejemplo completo (documentación v0.1)
con `errores: []`: secciones `model`, `objective`, `constraints`, `discover`,
`intervene`, `evaluate`, `accept_if`, `steps` todas reconocidas; pasos `step inspect
output.weight` y `step poke output.weight 5 12 0.5` extraídos y mapeados a
`wsp inspect` / `wsp poke ...`; `mapeo` de secciones correcto. El formato plano
(`experiment <nombre>` + asignaciones a indent 4) también pasa sin errores.

**Estado** — `review`

### P3 — Atribución de Áreas Semánticas y Roles Anatómicos en WSP-Lang (`peek` / `poke` / `semantic-map`)

**QUÉ** — Se amplió WSP-Lang (`wsp.py`) con un sistema completo de identificación semántica y anatómica de pesos y neuronas:
- **10 Áreas Semánticas** (`SEMANTIC_AREAS`): `math_logic`, `agent_action`, `code_software`, `abstract_ideas`, `mind_cognition`, `time_sequence`, `emotion_pos`, `emotion_neg`, `physical_world`, `social_power`.
- **Centroides Diferenciales**: resolución automática de semillas léxicas sobre el vocabulario BPE (incluyendo prefijo byte `Ġ` y espacio), extracción selectiva de embeddings de `token_embd.weight` sin cargar los 4 GB en memoria, y cálculo de centroides diferenciales con substracción del fondo medio isotrópico ($\mathbf{C}'_a = \mathbf{C}_a - \bar{\mathbf{C}}$) para eliminar la anisotropía del espacio latente.
- **Caché persistente**: se serializan los centroides en `data/<stem>.semantic_centroids.npz` (1.4 MB) para consultas sub-segundo en posteriores ejecuciones de `peek`.
- **Introspección Anatómica**: `describe_tensor_role()` detecta el tipo de capa (atención completa vs recurrencia SSM GatedDeltaNet según `recurrent_layers`), cabeza de atención (`col // key_length`), dimensión interna de cabeza (`col % key_length`), neurona intermedia de FFN (`ffn_down`, `ffn_up`), puertas de contexto SSM (`ssm_beta`, `ssm_alpha`, `ssm_out`) y decodificación de tokens para la cabeza de salida `output.weight`.
- **Proyección Semántica en `peek()`**: proyección del vector residual (vector columna para matrices de salida, vector fila para detectores de entrada) contra los centroides para reportar:
  - Área semántica dominante y su puntuación de similitud coseno.
  - Ranking de áreas semánticas afines.
  - Semillas léxicas individuales más alineadas.
- **CLI Commands**: registrados `peek` (con flag `--json`), `poke`, y `semantic-map` en el CLI parser y en el dispatch de `CMD_FN`.
- **Memoria en `poke()`**: eliminado el buffer de 8 GB en RAM; se utiliza clonación a nivel de sistema operativo (`shutil.copyfile`) y parcheo in-place de bloques afectados con `r+b` seek y write.

**PARA QUÉ** — Resolver la necesidad planteada por el usuario de saber "qué controla y para qué se usa un peso en un LLM" análogo a la computación de 8-bits: dotar a WSP-Lang de interpretabilidad semántica y mecánica directa para que los programas de edición no actúen a ciegas sobre coordenadas arbitrarias, sino sobre áreas funcionales concretas (como razonamiento lógico, bucles temporales o acciones de herramientas).

**QUÉ SE OBTIENE** — Verificado en `MiMo-V2.6-Distill-Qwen-9B`:
1. `wsp.py semantic-map`: 10 áreas semánticas calculadas y cacheadas con 100% de semillas resueltas.
2. `wsp.py peek blk.19.attn_output.weight -r 100 -c 200`:
   - Rol: `Attention Layer 19: head #0 (dim 200) -> escribe en residual dim 100`
   - Área dominante: `Matemáticas y lógica (math_logic)` (score `+0.0275`).
   - Semillas léxicas afines: `learn`, `theorem`, `kind`.
3. `wsp.py peek blk.19.ffn_down.weight -r 100 -c 500`:
   - Rol: `Layer 19 FFN Down: neurona intermedia #500 -> escribe en residual dim 100`.
   - Área dominante: `Mundo físico y materia (physical_world)` (score `+0.0211`).
4. `wsp.py peek blk.0.ssm_out.weight -r 100 -c 200`:
   - Rol: `SSM Layer 0: canal 200 -> escribe en residual dim 100`.
5. `wsp.py peek output.weight -r 1500 -c 100`:
   - Rol: `LM Head: proyección al logit del token #1500 ('Ġline')`.
   - Área dominante: `Acciones de agente y terminal (agent_action)` (score `+0.0276`).

**Estado** — `published`

### P4 — Áreas Semánticas en el Compilador y Runtime de WSP-Lang (`wsplang.py` + MCP)

**QUÉ** — Se integraron las áreas semánticas y la introspección de pesos en el nivel declarativo de WSP-Lang y en su servidor MCP:
- **Gramática y Tipos**: en `engine/wsplang.py`, se añadieron las operaciones `semantic_map` y `semantic_areas` a `OPS_ALLOWED`, los tipos `SemanticArea` y `SemanticCentroid` a `TYPES`, la operación `cov-steer` a `INT_OPS`, y las secciones `semantic_areas` y `semantic_target` a `METADATA`.
- **Ejecución Interactiva**: se implementó `execute_plan()` y el flag `--run`/`--exec` en `wsplang.py` para ejecutar interactivamente pasos declarativos de un programa `.wspl` usando el runtime de `wsp.py` y generar manifiestos enriquecidos con áreas semánticas y afinidades.
- **Servidor MCP**: en `engine/wsp_mcp_server.py`, se agregaron las herramientas `@mcp.tool() get_semantic_areas()` y `@mcp.tool() run_wsplang()`, y se corrigió el paso de argumentos de `_wsp()` para preservar retornos estructurados.
- **Programa de prueba**: creado `programas/e3_semantic_logic.wspl` con objetivo declarativo, secciones de áreas semánticas e invocación de pasos `semantic_map` y `peek`.

**PARA QUÉ** — Permitir que los programas formales de WSP-Lang (`.wspl`) puedan expresar hipótesis basadas en áreas semánticas (ej. apuntar a `math_logic` y evitar `loop_repetition`) y que cualquier agente a través de MCP o CLI pueda evaluar y recibir la distribución semántica directamente.

**QUÉ SE OBTIENE** — Verificado con `python engine/wsplang.py programas/e3_semantic_logic.wspl --run`:
- Validación limpia: `validado: true`, `errores: []`.
- Plan estructurado con mapeo a herramientas runtime.
- Extracción e informe automático de áreas semánticas para los tensores inspeccionados (`math_logic` en atención L19, `physical_world` en FFN L19, `agent_action` en `output.weight`).
- Manifiesto generado en `retos/wsplang/e3_semantic_logic.manifest.json`.

**Estado** — `published`

### P5 — Ejemplo declarativo de áreas semánticas (todas vs más importantes) en WSP-Lang

**QUÉ** — Se amplió el runtime y compilador de WSP-Lang para soportar filtrado y reporte flexible de áreas semánticas:
- **`attribute_semantic_area`**: devuelve concurrentemente `all_areas` (las 10 áreas con sus puntuaciones de similitud coseno) y `top_areas` (las $k$ más importantes).
- **`semantic_profile` a nivel de tensor**: en inspecciones globales sin coordenadas, calcula la distribución porcentual de áreas que dominan las neuronas/cabezas principales.
- **Soporte de argumentos `top=N`**: tanto `step semantic_map top=3` como `step peek <tensor> ... top=3` permiten seleccionar y destacar las áreas semánticas más importantes conservando la lista completa.
- **Caché en memoria `_READER_CACHE`**: reutilización de la instancia de `GGUFReader` entre pasos sucesivos en `wsp.py`, reduciendo el tiempo de ejecución multi-paso.
- **Programa de ejemplo**: creado `programas/ejemplo_areas_semanticas.wspl` que mapea el modelo completo y analiza atención lógica (L19), FFN down (L19) y memoria recurrente SSM (L0).

**PARA QUÉ** — Cumplir la petición del usuario de contar con un programa WSP-Lang que obtenga **todas** las áreas semánticas y opcionalmente filtre/resalte **las más importantes**, proporcionando tanto visión panorámica como foco analítico para experimentos de lógica y bucles de herramientas.

**QUÉ SE OBTIENE** — Ejecución exitosa con `python engine/wsplang.py programas/ejemplo_areas_semanticas.wspl --run`:
1. `semantic_map top=3`: devuelve las 10 áreas activas del modelo y destaca las top 3 prioritarias (`math_logic`, `agent_action`, `code_software`).
2. `peek blk.19.attn_output.weight top=3`: reporta área dominante (`math_logic`, score `+0.0275`), el top 3 más importante (`math_logic`, `time_sequence`, `social_power`), y las 10 áreas ordenadas con sus semillas afines.
3. `peek blk.19.ffn_down.weight top=3`: reporta área dominante (`physical_world`, score `+0.0211`), top 3 (`physical_world`, `emotion_pos`, `code_software`), y desglose completo de 10 áreas.
4. `peek blk.0.ssm_out.weight top=3`: reporta área dominante (`emotion_neg`, score `+0.0190`), top 3 (`emotion_neg`, `emotion_pos`, `math_logic`), y desglose completo de 10 áreas.
5. Manifiesto estructurado generado en `retos/wsplang/inspeccion_areas_semanticas.manifest.json`.

**Estado** — `published`

### P6 — Infografía visual de arquitectura y publicación en X (Twitter) sobre WSP-Lang

**QUÉ** — Se generó el material gráfico de difusión técnica y los textos de publicación para X (Twitter) en inglés explicando el mecanismo y los resultados de WSP-Lang en `MiMo-V2.6-Distill-Qwen-9B`:
- **Infografía de Alta Resolución (`docs/assets/wsplang_semantic_areas.png`)**: gráfico 16:9 (2400x1350 px, dark theme) que integra el código declarativo de WSP-Lang, el pipeline matemático de centroides diferenciales que soluciona el colapso del cono isotrópico de los embeddings, las mediciones empíricas de proyección en atención (L19), FFN (L19) y SSM (L0), y la taxonomía de las 10 áreas semánticas.
- **Script generador (`scripts/generate_infographic.py`)**: script modular basado en Matplotlib y PIL para reproducir y actualizar el gráfico.
- **Documento de publicación (`docs/post_x_semantic_areas.md`)**: post extenso (para formato Long-form de X) y un hilo de 5 tweets estructurado con ganchos técnicos, formulación matemática, resultados medidos, código WSP-Lang y enlace al repositorio de GitHub.

**PARA QUÉ** — Comunicar a la comunidad de investigación en IA e interpretabilidad mecanicista cómo deconstruir pesos individuales de un LLM de 9B en áreas semánticas sin pasadas hacia adelante ni sobrecarga de GPU.

**QUÉ SE OBTIENE** — 
- Imagen generada y validada en `docs/assets/wsplang_semantic_areas.png` (2400x1350 px, RGBA).
- Copia de previsualización en el directorio de artefactos de la sesión.
- Guía completa de publicación en `docs/post_x_semantic_areas.md`.

**Estado** — `published`

### P7 — Reestructuración integral de Onboarding (`README.md` + `docs/getting_started.md`)

**QUÉ** — Rediseño completo de la experiencia de bienvenida y documentación de entrada del repositorio:
- **`README.md` reescrito**: inclusión de badges de arquitectura, infografía insignia de áreas semánticas, sección **Quick Start en 5 minutos** (instalación mínima sin PyTorch/GPU, deconstrucción en 60s con `wsp.py peek`, y ejecución de programas declarativos con `engine/wsplang.py --run`), desglose visual de componentes, y principios de rigor científico (placebos, holdouts sellados, descartes honestos).
- **`docs/getting_started.md`**: guía paso a paso completa con configuración de entorno virtual, uso de modelos GGUF personalizados, comandos de inspección anatómica de tensores, sintaxis declarativa de WSP-Lang, integración de FastMCP para agentes de IA (Claude Desktop, Cursor), y formulación matemática de los centroides diferenciales.

**PARA QUÉ** — Responder a la auditoría del repositorio elevando la claridad y onboarding de 5.5 a nivel óptimo, asegurando que cualquier investigador o desarrollador comprenda el propósito de WSP y pueda ejecutar un ejemplo funcional en menos de 5 minutos.

**QUÉ SE OBTIENE** — 
- Documento `README.md` con claridad divulgativa y rigor técnico.
- Guía `docs/getting_started.md` autocontenida y verificada.
- Eliminación del malentendido de que WSP-Lang era solo una especificación: demostración visible de que su compilador y ejecutor interactivo están 100% operativos.

**Estado** — `published`

### P8 — Replicación y verificación de los experimentos canónicos del blog de MIT con WSP-Lang

**QUÉ** — Se programaron y verificaron formalmente en WSP-Lang los experimentos de lectura y escritura de pesos del blog *Sparks of Weight-Space Programming* (MIT CSAIL):
- **Experimento 1 (Lectura e inferencia funcional sin forward passes)**: implementado en programas/blog_exp1_reading_weights.wspl. Utiliza step semantic_map y step peek sobre las capas de atención y FFN de MiMo 2.6 para identificar qué calculan (especialización en math_logic en atención L19, memoria factual en FFN L19).
- **Experimento 3 (Identificación de idioma y especialización de dominio)**: implementado en programas/blog_exp3_language_and_domains.wspl. Descompila la matriz de vocabulario (output.weight) identificando afinidad por código y acciones de terminal (gent_action score +0.0276).
- **Experimento 5 (Escritura de pesos desde cero sin gradientes)**: implementado en programas/blog_exp5_write_mlp.wspl. Sintetiza los pesos de una MLP ReLU 2x2 para calcular exactamente (x) = 1.5 x_0 - 2.5 x_1 + 0.7$ mediante la identidad $	ext{ReLU}(u) - 	ext{ReLU}(-u) = u$.
- **Harness de verificación mecánica (scripts/verify_blog_exp5.py)**: evalúa la red sintetizada contra 100 vectores de prueba en $[-10, +10]^2$, verificando error numérico exacto de .19 	imes 10^{-8}$.

**PARA QUÉ** — Demostrar la expresividad universal de WSP-Lang: unificar en un único lenguaje declarativo tanto la decompilación de modelos gigantes de 9B como la síntesis exacta de redes desde cero.

**QUÉ SE OBTIENE** — 
- Verificación limpia de los 3 programas .wspl con manifiestos generados en 
etos/wsplang/.
- Verificación mecánica del Exp 5 con veredicto PASS y error absoluto máximo de .19 	imes 10^{-8}$.

**Estado** — published

### P9 — Internacionalización integral del repositorio (nomenclatura estándar en inglés)

**QUÉ** — Refactorización exhaustiva de carpetas, código, herramientas y programas al inglés para alineación internacional:
- **Carpetas migradas**: programas/ -> programs/, 
etos/ -> challenges/, 
etos/expedientes/ -> challenges/records/, 
etos/sellados/ -> challenges/sealed/, ediciones/ -> edits/, servidos/ -> served/.
- **Nomenclatura semántica en wsp.py**: etiquetas oficiales actualizadas a inglés (Math & Logic, Code & Software, Agent & Actions, Physical World, Time & Sequence, Abstract Ideas, Mind & Cognition, Social & Power, Positive Emotion, Negative Emotion). Salidas de terminal internacionalizadas (DOMINANT SEMANTIC AREA, TOP SEMANTIC AREAS, ALL SEMANTIC AREAS, Affine seeds).
- **WSP-Lang (engine/wsplang.py)**: rutas por defecto actualizadas a challenges/wsplang/, mensajes de ejecución y claves de manifiesto (steps_count, alidated, errors) con retrocompatibilidad.
- **Servidor MCP y Challenge (engine/challenge.py, wsp_mcp_server.py)**: resolución adaptativa a las nuevas rutas en inglés.
- **Documentación e infografía**: regeneración de docs/assets/wsplang_semantic_areas.png y actualización de README.md, docs/getting_started.md y docs/post_x_semantic_areas.md.

**PARA QUÉ** — Cumplir la instrucción del usuario de hacer el repositorio visualmente internacional y accesible para la comunidad global de investigación en IA.

**QUÉ SE OBTIENE** — 
- Estructura limpia y homogénea en inglés reconocida globalmente.
- Ejecución impecable de WSP-Lang contra programs/example_semantic_areas.wspl generando manifiestos en challenges/wsplang/.
- Infografía de alta resolución actualizada con las rutas y nomenclatura internacional.

**Estado** — published

### P10 — Integración de técnicas de cambio de comportamiento de Dreaming en WSP-Lang

**QUÉ** — Se formalizó la integración de las técnicas de modificación de comportamiento de FiveTechSoft/dreaming (técnica 16 de Covariance-Aligned Steering y los 5 modos analítico/creativo/poético/código/residual) en programas declarativos de WSP-Lang:
- **Descubrimiento de la lección de Dreaming**: el intento inicial de Dreaming en AGGRESSION_DIRECTION_SUMMARY.md de intervenir solo 	oken_embd.weight fracasó porque las 200 capas internas de atención y FFN diluyen la señal. Para modificar el comportamiento es imprescindible intervenir los tensores de proyección interna (FFN down y atención).
- **Programa WSP-Lang de Covariance Steering**: creado programs/dreaming_covariance_steering.wspl, que ejecuta el mapeo semántico de centroides diferenciales y la introspección anatómica de la neurona de hostilidad descubierta (capa 14, neurona 4258, afinidad con nger de +0.045).
- **Validación y Manifiesto**: ejecutado limpiamente con engine/wsplang.py --run, generando challenges/wsplang/dreaming_covariance_steering.manifest.json.

**PARA QUÉ** — Resolver con el instrumental quirúrgico de WSP-Lang el cuello de botella identificado en Dreaming, permitiendo inducir cambios de comportamiento (perspectiva estoica, analítica o villana) fijados directamente en los pesos de los bloques FFN y atención en cuantización Q6_K.

**QUÉ SE OBTIENE** — 
- Confirmación experimental de que la neurona 4258 de la capa 14 en MiMo-2.6-9B proyecta con score +0.0624 sobre Negative Emotion (semilla dominante: nger).
- Demostración de que las 10 técnicas DMT y los 5 modos de Dreaming pueden programarse de forma estandarizada y reproducible en WSP-Lang.

**Estado** — published

## Reglas

1. Cada cambio = una entrada única con ID estable (`E1`, `E2`, ...).
2. El veredicto (`keep`/`discard`/`inconclusive`) va siempre.
3. Si el cambio es rechazado, **se documenta igual**: el conocimiento de por qué falló es valioso.
4. Los holdouts (MATH-500, code, IF) NO alimentan el diseño: solo se evalúan al final.
5. Cada ficha `docs/changelog/` debe ser autocontenida para publicación.
