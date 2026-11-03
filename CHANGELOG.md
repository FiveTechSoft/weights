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

## Reglas

1. Cada cambio = una entrada única con ID estable (`E1`, `E2`, ...).
2. El veredicto (`keep`/`discard`/`inconclusive`) va siempre.
3. Si el cambio es rechazado, **se documenta igual**: el conocimiento de por qué falló es valioso.
4. Los holdouts (MATH-500, code, IF) NO alimentan el diseño: solo se evalúan al final.
5. Cada ficha `docs/changelog/` debe ser autocontenida para publicación.
