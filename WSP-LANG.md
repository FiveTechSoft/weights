# WSP-Lang 0.1

## Un lenguaje experimental para programar el espacio de pesos de los LLM

Formalización de la idea como primera versión de un lenguaje de programación para modelos neuronales.

La hipótesis central es:

> Un LLM contiene un programa computacional implícito en sus pesos. Podemos descubrir operaciones primitivas sobre ese programa y combinarlas para producir cambios de comportamiento verificables.

WSP-Lang no pretende ser todavía un lenguaje capaz de modificar cualquier modelo de forma segura. Es una especificación experimental para comenzar a construirlo dentro de FiveTechSoft/weights.

La primera versión tendrá tres objetivos:

1. Describir comportamientos que queremos modificar.
2. Expresar intervenciones localizadas sobre los pesos o las activaciones.
3. Ejecutar experimentos reproducibles que determinen si las intervenciones funcionan.

Principio de diseño: el agente puede proponer programas WSP-Lang, pero el runtime local valida y ejecuta únicamente operaciones permitidas.

Especificación — WSP-Lang 0.1 (propuesta experimental)

- Proyecto: FiveTechSoft/weights
- Modelo objetivo inicial: MiMo-V2.6-Distill-Qwen-9B
- Runtime: MCP local
- Modo inicial: investigación, intervenciones reversibles y evaluación

## 1. Modelo computacional

Un modelo se define como M=(A,W,τ), con A arquitectura, W parámetros y τ configuración de inferencia:

    y = M(x; W)

Un programa WSP es una secuencia P=(o1..on) que transforma el modelo:

    W' = P(W)

El objetivo no es minimizar una pérdida con gradientes, sino descubrir transformaciones que mejoren una capacidad determinada bajo restricciones explícitas.

## 2. Sintaxis inicial (declarativa)

```
experiment reduce_tool_loops:

    model:
        MiMo-V2.6-Distill-Qwen-9B

    objective:
        reduce_repeated_tool_calls

    constraints:
        preserve_planning
        preserve_language
        max_modified_parameters = 0.01%

    discover:
        candidate_directions

    intervene:
        project_out(candidate_direction)

    evaluate:
        loop_benchmark
        planning_benchmark
        language_benchmark

    accept_if:
        loop_rate < baseline.loop_rate
        planning_score >= baseline.planning_score
        language_score >= baseline.language_score
```

El programa describe la intención experimental, no una operación arbitraria sobre el modelo. El runtime resuelve componentes abstractos, valida tipos y autoriza la ejecución.

## 3. Tipos fundamentales

    Model, Tensor, Activation, Direction, Behavior, Hypothesis, Intervention, Benchmark, Experiment, Result

- **Model**: modelo completo (arquitectura, pesos, configuración).
- **Tensor**: identificado por nombre, forma, tipo numérico y ubicación.
- **Activation**: activación observada durante una ejecución.
- **Direction**: dirección o subespacio candidato en pesos o activaciones.
- **Behavior**: propiedad observable (`repeated_tool_calls`, `mathematical_reasoning`, `instruction_following`, `planning`, ...).
- **Hypothesis**: explicación falsable de un comportamiento (mecanismo candidato + predicción comprobable + condiciones de fallo).
- **Intervention**: modificación concreta, reversible y registrada.
- **Result**: métricas, configuración, datos y huellas del modelo de una evaluación.

## 4. Primitivas del lenguaje (8)

1. **inspect(model)** — estructura sin modificar: metadatos, dimensiones, estadísticas. [wsp.py: OK]
2. **observe(model, dataset, capture)** — traza comportamiento (activaciones, residuos, atención, salidas, tool-calls) solo sobre componentes autorizados. [wsp.py: capture_residual/hook OK]
3. **compare(traces_a, traces_b)** — diferencias estadísticas entre grupos de ejecuciones; no constituye por sí misma explicación causal. [wsp.py: act-scores OK]
4. **hypothesize(behavior, evidence)** — hipótesis comprobable con mecanismo, predicción y condiciones de fallo. [freeform, pendiente de formalizar]
5. **localize(hypothesis, search_space)** — candidatos (capa, tensor, canales, dirección); no presupone unicidad. [resuelto parcialmente: E1-E5]
6. **intervene(model, operation, target, strength)** — transforma siempre en copia experimental; el original permanece inmutable. [wsp.py: OK]
7. **evaluate(candidate, benchmark)** — evalúa el objetivo y las capacidades potencialmente degradadas. [rig: OK]
8. **rollback(candidate)** — restaura el estado previo vía manifiesto. [parcial: manifesto+origen, falta op explícita]

## 5. Operaciones matemáticas iniciales

- **Escalado de dirección**: W' = W + α·d (relación explícita entre dirección y subespacio intervenido).
- **Proyección**: P_d = d·dᵀ; W_d = W·P_d; reducción W' = W − α·W_d (solo con dims/interpretación compatibles con el tensor objetivo).
- **Intervención sobre activaciones**: h' = h − α·d·dᵀ·h — modifica la ejecución sin tocar pesos; útil para probar hipótesis antes de editar.
- **Composición**: P = P2∘P1; el orden NO es irrelevante (P2(P1(W)) ≠ P1(P2(W))) ⇒ el runtime registra el orden exacto.

## 6. Sistema de tipos y seguridad

```
validate(intervention):
    verify_model_identity()
    verify_target_exists()
    verify_target_is_allowed()
    verify_dimensions()
    verify_numeric_type()
    verify_magnitude_limit()
    verify_parameter_budget()
    verify_reversibility()
    return authorized
```

Reglas iniciales:
1. El checkpoint original es inmutable.
2. No se permite escritura arbitraria de archivos.
3. No se permite ejecución arbitraria de código Python libre.
4. No se editan tensores fuera de la lista autorizada (FORBIDDEN_TENSORS: head/lm_head).
5. Cada intervención lleva manifiesto.
6. Cada modificación es reversible.
7. Los benchmarks de confirmación (sellados) jamás se usan para descubrir.

La política de seguridad vive en el runtime, no solo en las instrucciones del agente. [mappings a wsp.py: FORBIDDEN_TENSORS/rangos OK; _budget_check como barrera OK; faltan reversibilidad explícita y no-ejecución-arbitraria en el parser]

## 7. Programa de ejemplo (E2_reduce_loops)

```
experiment E2_reduce_loops:
    model = load_model("MiMo-V2.6-Distill-Qwen-9B")
    baseline = evaluate(model, benchmark=loop_diagnostic)
    productive = observe(model, dataset=productive_tasks)
    repetitive = observe(model, dataset=repetitive_tasks)
    evidence = compare(productive, repetitive)
    hypothesis = hypothesize(behavior=repeated_tool_calls, evidence=evidence)
    candidates = localize(hypothesis, search_space=allowed_components)
    for candidate in candidates:
        intervention = intervene(model, operation=activation_projection, target=candidate)
        result = evaluate(intervention, benchmark=validation_suite)
        record(hypothesis, candidate, intervention, result)
    best = select_candidate(results, criteria=objective_and_regression)
    confirm(best, benchmark=sealed_holdout)
```

El programa no afirma que la dirección sea correcta: convierte una hipótesis en un experimento reproducible.

## 8. Criterio de aceptación

- ΔB > 0 en la métrica del comportamiento objetivo, y
- ΔR_i ≥ −ε_i para cada capacidad de regresión R_i (sin puntuación agregada que oculte degradaciones), y
- modified_parameters ≤ B_max, y
- superación de evaluación independiente (set fresco sellado).

## 9. Registro de conocimiento

```
ExperimentRecord:
    hypothesis, evidence, target, intervention, model_hash,
    benchmark_version, baseline_result, candidate_result,
    placebo_result, holdout_result, conclusion

conclusion ∈ {SUPPORTED, REJECTED, INCONCLUSIVE}
```

Un experimento fallido NO se elimina: se convierte en conocimiento reutilizable.(Ej registral: "weight_drift identifies behavior → rejected → lección: el drift solo es insuficiente para establecer causalidad comportamental" = nuestro E1.)

## 10. Evolución prevista: WSP-Lang 0.2

- **A. Descubrimiento de primitivas**: el agente propone operaciones nuevas a partir de resultados experimentales.
- **B. Composición automática**: combinar intervenciones de efectos compatibles demostrados.
- **C. Meta-programación**: el agente mejora el propio procedimiento de investigación (vía RSI). El objetivo no es auto-modificarse indiscriminadamente sino mejorar progresivamente el proceso de descubrimiento de modificaciones eficaces.

## Hipótesis final

Si existen regularidades computacionales reutilizables en el espacio de pesos, debería ser posible descubrir primitivas que construyan intervenciones complejas y verificables. La primera tarea de WSP-Lang NO es demostrar que existen: es construir el entorno para descubrirlas, probarlas y refutarlas.

**WSP-Lang 0.1 convierte la edición de pesos en un proceso de programación experimental.**

Siguiente pieza: parser + ejecutor mínimo dentro del MCP de FiveTechSoft/weights — intérprete que acepta programas como E2, valida cada operación y genera manifiesto y resultados automáticamentE.
