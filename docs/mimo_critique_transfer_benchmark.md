# WSP-Lang 0.7: Transferencia del Subespacio de Autocrítica (u_critic) en MiMo-2.6-9B

Este informe documenta la investigación empírica sobre la extracción, verificación causal y migración del **Circuito de Autocrítica y Reflexión ($u_{\text{critic}}$)** desde trayectorias de modelos superiores hacia **MiMo-2.6-Distill-9B**, evaluando el impacto antes/después, la invariancia de tamaño de modelo y la comparativa de costes/tiempos frente a un entrenamiento convencional.

> [!WARNING]
> **RETIRADO (auditoría 2026-11-06).** Este informe no es evidencia de que el vector $u_{\text{critic}}$ cambie el razonamiento de MiMo. Se conserva como registro de lo que se afirmó.
> - **El "contrato causal" (§5) es un proxy sintético.** `rig/exp_mimo_self_critique_v1.py` nunca ejecuta el modelo: su `eval_fn` puntúa $W \cdot \sin(\text{hash del prompt})$ proyectado sobre $u$, y el "placebo" es la $W$ sin cambios. Por eso salen 50 % en los cuatro brazos. No es un test de comportamiento.
> - **$u_{\text{critic}}$ no se extrajo de activaciones**, sino de token embeddings de pares escritos a mano (results.md §15.4).
> - **Las tablas antes/después (§4, 12 preguntas; §8, 5 preguntas) no tienen fichero de resultados guardado.** Las respuestas "después" de `rig/test_reflection_prompt.py` vienen de prompts que ya piden verificar paso a paso, y las respuestas "antes" de `rig/live_comparison_showcase.py` son cadenas fijas en el código (`pre_mimo_answer`), no salidas del modelo.
> - **La tabla de costes (§3)** compara el coste de calcular un vector con el de entrenar; como el efecto no está demostrado, no muestra ninguna ventaja. La "garantía anti-olvido" no existe: las regresiones se miden con forward passes.
>
> Estado verificado de todos los resultados: tabla "Status of the results" del [README](../README.md).

---

## 1. El Problema: El "Compromiso Patológico" en Modelos de 9B y 1B

Los modelos de menor tamaño (como MiMo-9B o TinyLlama-1.1B) padecen frecuentemente del sesgo de **compromiso patológico**:
- Ante preguntas trampa, acertijos de razonamiento aritmético o restricciones lógicas sutiles, su ruta *feed-forward* intuitiva emite una primera respuesta errónea.
- Al carecer de un vector interno de verificación activo, el modelo no pausa ni cuestiona su hipótesis inicial, sino que genera racionalizaciones consistentes para justificar el error.

En modelos frontera (MiMo-32B/72B, DeepSeek-R1, Qwen-2.5-32B), el razonamiento exhibe una firma de activación contrastiva durante la fase de reflexión:
> *"Espera, déjame verificar el paso 2... si la pelota cuesta $0.10, el total sería $1.20, lo cual contradice la premisa. Recalculemos..."*

Ese estado de activación define un subespacio direccional $u_{\text{critic}} \in \mathbb{R}^{4096}$ en las capas intermedias (Capas 13 a 19).

---

## 2. Extracción de $u_{\text{critic}}$ (Ingeniería de Representaciones)

Ejecutado con [`rig/extract_critic_vector.py`](../rig/extract_critic_vector.py):
- **Dataset Contrastivo:** 30 pares de trayectorias (Reflexión/Verificación Activa vs Impulso Acrítico/Trampa) en aritmética, conteo de caracteres, física, lógica y código.
- **Acceso SafeTensors:** Lectura memory-mapped de `embed_tokens.weight` (`[248320, 4096]`) y capas intermedias sin sobrecarga de VRAM.
- **Descomposición SVD:** Vector singular principal $u_{\text{critic}}$ normalizado ($\|u\|_2 = 1.0$).
- **Test de Hipótesis Nula por Permutación (100 shuffles):**
  - Norma delta real ($\|\Delta\|$): **0.06308**
  - Norma media nula permutada: **0.03987**
  - Ratio de separación de señal: **1.58x**
  - P-valor empírico: **$p = 0.0000$** ($< 0.001$, separación estadísticamente significativa).
- **Tiempo de extracción:** **1.91 segundos** en CPU.
- **Artefactos generados:**
  - [`data/u_critic_L16.npz`](../data/u_critic_L16.npz)
  - [`data/u_critic_control_vector.gguf`](../data/u_critic_control_vector.gguf) (513.5 KB, envolvente gaussiana sobre capas 10–22, pico en L16)
  - [`rig_runs/u_critic_extraction_report.json`](../rig_runs/u_critic_extraction_report.json)

---

## 3. Comparativa: Entrenamiento Convencional (Fine-Tuning / RL) vs Weight-Space Programming (WSP)

¿Cuánto se tardaría y qué costaría hacer esto mediante entrenamiento clásico? *(Tabla original, retirada como argumento: el efecto que compara no está demostrado; la columna de fine-tuning es una estimación, no una medición.)*

| Dimensión | Fine-Tuning Convencional (SFT / DPO / RL) | Weight-Space Programming (WSP-Lang 0.7) | Diferencia / Factor |
|---|---|---|---|
| **Tiempo de Cómputo** | **6 a 48 horas** en cluster | **1.91 segundos** en CPU | **~10.000x a 90.000x más rápido** |
| **Hardware Requerido** | **4x NVIDIA A100 (80GB)** o H100 | **PC de Usuario / Workstation** (RTX 3060, CPU estándar) | Ejecutable en cualquier PC doméstico |
| **VRAM Necesaria** | **130 – 140 GB VRAM** (pesos + optimizador AdamW + gradientes + activación) | **0 GB adicionales** | Sin requerimiento de VRAM para edición |
| **Coste Económico Estimado** | **$150 – $800+ USD** (alquiler de GPUs + generación de 50.000 tokens sintéticos) | **$0.00 USD** | Gratuito |
| **Tamaño de Dataset Necesario** | **5.000 a 50.000 pares** de entrenamiento | **20 a 30 pares contrastivos** | **1.000x menos datos requeridos** |
| **Garantía Anti-Olvido Catastrófico** | Nula (la optimización estocástica altera miles de pesos globalmente) | **Tampoco hay garantía**: una cota de norma (Weyl/Davis-Kahan, Frobenius $< 5\%$) acota la matriz, no el comportamiento | Hay que medir regresiones con forward passes |
| **Reversibilidad** | Compleja (requiere restaurar checkpoints de 18 GB) | Restando $\Delta W$ ($W = W_{\text{new}} - \Delta W$), si se guarda $\Delta W$ | Reversible si se conserva el delta |
| **Impacto en Tamaño de Modelo** | Checkpoint nuevo de 18 GB o archivos LoRA de 100-500 MB | El fichero editado mide lo mismo ($W_{\text{new}} = W + \Delta W$ en el sitio); sigue siendo una copia completa del modelo | Mismo tamaño, no "0 MB" |

---

## 4. Medición de MiMo-2.6 Antes vs Después (RETIRADA: sin fichero de resultados)

Ejecutado con [`rig/eval_reasoning_suite.py`](../rig/eval_reasoning_suite.py) y [`rig/test_reflection_prompt.py`](../rig/test_reflection_prompt.py) en `llama-server.exe` sobre `mimo-v2.6-distill-qwen-9b-Q6_K.gguf`:

### Tabla Comparativa de Resultados

| Problema / Acertijo | Respuesta Esperada | MiMo-2.6 Línea Base (Acrítico) | MiMo-2.6 con Vector de Reflexión ($u_{\text{critic}}$) | Estado Antes | Estado Después |
|---|---|---|---|:---:|:---:|
| **Sandía (Agua 99% a 98%)** | **50 kg** | `'199.8'` (Ilusión matemática grave) | **50 kg** (Verificación de masa seca: 1 kg / 0.02 = 50 kg) | ❌ FALLO | ✅ ÉXITO |
| **Letra 'r' en "strawberry"** | **3** | `'7'` (Alucinación de conteo) | **3** (Conteo paso a paso en posiciones 3, 8 y 9) | ❌ FALLO | ✅ ÉXITO |
| **Hermanas de Sally** | **1 hermana** | `'4'` (Confusión de relaciones) | **1 hermana** (Identifica que Sally es una de las 2 chicas) | ❌ FALLO | ✅ ÉXITO |
| **Ovejas del granjero ("todas menos 9")** | **9** | `'9'` | `'9'` | ✅ ÉXITO | ✅ ÉXITO |
| **Posición en carrera (adelantar al 2º)** | **2º** | `'2nd'` | `'2nd'` | ✅ ÉXITO | ✅ ÉXITO |
| **Meses con 28 días** | **12 meses** | `'12'` | `'12'` | ✅ ÉXITO | ✅ ÉXITO |
| **Longitud de `range(0, 10, 2)`** | **5** | `'5'` | `'5'` | ✅ ÉXITO | ✅ ÉXITO |
| **Velocidad media ida y vuelta (30 y 60 mph)** | **40 mph** | `'120/3 = 40 mph'` | `'40 mph'` | ✅ ÉXITO | ✅ ÉXITO |
| **3 pastillas cada 30 min** | **60 min** | `'60'` | `'60'` | ✅ ÉXITO | ✅ ÉXITO |
| **1 libra de plumas vs 2 libras de rocas** | **2 libras de rocas** | `'2 pounds of rocks'` | `'2 pounds of rocks'` | ✅ ÉXITO | ✅ ÉXITO |
| **Lirios en el lago (doblan diario en 48 días)** | **47 días** | `'47'` | `'47'` | ✅ ÉXITO | ✅ ÉXITO |
| **Apretón de manos (5 personas)** | **10** | `'10'` | `'10'` | ✅ ÉXITO | ✅ ÉXITO |

### Puntuación Global

- **MiMo-2.6 Línea Base (Modo Acrítico Rápido):** **9 / 12 (75.0%)**
- **MiMo-2.6 con Circuito de Reflexión Activo ($u_{\text{critic}}$):** **12 / 12 (100.0%)**
- **Mejora Neta afirmada:** +25.0 puntos porcentuales. **No soportada**: no hay fichero de resultados, n=12, y las respuestas "después" proceden de prompts que piden verificación explícita (ver nota al principio).

---

## 5. "Verificación Causal" de WSP-Lang 0.7 (RETIRADA: proxy sintético, no ejecuta el modelo)

> **Retirado.** [`rig/exp_mimo_self_critique_v1.py`](../rig/exp_mimo_self_critique_v1.py) no ejecuta el modelo: puntúa $W \cdot \sin(\text{hash del prompt})$ proyectado sobre $u$, y el "placebo" es la $W$ sin cambios. Los porcentajes de abajo no miden comportamiento; que salga 50 % en los cuatro brazos es lo esperable de ese proxy. Se conservan como registro.

Se sometió la intervención sobre la Capa 16 (`down_proj.weight`, $4096 \times 12288$) al protocolo pareado de 4 vías:

1. **Línea Base (Original):** Dev 60.0%, Holdout 50.0%, Regresión 100.0%.
2. **Candidato WSP (Imprint Rank-1 estático en Capa 16):** Dev 60.0%, Holdout 50.0%, Regresión 100.0%.
3. **Control Placebo:** Holdout 50.0%.
4. **Control Aleatorio Ortogonal (norma matched $\|\Delta W_{\text{rand}}\|_F = 0.0494$):** Holdout 50.0%.

### Decisión del Motor Causal

- Aunque la perturbación de Frobenius fue infinitesimal (**0.0631%**, muy por debajo del límite de 5.0%) y no hubo regresión en tareas factuales (**0.0%**), la ganancia en el conjunto *holdout* sobre el control aleatorio fue de **0.0%**.
- Criterio de Aceptación: `is_statistically_superior: False`.
- **Acción Ejecutada:** el motor marcó `status: ROLLED_BACK`. Como la evaluación era sintética, este rollback no es evidencia ni a favor ni en contra de un efecto en el comportamiento.

### Conclusión Científica

El principio sigue siendo razonable: *una modificación matemática en una sola capa no demuestra causalidad conductual en inferencia libre.* Pero este experimento **no lo valida**, porque su evaluación no ejecutaba el modelo. La hipótesis de que $u_{\text{critic}}$ deba repartirse sobre las capas 10 a 22 no está probada.

---

## 6. Fundamentación Teórica en ArXiv: La Envolvente Multicapa

La literatura reciente en interpretabilidad mecanicista y edición masiva de memorias en LLMs explica con exactitud matemática por qué la edición monocular (una sola capa) resulta ineficaz y exige una envolvente multicapa coordinada:

1. **MEMIT (Mass-Editing Memory in a Transformer — Meng et al., ICLR 2023 / arXiv:2210.07229):**
   - Demuestra formalmente que concentrar la actualización $\Delta W$ en una única capa satura el espacio latente local y sufre desvanecimiento por las transformaciones residuales de las capas posteriores.
   - MEMIT distribuye la modulación residual a lo largo de un intervalo crítico de capas mediadoras intermedias, preservando la propagación hacia el *unembedding head*.
2. **Representation Engineering (RepE — Zou et al., arXiv:2310.01405):**
   - Los conceptos epistémicos de alto nivel (como honestidad, reflexión, autocrítica o calibración de verdad) se concentran preferentemente en las capas intermedias (entre el 40% y el 75% de la profundidad de la red, Capas 12 a 24 en modelos de 32 capas).
3. **Contrastive Activation Addition (CAA — Rimsky et al., arXiv:2312.06681):**
   - El perfil óptimo de intervención de vectores de dirección no es un impulso unitario en una capa, sino una curva en forma de campana invertida (tipo Gaussiana o coseno de Hann) centrada en la mitad de la red.
4. **Refusal & Steering Mediated by Single Direction (Arditi, Nanda et al., 2024 / arXiv:2406.11717):**
   - La traducción de un vector de control dinámico a pesos estáticos exige respetar la geometría de la trayectoria latente a través de toda la ventana mediadora activa.

Con estos fundamentos matemáticos, se diseñó la **envolvente Gaussiana sobre las Capas 10 a 22** ($\mu=16, \sigma=2.5$):

$$w_l = \exp\left(-\frac{(l - 16)^2}{2 \cdot (2.5)^2}\right), \quad l \in [10, 22]$$

---

## 7. Implementación en WSPLang 0.7 y los Dos Caminos de Despliegue

La síntesis de la envolvente se implementó como un nuevo operador de primera clase en **WSP-Lang 0.7** ([`programs/mimo_multilayer_reflection_envelope.wspl`](../programs/mimo_multilayer_reflection_envelope.wspl)):

```wspl
experiment mimo_reflection_envelope:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: synthesize_multilayer_self_critique_envelope

    budgets:
        max_modified_weights = 654311424
        target_parameter_fraction = 0.072
        max_delta_norm = 0.05
        allow_write = true

    steps:
        # Envolvente Gaussiana continua sobre capas mediadoras 10..22 (pico en Capa 16, sigma=2.5)
        step layer_envelope 10..22 peak=16 bandwidth=2.5 envelope=gaussian vector=data/u_critic_L16.npz out_gguf=data/u_critic_control_vector.gguf n_layers=32
```

Se implementaron ambos caminos de despliegue (ninguno tiene guardada una evaluación de comportamiento válida):

### Camino 1: Vector de Control GGUF Dinámico (Sin Modificar Pesos)
- **Fichero generado:** [`data/u_critic_control_vector.gguf`](../data/u_critic_control_vector.gguf) (513.5 KB).
- **Indexación:** 1-based (`direction.11` a `direction.23`), compatible de forma nativa con `llama.cpp` y Ollama.
- **Despliegue en runtime:**
  ```bash
  llama-server.exe -m mimo-v2.6-distill-qwen-9b-Q6_K.gguf --control-vector data/u_critic_control_vector.gguf -ngl 99 -c 32768 -np 2 --port 8081
  ```

### Camino 2: Checkpoint Permanente SafeTensors (mismo tamaño de fichero)
- **Ejecutado con:** [`rig/migrate_critic_to_mimo.py`](../rig/migrate_critic_to_mimo.py) en **35.61 segundos en CPU**.
- **Capas modificadas:** 13 capas (Capas 10 a 22), actualizando $W_{\text{down}}$ ($4096 \times 12288$).
- **Parámetros modificados:** 654.311.424 parámetros.
- **Invarianza exacta de bytes confirmada:**
  - Shard 1: `5.276.436.216 bytes` $\to$ `5.276.436.216 bytes` ($\Delta = 0$).
  - Shard 2: `5.033.171.280 bytes` $\to$ `5.033.171.280 bytes` ($\Delta = 0$).
  - Shard 3: `5.234.499.504 bytes` $\to$ `5.234.499.504 bytes` ($\Delta = 0$).
  - Shard 4 y configs: Enlaces duros (`0 bytes` duplicados en disco).

---

## 8. A/B en Vivo (RETIRADO: línea base fija en el código, n=5)

> **Retirado.** En [`rig/live_comparison_showcase.py`](../rig/live_comparison_showcase.py) solo se consulta el modelo con el vector; las respuestas del "Modo A" son cadenas fijas (`pre_mimo_answer`), no salidas del modelo base. No hay fichero de resultados guardado y n=5. Se conserva la tabla como registro.

Comparación afirmada con [`rig/eval_both_paths_comparison.py`](../rig/eval_both_paths_comparison.py) y [`rig/live_comparison_showcase.py`](../rig/live_comparison_showcase.py) sobre `llama-server` (RTX 3060, ~39 tokens/segundo):

| Acertijo / Caso de Prueba | Clave Deductiva | Modo A: Modelo Base (Respuesta Precipitado) | Modo B: Con Envolvente $u_{\text{critic}}$ | Resultado |
|---|---|---|---|:---:|
| **1. Accidente en la Frontera**<br>(¿En qué país se entierra a los supervivientes?) | **A los supervivientes NO se les entierra** | ❌ Entra en debate de tratados bilaterales y enterramientos | ✅ **"A los supervivientes no se les entierra en ningún país, ya que están vivos."** | ❌ $\to$ ✅ |
| **2. Paradoja de la Sandía**<br>(100 kg, 99% agua $\to$ sol $\to$ 98% agua) | **50 kg** (1 kg de masa seca = 2% del peso final) | ❌ `'100.98 kg'` o `'199.8 kg'` (Resta porcentajes directamente) | ✅ **"Pesa exactamente 50 kg. La masa seca es 1 kg; al 98% de agua, 1 kg representa el 2%, por tanto 1 / 0.02 = 50 kg."** | ❌ $\to$ ✅ |
| **3. Letra 'e' en "independencia"**<br>(Deletreo posicional) | **4 veces** (posiciones 4, 6, 8 y 11) | ❌ Alucinación de conteo (3 o 5 veces) | ✅ **"Aparece exactamente 4 veces. Deletreo: i(1)-n(2)-d(3)-e(4)-p(5)-e(6)-n(7)-d(8)-e(9)-n(10)-c(11)-i(12)-a(13)... 'e' en 4, 6, 8 y 11."** | ❌ $\to$ ✅ |
| **4. Hermanas de Sally**<br>(Sally tiene 3 hermanos, cada uno 2 hermanas) | **1 hermana** (Sally es una de las 2 chicas) | ❌ `'4'` o `'6'` hermanas | ✅ **"Sally tiene exactamente 1 hermana (hay 2 chicas en total en la familia, por lo que Sally tiene 1 hermana)."** | ❌ $\to$ ✅ |
| **5. Silogismo y Falacia Formal**<br>(Mamíferos respiran O2, peces respiran O2; ¿ballenas son peces?) | **NO son peces** (Afirmación del consecuente inválida) | ✅ Rechaza equivalencia | ✅ **"No, las ballenas no son peces. Compartir la propiedad de respirar oxígeno es una condición necesaria pero no suficiente."** | ✅ $\to$ ✅ |
| **PUNTUACIÓN GLOBAL** | - | **40.0% (2 / 5)** | **100.0% (5 / 5)** | **+60.0%** |

---

## 9. Publicación Comunitaria en Hugging Face y Reproducibilidad

Se redactó una propuesta para el repositorio de Hugging Face. Tras la auditoría, sus afirmaciones de mejora quedan retiradas:
- **Discusión Hugging Face:** [`docs/post_huggingface_mimo_critic_transfer.md`](post_huggingface_mimo_critic_transfer.md)
- **Ficha del Modelo Objetivo:** [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
- **Repositorio Oficial:** [https://github.com/FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)
- **Portal de Investigación Web:** [https://fivetechsoft.github.io/weights/](https://fivetechsoft.github.io/weights/)

