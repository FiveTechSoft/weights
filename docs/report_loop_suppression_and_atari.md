# Informe Científico: Supresión de Bucles de Razonamiento y Descompilación de Políticas Atari

**Autores:** FiveTechSoft Research Team  
**Fecha:** Noviembre 2026  
**Licencia:** MIT License  
**Modelos Evaluados:** `Qwen3.5-0.8B` (BF16 y Q6_K), `MiMo-V2.6-Distill-Qwen-9B` (Q6_K GGUF)

---

## 1. Qwen3.5-0.8B y el edit de Astra (Exp 8): qué muestran los datos

> **Corrección de auditoría (2026-11-06).** Una versión anterior de esta sección afirmaba que, en GSM8K #2, el modelo editado "termina en menos de 100 tokens (`is_cut: false`)" y que la edición "extirpó el bucle". Es **falso**: en `exp8_control/results_eval_n5.json` las 10 respuestas (base y editado) se cortan en el límite de 512 tokens (`is_cut: true`, `cut_share = 1.0`). La conclusión de "90 % de los fallos" tampoco tenía medición detrás. La sección se ha reescrito con los datos guardados.

### 1.1. Lo que se difundió y lo que se puede afirmar
En redes sociales se difundió que *"editar el 0.016% de los pesos inyectó capacidades avanzadas de matemáticas a Qwen3.5-0.8B, subiendo su puntuación de 7.9% a 48.1% en MATH-500"*.

La hipótesis de **Jacob Lauritzen (@jacsebl)** es que la edición no enseña matemáticas nuevas, sino que reduce un patrón de duda recursiva que impide cerrar la respuesta. Nuestros datos son **compatibles** con esa hipótesis, pero **no la demuestran**: no medimos un mecanismo.

### 1.2. La edición reproducida
Los tensores editados son bit-idénticos a los de Astra (SHA-256 contra `exp8_control/p50c_edit/manifest.json`); difieren 130.188 entradas (0,016 %).

### 1.3. Smoke test GSM8K (n=5): no muestra la ruptura del bucle
Arnés: `exp8_control/base` frente a `exp8_control/edited`, `max_new_tokens=512`, resultados en `exp8_control/results_eval_n5.json`.

* Base: 0/5. Editado: 1/5.
* **Las 10 respuestas se cortan en el límite de 512 tokens** (`is_cut: true`, `cut_share = 1.0` en ambos modelos).
* **Problema #2** (*"A robe takes 2 bolts of blue fiber and half that much white fiber..."*, gold = 3):
  * Base: cortado sin predicción (`pred` vacío), FAIL. Su cola de generación sigue dudando entre interpretaciones de "half that much".
  * Editado: `pred = 3`, correcto, **pero también cortado a 512 tokens** (`is_cut: true`). Su cola de generación llega a "Total = 3" y sigue con "*Self-Correction/Alternative Interpretation:*". No terminó antes; el extractor tomó el número del texto cortado.

Con n=5 y todo cortado, este smoke test no permite decir que la edición rompa el bucle.

### 1.4. La evidencia real: MATH-500 holdout
MATH-500 holdout, Q6_K, n=20, seed 1337, temperatura 0.3, `max_tokens 2048` (`rig_runs/holdout-08b-{base,edit,placebo}-20.json`):

| Variante | Aciertos |
| :--- | :--- |
| Base | 1/20 |
| Edit (Astra) | **14/20** |
| Placebo | 0/20 |

Fisher: p ≈ 3,9e-5 edit vs base; p ≈ 3,3e-6 edit vs placebo. Es un efecto real en este modelo y este arnés, con n=20.

### 1.5. Límites
* Replicar la misma idea en un nanoGPT entrenado (results.md §36) **no supera el criterio 4 preregistrado**: una de 3 ediciones aleatorias de la misma norma dio el mismo resultado.
* El "loop fix" en MiMo-9B se **descartó**: holdout ciego n=40, bucles 17/40 con la edición frente a 20/40 con placebo, p = 0,654; GSM8K n=50, precisión 45 → 46.

## 2. Descompilación e Interpretación de Pesos de Juegos Atari (Simon Willison & MIT Exp 1)

### 2.1. Anatomía de los Prompts
Descargamos los prompts originales utilizados en el estudio:
* [`data/exp1_pong_prompt.txt`](../data/exp1_pong_prompt.txt) (115.269 caracteres).
* [`data/exp1_breakout_prompt.txt`](../data/exp1_breakout_prompt.txt) (115.478 caracteres).

Ambos archivos contienen una red neuronal MLP completa de **17.253 parámetros** entrenada con PPO en un grid de 10x10 con 5 canales:
* Capa 1 ($W_1$): $500 \times 32$
* Capa 2 ($W_2$): $32 \times 32$
* Capa 3 ($W_3$): $32 \times 5$ (Acciones: `0: noop`, `1: up`, `2: down`, `3: left`, `4: right`)

### 2.2. Por qué el Prompt Ocupa 104.794 Tokens
Debido a la tokenización BPE, cada número decimal como `0.054 -0.026` se fragmenta en 5–7 tokens. Los 17.253 números suman **104.794 tokens**.
* **Simon Willison:** Utilizó Qwen 27B en un Mac de 64GB de RAM con ventana de 128k, tardando entre **45 y 64 minutos** por juego para evaluar el prompt en CPU/Metal.
* **En hardware local estándar (GPU 12GB):** Un contexto de 104k en un servidor normal con ventana de 32k (`-c 32768`) arroja `HTTP 400 Context Exceeded`.

### 2.3. La Verificación Matemática de la Firma de Pesos
Usando WSP y [`rig/verify_atari_interpretability.py`](../rig/verify_atari_interpretability.py), calculamos la distribución de varianza y peso máximo de la capa de salida ($W_3$) para ambos juegos:

| Juego | Varianza Acciones Verticales (Up / Down) | Varianza Acciones Horizontales (Left / Right) | Peso Máximo Absoluto | Firma Mecanística |
| :--- | :--- | :--- | :--- | :--- |
| **Pong** | **$\text{std} = 0.896$ / $0.956$** (Poder: 1.852) | $\text{std} = 0.602$ / $0.601$ (Poder: 1.203) | Up: 1.95, Down: 1.90 | **Eje Vertical Domina (1.54x)**: la pala solo se mueve arriba/abajo. |
| **Breakout** | $\text{std} = 1.128$ / $1.138$ (Poder: 2.266) | **$\text{std} = 1.930$ / $1.546$** (Poder: 3.476) | Left: 5.34, Right: 3.06 | **Eje Horizontal Domina (1.53x)**: la pala en el fondo solo se mueve izquierda/derecha. |

### 2.4. Una consulta a MiMo-9B (anécdota, n=1)
Se hizo **una sola consulta** a MiMo-9B (puerto 8081). No recibió los pesos crudos, sino las **estadísticas de $W_3$ calculadas a mano** de la tabla anterior. Respondió "Pong" con una confianza del 90 % **autodeclarada por el propio modelo**, no medida. Es una anécdota: una consulta, un juego y la pista ya resumida en el prompt.

```text
Observations: Vertical action weights (up: max |w| = 1.95, down: max |w| = 1.898; std 0.896/0.956)
are both significantly larger than horizontal (left: 1.283, right: 1.282; std 0.602/0.601),
and the summed vertical activity (1.8522) exceeds horizontal activity (1.203).

Reasoning: The dominant axis of motion is vertical (up/down), which matches a paddle that moves
only along the vertical axis to intercept a ball — the defining control of Pong.
Breakout and Space Invaders both require horizontal movement.

Game: Pong
Confidence: 90%
```

---

## 3. Resumen de Conclusiones para la Comunidad

1. **Qwen3.5-0.8B:** la edición de Astra, reproducida bit a bit, mejora MATH-500 holdout de 1/20 a 14/20 (placebo 0/20). El smoke test GSM8K n=5 no muestra una ruptura del bucle (todas las respuestas cortadas). La misma idea en un nanoGPT entrenado no supera el criterio preregistrado.
2. **MiMo-9B:** el loop fix se descartó (17/40 vs 20/40 con placebo, p = 0,654). MiMo-9B sí tiene bucles (35/60 en la base, results.md).
3. **Atari:** las estadísticas de $W_3$ separan los ejes (1,54× Pong, 1,53× Breakout). La identificación del juego por MiMo es una anécdota de una consulta con estadísticas ya calculadas.
