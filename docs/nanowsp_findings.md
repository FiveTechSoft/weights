# NanoWSP: Lecciones y Límites al Programar un LLM desde Cero con WSPLang

**Fecha:** 4 de Noviembre de 2026  
**Investigador:** FiveTechSoft Research  
**Modelo:** `NanoWSP-80k` ($d=64$, $L=2$, $h=2$, $d_{ff}=128$, $V=100$, 80.704 parámetros)  
**Código y Runtime:** [`engine/nanowsp_runtime.py`](../engine/nanowsp_runtime.py)  
**Programa Declarativo:** [`programs/nanowsp_architecture.wspl`](../programs/nanowsp_architecture.wspl)  
**Tests de Verificación:** [`tests/test_nanowsp.py`](../tests/test_nanowsp.py)  

---

## 1. El Experimento

En lugar de limitarnos a la cirugía sobre checkpoints entrenados con SGD masivo (como MiMo-9B o Qwen3.5-0.8B), construimos un micro-Transformer autorregresivo completo (`NanoWSP`, 80.704 pesos en SafeTensors) y evaluamos si la programación en el espacio de pesos (**WSPLang** y **Fast Weight Programming**) permite sintetizar e imprimir comportamientos y memorias asociativas **directamente en las matrices de pesos sin backpropagation ni optimizadores gradient-based**.

---

## 2. Hallazgos Empíricos Clave

### 2.1 La Ilusión de la Impresión Hebbiana Ingenua (Interferencia de Claves Cercanas)
Al intentar imprimir dos reglas asociativas con el producto exterior de Hebb / rank-1 estándar ($\Delta W = \eta \cdot v \otimes k^T$):
1. Imprimimos `A->` $\to$ `'1'` en la Capa 1.
2. Imprimimos `B->` $\to$ `'2'` en la Capa 1.
3. **Resultado:** Al evaluar `A->`, el modelo devolvió `'2'` en lugar de `'1'`. Ambas entradas colapsaron a la última salida impresa.

**Causa mecanística descubierta:**
Al medir la similitud coseno entre las activaciones intermedias $k_A$ y $k_B$ en la Capa 1:
$$\cos(k_A, k_B) = \frac{k_A \cdot k_B}{\|k_A\| \|k_B\|} = 0.89407 \quad (\approx 90\%)$$
Debido a que `A->` y `B->` comparten 2 de 3 caracteres (`'-'` y `'>'`), en una red pequeña sin entrenamiento previo las representaciones latentes caen dentro de un cono isotrópico estrecho (el mismo colapso de cono documentado en modelos de 9B). La diafonía (*cross-talk*) entre $k_A$ y $k_B$ es del 89.4%, ahogando la primera memoria con la segunda.

---

### 2.2 La Solución de Desacoplo: Operador de Proyección de Moore-Penrose
Para resolver la diafonía entre múltiples asociaciones en el espacio de pesos sin backprop, formalizamos la síntesis multiasociativa exacta:
$$\Delta W = Y X^+ = Y (X^T X)^{-1} X^T$$
Donde $X = [k_1, k_2, \dots, k_M] \in \mathbb{R}^{d_{ff} \times M}$ y $Y = [v_1, v_2, \dots, v_M] \in \mathbb{R}^{d_{model} \times M}$.

La matriz inversa de Gram $(X^T X)^{-1}$ descorrelaciona activamente todas las memorias simultáneamente.
- **Resultado empírico (4 asociaciones simultáneas):**
  - `A->` $\to$ `1` (100% exacto)
  - `B->` $\to$ `2` (100% exacto)
  - `C->` $\to$ `3` (100% exacto)
  - `D->` $\to$ `4` (100% exacto)
  - **Precisión:** **100.0%** sin una sola iteración de descenso de gradiente.

---

### 2.3 La Barrera de Explosión del Condicionamiento Espectral (Límite de Capacidad)
Evaluamos empíricamente qué ocurre al incrementar el número de pares $M$ sintetizados en una matriz de $64 \times 128$:

| $M$ Pares | Condicionamiento $\kappa(X)$ | Norma Frobenius $\|\Delta W\|_F$ | Precisión de Retención |
|---|---|---|---|
| **$M = 5$** | $16.8$ | $68.31$ | **100.0%** |
| **$M = 10$** | $34.3$ | $111.92$ | **100.0%** |
| **$M = 20$** | $324.5$ | $208.08$ | **100.0%** |
| **$M = 30$** | $606.1$ | $639.46$ | **100.0%** |
| **$M = 50$** | $1366.0$ | $1026.12$ | **100.0%** |
| **$M = 60$** | $1811.5$ | $1407.94$ | **100.0%** |

**Degradación en tareas no relacionadas:**
A partir de $M \ge 30$, aunque la memorización forzada sobre los $M$ pares es del 100%, la norma $\|\Delta W\|_F$ supera $600.0$, violando completamente la cota de perturbación de Weyl y Davis-Kahan. Como consecuencia, las frases no relacionadas (`"Hello world"`, `"The sun is"`) colapsan y emiten únicamente los tokens sobreajustados.

---

## 3. Conclusiones y Limitaciones de WSPLang

1. **WSP es Óptimo para Cirugía Local ($M \le 5$, $\Delta W$ bajo):**
   Para extirpar un atractor de bucle, silenciar un concepto tóxico o redirigir un razonamiento específico, la perturbación de rango 1 respeta las cotas de Weyl y no altera el resto de la red.
2. **WSP No Reemplaza el Preentrenamiento Global:**
   Intentar sintetizar un LLM entero desde cero únicamente escribiendo productos exteriores en una matriz colisiona con el límite de condicionamiento de la matriz de Gram ($\kappa \to \infty$).
3. **El Problema de la No-Linealidad de la Atención:**
   Modificar la proyección de salida del MLP ($W_{out}$) es lineal con respecto a las activaciones intermedias. Sin embargo, modificar la atención $softmax(Q K^T / \sqrt{d}) V$ requiere intervenir simultáneamente $W_q$ y $W_k$, lo que exige operadores bilineales coordinados en la gramática de WSPLang 0.5.
