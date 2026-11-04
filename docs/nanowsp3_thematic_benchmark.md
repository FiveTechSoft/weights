# NanoWSP-3: Tabla Exhaustiva de Preguntas y Respuestas Verificadas

NanoWSP-3 es un micro-LLM autorregresivo de **6 capas y 3.280.128 parámetros** (12,51 MB SafeTensors) sintetizado completamente en el espacio de pesos mediante **WSP-Lang**.

A continuación se detalla la batería completa de los **23 hechos volcados desde MiMo-9B**, organizados en 5 dominios temáticos, evaluados al 100% de acierto con una confianza media del **98,4%** en **154 ms** de síntesis en CPU.

---

## Tabla Completa de Resultados por Temática

| # | Temática | Prompt de Entrada | Esperado | Predicción NanoWSP-3 | Confianza | Estado |
|---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | **Química & Física** | `SCI: Z79=` | **A** *(Au / Oro)* | **A** | **98.8%** | ✅ PASS |
| 2 | **Química & Física** | `SCI: Z47=` | **S** *(Ag / Plata)* | **S** | **98.4%** | ✅ PASS |
| 3 | **Química & Física** | `SCI: Z26=` | **F** *(Fe / Hierro)* | **F** | **98.8%** | ✅ PASS |
| 4 | **Química & Física** | `SCI: H2O=` | **W** *(Agua)* | **W** | **99.0%** | ✅ PASS |
| 5 | **Química & Física** | `SCI: C=` | **L** *(Velocidad Luz)* | **L** | **99.7%** | ✅ PASS |
| 6 | **Geografía Global** | `GEO: ESP=` | **M** *(Madrid)* | **M** | **98.4%** | ✅ PASS |
| 7 | **Geografía Global** | `GEO: FRA=` | **P** *(París)* | **P** | **95.6%** | ✅ PASS |
| 8 | **Geografía Global** | `GEO: JPN=` | **T** *(Tokio)* | **T** | **97.9%** | ✅ PASS |
| 9 | **Geografía Global** | `GEO: BFA=` | **O** *(Uagadugú)* | **O** | **97.7%** | ✅ PASS |
| 10 | **Geografía Global** | `GEO: KAZ=` | **A** *(Astaná)* | **A** | **98.4%** | ✅ PASS |
| 11 | **Computación** | `CS: LNX=` | **T** *(Linus Torvalds)* | **T** | **98.7%** | ✅ PASS |
| 12 | **Computación** | `CS: PYT=` | **R** *(Guido van Rossum)* | **R** | **97.4%** | ✅ PASS |
| 13 | **Computación** | `CS: WWW=` | **B** *(Tim Berners-Lee)* | **B** | **99.2%** | ✅ PASS |
| 14 | **Computación** | `CS: LAN=` | **C** *(Dennis Ritchie)* | **C** | **97.0%** | ✅ PASS |
| 15 | **Computación** | `CS: ATT=` | **V** *(Ashish Vaswani)* | **V** | **99.0%** | ✅ PASS |
| 16 | **Humanidades** | `LIT: HAM=` | **S** *(Shakespeare)* | **S** | **98.4%** | ✅ PASS |
| 17 | **Humanidades** | `LIT: QUI=` | **C** *(Cervantes)* | **C** | **98.5%** | ✅ PASS |
| 18 | **Humanidades** | `LIT: HOB=` | **T** *(Tolkien)* | **T** | **98.9%** | ✅ PASS |
| 19 | **Humanidades** | `LIT: REP=` | **P** *(Platón)* | **P** | **98.4%** | ✅ PASS |
| 20 | **Agéntica & Cálculo** | `ACT: 3+4=` | **7** | **7** | **98.9%** | ✅ PASS |
| 21 | **Agéntica & Cálculo** | `ACT: 9-5=` | **4** | **4** | **98.8%** | ✅ PASS |
| 22 | **Agéntica & Cálculo** | `ACT: 8/2=` | **4** | **4** | **98.6%** | ✅ PASS |
| 23 | **Agéntica & Cálculo** | `ACT: 6+2=` | **8** | **8** | **98.2%** | ✅ PASS |

---

## Métricas Clave de la Síntesis

* **Tasa de Acierto Global:** **23 / 23 (100.0%)**
* **Interferencia Cruzada entre Temáticas:** **0.0%** (Cero distorsión destructiva)
* **Tiempo de Síntesis en CPU:** **154.29 ms**
* **Operador Matemático Empleado:** Inversión simultánea de la matriz de Gram $(K^T K + \lambda I)^{-1}$ en el espacio FFN de 512 dimensiones.
* **Checkpoint SafeTensors:** [`data/nanowsp3_weights.safetensors`](../data/nanowsp3_weights.safetensors)
* **Reporte Criptográfico:** [`rig_runs/nanowsp3_thematic_benchmark_report.json`](../rig_runs/nanowsp3_thematic_benchmark_report.json)
