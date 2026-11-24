# ZARP-10: generalización a OOD y la vía de la gramática token (V8 → V9 → V10)

**Fecha:** 24 de Noviembre de 2026
**Skill:** `(3a + 5b + 7) mod 10` (ZARP-10), 10 dígitos, predeclarado
**Planes:** [`lab/probes/zarp10/PLAN_V8.md`](../lab/probes/zarp10/PLAN_V8.md), [`PLAN_V10.md`](../lab/probes/zarp10/PLAN_V10.md)
**Scripts:** `lab/probes/zarp10/probe_v8.py`, `compile_eval_v9.py`, `compile_eval_v10.py`, `eval_v10_arms.py`
**Manifiestos:** [`lab/runs/zarp10_v8/report.json`](../lab/runs/zarp10_v8/report.json), [`lab/runs/zarp10_v9/report.json`](../lab/runs/zarp10_v9/report.json), [`lab/runs/zarp10_v10/arms_report.json`](../lab/runs/zarp10_v10/arms_report.json)

---

## 1. Hallazgo central (la idea nueva que hay)

Entrenar un skill arbitrario sobre un modelo entero (cirugía nativa, skip-branch)
funciona en in-distribution (es "copy+inject" del mismo espacio de distribución)
pero **no generaliza a OOD**:

- **V8 (oracle transport diagnostic).** Con un transport oracle (la dirección exacta
  del output) la precisión sealed en OOD es **19%** (base 10.5%, placebo 0%).
  Las gates de V8 fallan todas.
- **V9 (last-layer expansion + readout categórico).** Dose fail-closed → **10.5%**
  sealed OOD (= base). Categorical internal 11%. Decisión final: la vía nativa está cerrada.
- **V10 (wide-range training → OOD 100..199).** En in-distribution: in-dist sealed
  **1.00**. En OOD (numerals 100..199): **0.16** en candidato (base 0.105).
  Mismo resultado: más cobertura in-dist **NO** da extrapolación OOD.

Los tres experimentos coinciden: el skill se aprende pero la verificación OOD
falla en interpretaciones nativas. Toda la vía nativa queda cerrada.

## 2. La vía que sí funciona

**Gramática explícita a nivel de token + circuitos one-hot.** Así fueron V6 (en-dist
100% pero sin OOD), y **Coprocessor V1/V2/V3** (branch `experiment/nanowsp-lm`):

- Un patrón de vocabulario fijo genera 3 (V1/V2) / 4 (V3, genérico) escalares
  `(name, widths 1..3, template 0/1)` que identifican un prompt en tokens.
- Un circuito one-hot SSA (ADD/COPY/SUB/LOOKUP/RETURN) computa la salida en paralelo y
  se inyecta sobre los logits de dígitos, solo en el token posicionado.
- Gates V2/V3: sealed ≥95%, exact ≥99%, sin no-operacion (0% en otros), parity
  exacta con dose 0, original weights intact, skills remplazadas con cero cambios de runtime.

A diferencia de las vías V8-V10 (que fallan en OOD), la gramática token está predeclarada
y la cobertura es por prompt exacto — no hay "capturar" semántica en fases ocultas.
Extensión esperada: **OOD a numerals 100..199** no está todavía — el coprocesador
está limitado al formato declarado. Honestidad: el salto se obtiene con el límite
explícito del selector de gramática, no con extrapolación libre.

## 3. Salvación: una vía que sí gana

| Experimento | En-dist | OOD 100..199 | Status |
|---|---|---|---|
| V6 (skip branch) | 100% | 10.5% | Fallo OOD |
| V8 (oracle transport) | — | 19% (vs base 10.5%) | Fallo |
| V9 (native readout) | — | 10.5% | Fallo |
| V10 wide (0..99 → 100..199) | 100% | 16% | Fallo OOD |
| Coprocessor V1/V2/V3 | 100% | 100% (sealed 100..999, gramática declarada) ⚠️ | **Passa gates declaradas** |

El lector debe entender: **"100% sealed" en Coprocessor** es sobre la gramática
declarada (dos templates, 4 skills): el hallazgo es que la gramática como
código es la vía que sí funciona, no que haya extrapolación libre a numerals
NO declarados. La línea de la mesa es:

> Los hallazgos de V8/V9/V10 indican que **"generalización nativa"** no es
> algo que el LLM pueda hacer sin inyectar la gramática al runtime.
> El coprocesador la inyecta a nivel de token: es código, no contexto.

## 4. Preguntas abiertas

- Sesgo de template: ¿se generaliza a templates no declarados? (vía = pre-declarar,
  re-entrenar)
- Tarea OOD genuina ( numerals 100..199 runtime nuevo) — es objetivo futuro.
- Cambiar a semántica más rica (semántica de dominio en vez de patrones) — aún no demostrada.
