# Coprocessor V3: registro-compiler genérico

**Fecha:** 24 Noviembre 2026
**Runner:** [`lab/probes/zarp10/coproc_v3.py`](../lab/probes/zarp10/coproc_v3.py)
**Spec:** [`programs/register/coproc_v3.json`](../programs/register/coproc_v3.json)
**Skills:** `SUM10`, `MIX10`, `COPY10`, `SPC10` (= `2x + y mod 10`, nuevo)
**Reemplazo:** `SPC10` → `SUB10` (a = SUB x y / RETURN a)
**Reporte:** [`lab/runs/coproc_v3/report.json`](../lab/runs/coproc_v3/report.json)
**Plan pre-declarado:** [`lab/probes/zarp10/PLAN_COPROC_V3.md`](../lab/probes/zarp10/PLAN_COPROC_V3.md)

---

## 1. Qué cambia V3 respecto a V2

V2 tenía hardcodeados: `NAMES`, `TEMPLATES`, `SOURCES`, labels (fórmulas
`(a+b, 2*a, 2*a-b)[skill]`). V3 los reemplaza por **un archivo spec JSON**:

- **Skills soportados** → cualquiera que exista como `.wspreg`.
- **Plantillas** → declaradas en el spec, uno por skill × plantilla.
- **Labels** → calculados por `simulate(source, a, b)`: el programa WSPLang es la fuente
  de verdad — no hay fórmulas hardcodeadas.
- **Reemplazo en caliente** → compila un programa nuevo en el slot SPC10
  sin tocar el runtime; los demás skills quedan intactos.

## 2. Resultado (CPU, 2026-11-24)

| Métrica | Valor |
|---|---|
| Sealed mean (100..999), 200 pairs × 8 groups | **0.99875** (base 0.0956, placebo 0.0) |
| Gate sealed (≥95% por group) | ✅ 8/8 groups |
| Routing (unique_activation == accuracy) | ✅ 200/200, 100% en los 8 groups |
| Arithmetic retention | ✅ 0.455 en ambos, delta = 0.0 |
| Prose retention (NLL) | ✅ 4.0217 en ambos, byte-identical |
| Zero-dose parity | ✅ 0.0 |
| Base hash intact | ✅ original_hash = `1bdc849c...` |
| Negatives (6510 prompts) | ✅ 0.0 en final/prefix/not/suffix/prose |
| Reemplazo SPC10→SUB10 | ✅ 100% (100 pairs) |
| Unchanged skills tras el reemplazo | ✅ 100% (6 groups: SUM10/MIX10/COPY10 × templates) |
| Ablation (W último stage de SPC10 → 0) | ✅ 0.0 (gate ≤ 0.8) |

**Las 13 gates están en verde.** Comandos de verificación:

```
python lab/probes/zarp10/coproc_v3.py --prose lab/data/shakespeare_char/input.txt --out lab/runs/coproc_v3
```

## 3. Qué NO significa

- **No es "generalización a sin-taxis arbitraria".** Los 100% sealed son sobre
  la gramática declarada (dos templates, 4 skills, same family, 100..999).
- **No es un LLM que razonará fuera de la gramática.** El coprocesador solo
  inyecta sobre patrones token-level exactos.
- **No es WSPLang ejecutando sobre weights del modelo.** Es un skill
  precompleado inyectado a nivel de token.

## 4. Reproducibilidad

- Spec JSON → runtime genérico; la referencia (`simulate`) usa la misma
  semántica (mod 10, SSA one-hot) que el circuito — verificada a nivel unit
  (5 programas, esperados vs. circuito).
- Las fórmulas hardcodeadas de V2 fueron eliminadas con test unit.
- El reemplazo SPC10→SUB10 prueba que el runtime no depende de los 3 skills
  originalmente fijos.

## 5. Detalle de integridad

- `zero_parity = 0.0` (dose 0).
- `artifact_bytes = 6136600` (vs. base de 6,138,176? — véase `report.json`)
- `runtime_sha256 = d412826e4f...` (hash del script fuente)
