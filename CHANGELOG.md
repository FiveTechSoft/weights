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

## Reglas

1. Cada cambio = una entrada única con ID estable (`E1`, `E2`, ...).
2. El veredicto (`keep`/`discard`/`inconclusive`) va siempre.
3. Si el cambio es rechazado, **se documenta igual**: el conocimiento de por qué falló es valioso.
4. Los holdouts (MATH-500, code, IF) NO alimentan el diseño: solo se evalúan al final.
5. Cada ficha `docs/changelog/` debe ser autocontenida para publicación.
