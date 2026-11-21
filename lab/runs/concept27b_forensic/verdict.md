# Forense ciego de conceptcountry/qwen-concept-27b — VEREDICTO (2026-11-09)

`model_report` (rows=128, HTTP-ranges, CPU, sin GPU) sobre los pesos publicados vs base `Qwen/Qwen3.8-27B`.
Predicción sellada ANTES: `sealed_prediction.md` (mismo directorio). Programa: `programs/concept27b_forensic.wspl`.

| Aspecto | Predicción sellada | Instrumento (report.json) | Concordancia |
|---|---|---|---|
| Capas tocadas | 9, alterno 32-48 (32,34,…,48) | "layers 32-48 (9 touched)" | Jaccard = 1.0 (criterio ≥0.7) |
| Familia de tensores | down_proj únicamente | mlp.down_proj.weight 9/64 | EXACTA |
| Rango del cambio por tensor | rank-1 repetido por capa | rank90 = 1 | EXACTO |
| Fracción relativa | ~0.5% en esas 9 matrices | 0.0000% global en el muestreo (100% concentrado en 9/64) | coherente |
| Método lógico | firma ARA/abliteration-parcial | method = **directional_ablation** | familia correcta |
| Extras fuera de pesos | nada significativo | tokenizer/config byte-diffs proxies; chat template ausente en el repo | pasable |

**Falso positivo declarado (lección J4):** el apartado "usable" del instrumento lo va de **decensoring**
("removes safety behaviour") — la firma structural de "dirección ablativa en down_proj profundo" coincide con la de
abliteración, pero la dirección aquí es de conducta (pensamiento largo). El clasificador acierta estructura
familia/lugar/rango/etc., no las intenciones. La calibración en segunda familia (Qwen3.8-27B) cumple el criterio de
estructura y primero falla de semántica de propósito.

**Procedencia:** fixes hechos para correrlo: `hf_files` descubre variantes en subfolder (bf16/) — `engine/weight_diff.py`;
dispatch `model_report` ya acepta sintaxis `hf:owner/repo` — `engine/wsplang.py` (antes 401 del prefijo en la tarjeta). Tests
112 en verdes tras los cambios.

Conclusión: primer blind test 27B de `model_report` en pesos reales con verdad conocida: estructura OBJEIVA
(capas/familia/rango exactas), semántica de intención pendiente de mejora (a la espera del atlas de direcciones H3).
