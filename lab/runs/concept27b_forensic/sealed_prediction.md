# Predicción sellada — forense ciego conceptcountry/qwen-concept-27b

Fecha-sello: 2026-11-09 19:4x (antes de ejecutar `model_report` sobre los pesos publicados).
Lo que se da por conocida es SOLO la receta publicada en el working paper (diseño del experimento "cómo
debería verse el delta"); el analyzer no lee este fichero ni tiene acceso a la receta — se comprueba si
NUESTRA huella (128 filas por tipo de tensor, HTTP-ranges, CPU) la reconoce sin ayuda.

## Predicción hecha ANTES del informe

- **Método que model_report debería clasificar**: intervención quirúrgica localizada y de baja fracción —
  firma tipo *abliteration/ARA parcial* por el lado del descriptivo (escritura aditiva rank-1 repetida
  en pocas capas), NO fine-tune completo y NO LoRA (no cubre módulos'app con rango alto en todo el modelo).
- **Dónde**: `down_proj` de **9 capas, consecutivas-alternadas en la mitad profunda** de un modelo de ~64
  capas (receta declarada: capas 32-48 de 64, alterno = 32,34,...,48); ninguna otra familia destacada.
- **Cuánto**: fracción de parámetros cambiados ~0.5% relativa en esas 9 matrices (‖ΔW‖/‖W‖≈0.5%),
  concentración aguda en pocas capas; resto del modelo igual a byte-level hullado (salvo requant).
- **Aprovechable**: Δ sintético con componente rank-1 compartida entre esas capas (una dirección común de
  conducta de pensamiento-largo, reusada capa a capa). Veredicto esperado del apartado transferencia:
  candidate única de tipo "dirección de comportamiento", cuerpo down_proj profundo.
- **Ficha/cabeceras**: base declarada `Qwen3.8-27B`; proxy cambió tokenizador/config? no se espera
  (receta de apenas 9 matrices) → `outside_weights` = nada significativo.

## Veredicto con el que se comparará

`lab/runs/concept27b_forensic/report.{md,json}` (se ejecuta a continuación). Concordancia esperada:
Jaccard de capas ≥0.7 con {32..48 step 2}; rango dobles ±; Spearman del protagonismo por capa ≥0.7.
