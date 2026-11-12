# Libro de predicciones

Toda predicción escrita **antes** de correr, con lo que salió. La tasa de acierto dice dónde comprendemos y dónde
solo describimos: una predicción fallida es ignorancia hecha visible, y es el dato más útil de este archivo.

Reglas: una fila por predicción pre-registrada (plan commiteado antes del dato). Se puntúa contra la regla del plan,
no contra la impresión posterior. Pendiente = corriendo o sin correr. No se reescribe una predicción después del
resultado: si el plan se enmendó antes de correr, se anota la enmienda. Cada sección nueva de `results.md` con
predicción añade su fila aquí.

Puntuación: ✅ acierta · ❌ falla · ◐ parcial (acierta el signo o la mitad de la regla) · ⏳ pendiente · — no corrido.

| nº | fecha | dónde se escribió | predicción | salió | nota | resultado |
|---|---|---|---|---|---|---|
| 1 | 2026-11-06 | `h4_plan.md` | H4 falla en el criterio 1: el carril compartido (A1) corta menos de 10 bucles que A0 | A1 18 cortes vs A0 24 (se pedía ≤ 14): falla | §39.7 | ✅ |
| 2 | 2026-11-06 | `h5_plan.md` | B_all corta ≥ 10 ítems menos; localización "repartida"; S_r no gana a R_r; H_ax nulo | H5 no se corrió (se pasó a H6) | — | — |
| 3 | 2026-11-06 | `plan_turbo27b.md` P1 | la descensura está en las capas que señale el salto de `down_proj`/`o_proj`/`out_proj` (predicho 26-63) | documentado 26-55, Jaccard 0.79 (≥ 0.7) | §43.20; el inicio exacto, el fin no se ve | ✅ |
| 4 | 2026-11-06 | `plan_turbo27b.md` P2 | rango 1 por tensor | mediana 13 direcciones para el 90 % | §43.20; en una suma con un fine-tune difuso solo asoma la primera dirección | ❌ |
| 5 | 2026-11-06 | `plan_turbo27b.md` P3 | el resto es difuso, con el perfil de las mejoras del 9B (Spearman ≥ 0.7) | Spearman 0.927 | §43.20 | ✅ |
| 6 | 2026-11-07 | `h6_plan.md` etapa 0 | E_2 (prolongar los primeros fotogramas) recupera 0.4-0.7; falla a 0.7 | recuperación −0.05 a 0.07, no gana al aleatorio | §40.2; falla, pero mucho más abajo de lo predicho | ◐ |
| 7 | 2026-11-07 | `h6_plan.md` H6a | r90 entre 51 y 102 | r90 = 20 | §40.3; la habilidad es de rango mucho más bajo que lo esperado | ❌ |
| 8 | 2026-11-07 | `h6_plan.md` H6b | el corte propio gana al aleatorio de igual norma por ≥ 20 puntos | gana por 12-14 (p < 1e-13) | §40.3; signo bien, margen no: el aleatorio proyecta el propio ΔW y conserva señal | ◐ |
| 9 | 2026-11-07 | `h6_plan.md` H6c | falla: el carril de las otras semillas recupera 0.5-0.8 de M2 | recupera 1.02 × M2 (umbral 0.8): pasa | §40.3; **falla por arriba**: el carril es de la tarea, no de la semilla | ❌ |
| 10 | 2026-11-07 | `h6_plan.md` | la recuperación sube más deprisa que la energía | recuperación r=10 0.71, r=20 0.91, r=51 0.99; la energía por r no está tabulada en §40.3 | no puntuable sin la curva de energía | — |
| 11 | 2026-11-07 | `h6_plan.md` H6d | pasa en parte: el lado de entrada de ΔW se alinea con las activaciones, recuperación 0.5-0.8 | no se corrió | — | — |
| 12 | 2026-11-07 | `h6_plan.md` H6e | un gradiente falla el 0.5, queda en 0.2-0.4 | recuperación −1.23 a −0.62: rompe el modelo | §40.4; signo bien, magnitud no: no era "poco", era destructivo | ◐ |
| 13 | 2026-11-07 | `h6_plan.md` add. c | G_r falla (0.1-0.3); G^X_r mejor pero < 0.5 (0.2-0.45) | H6f-a: GXm R 0.00-0.04, SXm 0.11-0.15 | §40.6; falla como se dijo, por debajo del rango | ◐ |
| 14 | 2026-11-07 | `h6_plan.md` add. d H6f-a | los dos fallan, 0.0-0.3, sin bajar de M0 | R 0.00-0.15, ninguno baja de M0 | §40.6 | ✅ |
| 15 | 2026-11-07 | `h6_plan.md` add. d H6f-b | pasa a 200 pasos, límite a 100 (0.7-1.0) | R 0.57 a 100, 0.76 a 200; **falla**: la base aleatoria de rango 20 aprende casi igual (0.63) | §40.6; el fallo vino por el control, que no estaba en la predicción | ❌ |
| 16 | 2026-11-08 | `h7_plan.md` | P_scale 0.1-0.3; P_lin 0.4-0.7 (falla); P_mlp algo mejor; multiplicar se predicen entre sí mejor que rev4 | P_scale 0.38, P_lin 0.23, P_mlp 0.13; parar en k (0.50) gana a todos | §41.3; el predictor más simple fue el mejor: el salto aprendido no existe con estos rasgos | ❌ |
| 17 | 2026-11-08 | `h8_plan.md` | (sin predicción numérica; pregunta A/B) | A: ninguna parte sirve sola; B: pares suman (0.89), siete no (0.60) | §42 | — |
| 18 | 2026-11-11 | `h6_plan.md` add. e H6g | BP1 R 0.2-0.5; si < 0.25 no se corre SPSA; Z3 falla | BP1 0.035, BP2 0.192: ningún brazo SPSA se corre | §44.1; falla por abajo: ni con gradiente exacto llega esa parametrización | ❌ |
| 19 | 2026-11-11 | `h6_plan.md` add. f H6h | W_all 0.3-0.6; G_all < 0.2; W_down 0.1-0.3; ninguno pasa limpio | W_all 0.01, G_all 0.08, W_down 0.04 | §44.2; G_all acierta; los dos ridge fallan por abajo: el precondicionado no ayuda | ◐ |
| 20 | 2026-11-11 | `act_map/PLAN.md` puerta 1 | R² 0.6-0.8 en el medio, más flojo al final; coseno *late* > 0.7 en más de la mitad; 0.8B → 9B más bajo; embeddings alto en los primeros estados | R² 0.731; cos ≈ 0.99 en todos; 0.8B → 9B 0.555 | §45 | ✅ |
| 21 | 2026-11-11 | `act_map/PLAN.md` puerta 2 | G 0.3-0.6, mejor a ¼ y peor a ¾; 0.8B → 9B pasa como mucho en una profundidad | G ≥ 0.5 en 3/3 profundidades y en las dos direcciones; no baja con la profundidad | §45.1; **falla por arriba** | ❌ |
| 22 | 2026-11-11 | `act_map/PLAN.md` puerta 3 | paso 0: coseno port/own ≥ 0.7; `own` pasa; `port` pasa P1-P5 a menor escala; P6 falla (feliz es vocabulario) | coseno 0.65; `own` falla C1/C2 (NO DECIDIDA); embport no lleva la dirección, port sí (léxico +2.1) | §45.2; falla en las cuatro partes; la más informativa: el mapa de embeddings no lleva la dirección | ❌ |
| 23 | 2026-11-11 | `happy/PLAN.md` repetición | pasa; la escala elegida puede ser mayor (la nueva capa 31 es menor) | falla C4/C5 (34/40 coherentes, GSM 5 vs 9); escala 0.25 igual; la capa 31 nueva es **mayor** (48.7 vs 38.6) | §45.3; falla en las dos partes, y la premisa ("menor") era falsa | ❌ |
| 24 | 2026-11-12 | `cc/PLAN.md` addendum | cc_v2 pasa dev y sellado; escala 1.0 otra vez | criba: escala 1.0 (38 vs 35 del antiguo); dev corriendo | — | ⏳ |
| 25 | 2026-11-12 | `cc/PLAN.md` "Name test" | fuera de matemáticas, cc_v2 ×1.0: HumanEval-20 acorta (N1, ~200-350 tokens) y pierde 3-6 correctas (N2 falla); lógica Horn 120 acorta −30 a −60 % con ≤ 6 de coste (N2 pasa); la base corta < 10 % (N3 no aplica); placebo dentro de ±10 % tokens y ±2/±6 correctas. Lectura: "concluye con coste". Premisas marcadas: p1-p3 en el plan | — | — | ⏳ |
| 26 | 2026-11-12 | `cc/PLAN.md` "Composition" | J (Jackrong ×0.5) en dev: GSM 86-93, MATH 31-37. J+C: redundante en GSM (±3 de J), MATH 2-6 por debajo de J, tokens −10 a −25 %, más colas en bucle → K1 "redundante", tirando a "interfiere" en MATH. Control J+placebo: GSM 3-10 por debajo de J, MATH ±3. Sorpresa si J+C gana a J por > 5 | — | — | ⏳ |

## Cuenta (26 filas; 19 puntuables)

| | n |
|---|---|
| ✅ acierta | 6 |
| ◐ parcial | 5 |
| ❌ falla | 8 |
| ⏳ / — | 7 |

Acierto limpio 6/19 = 32 %; acierto o parcial 11/19 = 58 %. Con "confianza baja" declarada en casi todas, 30 % es
lo que cabe esperar de rangos estrechos sobre fenómenos que no se entienden; el número que importa es **cómo** fallan.

## Lectura: dónde está la ignorancia

- **Fallamos por arriba cuando el modelo es más lineal/transportable de lo que creemos** (nº 9, 20-21): el carril es de
  la tarea y no de la semilla; el 0.8B usa estados del 9B a cualquier profundidad. Nuestro prior subestima cuánto
  comparten modelos y corridas distintas.
- **Fallamos por abajo cuando intentamos sintetizar en vez de reutilizar** (nº 6, 12-13, 16, 18-19): extrapolar,
  un gradiente, SPSA, ridge: todo queda por debajo del rango predicho, varias veces muy por debajo. Nuestro prior
  sobreestima lo que se puede construir sin que el optimizador lo recorra.
- **Fallamos cuando el control sorprende** (nº 8, 15): el aleatorio de igual norma conserva más de lo esperado. La
  predicción debería incluir el control, no solo el brazo.
- **Fallamos cuando una premisa no se midió** (nº 23: "la capa 31 nueva es menor" era falsa; nº 22: "feliz es
  vocabulario"). Comprobar las premisas de la predicción antes de predecir.
- **Acertamos en lo forense y lo estructural** (nº 1, 3, 5, 14, 20): dónde está un cambio, qué perfil tiene, qué
  no va a funcionar. Describir se nos da mejor que predecir magnitudes.

Mejora de método que sale de aquí: cada predicción nueva lleva (a) un número para el brazo **y** otro para el control,
(b) las premisas que asume, marcadas, y (c) un rango explícito de "me sorprendería si". Se aplica desde la fila 24.
