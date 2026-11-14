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
| 24 | 2026-11-12 | `cc/PLAN.md` addendum | cc_v2 pasa dev y sellado; escala 1.0 otra vez | criba: escala 1.0 (38 vs 35 del antiguo); dev PASA (83 / 35, placebo 26 / 5); sellado CONFIRMA (86 / 33 vs base 43 / 4; placebo 28 / 1) | §45.4; sin número para el control ni rango de sorpresa (la regla nueva se aplica desde la 25); la capa 31 corregida es mayor y aquí ayuda | ✅ |
| 25 | 2026-11-12 | `cc/PLAN.md` "Name test" | fuera de matemáticas, cc_v2 ×1.0: HumanEval-20 acorta (N1, ~200-350 tokens) y pierde 3-6 correctas (N2 falla); lógica Horn 120 acorta −30 a −60 % con ≤ 6 de coste (N2 pasa); la base corta < 10 % (N3 no aplica); placebo dentro de ±10 % tokens y ±2/±6 correctas. Lectura: "concluye con coste". Premisas marcadas: p1-p3 en el plan | HumanEval 18/20 (−2, N2 pasa), 229 tokens; lógica 119/120, −57 %; base no corta; placebo HumanEval en banda, **lógica fuera** (113, −7, 7 cortadas por bucle). Lectura: "el nombre se sostiene" | §45.6; acierta N1, los rangos de tokens y las premisas p1-p2; falla el coste de HumanEval (menor que lo predicho) y el control de lógica (el placebo daña) | ◐ |
| 26 | 2026-11-12 | `cc/PLAN.md` "Composition" | J (Jackrong ×0.5) en dev: GSM 86-93, MATH 31-37. J+C: redundante en GSM (±3 de J), MATH 2-6 por debajo de J, tokens −10 a −25 %, más colas en bucle → K1 "redundante", tirando a "interfiere" en MATH. Control J+placebo: GSM 3-10 por debajo de J, MATH ±3. Sorpresa si J+C gana a J por > 5 | J 88 / 35; J+C 89 / 38 (GSM p 1,0; MATH p 0,58) → K1 "redundante"; tokens J+C/J **0,39 / 0,41**; cortes 9 → 0 y 15 → 1; bucles 0 / 0. Control J+P 86 / 33 (−2 / −2, en banda) | §45.7; acierta J en rango, K1 y el control; falla el sentido en MATH (+3, no −2 a −6), los tokens (−60 %, no −10 a −25 %) y los bucles (cero, no más). Tercera vez que sobrestima el coste del vector | ◐ |
| 27 | 2026-11-12 | `cc_weights/PLAN.md` "Repeat with the corrected vector" | recompilar cc_v2 en `down_proj`: G0 ≈ 0.032, 30/31 `ffn_down` idénticos al primero; dev fusionado GSM 84-90, MATH 31-38, placebo fusionado 25-40 / 2-7, pasa C1-C4; sellado 2 (base 46/7): vec_v2 86-92 / 27-35, fusionado 86-92 / 26-34, placebo 25-40 / 2-7, confirma; HumanEval-20 fusionado 18-20. Premisas p1-p3 en el plan. Sorpresa si fusionado < vec_v2 por > 5 en MATH, placebo > base, G0 > 0.1 o < 28 capas idénticas | G0 0,032 (peor 0,043); 30/31 idénticos; capa 31 err 0,35 / cos 0,94; dev fusionado 85 / 30, placebo 27 / 3, pasa C1-C4 (C4 MATH 0/5 p 0,06); sellado 2: vec_v2 86 / 27, fusionado 89 / 25, placebo 34 / 5, confirma (C1 p 1,6e-11 / 7,6e-6; C4 p 0,55 / 0,69); HumanEval-20 19 | §45.8; acierta todo menos MATH del fusionado (un ítem por debajo del rango en dev y en sellado) y el error de la capa 31 (mejor que el predicho: p2 falla en el buen sentido). Nada en la zona de sorpresa | ◐ |
| 28 | 2026-11-13 | `cc/PLAN.md` "Prompt control" | base + frase de cierre en el prompt, dev a 2 048: GSM 62-75, MATH 12-22; cortes 20-35 / 30-40; bucles ≥ base; tokens GSM 1 100-1 400. Lectura R2: "la dirección hace más que un prompt" (cc_v2 83 / 35 por encima con p < 0,05 al menos en GSM). Control (frase neutra): ±3 de 50 / 6, cortes ±5. Premisas p1-p3 en el plan. Sorpresa si P_close ≥ cc_v2 − 3 en los dos, o P_close < base | P_close 37 / 4 (7/20 frente a base, p 0,019), cortes 63 / 45, tokens 1 675, bucles 2; cc_v2 46/0 y 31/0 por encima; control P_neutral 29 / 3 (−21, p 0,0015), cortes 71 | §45.9; falla el brazo, los cortes, los tokens y el control; acierta solo el sentido de R2 y bucles ≥ base. Zona de sorpresa: P_close < base. Premisa p3 falla (el modelo no sigue la instrucción de cerrar): el brazo no mide prompt frente a dirección | ❌ |
| 29 | 2026-11-13 | `cc/PLAN.md` "Placebo distribution" | 8 placebos de cc_v2 (semillas 1-8, misma norma por capa) a ×1.0 en la criba (64 MATH, 2 048; base 8/64, vector 38/64): mediana 5-9 aciertos, mín 2-5, máx ≤ 12; cortes 52-63; bucles 2-12; tokens 1 850-2 000. D2: 0 semillas sobre la base con p < 0,05 y 0 con ≥ 23 (mitad de la ganancia). D3: mediana ≤ 8, 0-3 semillas significativamente por debajo. Vector rango 1 de 9. Premisas p1-p3 en el plan (p1 comprobada: semilla 2026 regenerada idéntica). Sorpresa si alguna semilla ≥ 20, mediana ≥ 12 o vector no rango 1 | §45.10 | mediana 6,5, mín 3, máx 8; cortes 56-61; bucles 1-4; tokens 1 952-2 015. D2: 0 y 0. D3: mediana ≤ base, 0 significativas. Vector rango 1 de 9. Ninguna sorpresa | ◐ (D1-D3 y rango dentro; bucles mín 1 < 2 en dos semillas; tokens 2 009 y 2 015 > 2 000 en dos) |
| 30 | 2026-11-13 | `act_map/PLAN.md` "Gate 3b" | "concluir" 9B → 0.8B por los mapas de la puerta 1, 0.8B a 2 048: paso 0 cos(port, own) 0,4-0,7; dev base acaba 0-3/64 GSM; own y port acaban 5-25 a su mejor escala (own 0,5-1, port 1-2), embport ≤ base + 3. Test (GSM /100, MATH /50): base 0-5 / 0-3; own 5-25 / 2-10; port 5-25 / 2-8; embport y placebo ±3 de base; exp8 30-55 / 10-25; exp8_placebo ±3. Referencia = exp8 (own no llega a p < 0,05); Q1, Q2 (GSM), Q3 y Q5 pasan, **Q4 falla** (port 20-50 % de la ganancia de exp8): puerta 3b no pasa. Premisas p1-p4 en el plan. Sorpresa si port ≥ exp8 − 5 en GSM, port acaba ≤ base + 2 a toda escala, placebo > base con p < 0,05, u own ≥ exp8 | — | — | ⏳ |
| 31 | 2026-11-13 | `cc/PLAN.md` "Token-budget control" (escrito por otra sesión; integrado aquí tal cual, antes del dato) | GSM8K dev a 8 192: base 82-92/100, cortes ≤ 8; vector cc_v2 ×1.0 = su valor a 2 048 (83) ± 2, sigue cerrando mal sus 13; fusionado = vector ± 2. Sorpresa si base < 70, vector > base + 5, o los 3 ítems que la base acertaba a 2 048 y el vector cerró mal (50, 63, 82) salen bien con el vector a 8 192. Nota previa al dato: §39.6 ya midió la misma base a 8 192 (72/100, 24 cortadas); la predicción se dejó como estaba | §45.11 | base a 8 192: **72**/100, 24 cortadas (idéntica ítem a ítem a §39.6); vector a 8 192: 85, 4 cortadas, 13 acabadas mal (11 de las 13 de antes siguen mal); vector > base por 13 (16/3, p 0,0044); ítems 50, 63, 82 con el vector a 8 192: mal, mal, bien. Fusionado: **no medido**, el brazo sirvió la base (el script pasó el GGUF fusionado como control-vector; rechazado) | ❌ (base fuera de banda por 10; sorpresa declarada "vector > base + 5" ocurre; fusionado sin dato) |

## Cuenta (31 filas; 26 puntuables)

| | n |
|---|---|
| ✅ acierta | 6 |
| ◐ parcial | 9 |
| ❌ falla | 11 |
| ⏳ / — | 5 |

Acierto limpio 6/26 = 23 %; acierto o parcial 15/26 = 58 %. (Cuenta rehecha fila a fila el 2026-11-12: la tabla
anterior decía 7 ✅ / 8 ❌ con las mismas filas; el recuento por símbolos da 6 / 9.) Con "confianza baja" declarada en casi todas, 30 % es
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
(b) las premisas que asume, marcadas, y (c) un rango explícito de "me sorprendería si". Se aplica desde la fila 25 (la 24
se escribió antes de esta regla, sin número para el control).
