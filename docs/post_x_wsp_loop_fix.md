# Propuesta de Publicación para X (Twitter / Thread)

**Tema:** Desmitificando Weight-Space Programming (MIT Exp 8), Corrección Declarativa de Bucles en MiMo-9B con WSP-Lang y Discusión Oficial en Hugging Face.

---

### Versión en Hilo (Optimizado para X / Twitter)

**Tweet 1 (Hook & Hallazgo):**
¿Por qué editar solo el 0.016% de los pesos de Qwen3.5-0.8B multiplicó su puntuación en matemáticas?

No le inyectó nuevas fórmulas ni sabiduría. Extirpó un atractor patológico de auto-duda que bloqueaba al modelo.

Lo hemos reproducido empíricamente paso a paso, y lo hemos llevado a MiMo-9B 👇🧵

---

**Tweet 2 (La prueba empírica en Qwen-0.8B):**
En el problema #2 de GSM8K ("A robe takes 2 bolts of blue fiber and half that much white fiber..."):
- ❌ **Modelo Base:** 512 tokens en bucle de indecisión ("what does half mean? let's re-verify... on the other hand...") hasta cortarse por límite de tokens (0% acierto).
- ✅ **Modelo Editado (MIT Exp 8):** Cierra el pensamiento en 80 tokens y concluye: "Blue=2, White=1, Total=3" (+20% delta).

La cirugía de pesos fue una desobstrucción del canal de salida, no magia.

---

**Tweet 3 (¿Qué pasa en MiMo 2.6 Qwen 9B?):**
Comprobamos si este fallo existía en MiMo-9B:
- En matemáticas: NO existe. MiMo razona de forma concisa en 130 tokens y cierra el `<think>` limpiamente.
- En agentes (Tool-Calling): **SÍ existe**. Al usar herramientas (`<tool_call>`), MiMo entra en bucles idénticos el **70% de las veces** (invocando la misma función con los mismos argumentos una y otra vez ante errores).

---

**Tweet 4 (La solución declarativa con WSP-Lang & Cirugía de Pesos):**
Aislamos el atractor singular en la Capa 14 (`u_loop_L14.npz`) y aplicamos una sustracción de rango 1 en los pesos BF16 (`model.language_model.layers.14.mlp.down_proj.weight`).

Modificando solo 120.545 pesos (0,24% de la capa, 0,0013% del modelo total) desarmamos el bucle sin reentrenar.

Lo definimos declarativamente en **WSP-Lang 0.1**:
```wspl
experiment fix_mimo_tool_loops:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: suppress_agentic_tool_loops
    budgets:
        max_modified_weights = 1271011
        max_delta_norm = 0.04
```

---

**Tweet 5 (Publicación en Hugging Face & Open Source):**
Hemos abierto el hilo oficial en Hugging Face con el informe técnico y las 3 opciones para aplicarlo (PyTorch puro, WSPLang o Control Vector):

🤗 Discusión en Hugging Face:
https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/discussions/16

🔗 Repositorio GitHub (MIT):
https://github.com/FiveTechSoft/weights

🌐 Portal interactivo:
https://fivetechsoft.github.io/weights/

CC: @rikiyat @simonw @jarrodwatts @jacsebl @stochi0 @PawelHuryn @javierbits
