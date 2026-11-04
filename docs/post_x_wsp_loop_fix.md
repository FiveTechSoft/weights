# Propuesta de Publicación para X (Twitter / Thread)

**Tema:** Desmitificando Weight-Space Programming (MIT Exp 8) y Corrección Declarativa de Bucles en MiMo-9B con WSP-Lang.

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
- En agentes (Tool-Calling): **SÍ existe**. Al usar herramientas (`<tool_call>`), MiMo entra en bucles idénticos el **70% de las veces** (invocando la misma función con los mismos argumentos una y otra vez).

---

**Tweet 4 (La solución declarativa con WSP-Lang):**
Aislamos el vector contrastivo en la Capa 14 (`u_loop_L14.npz`) y escribimos la corrección en **WSP-Lang 0.1**, nuestro lenguaje declarativo para cirugía de pesos:

```wspl
experiment fix_mimo_tool_loops:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: suppress_agentic_tool_loops
    steps:
        step peek blk.14.ffn_down.weight row=100 col=4258 top=3
        step diff blk.14.ffn_down.weight base=mimo-v2.6-9b.gguf
        step semantic_map top=3
```
Modificamos únicamente el 2.5% de los pesos de la Capa 14 sin retropropagación ni reentrenamiento.

---

**Tweet 5 (Open Source & Reproducibilidad):**
Todo el código, los vectores de dirección, los prompts originales de Atari de Simon Willison y el compilador WSP-Lang están abiertos bajo Licencia MIT:

🔗 Repo: https://github.com/FiveTechSoft/weights
🌐 Portal en vivo: https://fivetechsoft.github.io/weights/
📄 Informe completo: https://github.com/FiveTechSoft/weights/blob/main/docs/report_loop_suppression_and_atari.md

CC: @rikiyat @simonw @jarrodwatts @jacsebl @stochi0 @PawelHuryn @javierbits
