# X (Twitter) Thread — English Version

**Topic:** Demystifying Weight-Space Programming (MIT Exp 8), Declarative Loop Suppression on MiMo-9B with WSP-Lang, and Official Hugging Face Release.

---

### Tweet 1 (Hook & Finding)
Why did editing just 0.016% of Qwen3.5-0.8B weights boost its math reasoning score?

It didn’t inject new formulas or knowledge. It surgically excised a pathological self-doubt attractor loop.

We reproduced it empirically step by step—and then brought the fix to MiMo-9B 👇🧵

---

### Tweet 2 (Empirical Proof on Qwen-0.8B)
On GSM8K Problem #2 ("A robe takes 2 bolts of blue fiber and half that much white fiber..."):
❌ Base Model: 512 tokens looping in self-doubt ("what does half mean? let's re-verify...") until token cutoff (0% acc).
✅ Edited Model (MIT Exp 8): Closes reasoning in 80 tokens: Total=3 (+20% delta).

Weight surgery removed an obstacle; it wasn't magic.

---

### Tweet 3 (What happens in MiMo 2.6 Qwen 9B?)
We tested whether this pathology exists in Xiaomi's MiMo-V2.6-Distill-Qwen-9B:
• Math reasoning: Clean. Solves GSM8K in ~130 tokens with 0 loops.
• Agent tool-calling: Acute pathology. 100% of targeted stress probes trap MiMo in infinite repetitive `<tool_call>` loops upon unexpected tool feedback.

---

### Tweet 4 (Declarative WSP-Lang Fix & SafeTensors Surgery)
We traced the loop attractor to Layer 14 down-proj (`u_loop_L14.npz`).
By subtracting a rank-1 projection, editing only 120,545 weights (0.24% of L14, 0.0013% of 9B params), we break the loop without retraining!

Defined declaratively in WSP-Lang 0.1:
```wspl
experiment fix_mimo_tool_loops:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: suppress_agentic_tool_loops
    budgets:
        max_modified_weights = 1271011
        max_delta_norm = 0.04
```

---

### Tweet 5 (Hugging Face Official Discussion & Open Source)
We just opened the official research discussion on Hugging Face with 3 deployment options (Pure PyTorch, WSPLang, or llama.cpp Control Vector):

🤗 HF Discussion:
https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/discussions/16

🔗 GitHub (MIT):
https://github.com/FiveTechSoft/weights

🌐 Web:
https://fivetechsoft.github.io/weights/

CC: @rikiyat @simonw @jarrodwatts @jacsebl @stochi0 @PawelHuryn @javierbits
