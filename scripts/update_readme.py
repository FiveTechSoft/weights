with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

target = "* **✅ Project NanoWSP-2 Micro-LLM (600k Parameters):** An advanced 4-layer autoregressive Transformer (599,168 weights, 4 heads, 256-byte vocabulary) synthesized completely without backpropagation. Delivers **100.0% recall** on symbolic arithmetic, relational knowledge graphs, and syntax closure, plus test-time self-healing loop breaking via Titans surprisal monitoring ([NanoWSP-2 Webpage](https://fivetechsoft.github.io/weights/nanowsp2.html) | [Showcase Script](scripts/demo_nanowsp2_capabilities.py))."

replacement = target + """\n> * **✅ Project NanoWSP-3 Thematic Micro-LLM (3.28M Parameters, 6 Layers):** A 6-layer autoregressive Transformer (3,280,128 Float32 weights, 8 heads, 256-byte ASCII vocabulary) synthesized in weight space. Integrates 5 modular thematic memory banks dumped from MiMo-9B (Química & Física, Geografía, Computación, Literatura y Herramientas Agénticas) with **100.0% recall across all 23 facts in 154 ms on CPU** via Gram-matrix orthogonalized MEMIT ([NanoWSP-3 Webpage](https://fivetechsoft.github.io/weights/nanowsp3.html) | [Showcase Script](scripts/demo_nanowsp3_capabilities.py)).\n> * **✅ Cross-Model Knowledge Transfer (MiMo-9B &rarr; TinyLlama-1.1B):** Orthogonal Procrustes latent alignment ($d_1 = 4096 \\to d_2 = 2048$, 82.5% cosine fidelity on 2,000 anchor words) and closed-form MEMIT knowledge imprinting. Corrected real-world factual hallucinations in TinyLlama (Atomic #79 $\\to$ Gold, Deepest Lake $\\to$ Baikal) from 0% to **100% exact recall in 18.4 ms** with 0% collateral damage on holdout tasks and 0 gradient steps ([Report](rig_runs/tinyllama_knowledge_transfer_benchmark.json))."""

assert target in text, "target not found in README.md"
text = text.replace(target, replacement)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)

print("README.md updated successfully!")
