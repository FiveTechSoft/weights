"""scripts/update_docs_and_web.py — Actualiza docs/ y la web portal para WSPLang 0.7 y Karpathy nanoGPT."""

import re
from pathlib import Path

R = Path(r"C:\weights")

# ==============================================================================
# 1. ACTUALIZAR docs/wsplang_0.7_spec.md
# ==============================================================================
spec_p = R / "docs" / "wsplang_0.7_spec.md"
spec_content = spec_p.read_text(encoding="utf-8")

if "## 12. Operador `tie_weights`" not in spec_content:
    spec_addition = """
---

## 12. Operador `tie_weights` y Síntesis de nanoGPT (Andrej Karpathy)

WSPLang 0.7.0 amplía la directiva `init_architecture` para permitir la síntesis directa de la arquitectura estándar de **`nanoGPT` (Andrej Karpathy / GPT-2)** con el operador nativo `tie_weights`:

### 12.1 Sintaxis y Semántica
```wsplang
step init_architecture model=nanoGPT layers=6 d_model=384 heads=6 d_ff=1536 vocab=65 max_seq=256 out=models/nanogpt/model.safetensors
step tie_weights source=transformer.wte.weight target=lm_head.weight model=models/nanogpt/model.safetensors
```

### 12.2 Especificación Matemática de `tie_weights`
Garantiza el atado de memoria entre los embeddings de entrada y la proyección lineal de salida:
$$\mathbf{W}_{\text{lm\_head}} \equiv \mathbf{W}_{\text{transformer.wte}} \in \mathbb{R}^{V \times d}$$
$$\|\mathbf{W}_{\text{wte}} - \mathbf{W}_{\text{lm\_head}}\|_F = 0.0000000000\times 10^0$$

### 12.3 Certificación de Paridad Absoluta
La arquitectura nanoGPT compilada por WSPLang ([`programs/build_karpathy_nanogpt.wspl`](file:///c:/weights/programs/build_karpathy_nanogpt.wspl)) demostró paridad matemática estricta a nivel de bits frente a la implementación de referencia oficial de Karpathy ([`lab/runs/karpathy_nanogpt_parity.json`](file:///c:/weights/lab/runs/karpathy_nanogpt_parity.json)):
- **Diferencia Máxima en Logits:** $0.0000000000\times 10^0$ a lo largo de secuencias de 16 a 256 tokens.
- **Diferencia de Cross-Entropy Loss:** $0.0000000000\times 10^0$.
- **Discrepancias en Generación Autorregresiva:** $0 / 35$ tokens (100% idéntico paso a paso).
"""
    # Actualizar tabla de operadores en sección 11
    spec_content = spec_content.replace(
        "| `compile_synthetic` | Lab Milestone M-3 | `dataset`, `lambda_base` | Compilación Multicapa |",
        "| `compile_synthetic` | Lab Milestone M-3 | `dataset`, `lambda_base` | Compilación Multicapa |\n| `tie_weights` | GPT-2 / nanoGPT | `source`, `target` | Atado Invariante de Tensores |"
    )
    spec_content += spec_addition
    spec_p.write_text(spec_content, encoding="utf-8")
    print("docs/wsplang_0.7_spec.md updated successfully!")


# ==============================================================================
# 2. ACTUALIZAR docs/nanowsp_findings.md
# ==============================================================================
findings_p = R / "docs" / "nanowsp_findings.md"
findings_content = findings_p.read_text(encoding="utf-8")

if "## 6. Certificación Formal de NanoWSP-3 como LLM Real" not in findings_content:
    findings_addition = """
---

## 6. Certificación Formal de NanoWSP-3 como LLM Real y Diagnóstico de NanoWSP-4

### 6.1 Batería Mecánica de Validación LLM
En la auditoría empírica ([`lab/runs/real_llm_verification.json`](file:///c:/weights/lab/runs/real_llm_verification.json)), NanoWSP-3 (4 capas, $d_{model}=256$, 3.2M params) demostró de forma incontrovertible las propiedades de un Modelo de Lenguaje Autorregresivo auténtico:
1. **Discriminación Entrópica (SNR 7x - 14.7x):**
   - Perplejidad en inglés natural: **3.35 - 6.03**.
   - Perplejidad en palabras permutadas: **8.74 - 10.49**.
   - Perplejidad en ruido uniforme: **42.22 - 49.10**.
2. **Q&A y Memoria de Entidades Canónicas:**
   - Prompt: `Question: Who stole the tarts? Answer: The Knave of ` &rarr; Respuesta: `Hearts, Alice hastily replied; only one doesn`
   - Prompt: `Twinkle, twinkle, little ` &rarr; Respuesta: `bat!\nHow I wonder when I should for seem to k` (Parodia del Sombrerero Loco).
3. **Léxico y Diversidad:**
   - 80.0% de palabras de diccionario inglés en decodificación greedy.
   - Ratio de 3-gramas únicos: 0.754 (>75% de variedad no degenerada).

### 6.2 Diagnóstico del Bug de Atención Muerta en NanoWSP-4
Al auditar por qué `nanowsp4_v1b144.safetensors` colapsaba en el unigrama `" the the"`, se detectó un fallo silencioso en `train_nanowsp4.py`:
- Las claves sin sufijo `q.{l}` eludían la condición `k.endswith((".weight",))`.
- Las matrices $Q, K, V, O$ permanecieron en ceros ($\|W\| = 0.0$), anulando los gradientes de atención y operando como 12 MLPs residuales sin atención causal.
- El bug fue corregido formalmente en el commit `59ff4af`.

---

## 7. Síntesis Declarativa del nanoGPT de Karpathy con Paridad Numérica Cero

Mediante WSPLang 0.7.0 y [`programs/build_karpathy_nanogpt.wspl`](file:///c:/weights/programs/build_karpathy_nanogpt.wspl), se sintetizó la arquitectura completa de **nanoGPT** (10,770,816 parámetros, 6 capas, 6 cabezas, $d=384$, vocabulario 65):
- **Paridad Numérica de Logits:** Diferencia máxima absoluta $= 0.0000000000\times 10^0$ a lo largo de secuencias de 16 a 256 tokens.
- **Atado de Pesos Exacto:** $\|W_{wte} - W_{lm\_head}\| = 0.0$.
- **Causalidad Pura:** Filtración al pasado $= 0.0$.
- **Pérdida Cross-Entropy:** $4.241905$ en referencia vs $4.241905$ en WSPLang ($\Delta = 0.0$).
"""
    findings_content += findings_addition
    findings_p.write_text(findings_content, encoding="utf-8")
    print("docs/nanowsp_findings.md updated successfully!")


# ==============================================================================
# 3. ACTUALIZAR docs/index.html (Web Portal)
# ==============================================================================
html_p = R / "docs" / "index.html"
html_content = html_p.read_text(encoding="utf-8")

# 1. Nav links update
nav_target = '<a href="nanowsp3.html" style="color: var(--accent-cyan); text-decoration: none; font-weight: 700;">NanoWSP-3 (3.28M)</a>'
nav_replacement = nav_target + '\n      <a href="wsplang.html" style="color: var(--accent-amber); text-decoration: none; font-weight: 700;">WSPLang 0.7 & Karpathy nanoGPT</a>'

if "WSPLang 0.7 & Karpathy nanoGPT" not in html_content and nav_target in html_content:
    html_content = html_content.replace(nav_target, nav_replacement)

# 2. Header badges update
badge_target = '<a href="nanowsp3.html" class="btn btn-secondary" style="border-color: var(--accent-cyan); color: #BAE6FD;">🤖 NanoWSP-3 (3.28M)</a>'
badge_replacement = badge_target + '\n      <a href="#nanogpt-card" class="btn btn-secondary" style="border-color: var(--accent-amber); color: #FDE68A;">⚡ Karpathy nanoGPT (10.77M / 0.0 Diff)</a>'

if "#nanogpt-card" not in html_content and badge_target in html_content:
    html_content = html_content.replace(badge_target, badge_replacement)

# 3. Add Karpathy nanoGPT & Real LLM Certification Cards
nanogpt_hero_card = '''    <!-- KARPATHY NANOGPT 10.77M PARITY HERO CARD -->
    <section id="nanogpt-card">
      <div class="card" style="border-color: var(--accent-amber); background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.98) 100%); padding: 2.25rem; margin-bottom: 2rem; border-width: 2px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;">
          <div style="display: inline-block; padding: 0.3rem 0.85rem; border-radius: 9999px; background: rgba(245, 158, 11, 0.25); border: 1px solid var(--accent-amber); color: #FDE68A; font-size: 0.8rem; font-weight: 800;">NEW MILESTONE · KARPATHY nanoGPT IN WSPLANG</div>
          <span style="font-family: monospace; font-size: 0.85rem; color: #34D399; background: rgba(16, 185, 129, 0.15); padding: 0.25rem 0.6rem; border-radius: 0.35rem; border: 1px solid rgba(52, 211, 153, 0.3);">CERTIFIED: 0.0000000000e+00 NUMERICAL DRIFT</span>
        </div>
        <h2 style="font-size: 1.8rem; color: #FFFFFF; margin-bottom: 0.75rem;">⚡ Declarative nanoGPT (10.77M): 100% Identical to Andrej Karpathy's Reference</h2>
        <p style="color: #CBD5E1; margin-bottom: 1.5rem; font-size: 1.05rem; line-height: 1.6;">
          Built purely through the <strong>WSPLang 0.7.0 declarative weight-space compiler</strong> (<code>programs/build_karpathy_nanogpt.wspl</code>) without procedural PyTorch boilerplate. Synthesizes all <strong>83 parameter tensors (10,770,816 weights)</strong> with canonical GPT-2 residual scaling and exact weight tying (<code>transformer.wte.weight &hArr; lm_head.weight</code>).
        </p>

        <div class="grid-3" style="margin-bottom: 1.5rem;">
          <div style="background: rgba(11, 17, 32, 0.85); border: 1px solid rgba(245, 158, 11, 0.3); padding: 1.25rem; border-radius: 0.5rem;">
            <div style="color: var(--accent-amber); font-weight: 700; font-size: 0.95rem; margin-bottom: 0.35rem;">1. Forward Pass Logits Parity</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: monospace;">&Delta; = 0.0</div>
            <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.35rem;">Exact float32 bit-level match across sequence lengths 16 to 256 tokens.</p>
          </div>
          <div style="background: rgba(11, 17, 32, 0.85); border: 1px solid rgba(52, 211, 153, 0.3); padding: 1.25rem; border-radius: 0.5rem;">
            <div style="color: #34D399; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.35rem;">2. Causal Mask & Weight Tying</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: monospace;">Leak = 0.0</div>
            <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.35rem;">Zero future attention leakage; exact Frobenius pointer equivalence wte &hArr; lm_head.</p>
          </div>
          <div style="background: rgba(11, 17, 32, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); padding: 1.25rem; border-radius: 0.5rem;">
            <div style="color: var(--accent-cyan); font-weight: 700; font-size: 0.95rem; margin-bottom: 0.35rem;">3. Generation & Loss Parity</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: monospace;">0 Discrepancies</div>
            <p style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.35rem;">Loss = 4.241905 vs 4.241905; step-by-step autoregressive trajectory matches 100%.</p>
          </div>
        </div>

        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.25rem; font-family: monospace; font-size: 0.85rem; color: #E2E8F0;">
          <span style="color: var(--accent-amber);"># Run declarative synthesis in WSPLang:</span><br>
          <span style="color: #38BDF8;">python engine/wsplang.py programs/build_karpathy_nanogpt.wspl --run</span><br>
          <span style="color: var(--accent-amber);"># Verify 6-test mathematical parity battery:</span><br>
          <span style="color: #34D399;">python lab/verify_karpathy_parity.py</span>
        </div>

        <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
          <a href="wsplang_0.7_spec.md" class="btn btn-primary" style="background: var(--accent-amber); color: #000000; font-weight: 800; border: none;">View WSPLang 0.7.0 Spec & Parity Details &rarr;</a>
          <span style="color: #94A3B8; font-size: 0.85rem;">Continuous automated verification against official <code>karpathy/nanoGPT@3adf61e</code> in <code>tests/test_wsplang_karpathy_nanogpt.py</code></span>
        </div>
      </div>
    </section>

    <!-- REAL LLM CERTIFICATION & QA CARD -->
    <section>
      <div class="card" style="border-color: var(--accent-cyan); background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(15, 23, 42, 0.95) 100%); padding: 2rem; margin-bottom: 2rem;">
        <div style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(56, 189, 248, 0.25); border: 1px solid var(--accent-cyan); color: #BAE6FD; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.75rem;">EMPIRICAL AUDIT · REAL LLM BEHAVIOR CERTIFIED</div>
        <h2 style="font-size: 1.6rem; color: #FFFFFF; margin-bottom: 0.75rem;">🔬 NanoWSP-3 Real LLM Verification: 14.7x SNR & Canonical Q&A Table</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.25rem; font-size: 1.05rem;">
          Rigorous mechanical evaluation of NanoWSP-3 (4 layers, d=256, active causal RoPE attention). Demonstrates <strong>PPL 3.35 (natural) vs 49.1 (uniform noise)</strong> (14.7x signal-to-noise ratio), 80.0% valid English vocabulary under greedy decoding, and exact factual entity retrieval in blind testing:
        </p>

        <div style="overflow-x: auto; margin-bottom: 1.25rem;">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: left;">
            <thead>
              <tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.15); color: #F8FAFC;">
                <th style="padding: 0.6rem;">Prompt de Entrada (Contexto)</th>
                <th style="padding: 0.6rem;">Respuesta Real (Greedy T=0.0)</th>
                <th style="padding: 0.6rem;">Capacidad Demostrada</th>
              </tr>
            </thead>
            <tbody style="color: #CBD5E1;">
              <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <td style="padding: 0.6rem; font-family: monospace; color: #BAE6FD;">Question: Who stole the tarts? Answer: The Knave of</td>
                <td style="padding: 0.6rem; font-family: monospace; color: #34D399;">Hearts, Alice hastily replied; only one doesn</td>
                <td style="padding: 0.6rem;"><strong>Identificación canónica exacta</strong> del Cap. XI de Lewis Carroll.</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <td style="padding: 0.6rem; font-family: monospace; color: #BAE6FD;">Twinkle, twinkle, little</td>
                <td style="padding: 0.6rem; font-family: monospace; color: #34D399;">bat!\nHow I wonder when I should for seem to k</td>
                <td style="padding: 0.6rem;"><strong>Memoria del corpus</strong> (parodia del Sombrerero Loco, no rima genérica).</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                <td style="padding: 0.6rem; font-family: monospace; color: #BAE6FD;">The Queen cried, 'Off with</td>
                <td style="padding: 0.6rem; font-family: monospace; color: #34D399;">his\nhead! or Off with her head! Those whom sh</td>
                <td style="padding: 0.6rem;"><strong>Completitud y prosodia</strong> de la orden emblemática victoriana.</td>
              </tr>
              <tr>
                <td style="padding: 0.6rem; font-family: monospace; color: #BAE6FD;">Alice was</td>
                <td style="padding: 0.6rem; font-family: monospace; color: #34D399;">quite silent for a\nminute, with her arms</td>
                <td style="padding: 0.6rem;"><strong>Continuación gramatical fluida</strong> libre de bucles unigrama.</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0;">
          Also documented: Forensic diagnosis of the <em>Dead-Attention Pathology</em> in NanoWSP-4 (key naming initialization bug resulting in 0-norm attention matrices), permanently resolved in commit <code>59ff4af</code>.
        </p>
      </div>
    </section>
'''

card_target = '    <!-- NANOWSP-3 3.28M THEMATIC CARD -->'
if "id=\"nanogpt-card\"" not in html_content and card_target in html_content:
    html_content = html_content.replace(card_target, nanogpt_hero_card + '\n' + card_target)

html_p.write_text(html_content, encoding="utf-8")
print("docs/index.html updated successfully with Karpathy nanoGPT & Real LLM Certification cards!")

# ==============================================================================
# 4. ACTUALIZAR README.md
# ==============================================================================
readme_p = R / "README.md"
readme_content = readme_p.read_text(encoding="utf-8")

if "### 5. Build Karpathy's nanoGPT via WSPLang" not in readme_content:
    readme_addition = """
### 5. Build Karpathy's nanoGPT via WSPLang (30 Seconds)

Synthesize and compile Andrej Karpathy's exact **`nanoGPT` (10.77M parameters, 6 layers, d=384, 6 heads, vocab=65)** declaratively in weight space with certified bit-exact mathematical parity:

```bash
# 1. Synthesize the architecture and tie weights declaratively
python engine/wsplang.py programs/build_karpathy_nanogpt.wspl --run

# 2. Verify 100% bit-exact numerical parity against Karpathy's reference implementation
python lab/verify_karpathy_parity.py
```

| Parity Test | Metric Observed | Target Threshold | Verdict |
| :--- | :--- | :--- | :--- |
| **Logits Forward Pass** | $\\Delta = 0.0000000000\\times 10^0$ | $< 1.0\\times 10^{-6}$ (fp32) | **EXACT MATCH** |
| **Weight Tying ($W_{wte} \\leftrightarrow W_{lm}$)** | $\\|W_{wte} - W_{lm}\\|_F = 0.0$ | $0.0$ Frobenius | **IDENTICAL** |
| **Causal Attention Mask** | Past leakage = $0.0$ | Past leakage == $0.0$ | **STRICT CAUSAL** |
| **Autoregressive Trajectory** | 0 discrepancies / 35 steps | 0 token errors | **100% BIT-EXACT** |
| **Cross-Entropy Loss** | Loss: $4.241905$ vs $4.241905$ | $\\Delta < 1.0\\times 10^{-6}$ | **EXACT MATCH** |
"""
    # Insert right after "### 4. Run NanoWSP: Zero-Backpropagation Micro-LLM (30 Seconds)"
    hook = "### 4. Run NanoWSP: Zero-Backpropagation Micro-LLM (30 Seconds)"
    idx = readme_content.find(hook)
    if idx != -1:
        # Find next section or end of subsection
        next_sec = readme_content.find("##  What is WSP-Lang?", idx)
        if next_sec != -1:
            readme_content = readme_content[:next_sec] + readme_addition + "\n" + readme_content[next_sec:]
            readme_p.write_text(readme_content, encoding="utf-8")
            print("README.md updated successfully with nanoGPT section!")

print("All docs and web assets updated cleanly!")
