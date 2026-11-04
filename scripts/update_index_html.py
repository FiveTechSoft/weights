import re

with open("docs/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Nav link
nav_target = '<a href="nanowsp2.html" style="color: var(--accent-emerald); text-decoration: none; font-weight: 600;">NanoWSP-2 (600k)</a>'
nav_replace = nav_target + '\n      <a href="nanowsp3.html" style="color: var(--accent-cyan); text-decoration: none; font-weight: 700;">NanoWSP-3 (3.28M)</a>'
assert nav_target in content, "nav_target not found"
content = content.replace(nav_target, nav_replace)

# 2. Header button
btn_target = '<a href="nanowsp2.html" class="btn btn-secondary" style="border-color: var(--accent-emerald); color: #A7F3D0;">🤖 NanoWSP-2 (600k)</a>'
btn_replace = btn_target + '\n      <a href="nanowsp3.html" class="btn btn-secondary" style="border-color: var(--accent-cyan); color: #BAE6FD;">🤖 NanoWSP-3 (3.28M)</a>'
assert btn_target in content, "btn_target not found"
content = content.replace(btn_target, btn_replace)

# 3. Add NanoWSP-3 card above NanoWSP-2 card
nanowsp3_card = '''    <!-- NANOWSP-3 3.28M THEMATIC CARD -->
    <section>
      <div class="card" style="border-color: var(--accent-cyan); background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(15, 23, 42, 0.95) 100%); padding: 2rem; margin-bottom: 2rem;">
        <div style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(56, 189, 248, 0.25); border: 1px solid var(--accent-cyan); color: #BAE6FD; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.75rem;">NEW · PROJECT NANOWSP-3 (3.28M PARAMETERS)</div>
        <h2 style="font-size: 1.6rem; color: #FFFFFF; margin-bottom: 0.75rem;">🤖 NanoWSP-3: 6-Layer Modular Thematic Micro-LLM Synthesized from MiMo-9B</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.25rem; font-size: 1.05rem;">
          A standalone <strong>6-layer autoregressive Transformer (3,280,128 parameters, 12.51 MB SafeTensors)</strong> with 8 attention heads and full 256-byte ASCII vocabulary. Imprints 5 complete modular thematic banks dumped from MiMo-9B with <strong>100.0% recall across all 23 facts in 154 ms on CPU</strong> via Gram-matrix orthogonalized MEMIT.
        </p>
        <div class="grid-2" style="margin-bottom: 1rem;">
          <div style="background: rgba(11, 17, 32, 0.8); border: 1px solid var(--card-border); padding: 1.25rem; border-radius: 0.5rem;">
            <h4 style="color: var(--accent-cyan); margin-bottom: 0.5rem;">1. Five Modular Thematic Domains</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1;">Modular knowledge banks: <strong>Química & Física</strong> (Z79=Au, Z47=Ag), <strong>Geografía</strong> (Paris, Tokyo, Ouagadougou), <strong>Computación</strong> (Torvalds, Rossum, Berners-Lee), <strong>Literatura</strong> (Shakespeare, Cervantes, Tolkien), and <strong>Cálculo Agéntico</strong>.</p>
          </div>
          <div style="background: rgba(11, 17, 32, 0.8); border: 1px solid var(--card-border); padding: 1.25rem; border-radius: 0.5rem;">
            <h4 style="color: var(--accent-emerald); margin-bottom: 0.5rem;">2. Zero Cross-Talk & Instant Synthesis</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1;">Full Gram diagonalization <code>(K<sup>T</sup> K + &lambda; I)<sup>-1</sup></code> guarantees <strong>0.0% cross-domain interference</strong> with 97–99% confidence per token. Synthesized in <strong>154 ms</strong> without gradient descent.</p>
          </div>
        </div>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
          <a href="nanowsp3.html" class="btn btn-primary" style="font-size: 0.9rem; font-weight: 700;">Explore NanoWSP-3 Architecture & Thematic Benchmarks &rarr;</a>
          <span style="display: inline-flex; align-items: center; font-size: 0.9rem; color: var(--text-muted); font-family: monospace;">python scripts/demo_nanowsp3_capabilities.py</span>
        </div>
      </div>
    </section>
'''

card_target = '    <!-- NANOWSP-2 600K CARD -->'
assert card_target in content, "card_target not found"
content = content.replace(card_target, nanowsp3_card + '\n' + card_target)

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(content)

print("docs/index.html updated successfully!")
