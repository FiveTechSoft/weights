# scratch/update_index.py
from pathlib import Path

html_path = Path("docs/index.html")
html = html_path.read_text(encoding="utf-8")

# 1. Update nav
old_nav = '''      <a href="index.html" style="color: var(--accent-cyan); text-decoration: none;">Overview</a>
      <a href="getting_started.html" style="color: var(--text-muted); text-decoration: none;">Getting Started</a>
      <a href="wsplang.html" style="color: var(--text-muted); text-decoration: none;">WSP-Lang Spec</a>'''

new_nav = '''      <a href="index.html" style="color: var(--accent-cyan); text-decoration: none;">Overview</a>
      <a href="wsplang.html" style="color: var(--text-muted); text-decoration: none;">WSP-Lang 0.3 Spec</a>
      <a href="https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/discussions/16" target="_blank" style="color: var(--text-muted); text-decoration: none;">🤗 HF Discussion #16</a>'''

if old_nav in html:
    html = html.replace(old_nav, new_nav)
    print("Nav updated.")
else:
    print("Warning: old_nav not found.")

# 2. Update hero buttons
old_btn = '<a href="wsplang.html" class="btn btn-secondary">📜 WSP-Lang 0.1 Spec</a>'
new_btn = '''<a href="wsplang.html" class="btn btn-secondary">📜 WSP-Lang 0.3 Spec (FWP)</a>
      <a href="https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/discussions/16" class="btn btn-secondary" target="_blank">🤗 HF Discussion #16</a>'''

if old_btn in html:
    html = html.replace(old_btn, new_btn)
    print("Hero buttons updated.")
else:
    print("Warning: old_btn not found.")

# 3. Add FWP breakthrough card
fwp_section = '''    <section>
      <div class="card" style="border-color: var(--accent-purple); background: linear-gradient(135deg, rgba(168, 85, 247, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%); padding: 2rem;">
        <div style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(168, 85, 247, 0.2); border: 1px solid var(--accent-purple); color: #D8B4FE; font-size: 0.75rem; font-weight: 700; margin-bottom: 0.75rem;">NEW IN WSP-LANG 0.3 · FAST WEIGHT PROGRAMMING</div>
        <h2 style="font-size: 1.6rem; color: #FFFFFF; margin-bottom: 0.75rem;">🧬 Linear Transformers are Secretly Fast Weight Programmers</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.25rem; font-size: 1.05rem;">
          Connecting Weight-Space Programming with the seminal framework by <strong>Kazuki Irie, Imanol Schlag, and Jürgen Schmidhuber</strong> (ICML 2021). Self-attention operates as an associative memory updated via outer products: <code>W_t = W_{t-1} + v_t ⊗ k_t^T</code>.
        </p>
        <div class="grid-2" style="margin-bottom: 1rem;">
          <div style="background: rgba(11, 17, 32, 0.8); border: 1px solid var(--card-border); padding: 1.25rem; border-radius: 0.5rem;">
            <h4 style="color: var(--accent-cyan); margin-bottom: 0.5rem;">1. The Delta-Rule Zero Cross-Talk Theorem</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1;">WSP-Lang 0.3 replaces blind projection subtractions with error-corrected unlearning: <code>W &larr; W - &eta;(u(u^T W) - v*)</code>. For any orthogonal concept <code>u_&perp;</code>, distortion is <strong>mathematically 0.0</strong>.</p>
          </div>
          <div style="background: rgba(11, 17, 32, 0.8); border: 1px solid var(--card-border); padding: 1.25rem; border-radius: 0.5rem;">
            <h4 style="color: var(--accent-purple); margin-bottom: 0.5rem;">2. Dual Architecture & Transient Overlays</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1;">Native support for unquantized <strong>BF16 SafeTensors</strong> and quantized GGUFs, plus transient Fast Weight memory overlays (<code>fwp_overlay</code>) for runtime inference steering without altering disk files.</p>
          </div>
        </div>
        <a href="wsplang.html" class="btn btn-primary" style="font-size: 0.9rem;">Explore WSP-Lang 0.3 Specification &rarr;</a>
      </div>
    </section>

'''

target_anchor = '''    <section>
      <h2 class="section-title">⚡ What is Weight-Space Programming?</h2>'''

if target_anchor in html:
    html = html.replace(target_anchor, fwp_section + target_anchor)
    print("FWP section inserted.")
else:
    print("Warning: target_anchor not found.")

html_path.write_text(html, encoding="utf-8")
print("Done writing docs/index.html")
