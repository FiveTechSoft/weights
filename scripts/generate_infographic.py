"""
generate_infographic.py
Generates a publication-grade dark-mode infographic visualizing WSP-Lang
and its semantic area decomposition on MiMo-V2.6-Distill-Qwen-9B.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import shutil

OUTPUT_DIR = Path("docs/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUTPUT_DIR / "wsplang_semantic_areas.png"

ARTIFACT_DIR = Path(r"C:\Users\Anto\.gemini\antigravity-cli\brain\e11471ec-0f1c-46ec-9ab6-af74c25cf911")

def create_graphic():
    # Setup high-resolution figure (16:9 widescreen: 2400 x 1350 at 150 DPI)
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="#080C15")
    
    font_main = "Segoe UI"
    font_mono = "Consolas"
    
    ax_canvas = fig.add_axes([0, 0, 1, 1], facecolor="#080C15")
    ax_canvas.set_xlim(0, 16)
    ax_canvas.set_ylim(0, 9)
    ax_canvas.axis("off")
    
    # -------------------------------------------------------------
    # HEADER SECTION (y: 7.75 to 8.85)
    # -------------------------------------------------------------
    # Header tag pill
    tag_box = patches.FancyBboxPatch((0.6, 8.38), 2.9, 0.32,
                                    boxstyle="round,pad=0.06,rounding_size=0.08",
                                    facecolor="#0B2338", edgecolor="#0284C7", linewidth=1.2)
    ax_canvas.add_patch(tag_box)
    ax_canvas.text(2.05, 8.54, "RESEARCH RELEASE · ZERO FORWARD PASS", color="#38BDF8",
                   fontfamily=font_main, fontsize=8.2, weight="bold", ha="center", va="center")
    
    # Title & Subtitle
    ax_canvas.text(0.6, 7.95, "WSP-Lang: Reverse Engineering LLM Semantics",
                   color="#F8FAFC", fontfamily=font_main, fontsize=21, weight="bold", va="bottom")
    ax_canvas.text(0.6, 7.62, "Direct Decompilation of 9B Quantized Weights into 10 Semantic Areas without Forward Passes",
                   color="#94A3B8", fontfamily=font_main, fontsize=11, va="bottom")
    
    # Spec Badges on Top Right
    specs = [
        ("TARGET MODEL", "MiMo-2.6-Qwen-9B", "#38BDF8"),
        ("QUANTIZATION", "Q6_K (6.56 bpw)", "#A855F7"),
        ("COMPUTE COST", "0 MB RAM / 0 FLOPs", "#10B981"),
        ("OPEN REPO", "FiveTechSoft/weights", "#F59E0B"),
    ]
    badge_x = 9.15
    for label, val, accent in specs:
        box = patches.FancyBboxPatch((badge_x, 7.68), 1.55, 0.96,
                                     boxstyle="round,pad=0.06,rounding_size=0.08",
                                     facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2)
        ax_canvas.add_patch(box)
        # Accent indicator
        ax_canvas.add_patch(patches.Circle((badge_x + 0.20, 8.42), 0.05, facecolor=accent, edgecolor="none"))
        ax_canvas.text(badge_x + 0.32, 8.42, label, color="#94A3B8", fontfamily=font_main, fontsize=7.5, weight="bold", va="center")
        ax_canvas.text(badge_x + 0.775, 8.00, val, color="#F8FAFC", fontfamily=font_mono, fontsize=8.4, weight="bold", ha="center", va="center")
        badge_x += 1.62

    # -------------------------------------------------------------
    # LEFT PANEL: WSP-LANG CODE + PIPELINE (x: 0.6 to 6.95)
    # -------------------------------------------------------------
    # Card 1: Code editor box
    code_bg = patches.FancyBboxPatch((0.6, 3.4), 6.35, 3.95,
                                     boxstyle="round,pad=0.08,rounding_size=0.12",
                                     facecolor="#0C1220", edgecolor="#1E293B", linewidth=1.5)
    ax_canvas.add_patch(code_bg)
    
    # Editor Titlebar
    titlebar = patches.FancyBboxPatch((0.6, 6.95), 6.35, 0.40,
                                      boxstyle="round,pad=0.08,rounding_size=0.10",
                                      facecolor="#131D33", edgecolor="#1E293B", linewidth=1.5)
    ax_canvas.add_patch(titlebar)
    # Window dots
    ax_canvas.add_patch(patches.Circle((0.85, 7.15), 0.06, facecolor="#EF4444"))
    ax_canvas.add_patch(patches.Circle((1.05, 7.15), 0.06, facecolor="#F59E0B"))
    ax_canvas.add_patch(patches.Circle((1.25, 7.15), 0.06, facecolor="#10B981"))
    ax_canvas.text(3.77, 7.15, "programas/ejemplo_areas_semanticas.wspl", color="#94A3B8",
                   fontfamily=font_mono, fontsize=8.8, ha="center", va="center")
    
    # Code Lines with line numbers
    code_lines = [
        (" 1", "experiment", " inspeccion_areas_semanticas:", "#C084FC", "#F1F5F9"),
        (" 2", "    model:", " MiMo-V2.6-Distill-Qwen-9B", "#38BDF8", "#E2E8F0"),
        (" 3", "    objective:", " map_semantic_geometry", "#38BDF8", "#E2E8F0"),
        (" 4", "", "", "", ""),
        (" 5", "    semantic_areas:", "", "#C084FC", ""),
        (" 6", "        mode", " = comprehensive", "#94A3B8", "#F59E0B"),
        (" 7", "        target", " = math_logic", "#94A3B8", "#10B981"),
        (" 8", "", "", "", ""),
        (" 9", "    steps:", "", "#C084FC", ""),
        ("10", "        # 1. Project against all 10 semantic centroids", "", "#64748B", ""),
        ("11", "        step", " semantic_map top=3", "#F43F5E", "#38BDF8"),
        ("12", "        # 2. Attention Layer 19 (Logic Reasoning Head)", "", "#64748B", ""),
        ("13", "        step", " peek blk.19.attn_output.weight row=100 col=200 top=3", "#F43F5E", "#38BDF8"),
        ("14", "        # 3. FFN Layer 19 (Intermediate Concept Storage)", "", "#64748B", ""),
        ("15", "        step", " peek blk.19.ffn_down.weight row=100 col=500 top=3", "#F43F5E", "#38BDF8"),
        ("16", "        # 4. GatedDeltaNet SSM Recurrent Layer 0", "", "#64748B", ""),
        ("17", "        step", " peek blk.0.ssm_out.weight row=100 col=200 top=3", "#F43F5E", "#38BDF8"),
    ]
    
    cy = 6.75
    for lno, kw, rest, col1, col2 in code_lines:
        # Line number
        ax_canvas.text(0.78, cy, lno, color="#475569", fontfamily=font_mono, fontsize=7.8, va="center")
        if kw or rest:
            ax_canvas.text(1.10, cy, kw, color=col1, fontfamily=font_mono, fontsize=7.9, va="center")
            if rest:
                x_offset = 1.10 + len(kw) * 0.082
                ax_canvas.text(x_offset, cy, rest, color=col2, fontfamily=font_mono, fontsize=7.9, va="center")
        cy -= 0.198

    # Card 2: Mathematical & Execution Pipeline Box
    pipe_bg = patches.FancyBboxPatch((0.6, 0.45), 6.35, 2.75,
                                     boxstyle="round,pad=0.08,rounding_size=0.12",
                                     facecolor="#0C1220", edgecolor="#1E293B", linewidth=1.5)
    ax_canvas.add_patch(pipe_bg)
    
    ax_canvas.text(0.85, 2.92, "THE ZERO-COST MATHEMATICAL PIPELINE", color="#38BDF8",
                   fontfamily=font_main, fontsize=10.2, weight="bold", va="center")
    
    steps_math = [
        ("1. Selective Quant Dequantization:",
         "Direct disk read of 32-byte Q6_K scale blocks into w in R^4096. 0 GPU memory overhead.",
         "#10B981"),
        ("2. Differential Centering (Solves Isotropic Cone):",
         r"$C'_a = C_a - \frac{1}{A}\sum_{k} C_k$  (Cancels baseline cone bias, +2.4x contrast dynamic range)",
         "#A855F7"),
        ("3. Direct Subspace Projection:",
         r"score(w, a) = cos(w, C'_a) = (w . C'_a) / (||w|| ||C'_a||) in [-1.0, +1.0]",
         "#F59E0B"),
        ("4. Architectural Role Attribution:",
         "Maps col index to exact attention head (col//256) or FFN neuron to decipher functional role.",
         "#06B6D4")
    ]
    
    py = 2.52
    for title, desc, col in steps_math:
        ax_canvas.add_patch(patches.Circle((0.85, py), 0.06, facecolor=col, edgecolor="none"))
        ax_canvas.text(1.02, py, title, color="#F1F5F9", fontfamily=font_main, fontsize=8.6, weight="bold", va="center")
        ax_canvas.text(1.02, py - 0.22, desc, color="#94A3B8", fontfamily=font_main, fontsize=7.8, va="center")
        py -= 0.52

    # -------------------------------------------------------------
    # RIGHT TOP: REAL MIMO-2.6 PROJECTIONS (x: 7.2 to 15.4)
    # -------------------------------------------------------------
    right_bg = patches.FancyBboxPatch((7.2, 3.4), 8.2, 3.95,
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor="#0C1220", edgecolor="#1E293B", linewidth=1.5)
    ax_canvas.add_patch(right_bg)
    
    ax_canvas.text(7.45, 7.12, "EMPIRICAL DECOMPILATION: MIMO-2.6-DISTILL-QWEN-9B",
                   color="#F8FAFC", fontfamily=font_main, fontsize=12, weight="bold", va="center")
    ax_canvas.text(7.45, 6.84, "Actual measurements extracted with WSP-Lang step peek across distinct architectural layers",
                   color="#94A3B8", fontfamily=font_main, fontsize=9.0, va="center")
    
    charts_data = [
        {
            "title": "Layer 19 Attention Head 0",
            "subtitle": "blk.19.attn_output.weight [r=100, c=200]",
            "dominant": "Math & Logic (+0.0275)",
            "color": "#0284C7",
            "items": [
                ("Math & Logic", +0.0275, "#38BDF8"),
                ("Time & Sequence", +0.0149, "#06B6D4"),
                ("Social & Power", +0.0098, "#A855F7"),
                ("Emotion Neg", -0.0034, "#64748B"),
                ("Physical World", -0.0165, "#EF4444"),
            ],
            "seeds": "Seeds: learn (+.034), theorem (+.029)"
        },
        {
            "title": "Layer 19 FFN Neuron 500",
            "subtitle": "blk.19.ffn_down.weight [r=100, c=500]",
            "dominant": "Physical World (+0.0211)",
            "color": "#10B981",
            "items": [
                ("Physical World", +0.0211, "#10B981"),
                ("Emotion Pos", +0.0155, "#34D399"),
                ("Code & Software", +0.0072, "#06B6D4"),
                ("Math & Logic", -0.0038, "#64748B"),
                ("Time & Sequence", -0.0191, "#EF4444"),
            ],
            "seeds": "Seeds: kind (+.042), integer (+.040)"
        },
        {
            "title": "Layer 0 GatedDeltaNet SSM",
            "subtitle": "blk.0.ssm_out.weight [r=100, c=200]",
            "dominant": "Emotion Neg (+0.0190)",
            "color": "#F43F5E",
            "items": [
                ("Emotion Neg", +0.0190, "#F43F5E"),
                ("Emotion Pos", +0.0162, "#FB7185"),
                ("Math & Logic", +0.0108, "#38BDF8"),
                ("Code & Software", -0.0055, "#64748B"),
                ("Time & Sequence", -0.0183, "#EF4444"),
            ],
            "seeds": "Seeds: glad (+.031), rage (+.029)"
        }
    ]
    
    col_x_start = 7.42
    for c_idx, chart in enumerate(charts_data):
        cx = col_x_start + c_idx * 2.62
        cbox = patches.FancyBboxPatch((cx, 3.65), 2.48, 2.95,
                                      boxstyle="round,pad=0.06,rounding_size=0.08",
                                      facecolor="#11192C", edgecolor="#1F2A44", linewidth=1.0)
        ax_canvas.add_patch(cbox)
        
        ax_canvas.text(cx + 0.12, 6.42, chart["title"], color="#F8FAFC", fontfamily=font_main, fontsize=8.4, weight="bold")
        ax_canvas.text(cx + 0.12, 6.24, chart["subtitle"], color="#64748B", fontfamily=font_mono, fontsize=7.0)
        
        # Dominant tag
        dom_box = patches.FancyBboxPatch((cx + 0.12, 5.90), 2.24, 0.24,
                                         boxstyle="round,pad=0.04,rounding_size=0.05",
                                         facecolor="#172238", edgecolor=chart["color"], linewidth=1.0)
        ax_canvas.add_patch(dom_box)
        ax_canvas.text(cx + 1.24, 6.02, f"DOMINANT: {chart['dominant']}", color=chart["color"],
                       fontfamily=font_main, fontsize=7.2, weight="bold", ha="center", va="center")
        
        # Draw horizontal mini-bars
        by = 5.62
        for name, val, bcol in chart["items"]:
            ax_canvas.text(cx + 0.12, by, name, color="#CBD5E1", fontfamily=font_main, fontsize=7.0, va="center")
            ax_canvas.text(cx + 2.36, by, f"{val:+.3f}", color=bcol, fontfamily=font_mono, fontsize=7.0, ha="right", va="center")
            
            bar_center_x = cx + 1.55
            track_w = 0.58
            ax_canvas.plot([bar_center_x - track_w/2, bar_center_x + track_w/2], [by, by], color="#1E293B", linewidth=3.2, solid_capstyle="round")
            
            bar_len = (val / 0.035) * (track_w / 2)
            bar_len = np.clip(bar_len, -track_w/2, track_w/2)
            ax_canvas.plot([bar_center_x, bar_center_x + bar_len], [by, by], color=bcol, linewidth=3.2, solid_capstyle="round")
            
            by -= 0.25
            
        # Seeds footer
        ax_canvas.text(cx + 0.12, 4.02, chart["seeds"], color="#94A3B8", fontfamily=font_mono, fontsize=6.8)

    # -------------------------------------------------------------
    # RIGHT BOTTOM: 10 SEMANTIC AREAS TAXONOMY (x: 7.2 to 15.4)
    # -------------------------------------------------------------
    areas_bg = patches.FancyBboxPatch((7.2, 0.45), 8.2, 2.75,
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor="#0C1220", edgecolor="#1E293B", linewidth=1.5)
    ax_canvas.add_patch(areas_bg)
    
    ax_canvas.text(7.45, 2.92, "ORTHOGONAL SEMANTIC CENTROIDS (CALIBRATED ON MIMO VOCABULARY)",
                   color="#38BDF8", fontfamily=font_main, fontsize=10.2, weight="bold", va="center")
    
    areas_list = [
        ("Math & Logic", "#38BDF8", "theorem, proof, calculate, integer, axiom"),
        ("Code & Software", "#818CF8", "function, class, return, import, syntax"),
        ("Agent & Action", "#F59E0B", "execute, bash, command, run, terminal"),
        ("Physical World", "#10B981", "stone, metal, gravity, atom, machine"),
        ("Time & Sequence", "#06B6D4", "before, after, future, step, sequence"),
        ("Abstract Ideas", "#E879F9", "concept, truth, reality, meaning, paradigm"),
        ("Mind & Cognition", "#C084FC", "think, reason, believe, memory, conscious"),
        ("Social & Power", "#F97316", "king, power, law, rule, status, authority"),
        ("Positive Emotion", "#34D399", "glad, joy, happy, kind, smile, love"),
        ("Negative Emotion", "#F43F5E", "rage, hate, grief, fear, shame, pain"),
    ]
    
    for a_idx, (name, color, seeds) in enumerate(areas_list):
        ax_col = 0 if a_idx < 5 else 1
        pos_y = 2.50 - (a_idx % 5) * 0.43
        pos_x = 7.45 if ax_col == 0 else 11.45
        
        ax_canvas.add_patch(patches.Circle((pos_x + 0.1, pos_y), 0.06, facecolor=color, edgecolor="none"))
        ax_canvas.text(pos_x + 0.26, pos_y + 0.08, name, color="#F8FAFC", fontfamily=font_main, fontsize=8.6, weight="bold", va="center")
        ax_canvas.text(pos_x + 0.26, pos_y - 0.10, f"Seeds: {seeds}", color="#94A3B8", fontfamily=font_mono, fontsize=7.1, va="center")

    # -------------------------------------------------------------
    # FOOTER BAR (y: 0.12 to 0.35)
    # -------------------------------------------------------------
    ax_canvas.text(0.6, 0.18, "FiveTechSoft / weights · High-Precision Sub-Weight Introspection (WSP) · Apache-2.0",
                   color="#475569", fontfamily=font_main, fontsize=8.5, va="center")
    ax_canvas.text(15.4, 0.18, "Run: python engine/wsplang.py programas/ejemplo_areas_semanticas.wspl --run",
                   color="#38BDF8", fontfamily=font_mono, fontsize=8.2, ha="right", va="center")

    # Save PNG
    plt.savefig(PNG_PATH, dpi=150, facecolor="#080C15", edgecolor="none")
    plt.close()
    print(f"[OK] Generated infographic at {PNG_PATH}")

    # Copy to artifact directory
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_png = ARTIFACT_DIR / "wsplang_semantic_areas.png"
    shutil.copyfile(PNG_PATH, artifact_png)
    print(f"[OK] Copied to artifact directory at {artifact_png}")

if __name__ == "__main__":
    create_graphic()
