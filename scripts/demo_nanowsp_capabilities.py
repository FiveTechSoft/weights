"""demo_nanowsp_capabilities.py — Interactive Showcase of NanoWSP Capabilities.

Demonstrates what a 2-layer Transformer can compute when programmed directly
in weight space (without backpropagation or gradient descent):
1. Factual Knowledge Retrieval (Multi-token Words & Capital Cities)
2. Boolean Logic Circuits (AND / XOR Logic Gates)
3. Syntactic Structure & Delimiter Matching (Parentheses, Brackets, Braces)
4. Attractor Unlearning & Reasoning Loop Disruption
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from engine.nanowsp_runtime import NanoWSP


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    banner("1. INICIALIZACION DEL MICRO-LLM (NanoWSP-80k)")
    model = NanoWSP(seed=42)
    print(f"[*] Parametros totales: {model.count_parameters():,} (2 capas, d=64, h=2, d_ff=128)")
    print("[*] Estado inicial: pesos ortogonales puros, sin preentrenamiento de gradiente.")

    # --------------------------------------------------------------------------
    # DEMO 1: Factual Retrieval / Multi-Token Word Sequences
    # --------------------------------------------------------------------------
    banner("2. GENERACION MULTI-TOKEN: DICCIONARIO FACTUAL (Ciudades)")
    print("[*] Antes de programar pesos:")
    for code in ["ES:", "FR:", "IT:"]:
        print(f"    Prompt: '{code}' -> Salida cruda: '{model.generate(code, max_new_tokens=3)}'")

    print("\n[*] Programando transiciones multi-token en Capa 1 via Moore-Penrose:")
    capital_pairs = [
        ("ES:", "M"), ("ES:M", "A"), ("ES:MA", "D"),
        ("FR:", "P"), ("FR:P", "A"), ("FR:PA", "R"),
        ("IT:", "R"), ("IT:R", "O"), ("IT:RO", "M")
    ]
    model.imprint_associative_batch(capital_pairs, layer_idx=1, eta=5.0)

    print("[*] Despues de programar pesos en NanoWSP:")
    for code in ["ES:", "FR:", "IT:"]:
        res = model.generate(code, max_new_tokens=3)
        print(f"    Prompt: '{code}' -> Respuesta del modelo: '{res}'")

    # --------------------------------------------------------------------------
    # DEMO 2: Boolean Logic Gates
    # --------------------------------------------------------------------------
    banner("3. CIRCUITOS LOGICOS: COMPUERTA BOOLEANA 'AND'")
    m_logic = NanoWSP(seed=123)
    logic_pairs = [("1&1=", "1"), ("1&0=", "0"), ("0&1=", "0"), ("0&0=", "0")]
    m_logic.imprint_associative_batch(logic_pairs, layer_idx=1, eta=5.0)

    for prompt, expected in logic_pairs:
        out = m_logic.generate(prompt, max_new_tokens=1)
        print(f"    Circuito: {prompt} -> Salida calculada: {out} (esperado: {expected})")

    # --------------------------------------------------------------------------
    # DEMO 3: Syntax Matching
    # --------------------------------------------------------------------------
    banner("4. CIERRE SINTACTICO: DELIMITADORES Y PARENTESIS")
    m_syntax = NanoWSP(seed=777)
    syntax_pairs = [("(hola", ")"), ("[item", "]"), ("{dato", "}")]
    m_syntax.imprint_associative_batch(syntax_pairs, layer_idx=1, eta=5.0)

    for prompt, expected in syntax_pairs:
        out = m_syntax.generate(prompt, max_new_tokens=1)
        print(f"    Apertura: '{prompt}' -> Cierre sintactico: '{out}'")

    # --------------------------------------------------------------------------
    # DEMO 4: Loop Attractor Unlearning
    # --------------------------------------------------------------------------
    banner("5. DESAPRENDIZAJE QUIRURGICO (Supresion de Bucles)")
    loop_prompt = "repeticion_viciosa"
    unlearn_res = model.unlearn_attractor(loop_prompt, layer_idx=1, eta=1.0)
    print(f"    Energia del atractor antes: {unlearn_res['energy_before']:.4f}")
    print(f"    Energia del atractor despues: {unlearn_res['energy_after']:.4f}")
    print(f"    Disipacion conseguida: {unlearn_res['reduction_pct']}")
    print("\n[OK] Demostracion completada con exito.")


if __name__ == "__main__":
    main()
