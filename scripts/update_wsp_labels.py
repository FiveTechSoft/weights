"""
update_wsp_labels.py
Updates wsp.py and engine/wsplang.py to English nomenclature for international standard.
"""

from pathlib import Path

def update_file(path, replacements):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for old, new in replacements:
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")

# 1. wsp.py
wsp_replacements = [
    ('"label": "Matemáticas y lógica"', '"label": "Math & Logic"'),
    ('"label": "Acciones de agente y terminal"', '"label": "Agent & Actions"'),
    ('"label": "Código y programación"', '"label": "Code & Software"'),
    ('"label": "Conceptos abstractos"', '"label": "Abstract Ideas"'),
    ('"label": "Mente y cognición"', '"label": "Mind & Cognition"'),
    ('"label": "Tiempo y secuencia"', '"label": "Time & Sequence"'),
    ('"label": "Emoción positiva"', '"label": "Positive Emotion"'),
    ('"label": "Emoción negativa"', '"label": "Negative Emotion"'),
    ('"label": "Mundo físico y materia"', '"label": "Physical World"'),
    ('"label": "Social y autoridad"', '"label": "Social & Power"'),
    ('ÁREA SEMÁNTICA DOMINANTE:', 'DOMINANT SEMANTIC AREA:'),
    ('ÁREAS MÁS IMPORTANTES (TOP', 'TOP SEMANTIC AREAS (TOP'),
    ('TODAS LAS ÁREAS SEMÁNTICAS (', 'ALL SEMANTIC AREAS ('),
    ('Semillas léxicas más afines:', 'Top affine seeds:'),
    ('Coordenadas: fila=', 'Coordinates: row='),
    ('Norma fila:', 'Row norm:'),
    ('Norma col:', 'Col norm:'),
    ('POKE completado en', 'POKE completed at'),
    ("Forma: {res['shape']}  Tipo: {res['dtype']}", "Shape: {res['shape']}  Type: {res['dtype']}"),
    ("Rol: {arch.get('role', '')}", "Role: {arch.get('role', '')}"),
]
update_file("wsp.py", wsp_replacements)

# 2. engine/wsplang.py
wsplang_replacements = [
    ('DIR_PL = ROOT / "retos" / "wsplang"', 'DIR_PL = ROOT / "challenges" / "wsplang"'),
    ('RESULTADOS DE EJECUCIÓN WSP-Lang:', 'WSP-Lang EXECUTION RESULTS:'),
    ('> Paso: {op}', '> Step: {op}'),
    ('ÁREA SEMÁNTICA DOMINANTE:', 'DOMINANT SEMANTIC AREA:'),
    ('ÁREAS MÁS IMPORTANTES (TOP', 'TOP SEMANTIC AREAS (TOP'),
    ('TODAS LAS ÁREAS SEMÁNTICAS (', 'ALL SEMANTIC AREAS ('),
    ('Semillas afines:', 'Affine seeds:'),
    ('PERFIL SEMÁNTICO DEL TENSOR:', 'TENSOR SEMANTIC PROFILE:'),
    ('Áreas más importantes:', 'Top areas:'),
    ('Todas las áreas:', 'All areas:'),
    ('TODAS LAS ÁREAS SEMÁNTICAS DEL MODELO (', 'ALL MODEL SEMANTIC AREAS ('),
    ('ÁREAS MÁS IMPORTANTES SELECCIONADAS (TOP', 'SELECTED TOP SEMANTIC AREAS (TOP'),
    ('"pasos": len(steps)', '"steps_count": len(steps)'),
    ('"validado": len(errors) == 0', '"validated": len(errors) == 0, "validado": len(errors) == 0'),
    ('"errores": errors', '"errors": errors, "errores": errors'),
]
update_file("engine/wsplang.py", wsplang_replacements)

# 3. engine/challenge.py
challenge_replacements = [
    ('RETOS = ROOT / "retos"', 'RETOS = ROOT / "challenges" if (ROOT / "challenges").exists() else ROOT / "retos"'),
    ('EXP = RETOS / "expedientes"', 'EXP = RETOS / "records" if (RETOS / "records").exists() else RETOS / "expedientes"'),
    ('SELLADOS = RETOS / "sellados"', 'SELLADOS = RETOS / "sealed" if (RETOS / "sealed").exists() else RETOS / "sellados"'),
]
update_file("engine/challenge.py", challenge_replacements)
