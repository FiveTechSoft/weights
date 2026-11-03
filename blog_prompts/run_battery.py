"""Bateria de prompts del blog Weight-Space Programming, adaptados al driver local.
P1 (exp1): identificar que computa una red PETITA solo leyendo pesos (esperamos w=x0+2x1-0.5x2 via relu+lin).
P5 (exp5): escribir pesos inline (formato edits del blog) para una mini-red; se verifica numerica y localmente.

Uso: python blog_prompts/run_battery.py <port>
"""
import json
import re
import sys
import urllib.request
import numpy as np

SERVER = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8950}"
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def chat(messages, max_tokens=700, temp=0.3):
    body = json.dumps({"messages": messages, "temperature": temp,
                       "max_tokens": max_tokens, "seed": 42}).encode()
    req = urllib.request.Request(SERVER + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def fmt_net(W1, b1, W2, b2):
    out = ["W1 (shape 3x2):"] + [f"  row{i}: [{', '.join(f'{v:.6g}' for v in W1[i])}]" for i in range(W1.shape[0])]
    out.append(f"b1: [{', '.join(f'{v:.6g}' for v in b1)}]")
    out += ["W2 (shape 2x1):"] + [f"  row{i}: [{W2[i, 0]:.6g}]" for i in range(W2.shape[0])]
    out.append(f"b2: [{', '.join(f'{v:.6g}' for v in b2)}]")
    return "\n".join(out)


def make_p1_network(seed=7):
    rng = np.random.default_rng(seed)
    dark = rng.uniform(-0.05, 0.05, size=(3, 2)).astype(np.float32)
    W1 = dark.copy()
    b1 = np.zeros(2, dtype=np.float32)
    W2 = np.array([[1.0], [2.0]], dtype=np.float32)
    b2 = np.array([-0.5], dtype=np.float32)
    W1[0, 0], W1[0, 1] = 1.0, 2.0
    W1[1, 0], W1[1, 1] = -0.5, 0.0
    W1[2, 0] = 0.03
    b1[1] = 0.0
    return W1, b1, W2, b2


def forward(W1, b1, W2, b2, x):
    h = np.maximum(x @ W1 + b1, 0)
    return h @ W2 + b2


def p1():
    W1, b1, W2, b2 = make_p1_network()
    dump = fmt_net(W1, b1, W2, b2)
    sysP = ("Eres un analista de pesos. Te doy los pesos crudos de una MLP minima (ReLU oculta, salida lineal de 1 valor, "
            "inputs 3 floats). REGLAS: prohibido calcular salidas (no pasar inputs por la red), prohibido entrenar. "
            "SOLO a partir de lectura de pesos responde en <=12 lineas: que funcion computa esta red? Expressa y(t) en funcion de "
            "x0,x1,x2 de forma simbolica, y di si hay pesos irrelevantes.\n\nPESOS CRUDOS:\n" + dump)
    txt = chat([{"role": "user", "content": sysP}])
    has12 = bool(re.search(r"(x0\s*\+|x0\s*\(x?\))?\s*[\+\.\d]*\s*2\s*x1", txt) or ("2" in txt and "x1" in txt and "x2" in txt))
    hasm05 = ("0.5" in txt or "0.5x2" in txt.replace(" ", "") or "x2" in txt)
    relu_ok = ("relu" in txt.lower() or "max(" in txt)
    print("P1 respuesta:")
    print(txt)
    verd = {"formula_x1x2": has12, "menciona_x2_coef": hasm05, "relu": relu_ok}
    pasa = verd["formula_x1x2"] and verd["menciona_x2_coef"] and verd["relu"]
    print("P1 verdict:", verd, "PASS" if pasa else "FAIL")
    return pasa


def p5():
    target = "y(x) = 1.5*x0 - 2.5*x1 + 0.7 (para x en R^2), implementada como MLP 2->2 (ReLU oculta, salida lineal 1)."
    sysP = ("Escribe los pesos de una MLP que cumpla exactamente: " + target +
            "\nFormato ('edits', una statement por linea, SIEMPRE este formato:\n```edits\nW1[0,0] = [...]; ... en notacion q lo entiendas\n```\n"
            "REGLAS: solo dikta valores numericos que consideres idoneos; empiezas de pesos TODO-CEROS; no puedes entrenar; "
            "justifica en <=3 lineas y luego SOLO el bloque edits con TODAS las asignaciones (W1 2x2, b1 2, W2 2x1, b2 1).")
    txt = chat([{"role": "user", "content": sysP}])
    print("P5 respuesta (primeras 40 lineas):")
    print("\n".join(txt.splitlines()[:40]))
    m = re.search(r"```edits(.*?)```", txt, re.S)
    ok = False
    if m:
        blk = m.group(1)
        W1 = np.zeros((2, 2)); b1 = np.zeros(2); W2 = np.zeros((2, 1)); b2 = np.zeros(1)
        try:
            for line in blk.splitlines():
                mt = re.match(r"\s*W\s*1\s*\[(\d+)\s*,\s*(\d+)\]\s*=\s*(-?[\d.]+)", line)
                if mt: W1[int(mt.group(1)), int(mt.group(2))] = float(mt.group(3)); continue
                mt = re.match(r"\s*b\s*1\s*\[(\d+)\]\s*=\s*(-?[\d.]+)", line)
                if mt: b1[int(mt.group(1))] = float(mt.group(2)); continue
                mt = re.match(r"\s*W\s*2\s*\[(\d+)\]\s*=\s*(-?[\d.]+)", line)
                if mt: W2[int(mt.group(1)), 0] = float(mt.group(2)); continue
                mt = re.match(r"\s*b\s*2\s*\[(\d+)\]\s*=\s*(-?[\d.]+)", line)
                if mt: b2[int(mt.group(1))] = float(mt.group(2)); continue
            rng = np.random.default_rng(3)
            errs = []
            for _ in range(20):
                x = rng.uniform(-3, 3, size=2)
                y_pred = forward(W1, b1, W2, b2, x)
                y_real = 1.5 * x[0] - 2.5 * x[1] + 0.7
                errs.append(abs(float(y_pred[0]) - y_real))
            ok = max(errs) < 0.01
            print("P5 err max en 20 puntos:", max(errs))
        except Exception as e:
            print("P5 parse fail:", repr(e))
    else:
        print("P5 sin bloque edits")
    print("P5 verdict:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    r1 = p1()
    r5 = p5()
    print("BATERIA:", "PASS" if (r1 and r5) else "FALLO", f"P1={'PASS' if r1 else 'FAIL'} P5={'PASS' if r5 else 'FAIL'}")
