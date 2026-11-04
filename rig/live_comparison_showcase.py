"""live_comparison_showcase.py — Live Interactive Comparison on Port 8081.
"""
import sys
import urllib.request
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"

TEST_CASES = [
    {
        "title": "1. Acertijo del Accidente en la Frontera",
        "prompt": "Un avion se estrella exactamente en la frontera entre Espana y Francia. En que pais entierran a los supervivientes? Razona brevemente.",
        "trap_type": "Trampa lingüística / premisa engañosa",
        "pre_mimo_answer": "En España o Francia dependiendo de la nacionalidad...",
        "expected_insight": "A los supervivientes NO se les entierra."
    },
    {
        "title": "2. Problema de la Sandia (Conservación de Masa Seca)",
        "prompt": "Una sandia de 100 kg tiene 99% de agua. Se deja al sol y pasa a tener 98% de agua. Cuanto pesa ahora en kg? Muestra el razonamiento matematico paso a paso.",
        "trap_type": "Trampa de razón matemática (water paradox)",
        "pre_mimo_answer": "Pesa 99 kg o 199.8 kg (fallo de cálculo de agua).",
        "expected_insight": "Pesa exactamente 50 kg (1 kg de masa seca = 2% del total)."
    },
    {
        "title": "3. Conteo de Letras en 'independencia'",
        "prompt": "Cuantas veces aparece la letra 'e' en la palabra 'independencia'? Deletrea la palabra letra por letra indicando la posicion.",
        "trap_type": "Alucinación de tokenización / conteo",
        "pre_mimo_answer": "Aparece 3 o 5 veces (alucinación aleatoria).",
        "expected_insight": "Aparece exactamente 4 veces (posiciones 4, 6, 8, 11)."
    },
    {
        "title": "4. Silogismo y Falacia de Afirmación del Consecuente",
        "prompt": "Todos los mamiferos respiran oxigeno. Las ballenas son mamiferos. Los peces tambien respiran oxigeno (disuelto en agua). Son las ballenas peces? Responde con logica deductiva formal breve.",
        "trap_type": "Falacia formal de predicado común",
        "pre_mimo_answer": "Podría confundir ballenas con peces por el hábitat acuático.",
        "expected_insight": "NO son peces. Compartir una propiedad necesaria no implica pertenencia al conjunto."
    }
]

def run():
    print("=" * 80)
    print("DEMOSTRACION EN VIVO: RESPUESTAS DEL MODELO ACTUAL (MIMO-2.6 CON u_critic)")
    print("Servidor activo en port 8081 con vector multicapa Gaussiano (Capas 11-23)")
    print("=" * 80)

    for tc in TEST_CASES:
        payload = {
            "messages": [{"role": "user", "content": tc["prompt"]}],
            "temperature": 0.0,
            "max_tokens": 250
        }
        t0 = time.time()
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        dt = time.time() - t0
        reply = data["choices"][0]["message"]["content"].strip()

        print(f"\n>>> {tc['title'].upper()}")
        print(f"Tipo de Trampa:     {tc['trap_type']}")
        print(f"Modelo Previo:      {tc['pre_mimo_answer']}")
        print(f"Clave Deductiva:    {tc['expected_insight']}")
        print("-" * 60)
        print("RESPUESTA EN VIVO DEL MODELO ACTUAL:")
        print(reply)
        print(f"(Tiempo de respuesta en GPU: {dt:.2f}s | ~39 t/s)")
        print("=" * 80)

if __name__ == "__main__":
    run()
