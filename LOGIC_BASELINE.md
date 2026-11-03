# Prueba lógica local de MiMo

Runner: `rig/logic_probe.py`. No requiere librerías adicionales ni descarga datasets.

## Qué mide
120 problemas sintéticos de lógica proposicional/Horn con hechos y reglas universales. El programa calcula el cierre lógico por encadenamiento hacia delante y pregunta si una conclusión se sigue necesariamente. Hay 60 respuestas YES y 60 NO. Cada caso cambia el orden, nombres y formulación superficial de hechos/reglas.

Esto mide una capacidad concreta: aplicar reglas y no afirmar conclusiones no demostradas. No es una medida completa de lógica formal, sentido común o razonamiento general.

## Ejecutar en el PC
Con el servidor llama.cpp del MiMo original activo en el puerto 8090:

```powershell
python -m py_compile rig\\logic_probe.py
python rig\\logic_probe.py --selftest
python rig\\logic_probe.py --endpoint http://127.0.0.1:8090/v1/chat/completions --model mimo --n 120 --temperature 0 --parallel 1 --out rig_runs/logic-mimo-orig.json
```

El selftest debe terminar en `selftest: OK`. La ejecución completa guarda respuestas individuales, errores, configuración y hash del conjunto en `rig_runs/logic-mimo-orig.json`.

## Lectura del resultado
- Exactitud global: porcentaje de YES/NO correctos.
- Comparar también aciertos YES frente a NO: un sesgo a contestar siempre una opción no debe parecer razonamiento.
- Errores de servidor se excluyen del denominador y se reportan aparte.
- No editar pesos a partir de esta prueba. Primero conservar el baseline y analizar los fallos concretos.
