# MATH-500 baseline reproducible

Esta rama contiene un ajuste del runner `rig/math500.py` para medir el modelo original antes de cualquier edición WSP.

## 1. Actualizar el checkout (Windows)

Desde PowerShell, dentro de `C:\\weights`:

```powershell
git fetch origin
git switch wsp/math500-reproducible-baseline
git pull
```

## 2. Comprobar el evaluador

```powershell
python -m py_compile rig\\math500.py
python rig\\math500.py --selftest
```

El selftest debe terminar con `selftest: ... asserts OK`. Si falla, no ejecutar el benchmark.

## 3. Arrancar el GGUF original

Usa el GGUF Q6_K original de MiMo, no una edición. Sustituye la ruta de ejemplo por la ruta real del archivo:

```powershell
python wsp.py serve "C:\\RUTA\\AL\\MIMO-ORIGINAL-Q6_K.gguf" -p 8090 -a mimo -c 8192 -n 99
```

Espera el mensaje `UP port 8090`. El servidor debe responder en localhost.

## 4. Ejecutar MATH-500 completo

```powershell
python rig\\math500.py --endpoint http://127.0.0.1:8090/v1/chat/completions --model mimo --n 0 --seed 42 --temperature 0 --generation-seed 20261103 --max-tokens 8192 --parallel 1 --tag mimo_orig --sell
```

- `--n 0`: los 500 problemas.
- `--seed 42`: selección reproducible del conjunto.
- `--temperature 0`: baseline determinista.
- `--generation-seed`: semilla base; el runner suma el índice del problema.
- `--sell`: crea/reutiliza el lock de los 500 índices y hashes de respuestas.
- `--parallel 1`: evita saturar la RTX 3060 y mantiene ejecución serial.

Resultado: `rig_runs/r0-math-mimo_orig.json`. El JSON incluye configuración, índices, gold/pred por problema, respuestas correctas, errores de inferencia y finish_reason.

## 5. Qué necesito para continuar

Comparte el contenido de consola al terminar y el archivo `rig_runs/r0-math-mimo_orig.json` (puedes adjuntarlo aquí). No se modifica ningún peso en esta fase.

**Importante:** no ejecutes primero `--sell` con un subset pequeño. El sellado de este rig es deliberadamente de una única combinación seed/tamaño; para esta medición queremos reservarlo para los 500 problemas completos.
