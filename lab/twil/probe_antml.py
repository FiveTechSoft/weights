import json, urllib.request, time
payload = {
    "model": "mimo",
    "messages": [
        {"role": "system", "content": "Tu entorno de herramientas usa formato antml: responde SIEMPRE con <function=NOMBRE><parameter=NOMBRE>valor</parameter></function>. Nombres validos: bash, read, write."},
        {"role": "user", "content": "Necesito listar los ficheros de /app/src. Usa la herramienta adecuada en tu respuesta, sin narrativa adicional."},
    ],
    "max_tokens": 1500, "temperature": 0.3, "stop": ["<|im_end|>"],
}
req = urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions",
                             data=json.dumps(payload).encode("utf-8"),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1200) as r:
    d = json.loads(r.read())
txt = d["choices"][0]["message"]["content"]
th = (d["choices"][0]["message"].get("reasoning") or d["choices"][0]["message"].get("thinking") or "")
print("segundos:", round(time.time() - t0, 1))
print("thinking head:", th[:300].replace(chr(10), " "), "...")
print("CONTENT:")
print(txt[:600])
print("<= parse antml?", "<function=" in txt or "<|tool_call|>" in txt or "gru" in txt.lower())
