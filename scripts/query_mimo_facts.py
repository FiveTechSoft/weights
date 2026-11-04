import urllib.request
import json

prompts = [
    "The chemical element with atomic number 79 is",
    "The deepest lake in the world is Lake"
]

for p in prompts:
    req = urllib.request.Request(
        "http://127.0.0.1:8081/v1/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "prompt": f"Complete this sentence with the exact entity: {p}",
            "max_tokens": 15,
            "temperature": 0.0
        }).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"MiMo-9B on '{p}': {repr(data['choices'][0]['text'].strip())}")
    except Exception as e:
        print("Error:", e)
