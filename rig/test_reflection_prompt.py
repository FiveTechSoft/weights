import urllib.request
import json

prompts = [
    (
        "watermelon",
        "A 100 kg watermelon is 99% water. After sitting in the sun, it is 98% water. What is its new weight in kg?\nThink carefully, verify the non-water mass step-by-step, and state the final number."
    ),
    (
        "strawberry",
        "How many times does the letter 'r' appear in the word 'strawberry'?\nSpell out the word letter by letter, verify each position, and state the final count."
    ),
    (
        "sally",
        "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?\nAnalyze the family relations carefully, verify whether Sally herself is one of the sisters, and state the final number."
    )
]

for name, p in prompts:
    req = urllib.request.Request(
        "http://127.0.0.1:8081/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": p}],
            "temperature": 0.0,
            "max_tokens": 300
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    ans = json.load(res)["choices"][0]["message"]["content"].strip()
    print("=" * 40)
    print(f"PROMPT: {name}")
    print("=" * 40)
    print(ans)
    print()
