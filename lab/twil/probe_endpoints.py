import json, os, urllib.request, urllib.error
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def posts_body(endpoint, body, timeout=1200):
    req = urllib.request.Request(endpoint,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode("utf-8", "ignore")[:500], "code": e.code}
    except Exception as e:
        return {"error": repr(e)[:300]}

ep_v1 = "http://127.0.0.1:11434/v1/chat/completions"
ep_apa = "http://127.0.0.1:11434/api/chat"

plain = {"model": "mimo",
         "messages": [{"role": "user", "content": "Responde solo: OK"}],
         "max_tokens": 8, "temperature": 0.3}
w_pen = dict(plain)
w_pen.update({"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0})
print("v1 plain   :", json.dumps(posts_body(ep_v1, plain))[:200])
print("v1 penalties:", json.dumps(posts_body(ep_v1, w_pen))[:200])
apa_plain = {"model": "mimo", "messages": plain["messages"], "stream": False}
print("api plain  :", json.dumps(posts_body(ep_apa, apa_plain))[:200])
