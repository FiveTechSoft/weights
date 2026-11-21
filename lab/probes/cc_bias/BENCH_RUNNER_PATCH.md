# Minimal patch to engine/bench_runner.py for per-request extra fields (not applied by this probe)

`run_arm.py` works without it by replacing `bench_runner.ask` in the module. If the coordinator prefers a proper hook,
this is the whole change (one keyword threaded through `ask` and `run`; existing callers unchanged):

```diff
-def ask(url, text, max_tokens, lora=None, extra=""):
+def ask(url, text, max_tokens, lora=None, extra="", extra_body=None):
     """lora: [(id, scale), ...] for adapters loaded with llama-server --lora ... --lora-init-without-apply.
-    extra: text appended after SUFFIX (prompt-control arm, lab/probes/cc/PLAN.md); "" sends exactly text + SUFFIX."""
+    extra: text appended after SUFFIX (prompt-control arm, lab/probes/cc/PLAN.md); "" sends exactly text + SUFFIX.
+    extra_body: dict merged into the request body (e.g. {"logit_bias": [[248069, 6.0]]}, lab/probes/cc_bias)."""
     body = {"messages": [{"role": "user", "content": text + SUFFIX + extra}], "temperature": 0, "seed": 0,
             "max_tokens": max_tokens}
     if lora:
         body["lora"] = [{"id": int(i), "scale": float(s)} for i, s in lora]
+    if extra_body:
+        body.update(extra_body)
```

```diff
 def run(name, endpoint, gsm_n=100, math_n=50, parallel=2, max_tokens=8192, out_dir=None, log=print, lora=None,
-        gsm_idx=None, math_idx=None, extra=""):
+        gsm_idx=None, math_idx=None, extra="", extra_body=None):
 ...
-        r = ask(url, text, max_tokens, lora=lora, extra=extra)
+        r = ask(url, text, max_tokens, lora=lora, extra=extra, extra_body=extra_body)
 ...
-    report = {"model": name, "endpoint": url, "max_tokens": max_tokens, "lora": lora, "extra": extra,
+    report = {"model": name, "endpoint": url, "max_tokens": max_tokens, "lora": lora, "extra": extra,
+              "extra_body": extra_body,
```

Test to add to tests/test_bench_runner.py if applied: `ask` with `extra_body={"logit_bias": [[1, 2.0]]}` puts that key in
the JSON body (monkeypatch `urllib.request.urlopen` to capture the request and return a minimal response).
