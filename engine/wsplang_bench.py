"""engine/wsplang_bench.py - the standard benchmark section of WSP-Lang (docs/wsplang_0.8_bench_addendum.md,
lab/probes/wsplang_bench/PLAN.md). One path to check and compare any model, the ones WSP-Lang produced included.

A named suite shipped with the language (engine/bench_suites/*.json, digest-verified) fixes the items, the decoding,
the model class and the preservation battery. `bench_run` executes it: for the LLM classes against a running
llama-server through engine/bench_runner.py (GSM8K / MATH-500, two graders) and engine/preserve_bench.py (HumanEval
164, 40 local instruction checks, garbage rates); for the `nanowsp` class in-process on the CPU over the local WSPX
tasks with the runtime's own exact-match recall. Every run JSON carries the served model's identity (path + sha256,
extras such as control vectors or LoRAs, llama-server binary sha, suite sha, decoding) or is refused; an optional
provenance (a wsp.py GGUF manifest, a compile_control_vector report, an edit_column manifest or a commit_manifest
bundle) is verified against the served sha and stored, so a model WSP-Lang built is "base@sha + patches", not a file
name. `bench_compare` reads two (or three) run JSONs item by item; `bench_table` renders the registry.

Verdict rule (bench_compare, fixed 2026-11-18), for "edit vs its base" (the arm's provenance base sha equals the base
run's model sha, or the two runs share the served model and differ only in extras):
  "improves" only if (a) on at least one benchmark the arm beats the base with exact two-sided binomial p < 0.05 on
  BOTH graders, (b) the placebo arm, when given, does not gain significantly on any benchmark / grader (a placebo that
  harms does not block: it says the direction matters), (c) when a noise band is given (two runs of the same
  configuration), |arm - base| >= 2 x band on that benchmark, (d) the preservation battery, when both runs carry it,
  does not fail (preserve_bench.compare's `preserves`).
  "harms": the symmetric statement with the arm below the base.  Otherwise "same / not decidable", with the reasons.
For "two models" (no provenance link) only the paired statistics and the band are reported; no verdict word.
This is not the sealed protocol (lab/probes/sealed/README.md): a sealed suite is single use and its result is read
whatever it says; bench_compare only counts.
"""
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))

from engine import bench_runner  # noqa: E402
from engine import preserve_bench  # noqa: E402
from engine import sealed_set  # noqa: E402

VERSION = "0.8"
SUITES = ROOT / "engine" / "bench_suites"
REGISTRY = ROOT / "lab" / "runs" / "bench" / "index.json"
TABLE = ROOT / "docs" / "wsplang_benchmarks.md"
DEFAULT_SUITE = "dev-2048"
DEFAULT_BINARY = Path("C:/Users/Anto/llamacpp/bin/llama-server.exe")
DECODING_KEYS = ("max_tokens", "parallel", "temperature", "seed", "template", "prompt_suffix")
MODEL_CLASSES = ("llm-chat-think", "llm-chat", "nanowsp", "llm-logits")
WSPX_TASKS = ("recall", "parity8", "mod13", "fsm4", "noecho")


class SealedSpent(RuntimeError):
    pass


class IdentityError(RuntimeError):
    """The served model cannot be identified and unknown_model=true was not given."""


class ProvenanceError(RuntimeError):
    """The provenance manifest does not describe the served model."""


class ClassMismatch(RuntimeError):
    """The suite's model_class does not match what is being benchmarked."""


# ----------------------------------------------------------------------------- hashing helpers

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- suites

def _decoding(max_tokens, parallel=2):
    return {"max_tokens": int(max_tokens), "parallel": int(parallel), "temperature": 0, "seed": 0,
            "template": "model chat template (thinking as the template does it)", "prompt_suffix": bench_runner.SUFFIX}


def suite_digest(suite):
    """sha256 over the canonical items (or the sealed file's own sha), the decoding, graders, preservation, model class
    and think flag."""
    core = {"name": suite["name"], "model_class": suite.get("model_class", "llm-chat-think"), "think": suite.get("think", "auto"),
            "items": suite.get("items") if not suite.get("sealed_file") else {"sealed_file": suite["sealed_file"],
                                                                                 "sealed_sha256": suite["sealed_sha256"]},
            "decoding": {k: suite["decoding"].get(k) for k in DECODING_KEYS},
            "graders": suite["graders"], "preservation": suite["preservation"]}
    if suite.get("components") is not None:                 # llm-logits suites: files, shas, chunk / task counts
        core["components"] = suite["components"]
        core["gpu_args"] = suite.get("gpu_args")
        core["thresholds"] = suite.get("thresholds")
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def make_suite(name, gsm_idx, math_idx, max_tokens, parallel=2, preservation=True, sealed_file=None, sealed_sha256=None,
               model_class="llm-chat-think", think="auto"):
    if model_class not in MODEL_CLASSES:
        raise ValueError(f"model_class must be one of {MODEL_CLASSES}")
    pres = ({"humaneval": 164, "compliance": 40, "garbage": True, "max_tokens": int(max_tokens)} if preservation is True
            else preservation if isinstance(preservation, dict) else False)
    s = {"wsplang_bench_version": VERSION, "name": name, "model_class": model_class, "think": think,
         "items": {"gsm8k": [int(i) for i in gsm_idx], "math500": [int(i) for i in math_idx]},
         "decoding": _decoding(max_tokens, parallel),
         "graders": ["g1: rig/math500.equivalent on the last \\boxed{}",
                     "g2: GSM8K last number in the last box as float; MATH-500 simple_norm string compare"],
         "preservation": pres}
    if sealed_file:
        s["sealed_file"] = str(sealed_file)
        s["sealed_sha256"] = sealed_sha256
        s["single_use"] = True
    s["sha256"] = suite_digest(s)
    return s


def make_nanowsp_suite(name="nanowsp-wspx", tasks=WSPX_TASKS):
    """The NanoWSP class: the local WSPX tasks (lab/data/wspx_<task>.json: train / holdout prompt -> single-character
    target), exact-match recall as engine.nanowsp_runtime.NanoWSP.evaluate_recall computes it (generate one character,
    compare). Items are identified by the task file's sha256 so the suite digest pins the data."""
    items = {}
    for t in tasks:
        p = ROOT / "lab" / "data" / f"wspx_{t}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        items[t] = {"file": f"lab/data/wspx_{t}.json", "sha256": sha_file(p), "n_train": len(d["train"]),
                    "n_holdout": len(d["holdout"])}
    s = {"wsplang_bench_version": VERSION, "name": name, "model_class": "nanowsp", "think": False, "items": items,
         "decoding": {"max_tokens": 1, "parallel": 1, "temperature": 0, "seed": 0, "template": "raw prompt, one character",
                      "prompt_suffix": ""},
         "graders": ["exact match of the generated character with the target (NanoWSP.evaluate_recall)"],
         "preservation": False}
    s["sha256"] = suite_digest(s)
    return s


def load_suite(path):
    """Reads a suite file, verifies its digest and, for a sealed wrapper, loads the items from the sealed file (whose own
    hash engine.sealed_set.load verifies) and remembers its `use` field."""
    path = Path(path)
    s = json.loads(path.read_text(encoding="utf-8"))
    if s.get("sha256") != suite_digest(s):
        raise ValueError(f"{path}: suite digest does not match its items / decoding (tampered or hand-edited)")
    if s.get("sealed_file"):
        sealed_path = ROOT / s["sealed_file"] if not Path(s["sealed_file"]).is_absolute() else Path(s["sealed_file"])
        sealed = sealed_set.load(sealed_path)
        if sealed["sha256"] != s["sealed_sha256"]:
            raise ValueError(f"{path}: sealed file sha256 changed since the suite was written")
        s["items"] = {"gsm8k": list(sealed["gsm8k"]), "math500": list(sealed["math500"])}
        s["sealed_use"] = sealed.get("use", "")
    s.setdefault("model_class", "llm-chat-think")
    s.setdefault("think", "auto")
    s["path"] = str(path)
    return s


def resolve_suite(name_or_path):
    """`suite=dev-2048` -> engine/bench_suites/dev-2048.json; an explicit path is accepted as is."""
    p = Path(str(name_or_path))
    if p.suffix == ".json" and p.exists():
        return load_suite(p)
    cand = SUITES / f"{name_or_path}.json"
    if cand.exists():
        return load_suite(cand)
    raise FileNotFoundError(f"no suite {name_or_path!r}: not a file and not in {SUITES}")


def dev_items(gsm_n=100, math_n=50):
    import math500 as m5
    return list(range(gsm_n)), list(m5.select_indices(2026, math_n, len(m5.load_items())))


def write_shipped_suites(out_dir=SUITES):
    """The suites shipped with the language. dev-2048 / dev-8192: bench_runner's default items at two budgets
    (llm-chat-think). sealed-1/2/3: wrappers over data/sealed_confirm*.json (single use enforced by their `use` field).
    nanowsp-wspx: the NanoWSP class over the local WSPX tasks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g, m = dev_items()
    written = []
    for mt in (2048, 8192):
        s = make_suite(f"dev-{mt}", g, m, mt)
        (out_dir / f"dev-{mt}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        written.append(out_dir / f"dev-{mt}.json")
    for n, fname in ((1, "sealed_confirm.json"), (2, "sealed_confirm2.json"), (3, "sealed_confirm3.json")):
        sp = ROOT / "data" / fname
        if not sp.exists():
            continue
        sealed = sealed_set.load(sp)
        s = make_suite(f"sealed-{n}", sealed["gsm8k"], sealed["math500"], 2048, sealed_file=f"data/{fname}",
                       sealed_sha256=sealed["sha256"])
        s.pop("items")  # a wrapper carries no items of its own: they come from the sealed file at load time
        s["sha256"] = suite_digest(s)
        (out_dir / f"sealed-{n}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        written.append(out_dir / f"sealed-{n}.json")
    s = make_nanowsp_suite()
    (out_dir / "nanowsp-wspx.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    written.append(out_dir / "nanowsp-wspx.json")
    from engine import dist_shift
    s = dist_shift.make_logits_suite(text="data/bench/wikitext2_test.txt", hellaswag="data/bench/hellaswag_val_full.txt",
                                     winogrande="data/bench/winogrande-debiased-eval.csv", mmlu="data/bench/mmlu-validation.bin")
    (out_dir / "dist-shift.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    written.append(out_dir / "dist-shift.json")
    return written


# ----------------------------------------------------------------------------- identity and provenance

def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def _get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url), timeout=30).read())


def server_identity(port, control_vector=None, lora=None, scale=None, binary=None):
    """What the server serves: /props model_path (+ sha256 when it is a local file), the extras the caller declares
    (llama-server does not report control vectors or LoRA files in /props), the binary's sha. model_sha None when the
    path is not a local file; model_path None when /props is unavailable."""
    base = bench_runner.endpoint_url(port).split("/v1/")[0]
    model_path, model_sha = None, None
    try:
        props = _get(base + "/props")
        model_path = props.get("model_path")
    except Exception:
        try:
            models = _get(base + "/v1/models")
            model_path = (models.get("data") or [{}])[0].get("id")
        except Exception:
            model_path = None
    if model_path and Path(model_path).is_file():
        model_sha = sha_file(model_path)
    extras = {}
    for kind, val in (("control_vector", control_vector), ("lora", lora)):
        if val:
            p = Path(str(val))
            extras[kind] = {"path": str(val), "sha256": sha_file(p) if p.is_file() else None, "declared_by": "caller"}
    if scale is not None:
        extras["scale"] = float(scale)
    bp = Path(binary) if binary else DEFAULT_BINARY
    return {"model_path": model_path, "model_sha": model_sha, "extras": extras,
            "binary": {"path": str(bp), "sha256": sha_file(bp) if bp.is_file() else None}}


def read_provenance(path):
    """Normalises the manifests WSP-Lang writes into {kind, output_sha, base_sha, patches, manifest_path, manifest_sha}.
    Supported: wsp.py GGUF manifest (source_fp.sha, out), compile_control_vector report (files.merged_gguf_sha256 or
    merged sha fields), edit_column manifest (base_tensor_hash / out sha), commit_manifest bundle (base hash +
    ordered patches + final hashes). Unknown shapes are kept as kind "unknown" with no checks possible."""
    p = Path(path)
    m = json.loads(p.read_text(encoding="utf-8"))
    out = {"kind": "unknown", "output_sha": None, "base_sha": None, "patches": [], "manifest_path": str(p),
           "manifest_sha": sha_file(p)}
    if "source_fp" in m and "ops" in m:                                   # wsp.py edit manifest
        out["kind"] = "wsp_edit"
        out["base_sha"] = (m.get("source_fp") or {}).get("sha")
        out["output_sha"] = m.get("output_sha256") or m.get("out_sha256")
        out["patches"] = [f"{o.get('type')}:{o.get('tensor')}" for o in m["ops"]]
    elif "inputs" in m and "vector" in m and "G0" in m:                  # compile_control_vector report
        out["kind"] = "compile_control_vector"
        out["base_sha"] = (m.get("inputs") or {}).get("base_gguf_sha256")
        files = m.get("files") or {}
        # Older compile reports list their outputs as a plain list of paths and record no merged sha: output_sha stays
        # None (the served sha is then unchecked against the manifest; the base link still comes from "inputs").
        out["output_sha"] = (files.get("merged_gguf_sha256") or files.get("merged_sha256")) if isinstance(files, dict) else None
        out["patches"] = [f"compile:{Path(str((m.get('inputs') or {}).get('vector', ''))).name}"]
    elif "wsplang_patch_version" in m or ("patches" in m and "base_checkpoint_hash" in m):  # weightpatch / bundle
        out["kind"] = "patch_bundle" if "patches" in m else "weightpatch"
        out["base_sha"] = m.get("base_checkpoint_hash") or m.get("base_tensor_hash")
        out["output_sha"] = m.get("final_checkpoint_hash") or m.get("output_sha256")
        pats = m.get("patches") or ([m["id"]] if m.get("id") else [])
        out["patches"] = [x["id"] if isinstance(x, dict) and "id" in x else (x.get("patch") or "?") if isinstance(x, dict) else str(x)
                          for x in pats]
    elif "op" in m and str(m.get("op", "")).startswith("edit_"):            # edit_column manifest / report
        out["kind"] = m["op"]
        out["base_sha"] = m.get("base_checkpoint_hash") or m.get("original_hash") or (m.get("hashes") or {}).get("before")
        out["output_sha"] = m.get("edited_checkpoint_hash") or m.get("edited_hash") or (m.get("hashes") or {}).get("after")
        out["patches"] = [f"{m['op']}:{m.get('tensor')}:{m.get('column', m.get('columns'))}"]
    return out


def sha_match(a, b):
    """Manifests store full sha256 or a 32-hex fingerprint prefix (wsp.py source_fp): equal when one is a prefix of the
    other and both are at least 16 hex characters."""
    a, b = (str(a or "")).lower(), (str(b or "")).lower()
    return bool(a) and bool(b) and min(len(a), len(b)) >= 16 and (a.startswith(b) or b.startswith(a))


def check_provenance(prov, identity):
    """A manifest describes the served model only if its output sha equals the served sha, when both exist."""
    if prov.get("output_sha") and identity.get("model_sha") and not sha_match(prov["output_sha"], identity["model_sha"]):
        raise ProvenanceError(f"provenance {prov['manifest_path']} output sha {prov['output_sha'][:12]} != served model sha "
                              f"{identity['model_sha'][:12]}")
    return True


# ----------------------------------------------------------------------------- running: LLM classes

def _make_ask(extra_body):
    """bench_runner.ask with the suite's request plus the arm's extra fields (e.g. logit_bias), body otherwise identical."""
    def ask(url, text, max_tokens, lora=None, extra=""):
        body = {"messages": [{"role": "user", "content": text + bench_runner.SUFFIX + extra}], "temperature": 0,
                "seed": 0, "max_tokens": max_tokens}
        if lora:
            body["lora"] = [{"id": int(i), "scale": float(s)} for i, s in lora]
        if extra_body:
            body.update(extra_body)
        r = _post(url, body)
        ch = r["choices"][0]
        msg = ch["message"]
        content, reasoning = bench_runner.split_think(msg.get("content") or "", msg.get("reasoning_content") or "")
        return {"content": content, "reasoning": reasoning, "finish": ch["finish_reason"],
                "tokens": r["usage"]["completion_tokens"],
                "tok_s": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}
    return ask


def _run_llm(suite, port, arm, work, extra_body, lora, log):
    dec = suite["decoding"]
    saved_ask = bench_runner.ask
    bench_runner.ask = _make_ask(extra_body)
    try:
        rep = bench_runner.run(f"{arm}", port, out_dir=work, max_tokens=dec["max_tokens"], parallel=dec["parallel"],
                               gsm_idx=suite["items"]["gsm8k"], math_idx=suite["items"]["math500"], lora=lora, log=log)
    finally:
        bench_runner.ask = saved_ask
    pres_rep = None
    pres = suite.get("preservation")
    if pres:
        pres_rep = preserve_bench.run(f"{arm}_preserve", port, work, max_tokens=pres.get("max_tokens", dec["max_tokens"]),
                                      parallel=dec["parallel"], he_n=pres.get("humaneval", 164),
                                      comp_n=pres.get("compliance", 40), lora=lora, log=log,
                                      ifeval_n=int(pres.get("ifeval") or 0), trivia_n=int(pres.get("triviaqa") or 0),
                                      short_max_tokens=pres.get("short_max_tokens"))
    think_share = None
    if rep["items"]:
        think_share = round(sum(r.get("reasoning_chars", 0) > 0 for r in rep["items"]) / len(rep["items"]), 3)
    return {"benchmarks": rep["summary"], "items": rep["items"], "served": rep.get("served"),
            "preservation": pres_rep["summary"] if pres_rep else {},
            "preservation_items": pres_rep["items"] if pres_rep else [], "think_share": think_share}


# ----------------------------------------------------------------------------- running: nanowsp class

def nanowsp_checkpoint_file(path):
    """A NanoWSP checkpoint is a safetensors file; a ternary checkpoint dir (engine/ternary_write.py) carries its float
    materialisation as <dir>/model.safetensors. Whatever the float runtime loads is accepted."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        for cand in (p / "model.safetensors",):
            if cand.exists():
                return cand
        files = sorted(p.glob("*.safetensors"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(f"no NanoWSP checkpoint at {path}")


def load_nanowsp(path):
    """Config inferred from the tensors (d_model, vocab, max_seq_len, n_layers, d_ff); heads as the runtime's default
    split (n_heads 2, d_head = d_model / 2, as lab/probes use)."""
    from safetensors.torch import load_file
    from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig
    f = nanowsp_checkpoint_file(path)
    t = load_file(str(f))
    vocab, d = t["token_embd.weight"].shape
    n_layers = sum(1 for k in t if k.endswith(".attn_norm.weight"))
    mi = t["layers.0.mlp_in.weight"].shape
    d_ff = mi[0] if mi[1] == d else mi[1]
    cfg = NanoWSPConfig(vocab_size=int(vocab), d_model=int(d), n_layers=int(n_layers), n_heads=2, d_head=int(d) // 2,
                        d_ff=int(d_ff), max_seq_len=int(t["pos_embd.weight"].shape[0]))
    m = NanoWSP(cfg)
    m.load_checkpoint(f)
    return m, f


def _run_nanowsp(suite, model_path, log):
    m, f = load_nanowsp(model_path)
    items, summary = [], {}
    for task, meta in suite["items"].items():
        p = ROOT / meta["file"]
        if sha_file(p) != meta["sha256"]:
            raise ValueError(f"{p}: task file changed since the suite was written")
        d = json.loads(p.read_text(encoding="utf-8"))
        for split in ("train", "holdout"):
            ok = 0
            for it in d[split]:
                out = m.generate(it["prompt"], max_new_tokens=1)
                hit = out.endswith(it["target"])
                ok += int(hit)
                items.append({"bench": f"{task}/{split}", "i": it["prompt"], "gold": it["target"],
                              "pred": out[len(it["prompt"]):], "ok": hit, "ok_independent": hit, "finish": "stop",
                              "tokens": 1, "reasoning_chars": 0})
            n = len(d[split])
            summary[f"{task}/{split}"] = {"n": n, "acc": round(ok / max(n, 1), 3), "acc_independent": round(ok / max(n, 1), 3),
                                          "graders_disagree": 0, "cut_at_max": 0, "loop_tails": 0, "mean_tokens": 1,
                                          "mean_tok_s": 0.0}
        log(f"[bench_run] {task}: train {summary[f'{task}/train']['acc']} holdout {summary[f'{task}/holdout']['acc']}")
    return {"benchmarks": summary, "items": items, "served": {"model_path": str(f), "sha256": sha_file(f)},
            "preservation": {}, "preservation_items": [], "think_share": 0.0}


# ----------------------------------------------------------------------------- bench_run

def _run_logits(suite, model, base, binary, runner, logits_dir, control_vector, scale, log):
    from engine import dist_shift
    bench, meta = dist_shift.run_dist_shift(suite, model, base=base, binary=binary, runner=runner, logits_dir=logits_dir,
                                            control_vector=control_vector, scale=scale, log=log)
    bp = Path(binary) if binary else dist_shift.DEFAULT_PERPLEXITY
    extras = {}
    if control_vector:
        p = Path(str(control_vector))
        extras["control_vector"] = {"path": str(control_vector), "sha256": sha_file(p) if p.is_file() else None,
                                    "declared_by": "caller"}
    if scale is not None:
        extras["scale"] = float(scale)
    identity = {"model_path": str(model), "model_sha": sha_file(model), "extras": extras,
                "binary": {"path": str(bp), "sha256": sha_file(bp) if bp.is_file() else None},
                "base_sha": meta["base_sha"], "base_logits": meta["base_logits"], "data_sha": meta["data_sha"]}
    return {"benchmarks": bench, "items": [], "served": {"model_path": str(model), "sha256": identity["model_sha"]},
            "preservation": {}, "preservation_items": [], "think_share": None, "commands": meta["commands"]}, identity


def bench_run(suite, port=None, out=None, arm=None, extra_body=None, lora=None, force_reuse=False, log=print,
              work_dir=None, binary=None, model=None, control_vector=None, scale=None, provenance=None,
              unknown_model=False, base=None, runner=None, logits_dir=None):
    """Runs a suite (dict from resolve_suite / make_suite). LLM classes: against the server on `port`; nanowsp class:
    `model=<checkpoint file or dir>` in-process; llm-logits class: `model=<gguf>` through llama-perplexity (`base=<gguf>`
    for the KL component, `binary=` the llama-perplexity path, `runner` / `logits_dir` for tests). Writes `out` (JSON)
    and returns it."""
    if suite.get("sealed_use") and "spent" in str(suite["sealed_use"]).lower() and not force_reuse:
        raise SealedSpent(f"suite {suite['name']} is a sealed set already spent ({suite['sealed_use']}); "
                          "a sealed set is used once; force_reuse=true runs it anyway and records that")
    cls = suite.get("model_class", "llm-chat-think")
    file_classes = ("nanowsp", "llm-logits")
    if cls in file_classes and not model:
        raise ClassMismatch(f"suite {suite['name']} is of class {cls}: it needs model=<file>, not a server")
    if cls not in file_classes and model and not port:
        raise ClassMismatch(f"suite {suite['name']} is of class {cls}: it needs a running server (port=), "
                            "model= is for the nanowsp and llm-logits classes")
    if cls not in file_classes and not port:
        raise ValueError("bench_run needs port=<llama-server port> for an LLM suite")
    out = Path(out) if out else ROOT / "lab" / "runs" / "bench" / f"{arm}.json"
    arm = arm or out.stem
    work = Path(work_dir) if work_dir else out.parent / f"{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cls == "llm-logits":
        res, identity = _run_logits(suite, model, base, binary, runner, logits_dir, control_vector, scale, log)
    elif cls == "nanowsp":
        res = _run_nanowsp(suite, model, log)
        identity = {"model_path": res["served"]["model_path"], "model_sha": res["served"]["sha256"], "extras": {},
                    "binary": {"path": "in-process (engine/nanowsp_runtime.py)", "sha256": sha_file(ROOT / "engine" / "nanowsp_runtime.py")}}
    else:
        identity = server_identity(port, control_vector=control_vector, lora=lora, scale=scale, binary=binary)
        if not identity["model_path"] and not unknown_model:
            raise IdentityError("the server reports no model path (/props and /v1/models): refusing to write a run "
                                "without identity; pass unknown_model=true to record it as unidentified")
        res = _run_llm(suite, port, arm, work, extra_body, lora, log)
        if not identity["model_path"] and (res.get("served") or {}).get("model_path"):
            identity["model_path"] = res["served"]["model_path"]
            identity["model_sha"] = res["served"].get("sha256")
    identity["suite_sha"] = suite["sha256"]
    identity["decoding"] = suite["decoding"]
    identity["unknown_model"] = bool(unknown_model and not identity.get("model_path"))
    prov = None
    if provenance:
        prov = read_provenance(provenance)
        check_provenance(prov, identity)
    run = {"wsplang_bench_version": VERSION, "suite": suite["name"], "suite_sha": suite["sha256"],
           "model_class": cls, "think": suite.get("think", "auto"), "arm": arm, "identity": identity,
           "provenance": prov, "served": res.get("served"), "binary": identity["binary"], "decoding": suite["decoding"],
           "extra_body": extra_body, "lora": lora, "force_reuse": bool(force_reuse), "sealed_use": suite.get("sealed_use"),
           "benchmarks": res["benchmarks"], "preservation": res["preservation"], "think_share": res.get("think_share"),
           "items": res["items"], "preservation_items": res["preservation_items"],
           "thresholds": suite.get("thresholds"), "commands": res.get("commands"),
           "seconds": round(time.time() - t0), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "imported_from": None}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[bench_run] {suite['name']} {arm}: {json.dumps(res['benchmarks'])}")
    return run


def bench_run_legacy(port, name, gsm_n=100, math_n=50, max_tokens=8192, parallel=2, out=None, log=print, work_dir=None,
                     unknown_model=True):
    """The evaluate_benchmark op's arguments on the same path: dev-2048 when they match it, else an ad-hoc suite with
    those items and budget (no preservation), so both ops produce the same JSON. evaluate_benchmark predates the
    identity rule, so it records an unidentified server instead of refusing."""
    g, m = dev_items(gsm_n, math_n)
    try:
        dev = resolve_suite(DEFAULT_SUITE)
    except (FileNotFoundError, ValueError):
        dev = None
    if dev and dev["items"] == {"gsm8k": g, "math500": m} and dev["decoding"]["max_tokens"] == int(max_tokens) \
            and dev["decoding"]["parallel"] == int(parallel):
        suite = dev
    else:
        suite = make_suite("adhoc", g, m, max_tokens, parallel, preservation=False)
    out = Path(out) if out else ROOT / "lab" / "runs" / "bench" / f"{name}.json"
    return bench_run(suite, port=port, out=out, arm=name, log=log, work_dir=work_dir, unknown_model=unknown_model)


# ----------------------------------------------------------------------------- comparing

def binom_two_sided(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, j) for j in range(0, k + 1)) / 2 ** n)


def _by_key(run):
    return {(r["bench"], r["i"]): r for r in run["items"]}


def _cells(base, arm):
    b, a = _by_key(base), _by_key(arm)
    benches = []
    for k in b:
        if k[0] not in benches:
            benches.append(k[0])
    out = {}
    for bench in benches:
        keys = [k for k in b if k[0] == bench and k in a]
        if not keys:
            continue
        bb, aa = [b[k] for k in keys], [a[k] for k in keys]
        cell = {}
        for g, f in (("g1", lambda r: bool(r["ok"])), ("g2", lambda r: bool(r["ok_independent"]))):
            win = sum(f(y) and not f(x) for x, y in zip(bb, aa))
            loss = sum(f(x) and not f(y) for x, y in zip(bb, aa))
            cell[g] = {"n": len(keys), "base": sum(map(f, bb)), "arm": sum(map(f, aa)), "win": win, "loss": loss,
                       "p": round(binom_two_sided(win, loss), 4)}
        tb, ta = sum(r["tokens"] for r in bb), sum(r["tokens"] for r in aa)
        cell["extra"] = {
            "lost_base_ok_stop": sum(x["ok"] and x["finish"] == "stop" and not y["ok"] for x, y in zip(bb, aa)),
            "gained": sum((not x["ok"]) and y["ok"] for x, y in zip(bb, aa)),
            "cut_base": sum(r["finish"] == "length" for r in bb), "cut_arm": sum(r["finish"] == "length" for r in aa),
            "nothink_arm": sum(r.get("reasoning_chars", 1) == 0 for r in aa),
            "tokens_base": tb, "tokens_arm": ta, "token_change": round(ta / tb - 1, 4) if tb else None}
        out[bench] = cell
    return out


def noise_band(run_a, run_b):
    """Two runs of the same configuration -> per benchmark: discordant items (g1) and |difference in correct|."""
    cells = _cells(run_a, run_b)
    return {bench: {"discordant": c["g1"]["win"] + c["g1"]["loss"], "abs_delta": abs(c["g1"]["arm"] - c["g1"]["base"]),
                    "n": c["g1"]["n"]} for bench, c in cells.items()}


def _sig_up(cell):
    return all(cell[g]["arm"] > cell[g]["base"] and cell[g]["p"] < 0.05 for g in ("g1", "g2"))


def _sig_down(cell):
    return all(cell[g]["arm"] < cell[g]["base"] and cell[g]["p"] < 0.05 for g in ("g1", "g2"))


def _any_sig_up(cells):
    """A placebo blocks only when it *gains*: a same-norm placebo that harms the base (the 9B, results §43.17) says the
    direction matters, which is the point of the control."""
    return any(cells[b][g]["p"] < 0.05 and cells[b][g]["arm"] > cells[b][g]["base"] for b in cells for g in ("g1", "g2"))


def _model_sha(run):
    return ((run.get("identity") or {}).get("model_sha")) or ((run.get("served") or {}).get("sha256"))


def relation(base, arm):
    """'edit vs its base' when the arm's provenance base sha equals the base run's model sha, or both runs serve the
    same file and differ only in extras (control vector, LoRA, request fields); else 'two models'."""
    bs = _model_sha(base)
    prov = arm.get("provenance") or {}
    if sha_match(prov.get("base_sha"), bs):
        return "edit vs its base"
    if sha_match(_model_sha(arm), bs):
        return "edit vs its base"
    if sha_match((arm.get("identity") or {}).get("base_sha"), bs):   # llm-logits: the KL base named at run time
        return "edit vs its base"
    return "two models"


def _compare_logits(base, arm, rel):
    """The llm-logits class: no items. PPL ratio, the arm's KL to its base, top-1 agreement, and the accuracy deltas
    with the tools' own errors. Verdict 'preserves distribution' / 'distribution shifted' only when the suite's
    thresholds are calibrated (kl_mean, same_top_p not null); 'not calibrated yet' otherwise."""
    b, a = base.get("benchmarks") or {}, arm.get("benchmarks") or {}
    kl = a.get("kl") or {}
    out = {"wsplang_bench_version": VERSION, "suite": arm.get("suite"), "relation": rel, "base": base.get("arm"),
           "arm": arm.get("arm"), "base_model_sha": (_model_sha(base) or "")[:12], "arm_model_sha": (_model_sha(arm) or "")[:12],
           "placebo": None, "band": None, "benchmarks": {}, "placebo_benchmarks": None, "preservation": None, "reasons": [],
           "logits": {}}
    L = out["logits"]
    pb, pa = (b.get("perplexity") or {}).get("ppl"), (a.get("perplexity") or {}).get("ppl")
    L["ppl_base"], L["ppl_arm"] = pb, pa
    L["ppl_ratio"] = (pa / pb) if pa is not None and pb else None
    for k in ("kl_mean", "kl_mean_err", "kl_p99", "kl_max", "same_top_p", "same_top_p_err"):
        L[k] = kl.get(k)
    L["delta"], L["err"] = {}, {}
    for t in ("hellaswag", "winogrande", "mmlu"):
        tb, ta = b.get(t) or {}, a.get(t) or {}
        if tb.get("acc") is not None and ta.get("acc") is not None:
            L["delta"][t] = round(ta["acc"] - tb["acc"], 6)
            eb = tb.get("acc_err") or ((tb["ci"][1] - tb["ci"][0]) / 2 if tb.get("ci") else None)
            ea = ta.get("acc_err") or ((ta["ci"][1] - ta["ci"][0]) / 2 if ta.get("ci") else None)
            L["err"][t] = {"base": eb, "arm": ea}
    if rel == "two models":
        out["verdict"] = "two models: statistics only"
        out["reasons"].append("no base link between the runs (the arm's KL base is not this base)")
        return out
    thr = arm.get("thresholds") or {}
    t_kl, t_top = thr.get("kl_mean"), thr.get("same_top_p")
    if t_kl is None or t_top is None or L["kl_mean"] is None or L["same_top_p"] is None:
        out["verdict"] = "not calibrated yet"
        out["reasons"].append("thresholds kl_mean / same_top_p are null in the suite: set them from the base-vs-base band "
                              "and the same-norm placebo before any verdict")
        return out
    ok = True
    if L["kl_mean"] > t_kl:
        ok = False
        out["reasons"].append(f"kl_mean {L['kl_mean']:.5f} > threshold {t_kl}")
    if L["same_top_p"] < t_top:
        ok = False
        out["reasons"].append(f"same_top_p {L['same_top_p']:.2f} < threshold {t_top}")
    out["verdict"] = "preserves distribution" if ok else "distribution shifted"
    return out


def bench_compare(base, arm, placebo=None, band=None):
    """Paired comparison of two bench_run JSONs (dicts). Same suite sha required. Full verdict rule for
    'edit vs its base'; paired statistics only for 'two models'. llm-logits runs go through _compare_logits."""
    if base.get("suite_sha") and arm.get("suite_sha") and base["suite_sha"] != arm["suite_sha"]:
        raise ValueError("bench_compare: the two runs are not on the same suite (suite sha differs)")
    if arm.get("model_class") == "llm-logits" or base.get("model_class") == "llm-logits":
        if arm.get("model_class") != base.get("model_class"):
            raise ValueError("bench_compare: an llm-logits run can only be compared with another llm-logits run")
        return _compare_logits(base, arm, relation(base, arm))
    cells = _cells(base, arm)
    rel = relation(base, arm)
    out = {"wsplang_bench_version": VERSION, "suite": arm.get("suite"), "relation": rel, "base": base.get("arm"),
           "arm": arm.get("arm"), "base_model_sha": (_model_sha(base) or "")[:12], "arm_model_sha": (_model_sha(arm) or "")[:12],
           "placebo": placebo.get("arm") if placebo else None, "band": band, "benchmarks": cells,
           "placebo_benchmarks": _cells(base, placebo) if placebo else None, "preservation": None, "reasons": []}
    if base.get("preservation_items") and arm.get("preservation_items"):
        out["preservation"] = preserve_bench.compare(base["preservation_items"], arm["preservation_items"])
    up = [b for b in cells if _sig_up(cells[b])]
    down = [b for b in cells if _sig_down(cells[b])]
    out["significant_up"], out["significant_down"] = up, down
    reasons = out["reasons"]
    if rel == "two models":
        out["verdict"] = "two models: paired statistics only"
        reasons.append("no provenance link between the runs; the verdict rule applies to an edit against its base")
        return out

    def band_ok(benches):
        if band is None:
            return benches
        ok = [b for b in benches if b in band and abs(cells[b]["g1"]["arm"] - cells[b]["g1"]["base"]) >= 2 * band[b]["abs_delta"]]
        for b in benches:
            if b not in ok:
                reasons.append(f"{b}: |delta| {abs(cells[b]['g1']['arm'] - cells[b]['g1']['base'])} below 2 x band "
                               f"{band.get(b, {}).get('abs_delta')}")
        return ok
    verdict = "same / not decidable"
    if up:
        up_ok = band_ok(up)
        if placebo and _any_sig_up(out["placebo_benchmarks"]):
            reasons.append("placebo arm gains significantly on some benchmark / grader")
            up_ok = []
        if out["preservation"] is not None and not out["preservation"]["preserves"]:
            reasons.append("preservation battery failed (HumanEval / compliance / garbage rule)")
            up_ok = []
        if up_ok:
            verdict = "improves"
    elif down:
        if band_ok(down):
            verdict = "harms"
    else:
        reasons.append("no benchmark significant on both graders")
    out["verdict"] = verdict
    return out


# ----------------------------------------------------------------------------- registry and table

def _load_index(index_path):
    p = Path(index_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"wsplang_bench_version": VERSION, "runs": [], "comparisons": []}


def _save_index(index_path, idx):
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx, indent=1, ensure_ascii=False), encoding="utf-8")


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _prov_label(run):
    prov = run.get("provenance")
    if not prov:
        return "external"
    base = (prov.get("base_sha") or "")[:8] or "?"
    n = len(prov.get("patches") or [])
    return f"base@{base} + {n} patch{'es' if n != 1 else ''} ({prov.get('kind')})"


def registry_add(index_path, run_path, model=None):
    """Adds one bench_run JSON to the registry (idempotent by path); `model` is the human label (defaults to the served
    file name or the arm)."""
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    idx = _load_index(index_path)
    rel = _rel(run_path)
    idx["runs"] = [r for r in idx["runs"] if r["path"] != rel]
    served = run.get("served") or {}
    ident = run.get("identity") or {}
    b = run.get("benchmarks") or {}
    pres = run.get("preservation") or {}
    sha = ident.get("model_sha") or served.get("sha256") or ""

    def acc(bench, key):
        s = b.get(bench)
        return round(s[key] * s["n"]) if s else None
    label = model or (Path(ident.get("model_path") or served.get("model_path") or "").name or run["arm"])
    logits = None
    if run.get("model_class") == "llm-logits":
        kl = b.get("kl") or {}
        logits = {"ppl": (b.get("perplexity") or {}).get("ppl"), "kl_mean": kl.get("kl_mean"), "same_top_p": kl.get("same_top_p"),
                  "hellaswag": (b.get("hellaswag") or {}).get("acc"), "winogrande": (b.get("winogrande") or {}).get("acc"),
                  "mmlu": (b.get("mmlu") or {}).get("acc")}
        b = {}                                             # no per-item accuracies for this class
    idx["runs"].append({
        "path": rel, "model": label, "model_sha": sha[:12], "model_class": run.get("model_class", "llm-chat-think"),
        "provenance": _prov_label(run), "suite": run["suite"], "suite_sha": run.get("suite_sha", "")[:12], "arm": run["arm"],
        "gsm8k_g1": acc("gsm8k", "acc"), "gsm8k_g2": acc("gsm8k", "acc_independent"),
        "math500_g1": acc("math500", "acc"), "math500_g2": acc("math500", "acc_independent"),
        "other": {k: s.get("acc") for k, s in b.items() if k not in ("gsm8k", "math500")},
        "cuts": sum(s.get("cut_at_max", 0) for s in b.values()) if b else None,
        "mean_tokens": {k: s.get("mean_tokens") for k, s in b.items() if k in ("gsm8k", "math500")} if b else None,
        "humaneval": (pres.get("humaneval") or {}).get("ok"), "compliance": (pres.get("compliance") or {}).get("ok"),
        "garbage": ((pres.get("garbage") or {}).get("repeat_16gram", 0) + (pres.get("garbage") or {}).get("nonascii_run", 0))
        if pres.get("garbage") else None,
        "ifeval_strict": (pres.get("ifeval") or {}).get("prompt_strict"), "ifeval_loose": (pres.get("ifeval") or {}).get("prompt_loose"),
        "ifeval_scorable": (pres.get("ifeval") or {}).get("scorable"), "triviaqa": (pres.get("triviaqa") or {}).get("em"),
        "extras": (ident.get("extras") or {}) or None, "extra_body": run.get("extra_body"), "logits": logits,
        "date": (run.get("created") or "")[:10], "imported_from": run.get("imported_from")})
    idx["runs"].sort(key=lambda r: (r["model_class"], r["model"], r["suite"], r["arm"]))
    _save_index(index_path, idx)
    return idx


def registry_add_comparison(index_path, compare_path):
    c = json.loads(Path(compare_path).read_text(encoding="utf-8"))
    idx = _load_index(index_path)
    rel = _rel(compare_path)
    idx["comparisons"] = [x for x in idx["comparisons"] if x["path"] != rel]
    ps = {f"{b}/{g}": c["benchmarks"][b][g]["p"] for b in c.get("benchmarks", {}) for g in ("g1", "g2") if g in c["benchmarks"][b]}
    idx["comparisons"].append({"path": rel, "suite": c.get("suite"), "relation": c.get("relation", ""), "base": c["base"],
                               "arm": c["arm"], "placebo": c.get("placebo"), "verdict": c["verdict"], "p": ps,
                               "band": c.get("band"), "reasons": c.get("reasons", []),
                               "logits": {k: (c.get("logits") or {}).get(k) for k in ("ppl_ratio", "kl_mean", "same_top_p", "delta")}
                               if c.get("logits") else None})
    _save_index(index_path, idx)
    return idx


RULE_TEXT = (
    "**Verdict rule (`bench_compare`, fixed 2026-11-18), for an edit against its base** (the arm's provenance names the "
    "base run's model sha, or both runs serve the same file and differ only in extras). *improves* only if (a) on at "
    "least one benchmark the arm beats the base with exact two-sided binomial p < 0.05 on both graders, (b) the placebo "
    "arm, when given, does not gain significantly on any benchmark / grader (a placebo that harms does not block), (c) "
    "when a noise band is given (two runs of the same "
    "configuration), |arm − base| ≥ 2 × band on that benchmark, (d) the preservation battery, when both runs carry it, "
    "does not fail. *harms* is the symmetric statement. Otherwise *same / not decidable*, with the reasons. Two "
    "unrelated models get the paired statistics and the band only, no verdict word. This table counts; the sealed "
    "protocol (`lab/probes/sealed/README.md`) decides claims.")


def render_table(index_path, out_md):
    idx = _load_index(index_path)
    L = ["# WSP-Lang benchmarks", "",
         f"Generated by the `bench_table` op from `{_rel(index_path)}` (WSP-Lang benchmarks {VERSION}; suites in "
         "`engine/bench_suites/`, spec `docs/wsplang_0.8_bench_addendum.md`). Every row is one `bench_run` JSON: the "
         "served model's sha256, its provenance (when WSP-Lang built it), the suite digest, the decoding and the "
         "per-item records are in the file.", "", RULE_TEXT, "",
         "## Runs", "",
         "| model (sha) | class | provenance | suite | arm | GSM8K g1 / g2 | MATH-500 g1 / g2 | other | cuts | "
         "mean tokens (gsm / math) | HumanEval /164 | compliance /40 | garbage | IFEval strict / loose (scorable) | TriviaQA em /200 | "
         "ppl | KL mean | top-1 agree % | HS | WG | MMLU | date | source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f3(v, nd=3):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    for r in idx["runs"]:
        mt = r.get("mean_tokens") or {}
        lg = r.get("logits") or {}
        other = "; ".join(f"{k} {v}" for k, v in (r.get("other") or {}).items()) or "-"
        model = f"{r['model']} ({r['model_sha']})" if r.get("model_sha") else f"{r['model']} (no sha)"
        L.append(f"| {model} | {r.get('model_class', '-')} | {r.get('provenance', 'external')} | {r['suite']} | {r['arm']} | "
                 f"{f3(r['gsm8k_g1'])} / {f3(r['gsm8k_g2'])} | {f3(r['math500_g1'])} / {f3(r['math500_g2'])} | {other} | "
                 f"{f3(r['cuts'])} | {f3(mt.get('gsm8k'))} / {f3(mt.get('math500'))} | "
                 f"{r['humaneval'] if r['humaneval'] is not None else '-'} | "
                 f"{r['compliance'] if r['compliance'] is not None else '-'} | "
                 f"{r['garbage'] if r['garbage'] is not None else '-'} | "
                 f"{(str(r.get('ifeval_strict')) + ' / ' + str(r.get('ifeval_loose')) + ' (' + str(r.get('ifeval_scorable')) + ')') if r.get('ifeval_strict') is not None else '-'} | "
                 f"{r.get('triviaqa') if r.get('triviaqa') is not None else '-'} | "
                 f"{f3(lg.get('ppl'))} | {f3(lg.get('kl_mean'), 5)} | {f3(lg.get('same_top_p'), 2)} | {f3(lg.get('hellaswag'))} | "
                 f"{f3(lg.get('winogrande'))} | {f3(lg.get('mmlu'))} | {r['date']} | "
                 f"{'imported: ' + r['imported_from'] if r.get('imported_from') else 'bench_run'} |")
    L += ["", "## Comparisons", "",
          "| suite | relation | base | arm | placebo | verdict | p (bench/grader) | band | reasons |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in idx["comparisons"]:
        ps = "; ".join(f"{k} {v if v else '<0.0001'}" for k, v in c["p"].items())
        if not ps and c.get("logits"):
            lg = c["logits"]
            d = "; ".join(f"{k} {v:+.3f}" for k, v in (lg.get("delta") or {}).items())
            ps = (f"ppl× {f3(lg.get('ppl_ratio'))}; KL {f3(lg.get('kl_mean'), 5)}; top-1 {f3(lg.get('same_top_p'), 2)} %"
                  + (f"; Δ {d}" if d else ""))
        band = c.get("band")
        band_s = "; ".join(f"{b} ±{v['abs_delta']}" for b, v in band.items()) if band else "none"
        L.append(f"| {c.get('suite') or '-'} | {c.get('relation') or '-'} | {c['base']} | {c['arm']} | {c.get('placebo') or '-'} | "
                 f"**{c['verdict']}** | {ps} | {band_s} | {'; '.join(c.get('reasons', [])) or '-'} |")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    return Path(out_md)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "write-suites":
        for p in write_shipped_suites():
            print(p)
    elif len(sys.argv) > 1 and sys.argv[1] == "table":
        print(render_table(REGISTRY, TABLE))
    else:
        print(__doc__)
