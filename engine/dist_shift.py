"""engine/dist_shift.py - the label-free part of WSP-Lang benchmarks (lab/probes/dist_shift/PLAN.md): llama.cpp's own
`llama-perplexity` as the measuring tool, so a model's general distribution is compared without any task labels.

Components (all teacher-forced, no sampling, deterministic for a given GGUF, binary and file):
  perplexity   `-f TEXT -c CTX --chunks N`                         -> PPL +/- err on a fixed text (WikiText-2 test)
  kl           `--kl-divergence --kl-divergence-base BASE.bin`    -> KL(base || model) per token over the same chunks,
               BASE.bin written once per base with `--save-all-logits` (the two flags are aliases: without
               --kl-divergence the file is written, with it the file is read; verified in --help and on the CPU,
               2026-11-18). Mean / percentiles / max of the KL, PPL(Q)/PPL(base), and "Same top p" (top-1 agreement).
  hellaswag    `-f FILE --hellaswag --hellaswag-tasks N`           -> acc_norm with a 95 % interval (last task row)
  winogrande   `-f FILE --winogrande --winogrande-tasks N`         -> accuracy (last task row)
  mmlu         `-bf FILE --multiple-choice --multiple-choice-tasks N` -> "Final result: acc +/- err" (binary file: -bf)
The task selections are the tool's own and do not depend on `-s` (checked twice on the CPU); the subset is fixed by N.
A control vector enters every scored command as `--control-vector-scaled FILE:SCALE` (never the base-logits save).

GPU flags are the caller's: `gpu_args` in the suite (default -ngl 99). Nothing here starts a server.
"""
import re
import shutil
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERPLEXITY = Path(os.environ.get("LLAMA_PERPLEXITY_BIN") or shutil.which("llama-perplexity") or shutil.which("llama-perplexity.exe") or "llama-perplexity")
DEFAULT_GPU = ("-ngl", "99")


# ----------------------------------------------------------------------------- command lines

def _s(x):
    return str(x)


def cmd_perplexity(bin_, gguf, text, chunks, ctx, gpu_args=DEFAULT_GPU):
    return [_s(bin_), "-m", _s(gguf), "-f", _s(text), "-c", str(int(ctx)), "--chunks", str(int(chunks))] + list(gpu_args)


def cmd_save_logits(bin_, gguf, text, out_logits, chunks, ctx, gpu_args=DEFAULT_GPU):
    return [_s(bin_), "-m", _s(gguf), "-f", _s(text), "-c", str(int(ctx)), "--chunks", str(int(chunks)),
            "--save-all-logits", _s(out_logits)] + list(gpu_args)


def cmd_kl(bin_, gguf, base_logits, chunks, ctx, gpu_args=DEFAULT_GPU):
    return [_s(bin_), "-m", _s(gguf), "-c", str(int(ctx)), "--chunks", str(int(chunks)), "--kl-divergence",
            "--kl-divergence-base", _s(base_logits)] + list(gpu_args)


def cmd_hellaswag(bin_, gguf, file, tasks, ctx, gpu_args=DEFAULT_GPU):
    return [_s(bin_), "-m", _s(gguf), "-f", _s(file), "--hellaswag", "--hellaswag-tasks", str(int(tasks)),
            "-c", str(int(ctx))] + list(gpu_args)


def cmd_winogrande(bin_, gguf, file, tasks, ctx, gpu_args=DEFAULT_GPU):
    return [_s(bin_), "-m", _s(gguf), "-f", _s(file), "--winogrande", "--winogrande-tasks", str(int(tasks)),
            "-c", str(int(ctx))] + list(gpu_args)


def cmd_multiple_choice(bin_, gguf, file, tasks, ctx, gpu_args=DEFAULT_GPU):
    # -bf: the task file is binary; text-mode -f truncates it on Windows (Ctrl-Z) -> "failed to read task positions"
    return [_s(bin_), "-m", _s(gguf), "-bf", _s(file), "--multiple-choice", "--multiple-choice-tasks", str(int(tasks)),
            "-c", str(int(ctx))] + list(gpu_args)


def vector_args(control_vector, scale):
    if not control_vector:
        return []
    return ["--control-vector-scaled", f"{control_vector}:{float(scale if scale is not None else 1.0)}"]


# ----------------------------------------------------------------------------- parsers

_NUM = r"([-+]?\d+(?:\.\d+)?)"


def _find(pattern, text, cast=float, required=True, name=None):
    m = re.search(pattern, text)
    if not m:
        if required:
            raise ValueError(f"llama-perplexity output has no {name or pattern!r}")
        return None
    return cast(m.group(1))


def parse_perplexity(text):
    ppl = _find(r"Final estimate: PPL = " + _NUM, text, name="final PPL")
    err = _find(r"Final estimate: PPL = [-+\d.]+ \+/- " + _NUM, text, required=False)
    chunks = _find(r"calculating perplexity over (\d+) chunks", text, int, required=False)
    return {"ppl": ppl, "ppl_err": err, "chunks": chunks}


def _stat(label, text, required=True):
    """'Mean    KLD:  -0.000000 ±   0.000001' -> (value, err or None); labels may have variable spacing."""
    pat = re.escape(label).replace(r"\ ", r"\s+") + r"\s*:\s*" + _NUM + r"(?:\s*±\s*" + _NUM + r")?"
    m = re.search(pat, text)
    if not m:
        if required:
            raise ValueError(f"llama-perplexity KL output has no {label!r}")
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.group(2) is not None else None)


def parse_kl(text):
    out = {}
    out["chunks"] = _find(r"kl_divergence: computing over (\d+) chunks", text, int, required=False)
    out["ppl_q"], out["ppl_q_err"] = _stat("Mean PPL(Q)", text)
    out["ppl_base"], out["ppl_base_err"] = _stat("Mean PPL(base)", text)
    out["ln_ppl_ratio_mean"], _ = _stat("Mean ln(PPL(Q)/PPL(base))", text, required=False)
    out["kl_mean"], out["kl_mean_err"] = _stat("Mean KLD", text)
    out["kl_max"], _ = _stat("Maximum KLD", text)
    out["kl_p99_9"], _ = _stat("99.9% KLD", text, required=False)
    out["kl_p99"], _ = _stat("99.0% KLD", text, required=False)
    out["kl_p95"], _ = _stat("95.0% KLD", text, required=False)
    out["kl_p90"], _ = _stat("90.0% KLD", text, required=False)
    out["kl_median"], _ = _stat("Median KLD", text, required=False)
    m = re.search(r"RMS Δp\s*:\s*" + _NUM + r"\s*±\s*" + _NUM, text) or re.search(r"RMS .p\s*:\s*" + _NUM + r"\s*±\s*" + _NUM, text)
    out["dp_rms"] = float(m.group(1)) if m else None
    m = re.search(r"Same top p:\s*" + _NUM + r"\s*±\s*" + _NUM, text)
    if not m:
        raise ValueError("llama-perplexity KL output has no 'Same top p'")
    out["same_top_p"], out["same_top_p_err"] = float(m.group(1)), float(m.group(2))
    return out


def _last_task_row(text, ncols_min=2):
    rows = [l for l in text.splitlines() if re.match(r"^\d+\t", l)]
    if not rows:
        raise ValueError("llama-perplexity task output has no task rows")
    return rows[-1].split("\t")


def parse_hellaswag(text):
    pool = _find(r"hellaswag_score : loaded (\d+) tasks", text, int, required=False)
    row = _last_task_row(text)
    tasks = int(row[0])
    acc = float(row[1].rstrip("%")) / 100.0
    ci = None
    m = re.search(r"\[\s*" + _NUM + r"%,\s*" + _NUM + r"%\]", row[2]) if len(row) > 2 else None
    if m:
        ci = (float(m.group(1)) / 100.0, float(m.group(2)) / 100.0)
    return {"tasks": tasks, "pool": pool, "acc": acc, "ci": ci}


def parse_winogrande(text):
    pool = _find(r"winogrande_score : loaded (\d+) tasks", text, int, required=False)
    row = _last_task_row(text)
    return {"tasks": int(row[0]), "pool": pool, "acc": float(row[1]) / 100.0}


def parse_multiple_choice(text):
    pool = _find(r"multiple_choice_score: there are (\d+) tasks", text, int, required=False)
    acc = _find(r"Final result:\s*" + _NUM, text, name="Final result")
    err = _find(r"Final result:\s*[-+\d.]+ \+/- " + _NUM, text, required=False)
    chance = _find(r"Random chance:\s*" + _NUM, text, required=False)
    rows = [l for l in text.splitlines() if re.match(r"^\d+\t", l)]
    tasks = int(rows[-1].split("\t")[0]) if rows else None
    return {"tasks": tasks, "pool": pool, "acc": acc / 100.0, "acc_err": err / 100.0 if err is not None else None,
            "chance": chance / 100.0 if chance is not None else None}


# ----------------------------------------------------------------------------- running

def run_subprocess(argv):
    """Default runner: the tool's stdout + stderr as text; raises with the tail on a non-zero exit."""
    cp = subprocess.run([str(a) for a in argv], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (cp.stdout or "") + "\n" + (cp.stderr or "")
    if cp.returncode != 0:
        raise RuntimeError(f"{Path(str(argv[0])).name} exited {cp.returncode}: {out[-800:]}")
    return out


def _resolve(p, root):
    p = Path(str(p))
    return p if p.is_absolute() else Path(root) / p


def run_dist_shift(suite, model, base=None, binary=None, runner=None, logits_dir=None, control_vector=None, scale=None,
                   log=print, root=ROOT):
    """Executes the suite's components on `model` (a GGUF file). KL needs `base` (a GGUF): its logits are saved once to
    logits_dir/<base_sha16>.bin and reused. Returns (benchmarks, meta) with meta = {base_sha, base_logits, data_sha,
    commands}."""
    from engine.wsplang_bench import sha_file
    runner = runner or run_subprocess
    bin_ = Path(binary) if binary else DEFAULT_PERPLEXITY
    root = Path(suite.get("root") or root)        # a suite written against another root (tests) carries it
    comps = suite["components"]
    gpu = tuple(suite.get("gpu_args") or DEFAULT_GPU)
    vec = vector_args(control_vector, scale)
    logits_dir = Path(logits_dir) if logits_dir else Path(root) / "data" / "bench" / "logits"
    model = Path(model)
    if not model.is_file():
        raise FileNotFoundError(f"dist-shift: model GGUF not found: {model}")
    data_sha, commands, bench = {}, [], {}
    files = {}
    for key, comp in comps.items():
        f = _resolve(comp["file"], root)
        if not f.is_file():
            raise FileNotFoundError(f"dist-shift component {key}: data file missing: {f}")
        files[key] = f
        data_sha[key] = sha_file(f)
        if comp.get("sha256") and comp["sha256"] != data_sha[key]:
            raise ValueError(f"dist-shift component {key}: {f} changed since the suite was written")
    meta = {"base_sha": None, "base_logits": None, "data_sha": data_sha, "commands": commands}

    def go(argv, parser, label):
        commands.append([str(a) for a in argv])
        log(f"[dist-shift] {label}: {' '.join(str(a) for a in argv[1:9])} ...")
        return parser(runner(argv))

    if "perplexity" in comps:
        c = comps["perplexity"]
        bench["perplexity"] = go(cmd_perplexity(bin_, model, files["perplexity"], c["chunks"], c["ctx"], gpu) + vec,
                                 parse_perplexity, "perplexity")
    if "kl" in comps:
        c = comps["kl"]
        if base is None:
            bench["kl"] = None
            log("[dist-shift] kl skipped: no base= given")
        else:
            base = Path(base)
            if not base.is_file():
                raise FileNotFoundError(f"dist-shift: base GGUF not found: {base}")
            meta["base_sha"] = sha_file(base)
            logits_dir.mkdir(parents=True, exist_ok=True)
            lp = logits_dir / f"{meta['base_sha'][:16]}.bin"
            meta["base_logits"] = str(lp)
            if not lp.exists():
                go(cmd_save_logits(bin_, base, files["kl"], lp, c["chunks"], c["ctx"], gpu), parse_perplexity,
                   "save base logits")                                       # no control vector on the base save
            bench["kl"] = go(cmd_kl(bin_, model, lp, c["chunks"], c["ctx"], gpu) + vec, parse_kl, "kl vs base")
    if "hellaswag" in comps:
        c = comps["hellaswag"]
        bench["hellaswag"] = go(cmd_hellaswag(bin_, model, files["hellaswag"], c["tasks"], c["ctx"], gpu) + vec,
                                parse_hellaswag, "hellaswag")
    if "winogrande" in comps:
        c = comps["winogrande"]
        bench["winogrande"] = go(cmd_winogrande(bin_, model, files["winogrande"], c["tasks"], c["ctx"], gpu) + vec,
                                 parse_winogrande, "winogrande")
    if "mmlu" in comps:
        c = comps["mmlu"]
        bench["mmlu"] = go(cmd_multiple_choice(bin_, model, files["mmlu"], c["tasks"], c["ctx"], gpu) + vec,
                           parse_multiple_choice, "mmlu")
    return bench, meta


# ----------------------------------------------------------------------------- suite

def make_logits_suite(text, hellaswag, winogrande, mmlu, ppl_chunks=100, kl_chunks=16, ctx=512, tasks=400, task_ctx=1024,
                      thresholds=None, name="dist-shift", root=ROOT, gpu_args=DEFAULT_GPU):
    """The `llm-logits` class: a GGUF file measured by llama-perplexity, no server. File shas are pinned when the file
    exists (None otherwise, so a missing WikiText file is visible and bench_run refuses to run until it is present and
    the suite is rewritten). thresholds: {kl_mean, same_top_p} for the 'preserves distribution' verdict; None = not
    calibrated yet (bench_compare says so)."""
    from engine.wsplang_bench import VERSION, sha_file, suite_digest
    root = Path(root)

    def rel(p):
        p = Path(str(p))
        try:
            return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    def comp(p, **kw):
        f = _resolve(p, root)
        d = {"file": rel(p), "sha256": sha_file(f) if f.is_file() else None}
        d.update(kw)
        return d
    comps = {"perplexity": comp(text, chunks=int(ppl_chunks), ctx=int(ctx)),
             "kl": comp(text, chunks=int(kl_chunks), ctx=int(ctx)),
             "hellaswag": comp(hellaswag, tasks=int(tasks), ctx=int(task_ctx)),
             "winogrande": comp(winogrande, tasks=int(tasks), ctx=int(task_ctx)),
             "mmlu": comp(mmlu, tasks=int(tasks), ctx=int(task_ctx))}
    s = {"wsplang_bench_version": VERSION, "name": name, "model_class": "llm-logits", "think": False,
         "components": comps, "items": {"components": sorted(comps)},
         "decoding": {"max_tokens": 0, "parallel": 1, "temperature": 0, "seed": 0,
                      "template": "raw text, teacher-forced logits (llama-perplexity); no chat template", "prompt_suffix": ""},
         "graders": ["perplexity on the fixed text", "KL(base || model) and top-1 agreement over the same chunks",
                     "HellaSwag acc_norm", "Winogrande accuracy", "MMLU multiple-choice accuracy"],
         "preservation": False, "gpu_args": list(gpu_args),
         "thresholds": thresholds if thresholds is not None else {"kl_mean": None, "same_top_p": None},
         "reproducibility": "task subsets are chosen by llama-perplexity's own fixed RNG (do not depend on -s; checked "
                            "twice on the CPU 2026-11-18); logits and perplexity are deterministic for a GGUF + binary"}
    if root.resolve() != ROOT.resolve():
        s["root"] = str(root)
    s["sha256"] = suite_digest(s)
    return s
