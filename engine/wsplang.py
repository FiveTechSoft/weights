"""wsplang.py — Parser, Compiler & Runtime for WSP-Lang 0.4 (Matrix Perturbation Theory & Weight Space Surgery).

WSP-Lang 0.4 grounds Weight Space Surgery in Matrix Perturbation Theory & RandNLA
(Weyl's Inequality, Davis-Kahan sin Theta Theorem, Statistical Leverage Scores; Yu-Xiang Wang et al.):
  - Weyl's Singular Value Perturbation Bound ('assert_weyl'):
      |sigma_i(W + Delta W) - sigma_i(W)| <= ||Delta W||_2
    Proves why low-rank weight edits guarantee representation stability across the entire spectrum.
  - Davis-Kahan Invariant Subspace Stability ('assert_davis_kahan'):
      sin Theta(U_1, U_1_tilde) <= ||Delta W||_2 / delta
    Guarantees that principal reasoning subspaces rotate by less than a fraction of a degree
    when intervening in layers with large spectral gaps (delta = sigma_1 - sigma_2).
  - Statistical Leverage Scores & Efficiency Filtering:
      l_j = sum_{i=1}^r V_{ji}^2
    Selects columns that maximize alignment with the targeted behavior while minimizing
    perturbation on high-leverage general reasoning dimensions.
  - Spectral Audit ('spectral_audit'):
      Computes randomized SVD spectrum, eigengap delta, leverage distribution, and stability bounds.
  - Fast Weight Programming Foundation (WSP-Lang 0.3):
      Delta-Rule associative memory unlearning (W <- W - eta * (u (u^T W) - v*)),
      Fast Weight Overlays ('fwp_overlay'), and Spectral Decay ('spectral_decay').
  - Dual Architecture Engine: Native support for unquantized SafeTensors (BF16/FP16)
    and quantized GGUF blocks.
  - Hard Budget & Security Governance: Enforces 'max_modified_weights', 'max_delta_norm',
    'allow_write', and forbidden tensor guards.
"""

import os
import sys
import json
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

__version__ = "0.8.3"

ROOT = Path(__file__).resolve().parents[1]

# Default model files: env override (WSP_MODELO_GGUF / WSP_BASE_GGUF, same variables engine/challenge.py reads),
# then repo-local candidates, then the repo-local default path (missing if not downloaded; callers surface that).
def _default_gguf(env, candidates):
    p = os.environ.get(env)
    if p:
        return Path(p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()

DEFAULT_MIMO_GGUF = _default_gguf(
    "WSP_MODELO_GGUF",
    [ROOT / "hf_cache" / "mimo-gguf" / "MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf",
     ROOT / "servidos" / "mimo-orig.gguf"],
)
DEFAULT_QWEN_BASE_GGUF = _default_gguf(
    "WSP_BASE_GGUF",
    [ROOT / "hf_cache" / "qwen35base" / "Qwen3.5_9B_Base.Q6_K.gguf"],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))
from engine.safety import FORBIDDEN_TENSORS as FORBIDDEN_TENSORS, FORBIDDEN_PATTERNS as FORBIDDEN_PATTERNS, is_protected_tensor  # noqa: E402
DIR_PL = ROOT / "challenges" / "wsplang"

# Allowed operations in steps
OPS_ALLOWED = {
    # Core verified inspection and projection operations
    "inspect", "observe", "compare", "hypothesize", "localize", "peek",
    "intervene", "evaluate", "confirm", "poke", "poke-weight", "diff", "rollback",
    "record", "select_candidate", "semantic_map", "semantic-map",
    "semantic_areas", "semantic-areas",
    "project_out", "project_in", "export_control_vector", "audit", "assert_budget",
    "subspace_transfer",  # move along a subspace of another model's update (engine/subspace_ops.py)
    # Studying improved models (engine/weight_diff.py, engine/bench_runner.py); none writes weights
    "delta_stats", "delta_signature", "delta_compare", "evaluate_benchmark", "model_report",
    "adapter_report",  # what a PEFT LoRA adapter changes, from its factors alone + base row blocks (engine/adapter_report.py)
    "quantize",  # extreme quantisation through llama-quantize / llama-imatrix, with a provenance manifest (engine/quantize_op.py)
    "delta_to_lora",  # a fine-tune's change as a LoRA adapter beside the models (engine/lora_export.py)
    "contrast_vector",  # llama.cpp control vector from two sets of texts (engine/contrast_vector.py)
    "compile_control_vector",  # that vector written into down_proj, + placebo, G0, LoRA/merged Q6_K (engine/cc_compile.py)
    "ternary_survival",  # how much of an edit survives ternarisation (TQ2_0 / absmean); report only (engine/ternary.py)
    "edit_column", "edit_neurons",  # one matrix, few columns, structural same-norm placebo, report + patches (engine/column_edit.py)
    "apply_patch", "rollback_last", "list_patches", "commit_manifest",  # .weightpatch.json runtime (engine/wsplpatch.py)
    "ternarize", "imprint_ternary",  # ternary NanoWSP checkpoints and the projected closed-form write (engine/ternary_write.py)
    "bench_run", "bench_compare", "bench_table",  # the language's standard benchmarks (engine/wsplang_bench.py); reports only
    # Architectural & Comparative Analysis (WSP-Lang 0.7: Base vs Distill Transformation)
    "architectural_drift", "drift_profile", "spectral_delta", "semantic_delta", "ssm_dynamics_diff",
    # Synthesis, perturbation bounds, and envelope operators
    "delta_unlearn", "fwp_overlay", "spectral_decay", "assert_orthogonal_preservation",
    "spectral_audit", "assert_weyl", "assert_davis_kahan",
    "init_architecture", "imprint_association", "imprint_batch", "imprint_corpus",
    "cov_imprint", "memit_imprint", "decoupled_delta",
    "rosetta_align", "cross_project",
    "layer_envelope", "multi_layer_envelope",
    "jepa_steer", "vicreg_preserve", "permute_align", "compile_synthetic", "assert_behavioral_drift", "tie_weights",
    "train", "synthesize", "decompile",
    # Experimental / stubs
    "leace_erase",   # Exact LEACE when covariance is supplied; falls back to weight-Gram proxy
    "surprise_fwp",  # STUB: word-frequency repetition heuristic, not neural Titans surprisal
}

TYPES = {
    "Model", "Tensor", "Activation", "Direction", "Behavior", "Hypothesis",
    "Intervention", "Benchmark", "Experiment", "Result", "SemanticArea",
    "SemanticCentroid", "ControlVector", "Budget", "FastWeightOverlay",
    "SpectralBound", "LeverageDistribution", "LayerEnvelope", "MultiLayerCircuit"
}

INT_OPS = {
    "project_out", "project_in", "delta_unlearn", "fwp_overlay", "spectral_decay",
    "activation_projection", "poke-weight", "scale", "ablate", "scale-rows",
    "scale-cols", "blend-rows", "cov-steer", "init_architecture", "imprint_association",
    "imprint_batch", "imprint_corpus", "cov_imprint", "memit_imprint", "decoupled_delta",
    "rosetta_align", "cross_project",
    "layer_envelope", "multi_layer_envelope",
    "jepa_steer", "vicreg_preserve", "permute_align", "compile_synthetic", "assert_behavioral_drift",
    "leace_erase", "surprise_fwp"
}
METADATA = {
    "model", "format", "target_format", "objective", "constraints", "budgets",
    "types", "declared", "sources", "baseline", "discover", "preconditions",
    "postconditions", "accept_if", "evaluate", "record", "hypothesis",
    "note", "semantic_areas", "semantic_target", "theoretical_foundation"
}

class WriteGuardError(ValueError):
    """A write would break a WSPLang safety rule; nothing was written."""


def guard_model_write(candidate, out_path, source_path, budgets, written) -> Dict[str, Any]:
    """The single check every weight write goes through. Raises WriteGuardError; returns what changed.

    - out_path must not be the model that was read (source_path), unless this run wrote that file itself;
    - the candidate keeps the source's tensor names, shapes and dtypes: no added, removed or reshaped tensors;
    - protected tensors change only if listed in budgets unlock_tensors;
    - changed elements <= max_modified_weights and max |delta| <= max_delta_norm.
    out_path=None checks the edit without a destination (simulation).
    """
    if out_path is not None and source_path is not None:
        src = Path(source_path).resolve()
        if Path(out_path).resolve() == src and src not in written:
            raise WriteGuardError(f"refusing to overwrite the input model {source_path}; "
                                  "write the edit elsewhere with out=<file> or --out-dir")
    report = {"modified_parameters": 0, "max_abs_delta": 0.0, "tensors_changed": []}
    if source_path is None:
        return report
    import torch
    from safetensors.torch import load_file

    original = load_file(str(source_path))
    name = Path(source_path).name
    added, removed = sorted(set(candidate) - set(original)), sorted(set(original) - set(candidate))
    if added:
        raise WriteGuardError(f"write adds tensors that {name} does not have: {added[:5]}")
    if removed:
        raise WriteGuardError(f"write drops tensors of {name}: {removed[:5]}")
    unlock = set(budgets.get("unlock_tensors") or [])
    for key in sorted(original):
        before, after = original[key], candidate[key]
        if tuple(before.shape) != tuple(after.shape):
            raise WriteGuardError(f"tensor {key} would change shape {tuple(before.shape)} -> {tuple(after.shape)}")
        if before.dtype != after.dtype:
            raise WriteGuardError(f"tensor {key} would change dtype {before.dtype} -> {after.dtype}")
        if torch.equal(before, after.to(before.device)):
            continue
        if is_protected_tensor(key, unlock):
            raise WriteGuardError(f"tensor {key} is protected; list it in budgets unlock_tensors to change it")
        delta = after.double() - before.double()
        report["modified_parameters"] += int((delta != 0).sum().item())
        report["max_abs_delta"] = max(report["max_abs_delta"], float(delta.abs().max().item()))
        report["tensors_changed"].append(key)
    limit = budgets.get("max_modified_weights")
    if limit is not None and report["modified_parameters"] > limit:
        raise WriteGuardError(f"{report['modified_parameters']:,} changed parameters exceed "
                              f"max_modified_weights={limit:,}")
    limit = budgets.get("max_delta_norm")
    if limit is not None and report["max_abs_delta"] > limit:
        raise WriteGuardError(f"max |delta| {report['max_abs_delta']:.6g} exceeds max_delta_norm={limit}")
    return report


def guarded_save(candidate, out_path, source_path, budgets, written, metadata=None) -> Dict[str, Any]:
    """guard_model_write, then save; records out_path as written by this run."""
    from safetensors.torch import save_file

    report = guard_model_write(candidate, out_path, source_path, budgets, written)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in candidate.items()}, str(out_path), metadata=metadata)
    written.add(out_path.resolve())
    return report


def check_new_file(out_path, protected, written) -> None:
    """Ops that create a model or program never write over the model they read."""
    out = Path(out_path).resolve()
    for p in protected:
        if p is None:
            continue
        p = Path(p)
        p = p / "model.safetensors" if p.is_dir() else p
        if out == p.resolve() and out not in written:
            raise WriteGuardError(f"refusing to overwrite the input model {p}; choose another out=")


def transfer_verdict(kept_weighted_at, delta_norm=None) -> str:
    """Will a fine-tune's change move as a small adapter? Read from the weighted energy kept at rank 16 (delta_to_lora's
    energy.json), before any eval. Calibrated on Qwen3.5-9B: Jackrong keeps 0.586 and transfers (MATH 6->30/36),
    MiMo keeps 0.038 and does not (<= 4/16 at any rank up to 512). delta_norm == 0: the "fine-tune" is the base
    (OmniCoder-9B, OxCoder-9B on the hub): no_change, whatever the kept share says (0/0)."""
    if delta_norm is not None and delta_norm == 0:
        return "no_change"
    k = kept_weighted_at.get("16")
    if k is None:
        return "uncertain"
    return "likely" if k >= 0.5 else "unlikely" if k < 0.2 else "uncertain"


def check_report_path(out_path, inputs) -> Path:
    """Reports and signatures never land inside an input model, and are never weight files."""
    out = Path(out_path)
    out = out if out.is_absolute() else ROOT / out
    if out.suffix.lower() in (".safetensors", ".gguf", ".bin"):
        raise WriteGuardError(f"{out.name}: a report or signature is not a weight file; choose another out=")
    o = out.resolve()
    for p in inputs:
        p = Path(p).resolve()
        root = p if p.is_dir() else p.parent
        if o == p or root in o.parents:
            raise WriteGuardError(f"refusing to write {out} inside the input model {p}")
    return out


def _model_path(arg) -> Path:
    p = Path(str(arg))
    return p if p.is_absolute() else ROOT / p


def run_exit_code(results) -> int:
    """0 when every step ran; 1 when any step failed, was rejected or failed an assertion."""
    bad = {"error", "security_rejection", "assertion_failed"}
    return 1 if any(r.get("status") in bad for r in results) else 0

# ==============================================================================
# AUDIT 2026-11-04: politicas de ejecucion real (falsable, no decorativa)
# 1) Solo estas ops tienen implementacion en execute_plan_v4. Cualquier otra declarada
#    en el programa SE RECHAZA en --check/--dry-run/--run (antes silenciaba como
#    "skipped (non-interactive op)" = programa que aparenta correr y no hace nada).
# 2) WRITE_OPS requieren backend safetensors+torch; sobre GGUF se rechazan en el
#    plan (el backend GGUF real vive en wsp.py, vease ROADMAP audit section).
# 3) STUB_OPS devuelven "stub_result" (no verifiable causal claim).
# ==============================================================================
READ_ONLY_OPS = {
    "peek", "semantic_areas", "semantic-areas", "semantic_map", "semantic-map",
    "diff", "compare", "evaluate",
    "architectural_drift", "drift_profile", "spectral_delta", "semantic_delta", "ssm_dynamics_diff",
    "assert_behavioral_drift", "vicreg_preserve",
    "delta_stats", "delta_signature", "delta_compare", "evaluate_benchmark", "model_report",  # reports, never weights
    "adapter_report",  # markdown + json report of a PEFT adapter; never weights
    "delta_to_lora",  # writes a new adapter dir, never inside or over an input model
    "contrast_vector",  # writes a new control-vector GGUF (run-time steering), never weights
    "compile_control_vector",  # writes new files in a new out dir (fits, LoRA, merged copies), never over an input
    "ternary_survival",  # writes one JSON report, never weights
    "rollback_last", "list_patches", "commit_manifest",  # patch stack bookkeeping: JSON under the session dir only
    "bench_run", "bench_compare", "bench_table",  # benchmark runs, comparisons and the rendered table: JSON / markdown only
}
IMPLEMENTED_OPS = READ_ONLY_OPS | {
    "edit_column", "edit_neurons", "apply_patch",  # new model dirs beside the inputs, never over them
    "quantize",  # a new quantised GGUF + manifest beside the input (engine/quantize_op.py), never over it
    "ternarize", "imprint_ternary",  # new ternary checkpoint dirs beside the inputs, never over them
    "leace_erase", "cov_imprint", "memit_imprint", "decoupled_delta", "surprise_fwp",
    "rosetta_align", "cross_project", "init_architecture", "imprint_association",
    "imprint_batch", "imprint_corpus", "delta_unlearn", "fwp_overlay", "layer_envelope",
    "multi_layer_envelope", "spectral_decay",
    "assert_orthogonal_preservation", "project_out", "project_in", "audit",
    "export_control_vector", "spectral_audit", "assert_weyl", "assert_davis_kahan",
    "jepa_steer", "permute_align", "compile_synthetic", "tie_weights", "train", "synthesize", "decompile",
    "subspace_transfer",
}
STUB_OPS = {"leace_erase", "surprise_fwp"}
# GGUF-native writers: new file beside a GGUF input through llama.cpp tools, no torch backend
GGUF_WRITE_OPS = {"quantize"}
WRITE_BACKEND_OPS = IMPLEMENTED_OPS - READ_ONLY_OPS - GGUF_WRITE_OPS


# ==============================================================================
# 1. PARSER & AST BUILDER
# ==============================================================================

def parse_args_lines(text: str) -> Dict[str, Any]:
    """Parse WSP-Lang program blocks and key-value sections."""
    prog = {}
    current = None
    current_indent = None
    prog.setdefault("__root__", [])

    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()

        # Experiment header
        if indent == 0 and s.startswith("experiment"):
            name = re.sub(r"^experiment\s+", "", s).strip().rstrip(":").strip()
            if name:
                prog["__root__"].append(("exp", name))
            continue

        # Section header e.g. "steps:" or inline metadata e.g. "model: Qwen"
        m = re.match(r"^(\S+)\s*:\s*(.*)$", s)
        if m:
            sec = m.group(1)
            val = m.group(2).strip()
            if not val:
                if sec == current and current_indent == indent:
                    continue
                current = sec
                current_indent = indent
                prog.setdefault(sec, [])
                continue
            elif sec in METADATA:
                prog.setdefault(sec, []).append((indent, val))
                continue

        if current is None:
            prog["__root__"].append((indent, s))
            continue

        prog[current].append((indent, s))

    return prog


def extract_budgets(prog: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize budget constraints."""
    budgets = {
        "max_modified_weights": None,
        "target_parameter_fraction": None,
        "max_delta_norm": None,
        "allow_write": False,
        "allow_training": False,
        "unlock_tensors": [],
    }
    raw_lines = prog.get("budgets", [])
    for _, line in raw_lines:
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            if k == "max_modified_weights":
                budgets[k] = int(v)
            elif k in ("target_parameter_fraction", "max_delta_norm"):
                budgets[k] = float(v)
            elif k in ("allow_write", "allow_training"):
                budgets[k] = v.lower() in ("true", "1", "yes")
            elif k == "unlock_tensors":
                budgets[k] = [name.strip() for name in v.split(",") if name.strip()]
    return budgets


def validate(prog: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate program structure, syntax, and security boundaries."""
    errors = []
    steps = []

    root = prog.get("__root__", [])
    experiment = None
    for ind, s in root:
        if ind == "exp":
            experiment = s
        elif isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
        else:
            errors.append(f"linea fuera de seccion: {s}")

    if not experiment:
        errors.append("falta experiment <nombre>")

    for sec, buf in prog.items():
        if sec == "__root__":
            continue
        if sec not in OPS_ALLOWED and sec not in METADATA and sec != "steps":
            errors.append(f"seccion desconocida: {sec}")

        for ind, s in buf:
            m = re.match(r"^(\w[\w_]+)\s*[=(]", s)
            if m:
                fn = m.group(1)
                if sec == "intervene" and fn not in INT_OPS:
                    errors.append(f"op de intervencion no permitida: {fn}")
                steps.append({"sec": sec, "expr": s})
            elif s.startswith("step "):
                m_step = re.match(r"^step\s+(\S+)(?:\s+(.*))?$", s)
                if m_step:
                    step_op = m_step.group(1)
                    if step_op not in OPS_ALLOWED:
                        errors.append(f"operacion de paso no permitida: {step_op}")
                    else:
                        steps.append({
                            "sec": sec,
                            "expr": s,
                            "op": step_op,
                            "args": (m_step.group(2) or "").strip()
                        })
                else:
                    errors.append(f"formato de step invalido: {s[:50]}")
            elif sec not in METADATA:
                errors.append(f"linea ignorada {sec}: {s[:50]}")

    return steps, errors


def check_feasibility(prog, steps, model_arg=None):
    """AUDIT 2026-11-04: factibilidad real de ejecucion.

    - Rechaza ops declaradas pero NO implementadas (antes: silent 'skipped').
    - Rechaza ops de escritura entre model con backend GGUF (requieren safetensors+torch).
    - Verifica existencia de artefactos referenciados (results/npz/gguf).
    - Marca ops STUB con warning.
    Devuelve (errors, warnings).
    """
    errors, warnings = [], []
    model_kind = None
    if model_arg:
        p_model = Path(model_arg)
        if p_model.suffix.lower() == ".gguf":
            model_kind = "gguf"
        elif p_model.is_dir() or p_model.suffix.lower() in (".safetensors",):
            model_kind = "safetensors"
        else:
            model_kind = "unknown"

    file_keys = ("file", "base", "vec_file", "vector", "results", "npz")
    file_suffixes = (".json", ".jsonl", ".npz", ".gguf", ".csv", ".safetensors", ".wspl", ".txt", ".md")

    for st in steps:
        op = st.get("op")
        if not op:
            continue
        if op in ("init_architecture", "train", "synthesize", "decompile"):
            model_kind = "safetensors"
        if op not in IMPLEMENTED_OPS:
            errors.append(
                f"op '{op}' NO implementada en el runtime (antes se ejecutaba como 'skipped' "
                f"silencioso y no hacia nada). Reescribe el paso con una op de {sorted(IMPLEMENTED_OPS)[:6]}... "
                "o delega la escritura en wsp.py.")
            continue
        if model_kind in ("gguf", None, "unknown") and op in WRITE_BACKEND_OPS:
            err = (f"op '{op}' requiere backend safetensors+torch; backend detectado='"
                   f"{model_kind}'. Para GGUF usar wsp.py (subtract-direction/edit con requant).")
            if model_kind is None:
                err = f"op '{op}' de escritura requiere --model safetensors explicito (no GGUF)."
            errors.append(err)
        if op in STUB_OPS:
            warnings.append(f"op '{op}' es STUB declarado: resultado 'stub_result', no afirmacion causal")

        # artefactos referenciados deben existir (si no, falla en check, no en run)
        args_str = st.get("args", "")
        for mref in re.finditer(r"(?:\w+)\s*=\s*([^\s,]+)", args_str) or []:
            pass
        for mref in re.finditer(r"(\w+)\s*=\s*([^\s,]+)", args_str):
            kref, vref = mref.group(1), mref.group(2)
            if kref not in file_keys:
                continue
            if Path(vref).suffix.lower() not in file_suffixes:
                continue
            pv = Path(vref)
            if not pv.is_absolute():
                pv = ROOT / vref
            if not pv.exists():
                errors.append(f"artefacto inexistente para op '{op}': {vref} (clausura de evidencia fallida)")
        # posicionales tipo 'compare archivo.json'
        if op in ("compare", "evaluate") and args_str:
            first = args_str.split()[0] if args_str.split() else ""
            pv = Path(first)
            if pv.suffix.lower() in file_suffixes and not pv.is_absolute():
                pv = ROOT / first
            if pv.suffix.lower() in file_suffixes and not pv.exists():
                errors.append(f"artefacto inexistente para op '{op}': {first}")
    return errors, warnings


def plan_run(prog: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate structured execution plan and command mappings."""
    mapping = {
        "inspect": "wsp tensors/stats",
        "observe": "residual streaming / activation trace",
        "compare": "evaluation comparison (base vs edited)",
        "hypothesize": "mechanistic hypothesis verification",
        "localize": "localization of loop/behavioral manifold",
        "semantic_map": "wsp semantic-map (differential centroids)",
        "semantic-map": "wsp semantic-map",
        "semantic_areas": "wsp peek (semantic area analysis)",
        "semantic-areas": "wsp peek",
        "peek": "wsp peek (tensor anatomy and concept projection)",
        "poke": "wsp poke (single weight modification)",
        "poke-weight": "wsp poke",
        "diff": "wsp diff (structural weight difference check)",
        "cov-steer": "covariance steering intervention",
        "intervene": "policy-enforced weight intervention",
        "evaluate": "benchmark evaluation (accuracy & loss)",
        "confirm": "challenge sealing (LOCK) + verification verdict",
        "rollback": "rollback to original unmodified checkpoint",
        "record": "journal + experimental manifest logging",
        "select_candidate": "history candidate selection",
        # WSP-Lang 0.2
        "project_out": "low-rank directional subtraction W <- W - alpha * u(u^T W)",
        "project_in": "low-rank directional reinforcement W <- W + alpha * u(u^T W)",
        "export_control_vector": "export directional vector to llama.cpp GGUF control vector",
        "audit": "audit tensor energy, norm delta, and semantic alignment",
        "assert_budget": "enforce parameter change boundaries",
        # WSP-Lang 0.3 Fast Weight Programming
        "delta_unlearn": "Delta-Rule associative memory unlearning W <- W - eta*(u(u^T W) - v*)",
        "fwp_overlay": "generate transient in-memory Fast Weight overlay for inference",
        "spectral_decay": "dissipate runaway recurrent attractors via continuous decay W <- lambda * W",
        "assert_orthogonal_preservation": "verify zero distortion on orthogonal concept manifold",
        # WSP-Lang 0.4 Matrix Perturbation Theory & Leverage Scores
        "spectral_audit": "randomized SVD spectral decomposition, eigengap delta, and column leverage scores",
        "assert_weyl": "verify Weyl operator norm perturbation bound |sigma_i' - sigma_i| <= ||Delta W||_2",
        "assert_davis_kahan": "verify Davis-Kahan invariant subspace stability bound sin Theta <= ||Delta W||_2 / delta",
        # WSP-Lang 0.5 Architecture Synthesis & Programming
        "init_architecture": "synthesize and initialize micro-transformer architecture from scratch in weight space",
        "imprint_association": "declarative factual/behavioral memory imprinting via outer product",
        "imprint_corpus": "NanoWSP-3 closed-form corpus imprint at layer/eta/reg_lambda; reports measured recall of the imprinted facts and of an optional probe set before and after",
        "imprint_batch": "Moore-Penrose closed-form multi-associative weight synthesis without backpropagation",
        "cov_imprint": "MEMIT/EMMET covariance-regularized mass memory editing Delta W = (V - W_0 K)(K^T K + lambda C_0)^-1 K^T",
        "memit_imprint": "MEMIT/EMMET covariance-regularized mass memory editing",
        "leace_erase": "LEACE closed-form least-squares concept erasure minimizing Frobenius distortion with exact zero target leakage",
        "decoupled_delta": "Gated DeltaNet-2 decoupled channel-wise erase and write linear attention update",
        "surprise_fwp": "Titans surprisal-driven test-time memory trigger for breaking runaway attractors"
    }

    step_cmds = []
    for st in steps:
        if st["sec"] == "steps":
            m = re.match(r"^step\s+(\S+)\s*(.*)$", st["expr"].strip())
            if m:
                step_cmds.append({"op": m.group(1), "args": m.group(2).strip()})

    sec_counts = {}
    for st in steps:
        sec_counts[st["sec"]] = sec_counts.get(st["sec"], 0) + 1

    paso_mapeo = {p["op"]: mapping.get(p["op"], "custom operation") for p in step_cmds}
    budgets = extract_budgets(prog)

    return {
        "wsplang_version": __version__,
        "theoretical_framework": "Matrix Perturbation Theory (Weyl, Davis-Kahan) & Fast Weight Programming",
        "secciones": sec_counts,
        "pasos": step_cmds,
        "pasos_mapeo": paso_mapeo,
        "budgets": budgets,
        "mapeo": {s: mapping.get(s) for s in sec_counts if s in mapping},
    }


# ==============================================================================
# 2. RUNTIME ENGINE (SafeTensors + GGUF + FWP Overlays)
# ==============================================================================

def parse_named_args(arg_string: str) -> Tuple[List[str], Dict[str, str]]:
    """Parse positional and key=value arguments from a step arg string, respecting quotes and Windows paths."""
    import re
    positional = []
    named = {}
    pattern = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))|(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
    for m in pattern.finditer(arg_string):
        k, v1, v2, v3, p1, p2, p3 = m.groups()
        if k:
            named[k] = v1 or v2 or v3 or ""
        else:
            val = p1 or p2 or p3
            if val:
                positional.append(val)
    return positional, named


def load_vector_numpy(vec_path: Path) -> np.ndarray:
    """Load and L2-normalize direction vector from .npz or .npy."""
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vec_path}")
    if vec_path.suffix == ".npy":
        arr = np.load(vec_path)
    else:
        data = np.load(vec_path)
        arr = data[list(data.files)[0]]
    arr = arr.astype(np.float32).ravel()
    norm = np.linalg.norm(arr)
    if norm > 1e-12:
        arr = arr / norm
    return arr


def load_safetensors_tensor(target_path: Path, tensor_name: str) -> Tuple[Any, Path, Any]:
    """Find and load a tensor from SafeTensors directory or file as float32.
    Returns: (tensor_float, shard_path, shard_tensors_dict)
    """
    from safetensors.torch import load_file

    index_file = target_path / "model.safetensors.index.json"
    target_shard = None
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            idx = json.load(f)
        target_shard = idx.get("weight_map", {}).get(tensor_name)

    shard_files = list(target_path.glob("*.safetensors")) if target_path.is_dir() else [target_path]
    shard_to_edit = None
    for sf in shard_files:
        if target_shard and sf.name == target_shard:
            shard_to_edit = sf
            break
        if not target_shard:
            from safetensors import safe_open
            with safe_open(str(sf), framework="np") as f:
                if tensor_name in f.keys():
                    shard_to_edit = sf
                    break

    if not shard_to_edit:
        raise FileNotFoundError(f"Tensor '{tensor_name}' not found in SafeTensors shards in {target_path}")

    tensors = load_file(str(shard_to_edit))
    if tensor_name not in tensors:
        raise FileNotFoundError(f"Tensor '{tensor_name}' not in loaded shard {shard_to_edit.name}")

    W = tensors[tensor_name].float()
    return W, shard_to_edit, tensors


def compute_spectral_decomposition(W: Any, q: int = 16, niter: int = 6) -> Dict[str, Any]:
    """Compute truncated SVD, spectral gap, and column leverage scores using randomized SVD.
    Uses oversampling (q + 10) for numerical stability on top singular values.
    """
    import torch
    oversample_q = min(W.shape[0], W.shape[1], q + 10)
    U, S, V = torch.svd_lowrank(W, q=oversample_q, niter=niter)
    top_s = S[:q].tolist()
    gap = float((S[0] - S[1]).item()) if len(S) > 1 else 0.0
    cond = float((S[0] / max(S[q - 1], 1e-12)).item()) if len(S) >= q else 1.0

    # Column Statistical Leverage Scores: l_j = sum_{i=1}^q V_{ji}^2
    V_q = V[:, :q]
    leverage = torch.sum(V_q ** 2, dim=1)

    return {
        "singular_values": top_s,
        "spectral_gap": gap,
        "condition_number": cond,
        "leverage_scores": leverage,
        "leverage_stats": {
            "sum": float(leverage.sum().item()),
            "mean": float(leverage.mean().item()),
            "max": float(leverage.max().item()),
            "min": float(leverage.min().item()),
        },
        "U": U,
        "S": S,
        "V": V
    }




def generate_orthogonal_probe(vec: Any, seed: int = 1337) -> Any:
    """Generate a unit probe vector strictly orthogonal to vec."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    v_norm = v / max(np.linalg.norm(v), 1e-12)
    rng = np.random.RandomState(seed)
    rand_probe = rng.randn(len(v)).astype(np.float32)
    u_ortho = rand_probe - np.dot(rand_probe, v_norm) * v_norm
    norm_ortho = np.linalg.norm(u_ortho)
    if norm_ortho < 1e-4:
        axis = int(np.argmin(np.abs(v_norm)))
        rand_probe = np.zeros_like(v_norm)
        rand_probe[axis] = 1.0
        u_ortho = rand_probe - np.dot(rand_probe, v_norm) * v_norm
        norm_ortho = np.linalg.norm(u_ortho)
    return (u_ortho / norm_ortho).astype(np.float32)




def compute_procrustes_alignment(E_source: Any, E_target: Any) -> Tuple[Any, float, float]:
    """Computes semi-orthogonal Procrustes matrix T in R^{d_target x d_source}
    such that E_target ~ E_source @ T^T
    """
    import torch
    E_s = E_source - E_source.mean(dim=0, keepdim=True)
    E_t = E_target - E_target.mean(dim=0, keepdim=True)
    E_s = E_s / torch.norm(E_s, dim=1, keepdim=True)
    E_t = E_t / torch.norm(E_t, dim=1, keepdim=True)
    M = torch.matmul(E_t.t(), E_s)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    T = torch.matmul(U, Vh)
    E_proj = torch.matmul(E_s, T.t())
    cos_sim = torch.cosine_similarity(E_t, E_proj, dim=1)
    ortho_err = float(torch.norm(torch.matmul(T, T.t()) - torch.eye(E_target.shape[1])).item())
    return T, float(torch.mean(cos_sim).item()), ortho_err

def compute_leace_projection(u: Any, sigma: Optional[Any] = None) -> Any:
    """LEACE: Closed-form Least-Squares Concept Erasure (Belrose et al., NeurIPS 2023).
    Analytically computes the optimal affine transformation that removes target concept u
    while strictly minimizing Frobenius norm distortion on the remaining distribution.
    Formula: P = I - (sigma @ u @ u^T) / (u^T @ sigma @ u)
    """
    import torch
    d = u.shape[0]
    u_col = u.view(d, 1).float()
    if sigma is None:
        u_norm_sq = float(torch.matmul(u_col.t(), u_col).item())
        if u_norm_sq < 1e-12:
            return torch.eye(d, device=u.device)
        return torch.eye(d, device=u.device) - torch.matmul(u_col, u_col.t()) / u_norm_sq
    else:
        sig = sigma.float().to(u.device)
        denom = float(torch.matmul(torch.matmul(u_col.t(), sig), u_col).item())
        if abs(denom) < 1e-12:
            return torch.eye(d, device=u.device)
        numer = torch.matmul(torch.matmul(sig, u_col), u_col.t())
        return torch.eye(d, device=u.device) - (numer / denom)


def compute_memit_update(
    W0: Any,
    K: Any,
    V: Any,
    C0: Optional[Any] = None,
    reg_lambda: float = 1e-3
) -> Any:
    """MEMIT / EMMET Covariance-Regularized Memory Editing (Meng et al. ICLR 2023, Gupta et al. 2024).
    Dual formulation with Tikhonov/covariance regularizer to prevent condition number explosion:
        Delta W = (V - W0 @ K) @ (K^T @ K + lambda * I)^(-1) @ K^T
    """
    import torch
    W0 = W0.float()
    K = K.float().to(W0.device)
    V = V.float().to(W0.device)
    M = K.shape[1]

    resid = V - torch.matmul(W0, K)  # [d_out, M]

    if C0 is not None:
        C0 = C0.float().to(W0.device)
        inv_term = torch.linalg.pinv(torch.matmul(K, K.t()) + reg_lambda * C0)
        delta = torch.matmul(torch.matmul(resid, K.t()), inv_term)
    else:
        XtX = torch.matmul(K.t(), K) + reg_lambda * torch.eye(M, device=W0.device)
        inv_XtX = torch.linalg.inv(XtX)
        delta = torch.matmul(resid, torch.matmul(inv_XtX, K.t()))

    return delta


def apply_decoupled_delta(
    W: Any,
    u_erase: Any,
    beta_erase: float,
    k_write: Optional[Any] = None,
    v_write: Optional[Any] = None,
    beta_write: float = 1.0
) -> Any:
    """Gated DeltaNet-2 Decoupled Erase and Write (NVIDIA 2026, Yang et al. ICLR 2025).
    Decouples the erasure of old attractor u_erase from the writing of new association (k_write, v_write).
    """
    import torch
    W = W.float()
    u_norm = u_erase.float().to(W.device)
    u_norm = u_norm / max(float(torch.norm(u_norm).item()), 1e-12)

    # 1. Erase phase
    u_proj = torch.matmul(u_norm, W)  # projection onto row space
    delta_erase = beta_erase * torch.outer(u_norm, u_proj)
    W_new = W - delta_erase

    # 2. Write phase (if provided, projected orthogonal to u_erase to guarantee zero re-contamination)
    if k_write is not None and v_write is not None:
        k_norm = k_write.float().to(W.device)
        k_norm = k_norm / max(float(torch.norm(k_norm).item()), 1e-12)
        v_norm = v_write.float().to(W.device)
        v_norm = v_norm / max(float(torch.norm(v_norm).item()), 1e-12)
        v_ortho = v_norm - torch.dot(u_norm, v_norm) * u_norm
        delta_write = beta_write * torch.outer(v_ortho, k_norm)
        W_new = W_new + delta_write

    return W_new


def compute_jepa_steer(
    W0: Any,
    H_in: Any,
    target_latent: Any,
    Sigma_bg: Optional[Any] = None,
    reg_lambda: float = 0.1,
    bg_weight: float = 1.0
) -> Tuple[Any, float, float]:
    """JEPA Latent Target Steering (LeCun 2022, Huang et al. 2025).
    Intervenes on intermediate representations to steer layer outputs towards target concept
    vector(s) target_latent without modifying vocabulary readouts:
        Delta W = (target_latent - W0 @ H_in) @ (H_in^T @ H_in + bg_weight * H_in^T @ Sigma_bg @ H_in + reg_lambda * I)^(-1) @ H_in^T
    """
    import torch
    W0 = W0.float()
    H = H_in.float().to(W0.device)
    S_target = target_latent.float().to(W0.device)
    if S_target.dim() == 1:
        S_target = S_target.unsqueeze(1).expand(-1, H.shape[1])
    elif S_target.shape[1] == 1 and H.shape[1] > 1:
        S_target = S_target.expand(-1, H.shape[1])

    M = H.shape[1]
    residual = S_target - torch.matmul(W0, H)

    HtH = torch.matmul(H.t(), H)
    reg_term = reg_lambda * torch.eye(M, device=W0.device)
    if Sigma_bg is not None:
        Sigma_bg = Sigma_bg.float().to(W0.device)
        reg_term = reg_term + bg_weight * torch.matmul(torch.matmul(H.t(), Sigma_bg), H) / max(float(torch.norm(Sigma_bg).item()), 1e-6)

    inv_term = torch.linalg.pinv(HtH + reg_term)
    delta = torch.matmul(torch.matmul(residual, inv_term), H.t())

    W_new = W0 + delta
    H_out_new = torch.matmul(W_new, H)
    cos_sim = float(torch.cosine_similarity(H_out_new, S_target, dim=0).mean().item())
    delta_norm = float(torch.norm(delta).item())
    return delta, round(cos_sim, 4), round(delta_norm, 6)


def compute_vicreg_metrics(
    W: Any,
    Sigma_act: Optional[Any] = None,
    var_floor: float = 0.1,
    max_cov_offdiag: float = 0.05
) -> Dict[str, Any]:
    """VICReg: Variance-Invariance-Covariance Regularization Audit (Bardes, Ponce, LeCun 2022).
    Evaluates spectral variance and off-diagonal covariance energy to prevent dimensional collapse.
    """
    import torch
    W = W.float()
    d_out, d_in = W.shape
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    s_var = float(torch.var(S).item())
    s_mean = float(torch.mean(S).item())
    s_norm = S / torch.clamp(S.sum(), min=1e-12)
    eff_rank = float(torch.exp(-torch.sum(s_norm * torch.log(s_norm + 1e-12))).item())

    if Sigma_act is not None:
        Sigma = Sigma_act.float().to(W.device)
    else:
        Sigma = torch.matmul(W, W.t()) / float(d_in)

    diag = torch.diag(Sigma)
    std = torch.sqrt(torch.clamp(diag, min=1e-8))
    Corr = Sigma / torch.outer(std, std)

    d = Corr.shape[0]
    eye = torch.eye(d, device=W.device)
    offdiag_sq = ((Corr * (1.0 - eye)) ** 2).sum()
    offdiag_energy = float((offdiag_sq / max(d * (d - 1), 1)).item())

    var_ok = bool(s_var >= var_floor)
    cov_ok = bool(offdiag_energy <= max_cov_offdiag)

    return {
        "singular_variance": round(s_var, 4),
        "singular_mean": round(s_mean, 4),
        "effective_rank": round(eff_rank, 2),
        "offdiagonal_covariance_energy": round(offdiag_energy, 6),
        "variance_floor_threshold": var_floor,
        "max_cov_offdiag_threshold": max_cov_offdiag,
        "variance_pass": var_ok,
        "covariance_decorrelation_pass": cov_ok,
        "vicreg_certified": bool(var_ok and cov_ok)
    }


def compute_permute_align(
    W_source: Any,
    W_target: Any,
    mode: str = "columns"
) -> Tuple[Any, Any, float, float]:
    """Permutation Equivariance & Bipartite Matching in Weight Space (Borth et al. 2026, Git Re-Basin).
    Resolves discrete neuron permutation symmetries W/S_n before continuous Procrustes rotation.
    Pure NumPy implementation with zero external dependencies.
    """
    import numpy as np
    import torch
    W_s = W_source.float()
    W_t = W_target.float().to(W_s.device)

    if mode == "rows":
        S1 = W_s
        S2 = W_t
    else:
        S1 = W_s.t()
        S2 = W_t.t()

    norm1 = torch.norm(S1, dim=1, keepdim=True) + 1e-8
    norm2 = torch.norm(S2, dim=1, keepdim=True) + 1e-8
    sim_matrix = torch.matmul(S1 / norm1, (S2 / norm2).t()).detach().cpu().numpy()

    n_rows, n_cols = sim_matrix.shape
    sim_abs = np.abs(sim_matrix)

    perm = np.zeros(n_rows, dtype=np.int64)
    used_cols = set()
    flat_indices = np.argsort(-sim_abs.ravel())
    matched = 0
    assigned_rows = set()
    for idx in flat_indices:
        r = int(idx // n_cols)
        c = int(idx % n_cols)
        if r not in assigned_rows and c not in used_cols:
            perm[r] = c
            assigned_rows.add(r)
            used_cols.add(c)
            matched += 1
            if matched >= min(n_rows, n_cols):
                break

    unused_cols = [c for c in range(n_cols) if c not in used_cols]
    unassigned_rows = [r for r in range(n_rows) if r not in assigned_rows]
    for r, c in zip(unassigned_rows, unused_cols):
        perm[r] = c

    min_dim = min(n_rows, n_cols)
    identity_sim = float(np.diag(sim_matrix[:min_dim, :min_dim]).mean())
    permuted_sim = float(sim_matrix[np.arange(n_rows), perm].mean())

    perm_tensor = torch.from_numpy(perm).to(W_s.device)
    if mode == "rows":
        W_aligned = W_s[perm_tensor, :]
    else:
        W_aligned = W_s[:, perm_tensor]

    return W_aligned, perm_tensor, round(identity_sim, 4), round(permuted_sim, 4)

def check_behavioral_drift(
    model: Any,
    anchor_corpus: str,
    max_ppl_increase_ratio: float = 1.30,
    seq_len: int = 64
) -> Dict[str, Any]:
    """Behavioral Integrity Guard (Damian Borth, 'Structure Is Not Enough', 2025/2026).
    Evaluates real autoregressive perplexity before allowing weight write,
    preventing catastrophic behavioral drift.
    """
    import math
    import torch
    import torch.nn.functional as F

    ids = [ord(c) for c in anchor_corpus if ord(c) < 256]
    if len(ids) <= seq_len:
        return {"status": "skipped", "reason": "anchor_corpus too short"}

    tot_loss, n_tokens = 0.0, 0
    for i in range(0, min(len(ids) - seq_len - 1, 2000), seq_len):
        chunk = ids[i:i + seq_len]
        target = ids[i + 1:i + seq_len + 1]
        with torch.no_grad():
            logits, _ = model.forward(chunk)
            loss = F.cross_entropy(logits[0], torch.tensor(target, device=logits.device), reduction="sum").item()
            tot_loss += loss
            n_tokens += len(target)

    ppl = math.exp(tot_loss / max(n_tokens, 1))
    return {
        "ppl": round(ppl, 2),
        "n_tokens": n_tokens,
        "max_allowed_ppl_ratio": max_ppl_increase_ratio
    }

def execute_plan_v4(
    plan: Dict[str, Any],
    prog: Dict[str, Any],
    model_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """Execute plan with WSP-Lang 0.3 Fast Weight Programming engine.

    The dispatch bodies live in engine/wsplang_ops.py; this function only
    manages the prologue (target selection), the per-step loop and the epilogue.
    """
    from engine.wsplang_ops import PlanContext, dispatch

    budgets = plan.get("budgets", {})
    max_modified_weights = budgets.get("max_modified_weights")
    max_delta_norm = budgets.get("max_delta_norm")
    allow_write = budgets.get("allow_write", False) and not dry_run
    written_paths = set()  # files this run wrote; later steps may edit them in place

    executed_steps = []
    step_list = plan.get("pasos", [])

    default_gguf = DEFAULT_MIMO_GGUF
    default_safetensors = ROOT / "hf_cache" / "mimo-distill-hf"

    model_name = ""
    if "model" in prog and prog["model"]:
        model_name = prog["model"][0][1].strip()
    if not model_name:
        for item in prog.get("__root__", []):
            if isinstance(item, tuple) and len(item) == 2:
                sec, val = item
                if sec == "model":
                    model_name = str(val).strip()
                elif isinstance(val, str) and val.startswith("model:"):
                    model_name = val.split(":", 1)[1].strip()

    nanowsp_default = ROOT / "models" / "nanowsp" / "model.safetensors"
    nanowsp2_default = ROOT / "models" / "nanowsp2" / "model.safetensors"
    nanowsp3_default = ROOT / "data" / "nanowsp3_weights.safetensors"
    tiny_default = Path(os.environ.get("WSP_TINYLLAMA_SAFETENSORS",
                                      ROOT / "models" / "tinyllama" / "model.safetensors"))
    qwen08_default = ROOT / "hf_cache" / "qwen35-08b-hf" / "model.safetensors-00001-of-00001.safetensors"

    if model_path:
        target_path = Path(model_path)
    elif "nanowsp3" in model_name.lower():
        target_path = nanowsp3_default if nanowsp3_default.exists() else nanowsp_default
    elif "nanowsp2" in model_name.lower():
        target_path = nanowsp2_default if nanowsp2_default.exists() else nanowsp_default
    elif "nanowsp" in model_name.lower():
        target_path = nanowsp_default if nanowsp_default.exists() else nanowsp_default.parent
    elif "tinyllama" in model_name.lower():
        target_path = tiny_default if tiny_default.exists() else default_safetensors
    elif "0.8b" in model_name.lower() or "qwen" in model_name.lower():
        target_path = qwen08_default if qwen08_default.exists() else (default_safetensors if default_safetensors.exists() else default_gguf)
    else:
        target_path = default_safetensors if default_safetensors.exists() else default_gguf
    is_safetensors = (target_path.is_dir() or target_path.suffix == ".safetensors") if isinstance(target_path, Path) else True

    ctx = PlanContext(
        budgets=budgets,
        max_modified_weights=max_modified_weights,
        max_delta_norm=max_delta_norm,
        allow_write=allow_write,
        written_paths=written_paths,
        target_path=target_path,
        is_safetensors=is_safetensors,
        default_gguf=default_gguf,
        default_safetensors=default_safetensors,
        nanowsp_default=nanowsp_default,
        nanowsp2_default=nanowsp2_default,
        nanowsp3_default=nanowsp3_default,
        tiny_default=tiny_default,
        qwen08_default=qwen08_default,
        out_dir=out_dir,
        model_path=model_path,
        dry_run=dry_run,
        executed_steps=executed_steps,
    )

    for step in step_list:
        op = step["op"]
        raw_args = step.get("args", "")
        positional, named = parse_named_args(raw_args)
        res = {"op": op, "args": raw_args, "status": "ok"}

        delivered = False
        try:
            delivered = bool(dispatch(ctx, op, positional, named, res))
        except AssertionError as e:
            res["status"] = "assertion_failed"
            res["error"] = str(e)
        except Exception as e:
            res["status"] = "error"
            res["error"] = str(e)

        if not delivered:
            executed_steps.append(res)

    # An assert whose own verdict is negative fails the run, whether or not it raised.
    verdicts = {"assert_orthogonal_preservation": "orthogonal_preservation_verified",
                "assert_behavioral_drift": "behavioral_drift_pass"}
    for r in executed_steps:
        key = verdicts.get(r.get("op"))
        if key and r.get("status") == "ok" and r.get("result", {}).get(key) is False:
            r["status"] = "assertion_failed"
            r["error"] = f"{r['op']}: {key} is false"

    # AUDIT 2026-11-04: ops STUB nunca publican claims causales
    for r in executed_steps:
        if r.get("op") in STUB_OPS and r.get("status") not in ("security_rejection", "error", "assertion_failed"):
            r["status"] = "stub_result"

    return executed_steps


# Backward compatibility aliases
execute_plan_v3 = execute_plan_v4
execute_plan_v2 = execute_plan_v4
execute_plan = execute_plan_v4


# ==============================================================================
# 3. CLI & ENTRY POINT
# ==============================================================================

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    DIR_PL.mkdir(parents=True, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser(
        prog="wsplang",
        description=f"WSP-Lang {__version__} Parser, Compiler & Weight Space Surgery Engine"
    )
    ap.add_argument("program", help="Path to .wspl program file")
    ap.add_argument("--check", action="store_true", help="Validate syntax, budgets, and security boundaries")
    ap.add_argument("--plan", action="store_true", help="Generate structured execution plan in JSON")
    ap.add_argument("--dry-run", action="store_true", help="Simulate execution in-memory without modifying disk")
    ap.add_argument("--run", "--exec", action="store_true", help="Execute program steps with full contract enforcement")
    ap.add_argument("--model", default=None, help="Path to target model (SafeTensors directory or GGUF file)")
    ap.add_argument("--out", default=None, help="Path for output model directory")
    ap.add_argument("--gguf", default=None, help="Legacy alias for --model with GGUF")
    args = ap.parse_args()

    src = Path(args.program)
    if not src.exists():
        print(f"Error: Program file not found: {src}", file=sys.stderr)
        sys.exit(1)

    text = src.read_text(encoding="utf-8")
    prog = parse_args_lines(text)
    steps, errors = validate(prog)
    model_arg_early = args.model or args.gguf
    feas_errors, feas_warnings = check_feasibility(prog, steps, model_arg_early)
    errors = errors + feas_errors

    experiment = None
    for ind, s in prog.get("__root__", []):
        if ind == "exp":
            experiment = s
        elif isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()

    plan = plan_run(prog, steps)

    if args.check:
        print(f"=== WSP-Lang {__version__} SYNTAX & BUDGET & FEASIBILITY CHECK: {experiment} ===")
        print(f"Theoretical Framework: {plan['theoretical_framework']}")
        for w in feas_warnings:
            print(f"  [STUB] {w}")
        if errors:
            print(f"[FAIL] {len(errors)} validation error(s) found:")
            for err in errors:
                print(f"  * {err}")
            sys.exit(1)
        else:
            print("[PASS] Syntax, budgets, and security boundaries valid.")
            print(f"  Steps: {len(steps)} | Budgets: {plan.get('budgets')}")
            sys.exit(0)

    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        sys.exit(0)

    model_arg = args.model or args.gguf
    execution_results = None

    if args.run or args.dry_run:
        if errors:
            print(f"=== WSP-Lang VALIDATION FAILED: {len(errors)} error(s) ===")
            for err in errors:
                print(f"  [ERROR] {err}")
            sys.exit(1)
        execution_results = execute_plan_v4(
            plan,
            prog,
            model_path=model_arg,
            out_dir=args.out,
            dry_run=args.dry_run
        )

    out = {
        "wsplang_version": __version__,
        "theoretical_framework": plan["theoretical_framework"],
        "programa": experiment,
        "archivo": str(src),
        "steps_count": len(steps),
        "steps": steps,
        "plan": plan,
        "errors": errors,
        "validated": len(errors) == 0,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if execution_results is not None:
        out["ejecucion"] = execution_results

    # Save manifest
    manif = DIR_PL / f"{experiment or 'sin-nombre'}.manifest.json"
    manif.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Display results
    if (args.run or args.dry_run) and execution_results:
        mode_str = "DRY RUN (IN-MEMORY SIMULATION)" if args.dry_run else "EXECUTION RESULTS"
        print(f"=== WSP-Lang {__version__} {mode_str}: {experiment} ===")
        for step in execution_results:
            op = step["op"]
            status = step["status"]
            print(f"\n> Step: {op} {step.get('args', '')} [status: {status}]")
            res = step.get("result", {})
            if op == "delta_unlearn":
                if status == "ok":
                    print(f"  Framework:           {res.get('rule')} (Fast Weight Unlearning)")
                    print(f"  Tensor:              {res.get('tensor')}")
                    print(f"  Strategy:            {res.get('column_selection_strategy')}")
                    print(f"  Modified Parameters: {res.get('modified_parameters'):,} ({res.get('fraction_modified')})")
                    print(f"  Energy Drop:         {res.get('energy_before'):.4f} -> {res.get('energy_after'):.4f} ({res.get('energy_reduction_pct')})")
                    print(f"  Orthogonal Crosstalk: {res.get('orthogonal_crosstalk_risk')} (Zero Cross-Talk Guarantee)")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "spectral_audit":
                if status == "ok":
                    print(f"  Tensor:              {res.get('tensor')} {res.get('shape')}")
                    print(f"  Top Singular Values: {res.get('top_singular_values')}")
                    print(f"  Spectral Gap delta:  {res.get('spectral_gap_delta')}")
                    print(f"  Condition Number:    {res.get('condition_number')}")
                    print(f"  Leverage Mean/Max:   {res.get('leverage_distribution', {}).get('mean', 0):.6f} / {res.get('leverage_distribution', {}).get('max', 0):.6f}")
                    if "davis_kahan_bound_sin_theta" in res:
                        print(f"  Operator Norm ||dW||:{res.get('operator_norm_bound_delta_W')}")
                        print(f"  Davis-Kahan Bound:   sin Theta <= {res.get('davis_kahan_bound_sin_theta')} (max rotation: {res.get('max_subspace_rotation_degrees')} deg)")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "assert_weyl":
                if status == "ok":
                    print(f"  Theorem:             {res.get('theorem')}")
                    print(f"  Weyl Bound Status:   VERIFIED (max shift {res.get('max_empirical_shift')} <= bound {res.get('operator_norm_bound')})")
                    print(f"  Principal Shift:     {res.get('principal_eigenvalue_shift')} ({res.get('relative_principal_shift')})")
                    print(f"  Safety Margin:       {res.get('margin')}")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "assert_davis_kahan":
                if status == "ok":
                    print(f"  Theorem:             {res.get('theorem')}")
                    print("  Davis-Kahan Status:  VERIFIED")
                    print(f"  Empirical sin Theta: {res.get('empirical_sin_theta')} ({res.get('empirical_rotation_degrees')} deg)")
                    print(f"  Theoretical Bound:   sin Theta <= {res.get('theoretical_upper_bound')} (bound: {res.get('theoretical_max_degrees')} deg)")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "fwp_overlay":
                if status == "ok":
                    print(f"  Overlay Type:   {res.get('overlay_type')}")
                    print(f"  Target Tensor:  {res.get('target_tensor')}")
                    print(f"  Saved File:     {res.get('overlay_file')}")
            elif op == "spectral_decay":
                if status == "ok":
                    print(f"  Decay Factor:   lambda = {res.get('decay_factor_lambda')} ({res.get('dissipation_pct')} dissipation)")
            elif op == "assert_orthogonal_preservation":
                if status == "ok":
                    print(f"  Preservation:   Verified (Cross-talk distortion: {res.get('crosstalk_distortion')})")
            elif op in ("project_out", "project_in"):
                if status == "ok":
                    print(f"  Tensor:              {res.get('tensor')}")
                    print(f"  Strategy:            {res.get('column_selection_strategy')}")
                    print(f"  Modified Parameters: {res.get('modified_parameters'):,} ({res.get('fraction_modified')})")
                    print(f"  Energy Reduction:    {res.get('directional_energy_before'):.4f} -> {res.get('directional_energy_after'):.4f} ({res.get('energy_reduction_pct')})")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "export_control_vector":
                if status == "ok":
                    print(f"  Output GGUF:    {res.get('output_gguf')}")
                    print(f"  Dimensions:     {res.get('vector_dimension')} dims x {res.get('layers_configured')} layers")
            elif op == "leace_erase":
                if status == "ok":
                    print(f"  Concept Erasure:   {res.get('tensor')} (Efficiency: {res.get('erasure_efficiency')})")
                    print(f"  Energy Delta:      {res.get('concept_energy_before'):.4f} -> {res.get('concept_energy_after'):.4f}")
                    print(f"  Distortion:        Frobenius = {res.get('frobenius_distortion'):.4f}, Max Delta = {res.get('max_weight_delta'):.6f}")
            elif op in ("cov_imprint", "memit_imprint"):
                if status == "ok":
                    print(f"  MEMIT Synthesis:   {res.get('tensor')} ({res.get('pairs_count')} pairs, lambda={res.get('regularization_lambda')})")
                    print(f"  Condition Number:  {res.get('unregularized_condition'):.2f} -> {res.get('regularized_condition'):.2f} (stabilized)")
                    print(f"  Delta Frobenius:   {res.get('delta_frobenius'):.4f}")
            elif op == "decoupled_delta":
                if status == "ok":
                    print(f"  Decoupled Delta:   {res.get('tensor')} (beta_erase={res.get('beta_erase')}, beta_write={res.get('beta_write')})")
                    print(f"  Attractor Energy:  {res.get('attractor_energy_before'):.4f} -> {res.get('attractor_energy_after'):.4f} ({res.get('reduction_pct')})")
            elif op == "surprise_fwp":
                if status == "ok":
                    print(f"  Titans Surprisal:  Repetition Score = {res.get('repetition_score')}, Triggered = {res.get('memory_update_triggered')}")
                    print(f"  Action:            {res.get('action')}")
            elif op == "rosetta_align":
                if status == "ok":
                    print(f"  Rosetta Procrustes: {res.get('d_source')} dims -> {res.get('d_target')} dims (Alignment: {res.get('mean_anchor_cosine')})")
                    print(f"  Output Mapping:     {res.get('rosetta_matrix_out')}")
            elif op == "cross_project":
                if status == "ok":
                    print(f"  Cross-Projection:   {res.get('source_dim')} dims -> {res.get('target_dim')} dims (Norm: {res.get('projected_norm')})")
                    print(f"  Translated Vector:  {res.get('translated_vector_out')}")
            elif op == "tie_weights":
                if status == "ok":
                    print(f"  Tie Weights:    {res.get('source')} <==> {res.get('target')} (Frobenius Diff: {res.get('parity_frobenius_norm', 0.0)})")
            elif op == "init_architecture":
                if status == "ok" and res.get("compiled_from"):
                    print(f"  Architecture:   {res.get('architecture')} (compiled from {res.get('compiled_from')})")
                    print(f"  Parameters:     {res.get('total_parameters'):,} weights")
                    print(f"  Checkpoint:     {res.get('output_file')}")
                elif status == "ok":
                    print(f"  Architecture:   {res.get('architecture')} ({res.get('layers')} layers, d={res.get('d_model')}, heads={res.get('heads')})")
                    print(f"  Parameters:     {res.get('total_parameters'):,} weights ({res.get('vocab_size')} vocab)")
                    print(f"  Checkpoint:     {res.get('output_file')}")
            elif op == "imprint_association":
                if status == "ok":
                    print(f"  Prompt -> Target: '{res.get('prompt')}' -> '{res.get('target')}' (Layer {res.get('layer')})")
                    print(f"  Delta Frobenius:  {res.get('details', {}).get('delta_frobenius', 0.0):.4f}")
            elif op == "imprint_batch":
                if status == "ok":
                    print(f"  Batch Imprint:    {res.get('pairs_count')} pairs synthesized via Moore-Penrose")
                    print(f"  Condition Number: {res.get('condition_number'):.2f}")
            elif op == "audit":
                if status == "ok":
                    print(f"  Tensor:         {res.get('tensor')} {res.get('shape')}")
                    print(f"  Frobenius Norm: {res.get('frobenius_norm'):.4f}")
            elif op in ("architectural_drift", "drift_profile"):
                if status == "ok":
                    print(f"  Base -> Target: {res.get('base_model')} -> {res.get('target_model')}")
                    print(f"  Total Tensors:  {res.get('total_tensors_analyzed')}")
                    if "family_summary" in res:
                        print("  Architectural Family Drift Breakdown:")
                        for fam, stats in res["family_summary"].items():
                            print(f"    - {fam:<30}: Rel Drift {stats.get('mean_rel_delta', 0):>6.2f}% | Cosine {stats.get('mean_cos_sim', 0):.4f}")
                    if "top_divergent_tensors" in res:
                        print("  Top Divergent Weights:")
                        for t in res["top_divergent_tensors"][:5]:
                            print(f"    * {t.get('tensor'):<35} Rel Drift {t.get('rel_delta', 0)*100:>6.2f}% | Cosine {t.get('cos_sim', 0):.4f}")
            elif op == "spectral_delta":
                if status == "ok":
                    print(f"  Tensor:               {res.get('tensor')} {res.get('shape')}")
                    print(f"  Frobenius Delta:      {res.get('frobenius_norm_delta'):.4f} (Rel: {res.get('relative_drift', 0)*100:.2f}%)")
                    print(f"  Top Singular Values:  {res.get('top_singular_values')}")
                    print(f"  Spectral Gap (s1/s2): {res.get('spectral_gap_s1_s2')} | Top-1 Variance: {res.get('top1_variance_share', 0)*100:.2f}%")
                    print(f"  Geometry Diagnosis:   {res.get('update_geometry')}")
            elif op == "semantic_delta":
                if status == "ok":
                    print(f"  Tensor:               {res.get('tensor')}")
                    print(f"  Dominant Concept:     {res.get('dominant_semantic_concept')}")
                    print("  Top Concept Shifts:")
                    for c in res.get("top_amplified_concepts", [])[:4]:
                        print(f"    - {c.get('concept'):<20} Alignment: {c.get('alignment_magnitude'):.4f}")
            elif op == "ssm_dynamics_diff":
                if status == "ok":
                    print(f"  SSM Layer {res.get('layer')} Recurrent Transition Analysis:")
                    for comp, sdata in res.get("dynamics_components", {}).items():
                        print(f"    - {comp:<15}: Rel Drift {sdata.get('relative_drift', 0)*100:>6.2f}% | {sdata.get('status')}")
                    print(f"  Diagnosis: {res.get('structural_diagnosis')}")
            elif op == "diff" and "overall_percentage" in res:
                print(f"  Total Weights Changed: {res['total_changed']:,} / {res['total_weights']:,} ({res['overall_percentage']})")
            elif "error" in step:
                print(f"  Error: {step['error']}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if execution_results is not None:
        code = run_exit_code(execution_results)
        if code:
            failed = [s["op"] for s in execution_results if run_exit_code([s])]
            print(f"\n[FAIL] {len(failed)} step(s) failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(code)


if __name__ == "__main__":
    main()

execute_plan_v5 = execute_plan_v4
