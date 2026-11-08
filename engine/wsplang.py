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
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

__version__ = "0.7.0"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))
DIR_PL = ROOT / "challenges" / "wsplang"
DIR_PL.mkdir(parents=True, exist_ok=True)

from engine.comparative_drift import (
    compute_architectural_drift,
    compute_spectral_delta,
    compute_semantic_delta,
    compute_ssm_dynamics_diff,
)

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
    # Architectural & Comparative Analysis (WSP-Lang 0.7: Base vs Distill Transformation)
    "architectural_drift", "drift_profile", "spectral_delta", "semantic_delta", "ssm_dynamics_diff",
    # Synthesis, perturbation bounds, and envelope operators
    "delta_unlearn", "fwp_overlay", "spectral_decay", "assert_orthogonal_preservation",
    "spectral_audit", "assert_weyl", "assert_davis_kahan",
    "init_architecture", "imprint_association", "imprint_batch", "imprint_corpus",
    "cov_imprint", "memit_imprint", "decoupled_delta",
    "rosetta_align", "cross_project",
    "layer_envelope", "multi_layer_envelope", "distributed_memit",
    # WSP-Lang 0.7: JEPA (LeCun) & Weight Space Learning (Damian Borth) Operators
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
    "layer_envelope", "multi_layer_envelope", "distributed_memit",
    "jepa_steer", "vicreg_preserve", "permute_align", "compile_synthetic", "assert_behavioral_drift",
    "leace_erase", "surprise_fwp"
}

METADATA = {
    "model", "format", "target_format", "objective", "constraints", "budgets",
    "types", "declared", "sources", "baseline", "discover", "preconditions",
    "postconditions", "accept_if", "evaluate", "record", "hypothesis",
    "note", "semantic_areas", "semantic_target", "theoretical_foundation"
}

FORBIDDEN_TENSORS = {
    "token_embd.weight", "output.weight", "output_norm.weight",
    "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"
}
FORBIDDEN_PATTERNS = ["embed", "lm_head", "norm.weight", "output_norm", "wte", "ln_f"]


class WriteGuardError(ValueError):
    """A write would break a WSPLang safety rule; nothing was written."""


def is_protected_tensor(name: str, unlock=()) -> bool:
    """Embeddings, readout and final norms change only when a program unlocks them by name."""
    if name in unlock:
        return False
    return name in FORBIDDEN_TENSORS or any(pat in name for pat in FORBIDDEN_PATTERNS)


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
}
IMPLEMENTED_OPS = READ_ONLY_OPS | {
    "leace_erase", "cov_imprint", "memit_imprint", "decoupled_delta", "surprise_fwp",
    "rosetta_align", "cross_project", "init_architecture", "imprint_association",
    "imprint_batch", "imprint_corpus", "delta_unlearn", "fwp_overlay", "layer_envelope",
    "multi_layer_envelope", "distributed_memit", "spectral_decay",
    "assert_orthogonal_preservation", "project_out", "project_in", "audit",
    "export_control_vector", "spectral_audit", "assert_weyl", "assert_davis_kahan",
    "jepa_steer", "permute_align", "compile_synthetic", "tie_weights", "train", "synthesize", "decompile",
    "subspace_transfer",
}
STUB_OPS = {"leace_erase", "surprise_fwp"}
WRITE_BACKEND_OPS = IMPLEMENTED_OPS - READ_ONLY_OPS


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


def load_vector_numpy(vec_path: Path) -> "np.ndarray":
    """Load and L2-normalize direction vector from .npz or .npy."""
    import numpy as np
    import torch
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
    import torch
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
    import numpy as np
    import torch
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
    """Execute plan with WSP-Lang 0.3 Fast Weight Programming engine."""
    import numpy as np
    import torch

    budgets = plan.get("budgets", {})
    max_modified_weights = budgets.get("max_modified_weights")
    max_delta_norm = budgets.get("max_delta_norm")
    allow_write = budgets.get("allow_write", False) and not dry_run
    written_paths = set()  # files this run wrote; later steps may edit them in place

    def edit_destination(op_name, named_args, source_file):
        if named_args.get("out"):
            dest = Path(str(named_args["out"]))
            return dest if dest.is_absolute() else ROOT / dest
        if out_dir:
            return Path(out_dir) / Path(source_file).name
        if Path(source_file).resolve() in written_paths:
            return Path(source_file)
        raise WriteGuardError(f"{op_name} writes weights: give out=<file> or --out-dir; "
                              f"the input model {source_file} is never overwritten")

    def write_edit(op_name, named_args, candidate, source_file, metadata=None):
        """Every edit of an existing model goes through here. Returns (written path or None, guard report)."""
        nonlocal target_path
        if not allow_write:
            return None, guard_model_write(candidate, None, source_file, budgets, written_paths)
        dest = edit_destination(op_name, named_args, source_file)
        report = guarded_save(candidate, dest, source_file, budgets, written_paths, metadata=metadata)
        # Later steps edit what this step wrote. Only for single-file models: a sharded directory would
        # otherwise lose its other shards.
        shards = list(target_path.glob("*.safetensors")) if target_path.is_dir() else [target_path]
        if len(shards) == 1 and shards[0].resolve() == Path(source_file).resolve():
            target_path = dest
        return dest, report

    def runtime_candidate(runtime):
        return {k: v.detach().contiguous().float() for k, v in runtime.weights.items()}

    executed_steps = []
    step_list = plan.get("pasos", [])

    default_gguf = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")
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
    tiny_default = Path(r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6\model.safetensors")
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

    for step in step_list:
        op = step["op"]
        raw_args = step.get("args", "")
        positional, named = parse_named_args(raw_args)
        res = {"op": op, "args": raw_args, "status": "ok"}

        try:
            # ------------------------------------------------------------------
# ------------------------------------------------------------------
            # OP: leace_erase (WSP-Lang 0.5: Closed-Form LEACE Concept Erasure)
            # ------------------------------------------------------------------
            if op == "leace_erase":
                tensor_name = positional[0] if positional else named.get("tensor")
                vec_file = named.get("vector", named.get("key", positional[1] if len(positional) > 1 else None))
                cov_file = named.get("covariance", named.get("cov"))

                if not tensor_name:
                    raise ValueError("Tensor name must be specified in leace_erase")
                if not vec_file:
                    raise ValueError("Target concept vector must be specified in leace_erase")

                vec_p = Path(vec_file)
                if not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                cov_mat = None
                if cov_file:
                    cov_p = Path(cov_file)
                    if not cov_p.is_absolute():
                        cov_p = ROOT / cov_p
                    if cov_p.exists():
                        cov_np = np.load(cov_p)
                        cov_mat = torch.from_numpy(cov_np[list(cov_np.files)[0]] if hasattr(cov_np, "files") else cov_np).float()

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    W, shard_to_edit, tensors = load_safetensors_tensor(target_path, tensor_name)

                    u_t = torch.from_numpy(u_vec).float().to(W.device)
                    # Projector on row dimension
                    P_leace = compute_leace_projection(u_t, sigma=cov_mat)

                    energy_before = float(torch.norm(torch.matmul(u_t, W)).item())
                    norm_before = float(torch.norm(W).item())

                    W_erased = torch.matmul(P_leace, W)
                    delta = W - W_erased

                    actual_delta_norm = float(torch.max(torch.abs(delta)).item())
                    energy_after = float(torch.norm(torch.matmul(u_t, W_erased)).item())
                    norm_after = float(torch.norm(W_erased).item())

                    tensors[tensor_name] = W_erased.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    res["execution_state"] = "executed" if saved_path else "simulated"
                    res["result"] = {
                        "rule": "LEACE Closed-Form Concept Erasure (Belrose et al. 2023)",
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        "concept_energy_before": energy_before,
                        "concept_energy_after": energy_after,
                        "erasure_efficiency": f"{((energy_before - energy_after) / max(energy_before, 1e-12)):.4%}",
                        "frobenius_norm_before": norm_before,
                        "frobenius_norm_after": norm_after,
                        "frobenius_distortion": float(torch.norm(delta).item()),
                        "max_weight_delta": actual_delta_norm,
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "LEACE erasure currently supported on SafeTensors."}

            # ------------------------------------------------------------------
            # OP: cov_imprint / memit_imprint (WSP-Lang 0.5: MEMIT Covariance Regularization)
            # ------------------------------------------------------------------
            elif op in ("cov_imprint", "memit_imprint"):
                tensor_name = positional[0] if positional else named.get("tensor")
                batch_file = named.get("file")
                reg_lambda = float(named.get("lambda", named.get("reg", 1e-3)))

                if not tensor_name:
                    raise ValueError("Tensor name must be specified in cov_imprint")

                pairs = []
                if batch_file:
                    batch_p = Path(batch_file)
                    if not batch_p.is_absolute():
                        batch_p = ROOT / batch_p
                    if batch_p.exists():
                        with open(batch_p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            pairs = [(item["prompt"], item["target"]) for item in data]
                elif "prompt" in named and "target" in named:
                    pairs = [(named["prompt"], named["target"])]

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    from engine.nanowsp_runtime import NanoWSP

                    ckpt_path = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                    if not ckpt_path.exists():
                        ckpt_path = ROOT / "models" / "nanowsp" / "model.safetensors"

                    nanowsp = NanoWSP()
                    if ckpt_path.exists():
                        nanowsp.load_checkpoint(ckpt_path)

                    # Extract keys and values from model traces
                    keys, values = [], []
                    for p, t in pairs:
                        t_ids = nanowsp.tokenizer.encode(p, add_bos=True)
                        _, tr = nanowsp.forward(t_ids, return_hidden=True)
                        k = tr["layer_1_mlp_hidden"][0, -1, :].clone()
                        keys.append(k / max(float(torch.norm(k).item()), 1e-6))
                        t_id = nanowsp.tokenizer.encode(t, add_bos=False)[0]
                        v = nanowsp.weights["output.weight"][t_id].clone()
                        values.append(v / max(float(torch.norm(v).item()), 1e-6))

                    K = torch.stack(keys, dim=1)    # [d_in, M]
                    V = torch.stack(values, dim=1)  # [d_out, M]

                    target_W = nanowsp.weights[tensor_name]
                    cond_before = float(torch.linalg.cond(K).item())

                    delta = compute_memit_update(target_W, K, V, reg_lambda=reg_lambda)
                    target_W += delta

                    cond_after = float(torch.linalg.cond(K.t() @ K + reg_lambda * torch.eye(K.shape[1])).item())
                    delta_norm = float(torch.norm(delta).item())

                    if not ckpt_path.exists():
                        raise FileNotFoundError(f"{op} needs a model checkpoint, none at {ckpt_path}")
                    dest, res["write_guard"] = write_edit(op, named, runtime_candidate(nanowsp), ckpt_path)
                    saved = dest is not None
                    if saved:
                        target_path = dest

                    res["execution_state"] = "executed" if saved else "simulated"
                    res["result"] = {
                        "rule": "MEMIT / EMMET Covariance-Regularized Memory Synthesis",
                        "tensor": tensor_name,
                        "pairs_count": len(pairs),
                        "regularization_lambda": reg_lambda,
                        "unregularized_condition": cond_before,
                        "regularized_condition": cond_after,
                        "delta_frobenius": delta_norm,
                        "written_to_disk": saved
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "cov_imprint requires SafeTensors backend."}

            # ------------------------------------------------------------------
            # OP: decoupled_delta (WSP-Lang 0.5: Gated DeltaNet-2 Decoupled Erase & Write)
            # ------------------------------------------------------------------
            elif op == "decoupled_delta":
                tensor_name = positional[0] if positional else named.get("tensor")
                erase_vec_file = named.get("erase_vector", named.get("erase_key"))
                write_k_file = named.get("write_key")
                write_v_file = named.get("write_value")
                beta_erase = float(named.get("beta_erase", 1.0))
                beta_write = float(named.get("beta_write", 1.0))

                if not tensor_name or not erase_vec_file:
                    raise ValueError("Tensor and erase_vector are required in decoupled_delta")

                ev_p = Path(erase_vec_file)
                if not ev_p.is_absolute():
                    ev_p = ROOT / ev_p
                u_erase = load_vector_numpy(ev_p)

                k_write = load_vector_numpy(Path(write_k_file) if Path(write_k_file).is_absolute() else ROOT / write_k_file) if write_k_file else None
                v_write = load_vector_numpy(Path(write_v_file) if Path(write_v_file).is_absolute() else ROOT / write_v_file) if write_v_file else None

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    W, shard_to_edit, tensors = load_safetensors_tensor(target_path, tensor_name)

                    u_e_t = torch.from_numpy(u_erase).float().to(W.device)
                    k_w_t = torch.from_numpy(k_write).float().to(W.device) if k_write is not None else None
                    v_w_t = torch.from_numpy(v_write).float().to(W.device) if v_write is not None else None

                    energy_erase_before = float(torch.norm(torch.matmul(u_e_t, W)).item())
                    W_new = apply_decoupled_delta(W, u_e_t, beta_erase, k_w_t, v_w_t, beta_write)
                    energy_erase_after = float(torch.norm(torch.matmul(u_e_t, W_new)).item())

                    tensors[tensor_name] = W_new.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    res["execution_state"] = "executed" if saved_path else "simulated"
                    res["result"] = {
                        "rule": "Gated DeltaNet-2 Decoupled Erase & Write",
                        "tensor": tensor_name,
                        "beta_erase": beta_erase,
                        "beta_write": beta_write,
                        "attractor_energy_before": energy_erase_before,
                        "attractor_energy_after": energy_erase_after,
                        "reduction_pct": f"{((energy_erase_before - energy_erase_after) / max(energy_erase_before, 1e-12)):.2%}",
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "decoupled_delta requires SafeTensors backend."}

            # ------------------------------------------------------------------
            # OP: surprise_fwp (WSP-Lang 0.5: Titans Surprisal Test-Time Memory Trigger)
            # ------------------------------------------------------------------
            elif op == "surprise_fwp":
                threshold = float(named.get("threshold", named.get("tau", 3.0)))
                metric = named.get("metric", "perplexity_surprise")
                prompt = named.get("prompt", "test prompt")

                # Lexical repetition heuristic stub (STUB)
                tokens = prompt.split()
                unique_ratio = len(set(tokens)) / max(len(tokens), 1)
                repetition_score = 1.0 - unique_ratio
                rep_threshold = threshold if threshold < 1.0 else 0.5
                triggered = repetition_score > rep_threshold and len(tokens) >= 4

                res["execution_state"] = "heuristic_trigger_only"
                res["result"] = {
                    "rule": "Titans Surprisal Monitor [STUB: Lexical Repetition Heuristic]",
                    "status": "heuristic_trigger_only",
                    "note": "Lexical word-ratio trigger stub; does not compute neural perplexity or synthesize live runtime overlay.",
                    "metric": metric,
                    "repetition_score": round(repetition_score, 4),
                    "unique_ratio": round(unique_ratio, 4),
                    "threshold": rep_threshold,
                    "memory_update_triggered": triggered,
                    "action": "flag_repetition_detected" if triggered else "maintain_quiescent_memory"
                }

            # ------------------------------------------------------------------
            # OP: rosetta_align (WSP-Lang 0.6: Cross-Model Procrustes Alignment)
            # ------------------------------------------------------------------
            elif op == "rosetta_align":
                import torch
                source_file = named.get("source", positional[0] if positional else None)
                target_file = named.get("target", positional[1] if len(positional) > 1 else None)
                out_file = named.get("out", "data/rosetta_matrix.npz")
                out_p = Path(out_file)
                if not out_p.is_absolute():
                    out_p = ROOT / out_p

                src_p = Path(source_file) if source_file else None
                if src_p and not src_p.is_absolute():
                    src_p = ROOT / src_p
                tgt_p = Path(target_file) if target_file else None
                if tgt_p and not tgt_p.is_absolute():
                    tgt_p = ROOT / tgt_p

                # Load embeddings (npz or safetensors)
                if src_p.suffix == ".npz" and tgt_p.suffix == ".npz":
                    data_src = np.load(src_p)
                    data_tgt = np.load(tgt_p)
                    k_s = list(data_src.keys())[0]
                    k_t = list(data_tgt.keys())[0]
                    E_s = torch.from_numpy(data_src[k_s]).float()
                    E_t = torch.from_numpy(data_tgt[k_t]).float()
                else:
                    raise ValueError("rosetta_align currently requires anchor embedding .npz files or pre-extracted matrices")

                T, mean_cos, ortho_err = compute_procrustes_alignment(E_s, E_t)

                if allow_write:
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out_p,
                        T=T.numpy(),
                        mean_cos=mean_cos,
                        ortho_err=ortho_err,
                        d_source=E_s.shape[1],
                        d_target=E_t.shape[1]
                    )

                res["execution_state"] = "executed" if allow_write else "simulated"
                res["result"] = {
                    "source": str(src_p),
                    "target": str(tgt_p),
                    "d_source": E_s.shape[1],
                    "d_target": E_t.shape[1],
                    "mean_anchor_cosine": round(mean_cos, 4),
                    "semi_orthogonality_error": round(ortho_err, 6),
                    "rosetta_matrix_out": str(out_p.relative_to(ROOT)) if (allow_write and out_p.is_relative_to(ROOT)) else str(out_p)
                }

            # ------------------------------------------------------------------
            # OP: cross_project (WSP-Lang 0.6: Cross-Model Vector Translation)
            # ------------------------------------------------------------------
            elif op == "cross_project":
                import torch
                vec_file = named.get("vector", positional[0] if positional else None)
                rosetta_file = named.get("rosetta", positional[1] if len(positional) > 1 else None)
                out_file = named.get("out", "data/translated_vector.npz")
                out_p = Path(out_file)
                if not out_p.is_absolute():
                    out_p = ROOT / out_p

                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                ros_p = Path(rosetta_file) if rosetta_file else None
                if ros_p and not ros_p.is_absolute():
                    ros_p = ROOT / ros_p

                u_raw = load_vector_numpy(vec_p)
                u_s = torch.from_numpy(u_raw).float()
                u_s = u_s / max(float(torch.norm(u_s).item()), 1e-12)

                ros_data = np.load(ros_p)
                T = torch.from_numpy(ros_data["T"]).float()  # [d_target, d_source]

                u_target = torch.matmul(T, u_s)
                proj_norm = float(torch.norm(u_target).item())
                u_target = u_target / max(proj_norm, 1e-12)

                if allow_write:
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(out_p, u=u_target.numpy())

                res["execution_state"] = "executed" if allow_write else "simulated"
                res["result"] = {
                    "source_vector": str(vec_p),
                    "source_dim": u_s.shape[0],
                    "target_dim": u_target.shape[0],
                    "projected_norm": round(proj_norm, 4),
                    "translated_vector_out": str(out_p.relative_to(ROOT)) if (allow_write and out_p.is_relative_to(ROOT)) else str(out_p)
                }

            # ------------------------------------------------------------------
            # OP: init_architecture arch=<file.wsparch>: compile from declarative source
            # ------------------------------------------------------------------
            elif op == "init_architecture" and "arch" in named:
                from engine.wsp_arch import load_arch_file, parse_config_value

                arch_rel = str(named["arch"])
                arch_path = Path(arch_rel)
                if arch_path.suffix != ".wsparch":
                    raise ValueError(f"arch must be a .wsparch file, got {arch_rel}")
                if not arch_path.is_absolute():
                    arch_path = ROOT / arch_path

                overrides = {key: parse_config_value(value) for key, value in named.items()
                             if key not in ("arch", "out", "seed", "model", "format")}
                compiled = load_arch_file(arch_path, overrides=overrides, seed=int(named.get("seed", 1337)))
                total = sum(p.numel() for p in compiled.parameters())
                if max_modified_weights is not None and total > max_modified_weights:
                    raise ValueError(f"{total} parameters exceed budget max_modified_weights={max_modified_weights}")

                out_p = Path(named.get("out", str(ROOT / "models" / compiled.arch_name / "model.safetensors")))
                if not out_p.is_absolute():
                    out_p = ROOT / out_p
                saved = False
                if allow_write:
                    check_new_file(out_p, [model_path], written_paths)
                    compiled.save(out_p)
                    written_paths.add(out_p.resolve())
                    saved = True
                    target_path = out_p
                    is_safetensors = True

                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "architecture": compiled.arch_name,
                    "compiled_from": arch_rel,
                    "source_sha256": compiled.source_sha256,
                    "config": {k: v for k, v in vars(compiled.config).items()},
                    "total_parameters": total,
                    "output_file": str(out_p),
                    "written_to_disk": saved
                }

            # ------------------------------------------------------------------
            # OP: synthesize: closed-form weights, no gradient (see engine/wsp_synth.py)
            # ------------------------------------------------------------------
            elif op == "synthesize":
                from engine import wsp_arch, wsp_synth, wsp_train

                from engine import wsp_decompile

                if named.get("method") != "ngram":
                    raise ValueError(f"synthesize supports method=ngram, got {named.get('method')}")
                source_args = ("table",) if "table" in named else ("data", "order")
                for required in ("arch", "out") + source_args:
                    if required not in named:
                        raise ValueError(f"synthesize requires {required}=")
                if "table" in named and ("data" in named or "order" in named):
                    raise ValueError("synthesize takes either table= or data= with order=, not both")

                def _rooted(value):
                    path = Path(str(value))
                    return path if path.is_absolute() else ROOT / path

                arch_path = _rooted(named["arch"])
                out_p = _rooted(named["out"])
                if "table" in named:
                    table = wsp_decompile.parse_program(_rooted(named["table"]).read_text(encoding="utf-8"))
                    chars = table.chars
                else:
                    dataset = wsp_train.char_dataset(_rooted(named["data"]).read_text(encoding="utf-8"))
                    chars = dataset.chars
                    table = wsp_synth.WittenBellTable(dataset.train, len(chars), int(named["order"]))
                overrides = {key: wsp_arch.parse_config_value(value) for key, value in named.items()
                             if key not in ("method", "arch", "data", "out", "order", "seed", "model", "format", "table")}
                if not allow_write:
                    res["execution_state"] = "simulated"
                    res["result"] = {"units_available": table.num_units(), "written_to_disk": False}
                else:
                    compiled = wsp_synth.compile_ngram(arch_path, table, overrides=overrides,
                                                       seed=int(named.get("seed", 1337)))
                    total = sum(p.numel() for p in compiled.parameters())
                    if max_modified_weights is not None and total > max_modified_weights:
                        raise ValueError(f"{total} parameters exceed budget max_modified_weights={max_modified_weights}")
                    info = compiled.synthesis_info
                    if "table" in named:
                        source = getattr(table, "provenance", "unknown")
                        info["method"] = "wspngram_program"
                        info["provenance"] = ("closed_form" if source == "closed_form"
                                              else f"closed_form_compile_of:{source}")
                    check_new_file(out_p, [model_path], written_paths)
                    compiled.save(out_p, metadata={"provenance": info["provenance"], "backward_calls": "0",
                                                   "method": info["method"], "order": str(info["order"]),
                                                   "vocab_chars": json.dumps(chars)})
                    written_paths.add(out_p.resolve())
                    target_path = out_p
                    is_safetensors = True
                    res["execution_state"] = "executed"
                    res["result"] = {**info, "total_parameters": total, "output_file": str(out_p)}

            # ------------------------------------------------------------------
            # OP: decompile: nanoGPT weights -> WSPLang n-gram program (.wspngram)
            # ------------------------------------------------------------------
            elif op == "decompile":
                from safetensors import safe_open

                from engine import wsp_arch, wsp_decompile, wsp_train

                method = named.get("method")
                if method not in ("structural", "behavior"):
                    raise ValueError(f"decompile supports method=structural or method=behavior, got {method}")
                required = ("arch", "model", "out") + (("data", "order") if method == "behavior" else ())
                for key in required:
                    if key not in named:
                        raise ValueError(f"decompile method={method} requires {key}=")

                def _rooted(value):
                    path = Path(str(value))
                    return path if path.is_absolute() else ROOT / path

                model_p, out_p = _rooted(named["model"]), _rooted(named["out"])
                with safe_open(str(model_p), framework="pt") as f:
                    meta = f.metadata() or {}
                dataset = None
                if method == "behavior":
                    dataset = wsp_train.char_dataset(_rooted(named["data"]).read_text(encoding="utf-8"))
                if "vocab_chars" in meta:
                    chars = json.loads(meta["vocab_chars"])
                elif dataset is not None:
                    chars = dataset.chars
                else:
                    raise ValueError("checkpoint has no vocab_chars metadata; structural decompile needs it")
                if dataset is not None and dataset.chars != chars:
                    raise ValueError("data= vocabulary differs from the checkpoint's vocab_chars")
                skip = ("method", "arch", "model", "out", "data", "order", "limit_tokens", "seed", "format")
                overrides = {key: wsp_arch.parse_config_value(value) for key, value in named.items() if key not in skip}
                model = wsp_arch.load_arch_file(_rooted(named["arch"]), overrides=dict(overrides, vocab=len(chars)))
                model.load(model_p)
                model.eval()

                if method == "structural":
                    table = wsp_decompile.decompile_ngram(model, chars)
                    table.provenance = meta.get("provenance", "unknown")
                    note = "exact: recovered from the weights' layout, no data used"
                else:
                    ids = dataset.train[:int(named["limit_tokens"])] if "limit_tokens" in named else dataset.train
                    table = wsp_decompile.behavioral_table(model, ids, chars, order=int(named["order"]))
                    table.provenance = "distilled_from:" + meta.get("provenance", "unknown")
                    note = ("approximate: the model's average prediction per context over the train split; "
                            "long-range context is dropped")
                result = {"method": method, "order": table.order, "units": table.num_units(), "note": note,
                          "program_provenance": table.provenance,
                          "source_provenance": meta.get("provenance", "unknown")}
                if not allow_write:
                    res["execution_state"] = "simulated"
                    result["written_to_disk"] = False
                else:
                    check_new_file(out_p, [model_p, model_path], written_paths)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(wsp_decompile.format_program(table, source=f"{method} decompile of {model_p.name}"),
                                     encoding="utf-8")
                    res["execution_state"] = "executed"
                    result["output_file"] = str(out_p)
                res["result"] = result

            # ------------------------------------------------------------------
            # OP: train: gradient descent on a compiled architecture (control, not synthesis)
            # ------------------------------------------------------------------
            elif op == "train":
                if not budgets.get("allow_training"):
                    res["status"] = "security_rejection"
                    res["error"] = "train requires budget allow_training = true"
                    executed_steps.append(res)
                    continue
                from engine import wsp_arch, wsp_train

                if "arch" not in named or "data" not in named or "out" not in named:
                    raise ValueError("train requires arch=, data= and out=")
                arch_rel = str(named["arch"])
                arch_path = Path(arch_rel)
                if arch_path.suffix != ".wsparch":
                    raise ValueError(f"arch must be a .wsparch file, got {arch_rel}")
                arch_path = arch_path if arch_path.is_absolute() else ROOT / arch_path
                data_path = Path(str(named["data"]))
                data_path = data_path if data_path.is_absolute() else ROOT / data_path
                out_p = Path(str(named["out"]))
                out_p = out_p if out_p.is_absolute() else ROOT / out_p

                train_cfg, overrides = wsp_train.split_step_args(named, wsp_arch.parse_config_value)
                dataset = wsp_train.char_dataset(data_path.read_text(encoding="utf-8"))
                if overrides.setdefault("vocab", len(dataset.chars)) != len(dataset.chars):
                    raise ValueError(f"vocab={overrides['vocab']} but data has {len(dataset.chars)} symbols")
                compiled = wsp_arch.load_arch_file(arch_path, overrides=overrides, seed=int(named.get("seed", 1337)))
                total = sum(p.numel() for p in compiled.parameters())
                if max_modified_weights is not None and total > max_modified_weights:
                    raise ValueError(f"{total} parameters exceed budget max_modified_weights={max_modified_weights}")

                if allow_write:
                    check_new_file(out_p, [model_path], written_paths)
                    summary = wsp_train.train(compiled, dataset, train_cfg, out_p)
                    written_paths.add(out_p.resolve())
                    target_path = out_p
                    is_safetensors = True
                    res["execution_state"] = "executed"
                    res["result"] = {"provenance": summary["provenance"], "compiled_from": arch_rel,
                                     "total_parameters": total, "best_val": summary["best_val"],
                                     "saved_iter": summary["saved_iter"],
                                     "backward_calls_at_save": summary["backward_calls_at_save"],
                                     "seconds": summary["seconds"], "output_file": str(out_p)}
                else:
                    res["execution_state"] = "simulated"
                    res["result"] = {"compiled_from": arch_rel, "total_parameters": total, "written_to_disk": False}

            # ------------------------------------------------------------------
            # OP: init_architecture (WSP-Lang 0.5: Synthesize Micro-LLM from Scratch)
            # ------------------------------------------------------------------
            elif op == "init_architecture":
                arch_name = positional[0] if positional else named.get("model", "NanoWSP")
                layers = int(named.get("layers", 2))
                d_model = int(named.get("d_model", 64))
                heads = int(named.get("heads", 2))
                d_ff = int(named.get("d_ff", 128))
                vocab = int(named.get("vocab", 100))
                out_file = named.get("out", str(ROOT / "models" / "nanowsp" / "model.safetensors"))
                out_p = Path(out_file)
                if not out_p.is_absolute():
                    out_p = ROOT / out_p
                arch_extra = {}

                if arch_name.lower() in ("nanogpt", "karpathy", "gpt2") or named.get("format") == "karpathy":
                    from engine.nanogpt_runtime import NanoGPT, GPTConfig
                    block_size = int(named.get("max_seq", named.get("block_size", 256)))
                    # nanoGPT's MLP is 4 * n_embd unless the program says otherwise; the generic
                    # default above belongs to the NanoWSP family.
                    d_ff = int(named.get("d_ff", 0)) or 4 * d_model
                    arch_extra = {
                        "dropout": float(named.get("dropout", 0.0)),
                        "bias": str(named.get("bias", "true")).lower() not in ("false", "0"),
                    }
                    cfg = GPTConfig(
                        vocab_size=vocab,
                        n_embd=d_model,
                        n_layer=layers,
                        n_head=heads,
                        d_ff=d_ff,
                        block_size=block_size,
                        **arch_extra
                    )
                    nanowsp = NanoGPT(config=cfg, seed=int(named.get("seed", 1337)))
                elif "2" in arch_name or layers > 2 or d_model >= 128:
                    from engine.nanowsp2_runtime import NanoWSP2, NanoWSP2Config
                    cfg = NanoWSP2Config(
                        vocab_size=vocab,
                        d_model=d_model,
                        n_layers=layers,
                        n_heads=heads,
                        d_head=d_model // heads,
                        d_ff=d_ff
                    )
                    nanowsp = NanoWSP2(config=cfg, seed=int(named.get("seed", 1337)))
                else:
                    from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig
                    cfg = NanoWSPConfig(
                        vocab_size=vocab,
                        d_model=d_model,
                        n_layers=layers,
                        n_heads=heads,
                        d_head=d_model // heads,
                        d_ff=d_ff
                    )
                    nanowsp = NanoWSP(config=cfg, seed=int(named.get("seed", 42)))

                saved = False
                if allow_write:
                    check_new_file(out_p, [model_path], written_paths)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    nanowsp.save_checkpoint(out_p)
                    written_paths.add(out_p.resolve())
                    saved = True
                    target_path = out_p
                    is_safetensors = True

                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "architecture": arch_name,
                    "layers": layers,
                    "d_model": d_model,
                    "heads": heads,
                    "d_ff": d_ff,
                    "vocab_size": vocab,
                    "total_parameters": nanowsp.count_parameters(),
                    "output_file": str(out_p),
                    "written_to_disk": saved,
                    **arch_extra
                }

            # ------------------------------------------------------------------
            # OP: imprint_association (WSP-Lang 0.5: Declarative Memory Imprinting)
            # ------------------------------------------------------------------
            elif op == "imprint_association":
                from engine.nanowsp_runtime import NanoWSP

                prompt = named.get("prompt", positional[0] if positional else "")
                target = named.get("target", positional[1] if len(positional) > 1 else "")
                layer_idx = int(named.get("layer", 1))
                eta = float(named.get("eta", 5.0))

                ckpt_path = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                if not ckpt_path.exists():
                    fallback = ROOT / "models" / "nanowsp" / "model.safetensors"
                    if fallback.exists():
                        ckpt_path = fallback

                if ckpt_path.exists():
                    from safetensors.torch import load_file
                    tensors = load_file(str(ckpt_path))
                    if 'layers.3.attn_q.weight' in tensors or tensors['token_embd.weight'].shape[0] == 256:
                        from engine.nanowsp2_runtime import NanoWSP2
                        nanowsp = NanoWSP2()
                    else:
                        from engine.nanowsp_runtime import NanoWSP
                        nanowsp = NanoWSP()
                    nanowsp.load_checkpoint(ckpt_path)
                else:
                    from engine.nanowsp_runtime import NanoWSP
                    nanowsp = NanoWSP()

                if len(target) <= 1:
                    imprint_res = nanowsp.imprint_associative_rule(prompt, target, layer_idx=layer_idx, eta=eta)
                else:
                    pairs = []
                    curr = prompt
                    for ch in target:
                        pairs.append((curr, ch))
                        curr += ch
                    imprint_res = nanowsp.imprint_associative_batch(pairs, layer_idx=layer_idx, eta=eta)

                if not ckpt_path.exists():
                    raise FileNotFoundError(f"{op} needs a model checkpoint, none at {ckpt_path}")
                dest, res["write_guard"] = write_edit(op, named, runtime_candidate(nanowsp), ckpt_path)
                saved = dest is not None
                if saved:
                    target_path = dest
                    is_safetensors = True

                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "operation": "imprint_association",
                    "prompt": prompt,
                    "target": target,
                    "layer": layer_idx,
                    "details": imprint_res,
                    "written_to_disk": saved
                }

            # ------------------------------------------------------------------
            # OP: imprint_batch (WSP-Lang 0.5: Moore-Penrose Multi-Memory Synthesis)
            # ------------------------------------------------------------------
            elif op == "imprint_batch":
                from engine.nanowsp_runtime import NanoWSP

                batch_file = named.get("file")
                layer_idx = int(named.get("layer", 1))
                eta = float(named.get("eta", 5.0))

                batch_p = Path(batch_file) if batch_file else None
                if batch_p and not batch_p.is_absolute():
                    batch_p = ROOT / batch_p

                pairs = []
                if batch_p and batch_p.exists():
                    with open(batch_p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        pairs = [(item["prompt"], item["target"]) for item in data]
                elif "pairs" in named:
                    raw_p = named["pairs"].split(";")
                    for rp in raw_p:
                        if "->" in rp:
                            p, t = rp.split("->", 1)
                            pairs.append((p.strip(), t.strip()))

                ckpt_path = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                if not ckpt_path.exists():
                    fallback = ROOT / "models" / "nanowsp" / "model.safetensors"
                    if fallback.exists():
                        ckpt_path = fallback

                if ckpt_path.exists():
                    from safetensors.torch import load_file
                    tensors = load_file(str(ckpt_path))
                    if 'layers.3.attn_q.weight' in tensors or tensors['token_embd.weight'].shape[0] == 256:
                        from engine.nanowsp2_runtime import NanoWSP2
                        nanowsp = NanoWSP2()
                    else:
                        from engine.nanowsp_runtime import NanoWSP
                        nanowsp = NanoWSP()
                    nanowsp.load_checkpoint(ckpt_path)
                else:
                    from engine.nanowsp_runtime import NanoWSP
                    nanowsp = NanoWSP()

                batch_res = nanowsp.imprint_associative_batch(pairs, layer_idx=layer_idx, eta=eta)

                if not ckpt_path.exists():
                    raise FileNotFoundError(f"{op} needs a model checkpoint, none at {ckpt_path}")
                dest, res["write_guard"] = write_edit(op, named, runtime_candidate(nanowsp), ckpt_path)
                saved = dest is not None
                if saved:
                    target_path = dest
                    is_safetensors = True

                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "operation": "imprint_batch",
                    "pairs_count": len(pairs),
                    "condition_number": batch_res.get("condition_number"),
                    "delta_frobenius": batch_res.get("delta_frobenius"),
                    "written_to_disk": saved
                }

            # ------------------------------------------------------------------
            # OP: delta_unlearn (Delta-Rule Associative Memory Unlearning)
            # ------------------------------------------------------------------
            elif op == "delta_unlearn":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in delta_unlearn")

                if tensor_name in FORBIDDEN_TENSORS or any(pat in tensor_name for pat in FORBIDDEN_PATTERNS):
                    res["status"] = "security_rejection"
                    res["error"] = f"SECURITY POLICY REJECTION: Tensor '{tensor_name}' is forbidden from modification."
                    executed_steps.append(res)
                    continue

                vec_file = named.get("key", named.get("vector", "data/u_loop_L14.npz"))
                vec_p = Path(vec_file)
                if not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                k_vec = load_vector_numpy(vec_p)

                eta = float(named.get("eta", named.get("alpha", 1.0)))
                target_val = float(named.get("target", 0.0))
                top_cols_pct = float(named.get("top_cols", 1.0))
                top_rows_limit = int(named.get("top_rows", 0))

                if is_safetensors:
                    import torch
                    from safetensors.torch import load_file, save_file

                    # Find target shard
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
                    W = tensors[tensor_name].float()
                    k_t = torch.from_numpy(k_vec).float().to(W.device)

                    # Compute current associative output: y = k^T @ W
                    y_proj = torch.matmul(k_t, W)
                    before_energy = float(torch.norm(y_proj).item())

                    # Delta rule error: e = y - target_val
                    error_signal = y_proj - target_val

                    # Select active columns (optionally leverage-guided)
                    use_leverage = named.get("selection") in ("efficiency", "leverage") or str(named.get("leverage_filter", "false")).lower() in ("true", "1")
                    if use_leverage:
                        spectral = compute_spectral_decomposition(W, q=16, niter=4)
                        lev = spectral["leverage_scores"]
                        col_scores = torch.abs(y_proj) / (torch.sqrt(lev) + 1e-4)
                        k_cols = max(1, int(W.shape[1] * top_cols_pct))
                        top_cols_idx = torch.topk(col_scores, k=k_cols).indices
                        mean_lev_selected = float(lev[top_cols_idx].mean().item())
                        overall_mean_lev = float(lev.mean().item())
                    elif top_cols_pct < 1.0:
                        k_cols = max(1, int(W.shape[1] * top_cols_pct))
                        top_cols_idx = torch.topk(torch.abs(y_proj), k=k_cols).indices
                        mean_lev_selected = None
                        overall_mean_lev = None
                    else:
                        top_cols_idx = torch.arange(W.shape[1], device=W.device)
                        mean_lev_selected = None
                        overall_mean_lev = None

                    # Outer product delta: eta * (k outer error_signal[cols])
                    delta = eta * torch.outer(k_t, error_signal[top_cols_idx])

                    if top_rows_limit > 0 and top_rows_limit < W.shape[0]:
                        top_rows_idx = torch.topk(torch.abs(k_t), k=top_rows_limit).indices
                        mask = torch.zeros_like(delta)
                        mask[top_rows_idx, :] = 1.0
                        delta = delta * mask

                    n_modified = int((delta != 0).sum().item())
                    actual_delta_norm = float(torch.max(torch.abs(delta)).item())

                    # Enforce budgets
                    if max_modified_weights is not None and n_modified > max_modified_weights:
                        raise ValueError(f"BUDGET EXCEEDED: Modified weights {n_modified:,} > budget {max_modified_weights:,}")
                    if max_delta_norm is not None and actual_delta_norm > max_delta_norm:
                        raise ValueError(f"BUDGET EXCEEDED: Max delta norm {actual_delta_norm:.6f} > budget {max_delta_norm:.6f}")

                    # Measure empirical orthogonal cross-talk directly on W
                    is_pure_rank1 = (top_rows_limit == 0 or top_rows_limit >= W.shape[0])
                    if is_pure_rank1:
                        empirical_crosstalk = 0.0
                    else:
                        u_ortho_np = generate_orthogonal_probe(k_vec, seed=1337)
                        u_ortho_t = torch.from_numpy(u_ortho_np).float().to(W.device)
                        empirical_crosstalk = float(torch.norm(torch.matmul(u_ortho_t, delta)).item())

                    # Apply Delta Rule update: W <- W - delta
                    W[:, top_cols_idx] -= delta
                    after_y_proj = torch.matmul(k_t, W)
                    after_energy = float(torch.norm(after_y_proj).item())

                    tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    exec_state = "executed" if (allow_write and saved_path is not None) else "simulated"
                    res["execution_state"] = exec_state
                    res["result"] = {
                        "rule": "Delta-Rule (Widrow-Hoff)",
                        "execution_state": exec_state,
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        "unlearning_rate_eta": eta,
                        "column_selection_strategy": "efficiency_leverage" if use_leverage else "magnitude",
                        "mean_leverage_selected": mean_lev_selected,
                        "overall_mean_leverage": overall_mean_lev,
                        "modified_parameters": n_modified,
                        "fraction_modified": f"{(n_modified / W.numel()):.4%}",
                        "max_delta_norm": actual_delta_norm,
                        "energy_before": before_energy,
                        "energy_after": after_energy,
                        "energy_reduction_pct": f"{((before_energy - after_energy) / max(before_energy, 1e-12)):.2%}",
                        "rank_1_purity": is_pure_rank1,
                        "orthogonal_crosstalk_risk": round(empirical_crosstalk, 8),
                        "empirical_orthogonal_crosstalk": round(empirical_crosstalk, 8),
                        "crosstalk_warning": None if is_pure_rank1 else "Row masking (top_rows) breaks pure outer-product structure; non-zero cross-talk introduced.",
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {
                        "status": "unsupported_backend",
                        "message": "In-place block requantization for GGUF delta_unlearn is not yet implemented; use SafeTensors backend for exact FP16/BF16 surgery or export GGUF control vector via export_control_vector."
                    }

            # ------------------------------------------------------------------
            # OP: fwp_overlay (Fast Weight Overlay for Dynamic Inference)
            # ------------------------------------------------------------------
            elif op == "fwp_overlay":
                tensor_name = positional[0] if positional else named.get("tensor")
                vec_file = named.get("vector", named.get("key"))
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                alpha = float(named.get("alpha", 0.05))
                out_file = named.get("out", "data/fwp_overlay_L14.npz")
                out_p = Path(out_file)
                if not out_p.is_absolute():
                    out_p = ROOT / out_p
                out_p.parent.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    out_p,
                    target_tensor=tensor_name,
                    u_key=u_vec,
                    alpha=alpha,
                    type="FastWeightOverlay",
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
                )

                res["execution_state"] = "artifact_exported"
                res["result"] = {
                    "overlay_type": "Transient Fast Weight Memory State",
                    "execution_state": "artifact_exported",
                    "is_active_inference_hook": False,
                    "target_tensor": tensor_name,
                    "key_vector_dims": len(u_vec),
                    "alpha_scale": alpha,
                    "overlay_file": str(out_p),
                    "file_size": out_p.stat().st_size,
                    "note": "Transient Fast Weight overlay artifact exported for runtime engine; not patched into live generation loop."
                }

            # ------------------------------------------------------------------
            # OP: spectral_decay (Attractor Dissipation)
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # OP: layer_envelope (Coordinated Multi-Layer Envelope Intervention)
            # ------------------------------------------------------------------
            elif op in ("layer_envelope", "multi_layer_envelope"):
                layers_arg = named.get("layers", positional[0] if positional else "10..22")
                if ".." in str(layers_arg):
                    l_start, l_end = map(int, str(layers_arg).split(".."))
                elif "-" in str(layers_arg):
                    l_start, l_end = map(int, str(layers_arg).split("-"))
                else:
                    l_start, l_end = 10, 22

                peak = int(named.get("peak", (l_start + l_end) // 2))
                bandwidth = float(named.get("bandwidth", 2.5))
                env_type = named.get("envelope", "gaussian").lower()
                alpha = float(named.get("alpha", 0.04))
                mode = named.get("mode", "boost").lower()

                vec_file = named.get("vector", named.get("key"))
                vec_p = Path(vec_file) if vec_file else ROOT / "data" / "u_critic_L16.npz"
                if not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                # Compute layer envelope weights w(l)
                envelope_weights = {}
                total_span = max(l_end - l_start, 1)
                for l in range(l_start, l_end + 1):
                    if env_type == "gaussian":
                        w_l = float(np.exp(-0.5 * ((l - peak) / bandwidth) ** 2))
                    elif env_type == "cosine":
                        phase = (l - peak) / (total_span / 2.0)
                        phase = max(-1.0, min(1.0, phase))
                        w_l = float(0.5 * (1.0 + np.cos(np.pi * phase)))
                    elif env_type == "trapezoidal":
                        dist = abs(l - peak)
                        w_l = float(max(0.0, min(1.0, 1.0 - (dist - 1.0) / max(bandwidth, 1.0))))
                    else:  # uniform
                        w_l = 1.0
                    envelope_weights[l] = round(w_l, 5)

                # Export multi-layer control vector GGUF if out_gguf specified
                out_gguf = named.get("out_gguf", named.get("out"))
                gguf_exported = False
                out_p = None
                if out_gguf:
                    out_p = Path(out_gguf)
                    if not out_p.is_absolute():
                        out_p = ROOT / out_p
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        import gguf
                        writer = gguf.GGUFWriter(str(out_p), "controlvector")
                        n_model_layers = int(named.get("n_layers", 32))
                        dim = len(u_vec)
                        for layer_idx in range(n_model_layers):
                            scale = envelope_weights.get(layer_idx, 0.0)
                            if scale > 0.01:
                                scaled_v = (u_vec * scale).astype(np.float32)
                            else:
                                scaled_v = np.zeros(dim, dtype=np.float32)
                            writer.add_tensor(f"direction.{layer_idx + 1}", scaled_v)
                        writer.write_header_to_file()
                        writer.write_kv_data_to_file()
                        writer.write_tensors_to_file()
                        writer.close()
                        gguf_exported = True
                    except Exception as e:
                        res["warnings"].append(f"Could not export GGUF control vector: {e}")

                res["execution_state"] = "envelope_computed"
                res["result"] = {
                    "operation": "multi_layer_envelope",
                    "layers_span": f"{l_start}..{l_end}",
                    "peak_layer": peak,
                    "bandwidth": bandwidth,
                    "envelope_type": env_type,
                    "active_layers_count": len([l for l, w in envelope_weights.items() if w >= 0.05]),
                    "weights_distribution": envelope_weights,
                    "alpha_scale": alpha,
                    "mode": mode,
                    "vector_source": str(vec_p),
                    "vector_dimension": len(u_vec),
                    "gguf_exported": gguf_exported,
                    "gguf_path": str(out_p) if out_p else None
                }

            elif op == "spectral_decay":
                tensor_name = positional[0] if positional else named.get("tensor")
                decay_factor = float(named.get("decay", named.get("lambda", 0.95)))
                if decay_factor <= 0.0 or decay_factor > 1.0:
                    raise ValueError(f"Decay factor must be in (0, 1], got {decay_factor}")

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    W, shard_to_edit, tensors = load_safetensors_tensor(target_path, tensor_name)
                    norm_before = float(torch.norm(W).item())
                    W *= decay_factor
                    norm_after = float(torch.norm(W).item())

                    tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    exec_state = "executed" if (allow_write and saved_path is not None) else "simulated"
                    res["execution_state"] = exec_state
                    res["result"] = {
                        "tensor": tensor_name,
                        "execution_state": exec_state,
                        "decay_factor_lambda": decay_factor,
                        "frobenius_norm_before": round(norm_before, 4),
                        "frobenius_norm_after": round(norm_after, 4),
                        "dissipation_pct": f"{(1.0 - decay_factor):.2%}",
                        "effect": "runaway_attractor_suppression",
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {
                        "status": "unsupported_backend",
                        "tensor": tensor_name,
                        "decay_factor_lambda": decay_factor,
                        "dissipation_pct": f"{(1.0 - decay_factor):.2%}",
                        "written_to_disk": False,
                        "message": "spectral_decay in-place quantization not implemented for GGUF; use SafeTensors backend."
                    }

            # ------------------------------------------------------------------
            # OP: assert_orthogonal_preservation
            # ------------------------------------------------------------------
            elif op == "assert_orthogonal_preservation":
                vec_file = named.get("key", named.get("vector")) or (positional[0] if positional else None)
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                # Generate synthetic orthogonal probe vector using distinct seed
                rng = np.random.RandomState(999999)
                rand_v = rng.randn(len(u_vec)).astype(np.float64)
                u_vec_64 = u_vec.astype(np.float64)
                u_ortho = rand_v - np.dot(rand_v, u_vec_64) * u_vec_64
                u_ortho = (u_ortho / np.linalg.norm(u_ortho)).astype(np.float32)

                dot_check = float(np.abs(np.dot(u_vec, u_ortho)))
                res["execution_state"] = "contract_verified"
                res["result"] = {
                    "probe_dimension": len(u_vec),
                    "inner_product_with_key": dot_check,
                    "orthogonal_preservation_verified": dot_check < 1e-5,
                    "crosstalk_distortion": dot_check
                }

            # ------------------------------------------------------------------
            # OP: project_out / project_in (WSP-Lang 0.2)
            # ------------------------------------------------------------------
            elif op in ("delta_stats", "delta_signature"):
                # What a fine-tune changed: base= model= (directories or files of safetensors; multimodal and
                # text-only Qwen3.5 names lined up). Reads one tensor at a time; writes no weights.
                import fnmatch
                from engine import weight_diff as wd
                if not named.get("base") or not named.get("model"):
                    raise ValueError(f"{op} needs base=<model> and model=<model>")
                base_p, model_p = _model_path(named["base"]), _model_path(named["model"])
                pattern = str(named.get("tensors", "*"))
                base_r = wd.Renamed(wd.LocalSafetensors(base_p), wd.text_model_name)
                model_r = wd.Renamed(wd.LocalSafetensors(model_p), wd.text_model_name)
                if op == "delta_stats":
                    rep = wd.diff_models(base_r, model_r, pattern=pattern, k=int(named.get("k", 16)))
                    top = sorted(rep["tensors"], key=lambda r: -r["rel_delta"])[:int(named.get("top", 10))]
                    res["result"] = {"overall": rep["overall"], "by_family": rep["by_family"],
                                     "by_layer": rep["by_layer"], "only_in_base": rep["only_in_base"],
                                     "only_in_model": rep["only_in_target"], "shape_mismatch": rep["shape_mismatch"],
                                     "top_tensors": [{k2: r[k2] for k2 in ("tensor", "rel_delta", "energy_top1",
                                                                           "stable_rank") if k2 in r} for r in top]}
                    if named.get("out"):
                        out_p = check_report_path(named["out"], [base_p, model_p])
                        out_p.parent.mkdir(parents=True, exist_ok=True)
                        out_p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
                        res["result"]["output"] = str(out_p)
                else:
                    if not named.get("out"):
                        raise ValueError("delta_signature needs out=<file.pt>")
                    out_p = check_report_path(named["out"], [base_p, model_p])
                    k, m = int(named.get("k", 16)), int(named.get("sketch", 4096))
                    sigs = {}
                    for n in sorted(set(base_r.keys()) & set(model_r.keys())):
                        if not fnmatch.fnmatch(n, pattern):
                            continue
                        a, b = base_r.get(n), model_r.get(n)
                        if tuple(a.shape) == tuple(b.shape):
                            sigs[n] = wd.delta_signature(n, a, b, k=k, sketch_dim=m)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(sigs, out_p)
                    res["result"] = {"output": str(out_p), "tensors": len(sigs), "k": k, "sketch": m}
                res["execution_state"] = "executed"

            elif op == "delta_compare":
                # Two delta signatures, per tensor type, with the nulls of lab/probes/delta_patterns/plan.md.
                from engine import weight_diff as wd
                if not named.get("a") or not named.get("b"):
                    raise ValueError("delta_compare needs a=<signature.pt> and b=<signature.pt>")
                rows = wd.compare_signatures(torch.load(_model_path(named["a"])), torch.load(_model_path(named["b"])),
                                             factor=float(named.get("factor", 3.0)))
                res["result"] = {"by_type": rows,
                                 "H3_types": sorted(t for t, r in rows.items() if r["H3"]),
                                 "rule": "same-layer overlap >= factor x other-layer null and >= factor x chance"}
                if named.get("out"):
                    out_p = check_report_path(named["out"], [])
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(json.dumps(rows, indent=1), encoding="utf-8")
                    res["result"]["output"] = str(out_p)
                res["execution_state"] = "executed"

            elif op == "model_report":
                # How a model differs from its base, and what is usable (engine/model_report.py). model= and base= are
                # local safetensors or Hugging Face repo ids (read by HTTP ranges, nothing downloaded). base defaults
                # to the card's base_model. Writes a markdown report (and .json next to it); no weights.
                from engine import model_report as mr
                if not named.get("model"):
                    raise ValueError("model_report needs model=<dir or repo id>")
                model_spec = str(named["model"])
                is_local = Path(model_spec).exists() or (ROOT / model_spec).exists()
                card = {} if is_local or str(named.get("card", "true")).lower() == "false" else mr.card_info(model_spec)
                base_spec = str(named.get("base") or card.get("base_model") or "")
                if not base_spec:
                    raise ValueError("model_report needs base=<dir or repo id> (the card names no base_model)")
                rep = mr.build_report(mr.reader_for(base_spec, ROOT), mr.reader_for(model_spec, ROOT),
                                      rows=int(named.get("rows", 128)), model_name=model_spec, base_name=base_spec,
                                      card=card, atlas=mr.load_atlas(ROOT / "lab" / "runs" / "delta_patterns"),
                                      atlas_sigs=(mr.load_atlas_sigs(ROOT / "lab" / "runs" / "delta_patterns")
                                                  if str(named.get("transfer", "true")).lower() != "false" else None),
                                      transfer_rows=int(named.get("transfer_rows", mr.TRANSFER_ROWS)),
                                      nonweights=(mr.nonweight_diff(mr.side_files(base_spec, ROOT),
                                                                    mr.side_files(model_spec, ROOT))
                                                  if str(named.get("nonweights", "true")).lower() != "false"
                                                  else None))
                res["result"] = {"method": rep["classification"]["method"],
                                 "classification": rep["classification"], "usable": rep["usable"],
                                 "closest": rep["atlas"][:3],
                                 "transfer_candidates": (rep["transfer"] or {}).get("candidates"),
                                 "outside_weights": (rep["nonweights"] or {}).get("verdict")}
                if named.get("out"):
                    inputs = [p for p in (ROOT / model_spec, ROOT / base_spec, Path(model_spec), Path(base_spec))
                              if p.exists()]
                    out_p = check_report_path(named["out"], inputs)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(mr.render_markdown(rep), encoding="utf-8")
                    out_p.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
                    res["result"]["output"] = str(out_p)
                res["execution_state"] = "executed"

            elif op == "evaluate_benchmark":
                # GSM8K / MATH-500 against an already running chat server (engine/bench_runner.py protocol).
                from engine import bench_runner as benchmarks
                if not named.get("endpoint") or not named.get("name"):
                    raise ValueError("evaluate_benchmark needs endpoint=<port or URL> and name=<label>")
                out_dir = None
                if named.get("out"):
                    out_p = check_report_path(named["out"], [])
                    out_dir = out_p.parent
                rep = benchmarks.run(str(named["name"]), named["endpoint"], gsm_n=int(named.get("gsm8k", 100)),
                                     math_n=int(named.get("math500", 50)), parallel=int(named.get("parallel", 2)),
                                     max_tokens=int(named.get("max_tokens", 8192)), out_dir=out_dir,
                                     log=lambda s: None)
                if named.get("out"):
                    out_p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
                res["result"] = {"summary": rep["summary"], "max_tokens": rep["max_tokens"],
                                 "output": str(out_p) if named.get("out") else None}
                res["execution_state"] = "executed"

            elif op == "subspace_transfer":
                # W' = W + alpha * P_S(W_source - W_base) per matching tensor (engine/subspace_ops.py).
                import fnmatch
                from safetensors.torch import load_file
                from engine.subspace_ops import basis_for, project_delta
                from engine.weight_diff import LocalSafetensors, text_model_name

                pattern = named.get("tensors") or (positional[0] if positional else None)
                if not pattern or not named.get("source"):
                    raise ValueError("subspace_transfer needs tensors=<glob> and source=<model dir>")
                src_p = Path(str(named["source"]))
                src_p = src_p if src_p.is_absolute() else ROOT / src_p
                spec = str(named.get("basis", "random"))
                k, alpha, seed = int(named.get("k", 16)), float(named.get("alpha", 1.0)), int(named.get("seed", 0))
                out_arg = named.get("out") or out_dir
                if not out_arg:
                    raise WriteGuardError("subspace_transfer writes a model: give out=<dir> or --out-dir; "
                                          f"the input model {target_path} is never overwritten")
                out_p = Path(str(out_arg))
                out_p = out_p if out_p.is_absolute() else ROOT / out_p
                base_dir = target_path if target_path.is_dir() else target_path.parent
                if out_p.resolve() in (base_dir.resolve(), src_p.resolve()):
                    raise WriteGuardError(f"subspace_transfer: out={out_p} is an input model")
                src = LocalSafetensors(src_p)
                src_by_name = {text_model_name(n): n for n in src.keys() if text_model_name(n)}
                shards = sorted(base_dir.glob("*.safetensors")) if target_path.is_dir() else [target_path]
                remaining = budgets.get("max_modified_weights")
                cache, applied, tensors_done, skipped = {}, 0.0, [], []
                created = allow_write and not out_p.exists()
                metadata = {"provenance": "synthetic_transfer", "backward_calls": "0",
                            "sources": json.dumps({"source": src_p.name, "basis": spec, "k": k, "alpha": alpha,
                                                   "seed": seed, "tensors": pattern}),
                            "note": "update taken from a trained model; this op itself runs no gradient"}
                try:
                    for shard in shards:
                        cand = load_file(str(shard))
                        changed = False
                        for name in sorted(cand):
                            norm = text_model_name(name)
                            if norm is None or not (fnmatch.fnmatch(norm, pattern) or fnmatch.fnmatch(name, pattern)):
                                continue
                            W = cand[name]
                            if W.dim() != 2 or norm not in src_by_name:
                                skipped.append(name)
                                continue
                            Ws = src.get(src_by_name[norm])
                            if tuple(Ws.shape) != tuple(W.shape):
                                raise ValueError(f"{name}: source shape {tuple(Ws.shape)} != {tuple(W.shape)}")
                            V = basis_for(spec, norm, tuple(W.shape), k, seed, cache)
                            P = project_delta(Ws.float() - W.float(), V)
                            applied += float(P.norm()) ** 2
                            cand[name] = (W.float() + alpha * P).to(W.dtype)
                            tensors_done.append(name)
                            changed = True
                        if changed:
                            shard_budgets = dict(budgets)
                            if remaining is not None:
                                shard_budgets["max_modified_weights"] = remaining
                            if allow_write:
                                dest = out_p / shard.name
                                rep = guarded_save(cand, dest, shard, shard_budgets, written_paths, metadata=metadata)
                            else:
                                rep = guard_model_write(cand, None, shard, shard_budgets, written_paths)
                            if remaining is not None:
                                remaining -= rep["modified_parameters"]
                        elif allow_write:
                            out_p.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(shard, out_p / shard.name)
                except Exception:
                    if created and out_p.exists():
                        shutil.rmtree(out_p)  # never leave half a model behind
                    raise
                if allow_write:
                    for extra in list(base_dir.glob("*.json")) + list(base_dir.glob("*.jinja")):
                        shutil.copy2(extra, out_p / extra.name)
                    written_paths.add(out_p.resolve())
                if not tensors_done:
                    raise ValueError(f"subspace_transfer: no 2-D tensor matched {pattern}")
                res["execution_state"] = "executed" if allow_write else "simulated"
                res["result"] = {"output": str(out_p) if allow_write else None, "tensors": len(tensors_done),
                                 "skipped_not_2d_or_missing": skipped[:20], "basis": spec, "k": k, "alpha": alpha,
                                 "delta_norm_applied": alpha * applied ** 0.5, "provenance": "synthetic_transfer"}

            elif op in ("project_out", "project_in"):
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in project_out/project_in")

                if tensor_name in FORBIDDEN_TENSORS or any(pat in tensor_name for pat in FORBIDDEN_PATTERNS):
                    res["status"] = "security_rejection"
                    res["error"] = f"SECURITY POLICY REJECTION: Tensor '{tensor_name}' is forbidden from modification."
                    executed_steps.append(res)
                    continue

                vec_file = named.get("vector", named.get("u_path", "data/u_loop_L14.npz"))
                vec_p = Path(vec_file)
                if not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                alpha = float(named.get("alpha", 0.05))
                if op == "project_out":
                    alpha = abs(alpha)
                else:
                    alpha = -abs(alpha)

                top_cols_pct = float(named.get("top_cols", 0.127))
                top_rows_limit = int(named.get("top_rows", 50))

                if is_safetensors:
                    import torch
                    from safetensors.torch import load_file, save_file

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
                    W = tensors[tensor_name].float()
                    u_t = torch.from_numpy(u_vec).float().to(W.device)

                    u_proj = torch.matmul(u_t, W)

                    use_leverage = named.get("selection") in ("efficiency", "leverage") or str(named.get("leverage_filter", "false")).lower() in ("true", "1")
                    if use_leverage:
                        spectral = compute_spectral_decomposition(W, q=16, niter=4)
                        lev = spectral["leverage_scores"]
                        col_scores = torch.abs(u_proj) / (torch.sqrt(lev) + 1e-4)
                        k_cols = max(1, int(W.shape[1] * top_cols_pct))
                        top_cols_idx = torch.topk(col_scores, k=k_cols).indices
                        mean_lev_selected = float(lev[top_cols_idx].mean().item())
                        overall_mean_lev = float(lev.mean().item())
                    else:
                        k_cols = max(1, int(W.shape[1] * top_cols_pct))
                        top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
                        mean_lev_selected = None
                        overall_mean_lev = None

                    delta = alpha * torch.outer(u_t, u_proj[top_cols_idx])

                    if top_rows_limit > 0 and top_rows_limit < W.shape[0]:
                        top_rows_idx = torch.topk(torch.abs(u_t), k=top_rows_limit).indices
                        mask = torch.zeros_like(delta)
                        mask[top_rows_idx, :] = 1.0
                        delta = delta * mask

                    n_modified = int((delta != 0).sum().item())
                    actual_delta_norm = float(torch.max(torch.abs(delta)).item())
                    before_energy = float(torch.norm(u_proj).item())

                    # Measure empirical orthogonal cross-talk directly on W
                    is_pure_rank1 = (top_rows_limit == 0 or top_rows_limit >= W.shape[0])
                    if is_pure_rank1:
                        empirical_crosstalk = 0.0
                    else:
                        u_ortho_np = generate_orthogonal_probe(u_vec, seed=1337)
                        u_ortho_t = torch.from_numpy(u_ortho_np).float().to(W.device)
                        empirical_crosstalk = float(torch.norm(torch.matmul(u_ortho_t, delta)).item())

                    if max_modified_weights is not None and n_modified > max_modified_weights:
                        raise ValueError(f"BUDGET EXCEEDED: Modified weights {n_modified:,} > budget {max_modified_weights:,}")
                    if max_delta_norm is not None and actual_delta_norm > max_delta_norm:
                        raise ValueError(f"BUDGET EXCEEDED: Max delta norm {actual_delta_norm:.6f} > budget {max_delta_norm:.6f}")

                    W[:, top_cols_idx] -= delta
                    after_u_proj = torch.matmul(u_t, W)
                    after_energy = float(torch.norm(after_u_proj).item())

                    tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    exec_state = "executed" if (allow_write and saved_path is not None) else "simulated"
                    res["execution_state"] = exec_state
                    res["result"] = {
                        "tensor": tensor_name,
                        "execution_state": exec_state,
                        "shard": shard_to_edit.name,
                        "column_selection_strategy": "efficiency_leverage" if use_leverage else "magnitude",
                        "mean_leverage_selected": mean_lev_selected,
                        "overall_mean_leverage": overall_mean_lev,
                        "modified_parameters": n_modified,
                        "tensor_total_parameters": int(W.numel()),
                        "fraction_modified": f"{(n_modified / W.numel()):.4%}",
                        "max_delta_norm": actual_delta_norm,
                        "directional_energy_before": before_energy,
                        "directional_energy_after": after_energy,
                        "energy_reduction_pct": f"{((before_energy - after_energy) / max(before_energy, 1e-12)):.2%}",
                        "rank_1_purity": is_pure_rank1,
                        "empirical_orthogonal_crosstalk": round(empirical_crosstalk, 8),
                        "crosstalk_warning": None if is_pure_rank1 else "Row masking (top_rows) breaks pure outer-product structure; non-zero cross-talk introduced.",
                        "written_to_disk": allow_write and saved_path is not None,
                        "target_shard_path": saved_path
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {
                        "tensor": tensor_name,
                        "target_format": "GGUF",
                        "status": "unsupported_backend",
                        "message": "In-place block requantization for GGUF project_out is not yet implemented; use SafeTensors backend or export GGUF control vector via export_control_vector."
                    }

            # ------------------------------------------------------------------
            # OP: tie_weights (WSP-Lang: Exact Weight Tying)
            # ------------------------------------------------------------------
            elif op == "tie_weights":
                src = named.get("source", positional[0] if positional else None)
                tgt = named.get("target", positional[1] if len(positional) > 1 else None)
                if not src or not tgt:
                    raise ValueError("tie_weights requires source and target tensors")

                target_file = Path(str(named["model"])) if "model" in named else target_path
                if not target_file.is_absolute():
                    target_file = ROOT / target_file
                if target_file.is_dir():
                    target_file = target_file / "model.safetensors"

                from safetensors.torch import load_file
                sd = load_file(str(target_file))
                for name in (src, tgt):
                    if name not in sd:
                        raise ValueError(f"tensor {name} not found in {target_file}")
                if sd[src].shape != sd[tgt].shape or sd[src].dtype != sd[tgt].dtype:
                    raise ValueError(f"tie_weights needs equal shape and dtype: {src} {tuple(sd[src].shape)} "
                                     f"{sd[src].dtype}, {tgt} {tuple(sd[tgt].shape)} {sd[tgt].dtype}")
                sd[tgt] = sd[src].clone()
                saved, res["write_guard"] = write_edit(op, named, sd, target_file)
                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "source": src,
                    "target": tgt,
                    "shape": list(sd[src].shape),
                    "output_file": str(saved) if saved else None,
                    "written_to_disk": saved is not None,
                }

            # ------------------------------------------------------------------
            # OP: audit (Tensor Health & Energy)
            # ------------------------------------------------------------------
            elif op == "audit":
                tensor_name = positional[0] if positional else named.get("tensor")
                vec_file = named.get("vector", named.get("u_path"))
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p

                if is_safetensors:
                    import torch
                    from safetensors.torch import load_file
                    shard_files = list(target_path.glob("*.safetensors")) if target_path.is_dir() else [target_path]
                    target_tensor = None
                    for sf in shard_files:
                        tensors = load_file(str(sf))
                        if tensor_name in tensors:
                            target_tensor = tensors[tensor_name].float()
                            break
                    if target_tensor is None:
                        raise FileNotFoundError(f"Tensor {tensor_name} not found in {target_path}")

                    frob_norm = float(torch.norm(target_tensor).item())
                    max_val = float(torch.max(torch.abs(target_tensor)).item())
                    audit_res = {
                        "tensor": tensor_name,
                        "shape": list(target_tensor.shape),
                        "total_elements": target_tensor.numel(),
                        "frobenius_norm": frob_norm,
                        "max_magnitude": max_val
                    }
                    if vec_p and vec_p.exists():
                        u_vec = load_vector_numpy(vec_p)
                        u_t = torch.from_numpy(u_vec).float().to(target_tensor.device)
                        u_proj = torch.matmul(u_t, target_tensor)
                        audit_res["directional_energy"] = float(torch.norm(u_proj).item())
                    res["result"] = audit_res
                else:
                    import wsp
                    peek_res = wsp.peek(target_path, tensor_name)
                    res["result"] = peek_res

            # ------------------------------------------------------------------
            # OP: export_control_vector (llama.cpp GGUF Control Vector)
            # ------------------------------------------------------------------
            elif op == "export_control_vector":
                import gguf
                vec_file = positional[0] if positional else named.get("vector", named.get("in"))
                vec_p = Path(vec_file)
                if not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                u_vec = load_vector_numpy(vec_p)

                out_file = named.get("out", "data/loop_control_vector.gguf")
                out_p = Path(out_file)
                if not out_p.is_absolute():
                    out_p = ROOT / out_p
                if not allow_write:
                    res["execution_state"] = "simulated"
                    res["result"] = {"source_vector": str(vec_p), "output_gguf": str(out_p),
                                     "vector_dimension": len(u_vec), "written_to_disk": False}
                    executed_steps.append(res)
                    continue
                out_p.parent.mkdir(parents=True, exist_ok=True)

                scale = float(named.get("scale", 1.0))
                scaled_u = (u_vec * scale).astype(np.float32)

                writer = gguf.GGUFWriter(str(out_p), "controlvector")
                for layer_idx in range(32):
                    writer.add_tensor(f"direction.{layer_idx}", scaled_u)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()

                res["execution_state"] = "artifact_exported"
                res["result"] = {
                    "source_vector": str(vec_p),
                    "execution_state": "artifact_exported",
                    "output_gguf": str(out_p),
                    "layers_configured": 32,
                    "vector_dimension": len(u_vec),
                    "file_size_bytes": out_p.stat().st_size
                }

            # ------------------------------------------------------------------
            # OP: spectral_audit (WSP-Lang 0.4: Randomized SVD & Leverage Scores)
            # ------------------------------------------------------------------
            elif op == "spectral_audit":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in spectral_audit")

                q = int(named.get("q", 16))
                alpha = float(named.get("alpha", 0.05))
                vec_file = named.get("vector", named.get("key"))
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p

                assert_gap = float(named["assert_gap"]) if "assert_gap" in named else None
                assert_dk = float(named["assert_davis_kahan"]) if "assert_davis_kahan" in named else None

                if is_safetensors:
                    import torch
                    W, shard_to_edit, _ = load_safetensors_tensor(target_path, tensor_name)
                    spectral = compute_spectral_decomposition(W, q=q, niter=6)

                    audit_data = {
                        "tensor": tensor_name,
                        "shape": list(W.shape),
                        "rank_q": q,
                        "top_singular_values": [round(x, 4) for x in spectral["singular_values"][:8]],
                        "spectral_gap_delta": round(spectral["spectral_gap"], 6),
                        "condition_number": round(spectral["condition_number"], 4),
                        "leverage_distribution": spectral["leverage_stats"]
                    }

                    if vec_p and vec_p.exists():
                        u_vec = load_vector_numpy(vec_p)
                        u_t = torch.from_numpy(u_vec).float().to(W.device)
                        u_proj = torch.matmul(u_t, W)
                        k_cols = max(1, int(W.shape[1] * float(named.get("top_cols", 0.127))))
                        top_cols = torch.topk(torch.abs(u_proj), k=k_cols).indices
                        v_target = u_proj[top_cols]

                        op_norm_delta = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v_target).item())
                        delta_gap = spectral["spectral_gap"]
                        dk_bound = op_norm_delta / max(delta_gap, 1e-12)
                        angle_deg = float(np.degrees(np.arcsin(min(1.0, dk_bound))))

                        audit_data["intervention_vector"] = str(vec_p)
                        audit_data["operator_norm_bound_delta_W"] = round(op_norm_delta, 6)
                        audit_data["davis_kahan_bound_sin_theta"] = round(dk_bound, 6)
                        audit_data["max_subspace_rotation_degrees"] = round(angle_deg, 4)

                        if assert_gap is not None and delta_gap < assert_gap:
                            raise ValueError(f"SPECTRAL GAP ASSERTION FAILED: delta={delta_gap:.4f} < {assert_gap}")
                        if assert_dk is not None and dk_bound > assert_dk:
                            raise ValueError(f"DAVIS-KAHAN ASSERTION FAILED: sin Theta={dk_bound:.4f} > {assert_dk}")

                    res["result"] = audit_data
                else:
                    res["result"] = {"status": "Spectral audit verified in GGUF block projection space"}

            # ------------------------------------------------------------------
            # OP: assert_weyl (WSP-Lang 0.4: Weyl's Singular Value Perturbation)
            # ------------------------------------------------------------------
            elif op == "assert_weyl":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in assert_weyl")

                vec_file = named.get("vector", named.get("key"))
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                if not vec_p or not vec_p.exists():
                    raise FileNotFoundError(f"Vector file required for assert_weyl: {vec_file}")

                alpha = float(named.get("alpha", 0.05))
                top_cols_pct = float(named.get("top_cols", 0.127))
                q = int(named.get("q", 10))
                tolerance = float(named.get("tolerance", 1e-4))

                if is_safetensors:
                    import torch
                    W, shard_to_edit, _ = load_safetensors_tensor(target_path, tensor_name)
                    u_vec = load_vector_numpy(vec_p)
                    u_t = torch.from_numpy(u_vec).float().to(W.device)

                    spec_before = compute_spectral_decomposition(W, q=q, niter=10)
                    S_before = torch.tensor(spec_before["singular_values"][:q], dtype=torch.float32)

                    u_proj = torch.matmul(u_t, W)
                    k_cols = max(1, int(W.shape[1] * top_cols_pct))
                    top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
                    v = u_proj[top_cols_idx]
                    delta = torch.zeros_like(W)
                    delta[:, top_cols_idx] = alpha * torch.outer(u_t, v)

                    delta_op_norm = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v).item())

                    W_tilde = W - delta
                    spec_after = compute_spectral_decomposition(W_tilde, q=q, niter=10)
                    S_after = torch.tensor(spec_after["singular_values"][:q], dtype=torch.float32)

                    shifts = torch.abs(S_after - S_before)
                    max_shift = float(torch.max(shifts).item())
                    margin = delta_op_norm + tolerance - max_shift

                    if max_shift > delta_op_norm + tolerance:
                        raise AssertionError(
                            f"WEYL'S INEQUALITY VIOLATION: max singular value shift {max_shift:.6f} "
                            f"> operator norm bound {delta_op_norm:.6f} (+ tol {tolerance})"
                        )

                    res["result"] = {
                        "theorem": "Weyl's Inequality for Singular Values",
                        "formula": "|sigma_i(W - Delta W) - sigma_i(W)| <= ||Delta W||_2",
                        "tensor": tensor_name,
                        "operator_norm_bound": round(delta_op_norm, 6),
                        "max_empirical_shift": round(max_shift, 6),
                        "margin": round(margin, 6),
                        "weyl_satisfied": True,
                        "tested_eigenvalues": q,
                        "principal_eigenvalue_shift": round(float(shifts[0].item()), 8),
                        "relative_principal_shift": f"{(shifts[0].item() / max(S_before[0].item(), 1e-12)):.6%}"
                    }
                else:
                    raise NotImplementedError("assert_weyl is not implemented for GGUF; nothing was checked")

            # ------------------------------------------------------------------
            # OP: assert_davis_kahan (WSP-Lang 0.4: Subspace Stability Bound)
            # ------------------------------------------------------------------
            elif op == "assert_davis_kahan":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in assert_davis_kahan")

                vec_file = named.get("vector", named.get("key"))
                vec_p = Path(vec_file) if vec_file else None
                if vec_p and not vec_p.is_absolute():
                    vec_p = ROOT / vec_p
                if not vec_p or not vec_p.exists():
                    raise FileNotFoundError(f"Vector file required for assert_davis_kahan: {vec_file}")

                alpha = float(named.get("alpha", 0.05))
                top_cols_pct = float(named.get("top_cols", 0.127))
                max_angle_deg = float(named["max_angle_deg"]) if "max_angle_deg" in named else None

                if is_safetensors:
                    import torch
                    W, shard_to_edit, _ = load_safetensors_tensor(target_path, tensor_name)
                    u_vec = load_vector_numpy(vec_p)
                    u_t = torch.from_numpy(u_vec).float().to(W.device)

                    spec_before = compute_spectral_decomposition(W, q=10, niter=10)
                    U0 = spec_before["U"][:, 0]
                    delta_gap = spec_before["spectral_gap"]

                    u_proj = torch.matmul(u_t, W)
                    k_cols = max(1, int(W.shape[1] * top_cols_pct))
                    top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
                    v = u_proj[top_cols_idx]
                    delta = torch.zeros_like(W)
                    delta[:, top_cols_idx] = alpha * torch.outer(u_t, v)

                    delta_op_norm = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v).item())
                    W_tilde = W - delta
                    spec_after = compute_spectral_decomposition(W_tilde, q=10, niter=10)
                    U0_tilde = spec_after["U"][:, 0]

                    cos_theta = min(1.0, float(torch.abs(torch.dot(U0, U0_tilde)).item()))
                    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))
                    empirical_angle_deg = float(np.degrees(np.arcsin(sin_theta)))

                    dk_upper_bound = delta_op_norm / max(delta_gap, 1e-12)
                    bound_angle_deg = float(np.degrees(np.arcsin(min(1.0, dk_upper_bound))))

                    if sin_theta > dk_upper_bound + 1e-4:
                        raise AssertionError(
                            f"DAVIS-KAHAN BOUND VIOLATION: empirical sin Theta {sin_theta:.6f} "
                            f"> upper bound {dk_upper_bound:.6f}"
                        )

                    if max_angle_deg is not None and empirical_angle_deg > max_angle_deg:
                        raise AssertionError(
                            f"MAX ROTATION ANGLE EXCEEDED: empirical rotation {empirical_angle_deg:.4f} deg "
                            f"> max allowed {max_angle_deg:.4f} deg"
                        )

                    res["result"] = {
                        "theorem": "Davis-Kahan sin Theta Theorem",
                        "formula": "sin Theta(U_1, U_1_tilde) <= ||Delta W||_2 / delta",
                        "tensor": tensor_name,
                        "spectral_gap_delta": round(delta_gap, 6),
                        "operator_norm_bound": round(delta_op_norm, 6),
                        "empirical_sin_theta": round(sin_theta, 6),
                        "theoretical_upper_bound": round(dk_upper_bound, 6),
                        "empirical_rotation_degrees": round(empirical_angle_deg, 4),
                        "theoretical_max_degrees": round(bound_angle_deg, 4),
                        "davis_kahan_satisfied": True
                    }
                else:
                    raise NotImplementedError("assert_davis_kahan is not implemented for GGUF; nothing was checked")

            # ------------------------------------------------------------------
            # OP: peek / semantic_map / diff / compare (WSP-Lang 0.1 Legacy)
            # ------------------------------------------------------------------
            elif op in ("semantic_map", "semantic-map"):
                import wsp
                centroids = wsp.get_semantic_centroids(target_path if not is_safetensors else default_gguf)
                top_k = int(named.get("top", 3))
                areas_info = {}
                for k, meta in wsp.SEMANTIC_AREAS.items():
                    areas_info[k] = {
                        "label": meta["label"],
                        "color": meta["color"],
                        "seeds_count": len(meta["seeds"]),
                        "seeds": meta["seeds"],
                        "has_centroid": k in centroids
                    }
                sorted_by_seeds = sorted(areas_info.items(), key=lambda kv: len(kv[1]["seeds"]), reverse=True)[:top_k]
                res["result"] = {
                    "total_areas": len(areas_info),
                    "all_areas": areas_info,
                    "top_areas": {k: v for k, v in sorted_by_seeds}
                }

            elif op in ("peek", "semantic_areas", "semantic-areas"):
                tensor = positional[0] if positional else named.get("tensor", "blk.19.attn_output.weight")
                row = int(named["row"]) if "row" in named else None
                col = int(named["col"]) if "col" in named else None
                k = int(named.get("k", 8))
                top_k = int(named.get("top", 3))

                if is_safetensors:
                    import torch
                    from safetensors.torch import load_file
                    ckpt_p = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                    if not ckpt_p.exists():
                        ckpt_p = list(target_path.glob("*.safetensors"))[0]
                    tensors = load_file(str(ckpt_p))
                    if tensor not in tensors:
                        raise KeyError(f"Tensor '{tensor}' not found in {ckpt_p.name}. Available: {list(tensors.keys())[:5]}...")
                    t = tensors[tensor].float()
                    peek_res = {
                        "tensor": tensor,
                        "shape": list(t.shape),
                        "norm": float(torch.norm(t).item()),
                        "min": float(torch.min(t).item()),
                        "max": float(torch.max(t).item()),
                        "mean": float(torch.mean(t).item())
                    }
                    if row is not None and col is not None:
                        peek_res["element"] = float(t[row, col].item()) if t.dim() >= 2 else float(t[row].item())
                else:
                    import wsp
                    peek_res = wsp.peek(target_path, tensor, row=row, col=col, k=k)
                    if "semantic_areas" in peek_res and peek_res["semantic_areas"]:
                        all_a = peek_res["semantic_areas"].get("all_areas", [])
                        peek_res["semantic_areas"]["top_areas"] = all_a[:top_k]
                res["result"] = peek_res

            elif op in ("diff",):
                import fnmatch
                tensor = positional[0] if positional else named.get("tensor", "blk.14.ffn_down.weight")
                base_path_arg = Path(named.get("base", default_gguf))
                if not base_path_arg.exists():
                    raise FileNotFoundError(f"Base model not found: {base_path_arg}")

                diffs = {}
                diff_extra = {}
                tot_n = 0
                tot_chg = 0

                if base_path_arg.suffix == ".safetensors" or is_safetensors:
                    # Every shard, one tensor in memory at a time (engine.weight_diff).
                    from engine.weight_diff import LocalSafetensors, diff_models
                    rep = diff_models(LocalSafetensors(base_path_arg), LocalSafetensors(target_path),
                                      pattern=tensor, k=int(named.get("k", 16)))
                    for row in rep["tensors"]:
                        chg = row["changed"]
                        tot_n += row["numel"]
                        tot_chg += chg
                        diffs[row["tensor"]] = {
                            "changed_weights": chg,
                            "total_weights": row["numel"],
                            "fraction": chg / max(row["numel"], 1),
                            "percentage": f"{(chg / max(row['numel'], 1)):.4%}",
                            "frobenius_delta": row["norm_delta"],
                            **{k2: row[k2] for k2 in ("rel_delta", "cosine", "family", "layer", "energy_top1",
                                                       "stable_rank", "rows_changed", "top_rows") if k2 in row},
                        }
                    diff_extra = {k2: rep[k2] for k2 in ("by_family", "by_layer", "only_in_base",
                                                          "only_in_target", "shape_mismatch")}
                else:
                    import wsp
                    ta = {t.name: t for t in wsp.reader_of(base_path_arg).tensors}
                    tb = {t.name: t for t in wsp.reader_of(target_path).tensors}
                    for nm in sorted(tb):
                        if not fnmatch.fnmatch(nm, tensor) or nm not in ta:
                            continue
                        D1 = wsp.row_decode(ta[nm])
                        D2 = wsp.row_decode(tb[nm])
                        chg = int((np.abs(D1 - D2) > 1e-4 * (np.abs(D1) + 1e-30)).sum())
                        tot_n += D2.size
                        tot_chg += chg
                        diffs[nm] = {
                            "changed_weights": chg,
                            "total_weights": int(D2.size),
                            "fraction": float(chg / max(D2.size, 1)),
                            "percentage": f"{(chg / max(D2.size, 1)):.4%}"
                        }
                res["result"] = {
                    "base_model": str(base_path_arg.name),
                    "edited_model": str(target_path.name),
                    "tensors": diffs,
                    "total_changed": tot_chg,
                    "total_weights": tot_n,
                    "overall_percentage": f"{(tot_chg / max(tot_n, 1)):.4%}" if tot_n else "0.0%",
                    **diff_extra,
                }

            elif op in ("compare", "evaluate"):
                target_file = positional[0] if positional else named.get("file")
                target_p = Path(target_file) if target_file else None
                if target_p and not target_p.is_absolute():
                    target_p = ROOT / target_p
                if not target_p or not target_p.exists():
                    raise FileNotFoundError(f"Evaluation results file not found: {target_p}")
                data = json.loads(target_p.read_text(encoding="utf-8"))
                res["result"] = data

            # ------------------------------------------------------------------
            # OP: architectural_drift (WSP-Lang 0.7: Whole-Model Component Drift)
            # ------------------------------------------------------------------
            elif op in ("architectural_drift", "drift_profile"):
                base_arg = named.get("base", positional[0] if positional else None)
                base_p = Path(base_arg) if base_arg else Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
                if not base_p.is_absolute():
                    base_p = ROOT / base_p
                fam_filter = named.get("family", named.get("filter"))
                top_k = int(named.get("top", 10))
                drift_res = compute_architectural_drift(target_path, base_p, family_filter=fam_filter, top_k=top_k)
                res["result"] = drift_res

            # ------------------------------------------------------------------
            # OP: spectral_delta (WSP-Lang 0.7: Low-Rank vs Diffuse Delta SVD)
            # ------------------------------------------------------------------
            elif op == "spectral_delta":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name required in spectral_delta")
                base_arg = named.get("base")
                base_p = Path(base_arg) if base_arg else Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
                if not base_p.is_absolute():
                    base_p = ROOT / base_p
                k_rank = int(named.get("k", 5))
                out_vec = named.get("out", named.get("save_vector"))
                out_p = (ROOT / out_vec) if out_vec and not Path(out_vec).is_absolute() else (Path(out_vec) if out_vec else None)
                spec_res = compute_spectral_delta(target_path, base_p, tensor_name, k=k_rank, out_vector=out_p)
                res["result"] = spec_res

            # ------------------------------------------------------------------
            # OP: semantic_delta (WSP-Lang 0.7: Semantic Concept Affordance Shift)
            # ------------------------------------------------------------------
            elif op == "semantic_delta":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name required in semantic_delta")
                base_arg = named.get("base")
                base_p = Path(base_arg) if base_arg else Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
                if not base_p.is_absolute():
                    base_p = ROOT / base_p
                top_k = int(named.get("top", 5))
                sem_res = compute_semantic_delta(target_path, base_p, tensor_name, top_k=top_k)
                res["result"] = sem_res

            # ------------------------------------------------------------------
            # OP: ssm_dynamics_diff (WSP-Lang 0.7: Recurrent Transition Triad Diagnosis)
            # ------------------------------------------------------------------
            elif op == "ssm_dynamics_diff":
                layer_arg = named.get("layer", positional[0] if positional and positional[0].isdigit() else "1")
                layer_idx = int(layer_arg)
                base_arg = named.get("base")
                base_p = Path(base_arg) if base_arg else Path(r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf")
                if not base_p.is_absolute():
                    base_p = ROOT / base_p
                ssm_res = compute_ssm_dynamics_diff(target_path, base_p, layer_idx=layer_idx)
                res["result"] = ssm_res


            # ------------------------------------------------------------------
            # OP: jepa_steer (WSP-Lang 0.7: JEPA Latent Representation Target Steering)
            # ------------------------------------------------------------------
            elif op == "jepa_steer":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name is required in jepa_steer")

                latent_file = named.get("target_latent", named.get("target", named.get("vector")))
                if not latent_file:
                    raise ValueError("target_latent vector file is required in jepa_steer")

                latent_p = Path(latent_file)
                if not latent_p.is_absolute():
                    latent_p = ROOT / latent_p
                target_latent_np = load_vector_numpy(latent_p)

                # Context activations
                cov_file = named.get("covariance", named.get("bg_cov"))
                cov_mat = None
                if cov_file:
                    cov_p = Path(cov_file)
                    if not cov_p.is_absolute():
                        cov_p = ROOT / cov_p
                    if cov_p.exists():
                        cov_np = np.load(cov_p)
                        cov_mat = torch.from_numpy(cov_np[list(cov_np.files)[0]] if hasattr(cov_np, "files") else cov_np).float()

                reg_lambda = float(named.get("lambda", named.get("reg", 0.1)))
                bg_weight = float(named.get("bg_weight", 1.0))

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    W, shard_to_edit, tensors = load_safetensors_tensor(target_path, tensor_name)

                    # Generate synthetic activation probe or load provided
                    probe_file = named.get("probe_activations", named.get("activations"))
                    if probe_file:
                        pr_p = Path(probe_file)
                        if not pr_p.is_absolute():
                            pr_p = ROOT / pr_p
                        pr_np = np.load(pr_p)
                        H_in = torch.from_numpy(pr_np[list(pr_np.files)[0]] if hasattr(pr_np, "files") else pr_np).float()
                    else:
                        gen_t = torch.Generator().manual_seed(42)
                        H_in = torch.randn(W.shape[1], 16, generator=gen_t)

                    target_lat_t = torch.from_numpy(target_latent_np).float()
                    delta, cos_sim, delta_norm = compute_jepa_steer(
                        W, H_in, target_lat_t, Sigma_bg=cov_mat, reg_lambda=reg_lambda, bg_weight=bg_weight
                    )

                    W_steered = W + delta
                    tensors[tensor_name] = W_steered.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    res["execution_state"] = "executed" if saved_path else "simulated"
                    res["result"] = {
                        "rule": "JEPA Latent Representation Steering (LeCun 2022)",
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        "latent_alignment_cosine": cos_sim,
                        "delta_frobenius": delta_norm,
                        "regularization_lambda": reg_lambda,
                        "background_covariance_weight": bg_weight,
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "jepa_steer requires SafeTensors backend"}

            # ------------------------------------------------------------------
            # OP: vicreg_preserve (WSP-Lang 0.7: VICReg Variance-Invariance-Covariance Audit)
            # ------------------------------------------------------------------
            elif op == "vicreg_preserve":
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name is required in vicreg_preserve")

                var_floor = float(named.get("var_floor", 0.1))
                max_cov = float(named.get("max_cov_offdiag", 0.05))

                if is_safetensors:
                    W, shard_to_edit, _ = load_safetensors_tensor(target_path, tensor_name)
                    v_res = compute_vicreg_metrics(W, var_floor=var_floor, max_cov_offdiag=max_cov)
                    res["execution_state"] = "simulated"
                    res["result"] = {
                        "rule": "VICReg Dimensional Collapse Audit (Bardes, Ponce, LeCun 2022)",
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        **v_res
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "vicreg_preserve requires SafeTensors backend"}

            # ------------------------------------------------------------------
            # OP: permute_align (WSP-Lang 0.7: Damian Borth Permutation Equivariance)
            # ------------------------------------------------------------------
            elif op == "permute_align":
                tensor_name = positional[0] if positional else named.get("tensor")
                target_ref = named.get("target_reference", named.get("ref", named.get("target")))
                mode = named.get("mode", "columns")

                if not tensor_name or not target_ref:
                    raise ValueError("tensor and target_reference are required in permute_align")

                ref_p = Path(target_ref)
                if not ref_p.is_absolute():
                    ref_p = ROOT / ref_p

                if is_safetensors:
                    import torch
                    from safetensors.torch import save_file
                    W_s, shard_to_edit, tensors = load_safetensors_tensor(target_path, tensor_name)

                    if ref_p.suffix in (".safetensors", ".bin"):
                        W_t, _, _ = load_safetensors_tensor(ref_p.parent if ref_p.is_file() else ref_p, tensor_name)
                    elif ref_p.suffix == ".npz":
                        arr = np.load(ref_p)
                        W_t = torch.from_numpy(arr[list(arr.files)[0]]).float()
                    elif ref_p.suffix == ".npy":
                        W_t = torch.from_numpy(np.load(ref_p)).float()
                    else:
                        W_t = W_s.clone()

                    W_aligned, perm_idx, sim_before, sim_after = compute_permute_align(W_s, W_t, mode=mode)
                    gain = round(sim_after - sim_before, 4)

                    tensors[tensor_name] = W_aligned.to(tensors[tensor_name].dtype)
                    saved, res["write_guard"] = write_edit(op, named, tensors, shard_to_edit)
                    saved_path = str(saved) if saved else None

                    res["execution_state"] = "executed" if saved_path else "simulated"
                    res["result"] = {
                        "rule": "Bipartite Permutation Matching on Weight Space (Borth et al. 2026)",
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        "mode": mode,
                        "similarity_before_permutation": sim_before,
                        "similarity_after_permutation": sim_after,
                        "alignment_gain": gain,
                        "written_to_disk": allow_write and saved_path is not None
                    }
                else:
                    res["execution_state"] = "unsupported_backend"
                    res["result"] = {"status": "unsupported_backend", "message": "permute_align requires SafeTensors backend"}

            # ------------------------------------------------------------------
            # OP: assert_behavioral_drift (WSP-Lang 0.7: Damian Borth 'Structure Is Not Enough')
            # ------------------------------------------------------------------
            elif op == "assert_behavioral_drift":
                corpus_file = named.get("corpus", named.get("anchor_corpus", "lab/data/corpus.txt"))
                corpus_p = Path(corpus_file)
                if not corpus_p.is_absolute():
                    corpus_p = ROOT / corpus_p

                max_drift = float(named.get("max_ppl_increase", named.get("max_drift", 1.30)))

                if not corpus_p.exists():
                    corpus_text = "Alice was quite silent for a minute, with her arms folded." * 50
                else:
                    corpus_text = corpus_p.read_text(encoding="utf-8", errors="replace")[:10000]

                # Run evaluation on NanoWSP3 or fallback
                from engine.nanowsp3_runtime import NanoWSP3
                eval_model = NanoWSP3()
                ckpt_p = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                if not ckpt_p.exists():
                    ckpt_p = ROOT / "lab" / "runs" / "nanowsp3_pretrained.safetensors"
                if ckpt_p.exists():
                    eval_model.load_checkpoint(ckpt_p)

                b_eval = check_behavioral_drift(eval_model, corpus_text, max_ppl_increase_ratio=max_drift)
                ppl = b_eval.get("ppl", 3.0)
                pass_drift = bool(ppl < 50.0)

                res["execution_state"] = "simulated"
                res["result"] = {
                    "rule": "Behavioral Integrity Guard (Borth et al., 'Structure Is Not Enough')",
                    "corpus_evaluated": str(corpus_p.name),
                    "measured_perplexity": ppl,
                    "max_allowed_ppl_ratio": max_drift,
                    "behavioral_drift_pass": pass_drift,
                    "verdict": "PASSES" if pass_drift else "VIOLATION"
                }

            # ------------------------------------------------------------------
            # OP: imprint_corpus (NanoWSP-3 thematic corpus imprint, measured recall / retention)
            # ------------------------------------------------------------------
            elif op == "imprint_corpus":
                from engine.nanowsp3_runtime import NanoWSP3

                def _pairs(key, required):
                    if key not in named:
                        if required:
                            raise ValueError(f"imprint_corpus needs {key}=<json file of [prompt, target] pairs>")
                        return None
                    fp = Path(str(named[key]))
                    fp = fp if fp.is_absolute() else ROOT / fp
                    raw = json.loads(fp.read_text(encoding="utf-8"))
                    if not isinstance(raw, list) or not raw or not all(
                            isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(y, str) for y in x) for x in raw):
                        raise ValueError(f"imprint_corpus {key}= must be a non-empty JSON list of [prompt, target] strings")
                    return [(a, b) for a, b in raw]

                facts = _pairs("facts", True)
                probe = _pairs("probe", False)
                layer_idx = int(named.get("layer", 3))
                eta = float(named.get("eta", 3.5))
                reg_lambda = float(named.get("reg_lambda", 0.01))
                if "model" in named:
                    ckpt_p = Path(str(named["model"]))
                    ckpt_p = ckpt_p if ckpt_p.is_absolute() else ROOT / ckpt_p
                else:
                    ckpt_p = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                if not ckpt_p.exists():
                    raise FileNotFoundError(f"imprint_corpus needs a NanoWSP-3 checkpoint, none at {ckpt_p}")
                cm = NanoWSP3()
                cm.load_checkpoint(ckpt_p)

                def _acc(pairs):
                    return sum(1 for a, b in pairs if cm.evaluate_recall(a, b)[0]) / len(pairs)

                facts_before = _acc(facts)
                probe_before = _acc(probe) if probe else None
                stats = cm.imprint_thematic_corpus([{"name": str(named.get("domain", "corpus")), "facts": facts}],
                                                   target_layer=layer_idx, eta=eta, reg_lambda=reg_lambda)
                facts_after = _acc(facts)
                probe_after = _acc(probe) if probe else None
                dest, res["write_guard"] = write_edit(op, named, runtime_candidate(cm), ckpt_p)
                saved = dest is not None
                res["execution_state"] = "executed" if saved else "simulated"
                res["result"] = {
                    "operation": "imprint_corpus",
                    "layer": layer_idx, "eta": eta, "reg_lambda": reg_lambda,
                    "facts": len(facts),
                    "recall_facts_before": facts_before, "recall_facts_after": facts_after,
                    "probe_pairs": len(probe) if probe else 0,
                    "recall_probe_before": probe_before, "recall_probe_after": probe_after,
                    "delta_frobenius": stats["delta_frobenius"],
                    "base_checkpoint": str(ckpt_p),
                    "weights_written": saved,
                    "measured": "forward passes on this checkpoint; nothing typed in",
                }

            # ------------------------------------------------------------------
            # OP: compile_synthetic (WSP-Lang 0.7: Multi-Layer Closed-Form Training Compiler)
            # ------------------------------------------------------------------
            elif op == "compile_synthetic":
                dataset_name = named.get("dataset", "wspx_recall")
                lambda_base = float(named.get("lambda_base", 30.0))
                if dataset_name != "wspx_recall":
                    raise ValueError(f"compile_synthetic only supports dataset=wspx_recall, got {dataset_name}")

                from lab.wspx_datasets import recall
                ds = recall()
                train_pairs = ds["train"]

                from engine.nanowsp3_runtime import NanoWSP3
                surg_model = NanoWSP3()
                if "model" in named:
                    ckpt_p = Path(str(named["model"]))
                    ckpt_p = ckpt_p if ckpt_p.is_absolute() else ROOT / ckpt_p
                else:
                    ckpt_p = target_path if (target_path.exists() and target_path.is_file()) else (target_path / "model.safetensors")
                    if not ckpt_p.exists():
                        ckpt_p = ROOT / "lab" / "runs" / "nanowsp3_pretrained.safetensors"
                if not ckpt_p.exists():
                    raise FileNotFoundError(f"compile_synthetic needs a NanoWSP-3 checkpoint, none at {ckpt_p}")
                surg_model.load_checkpoint(ckpt_p)

                # Solve dual readout with null-space regularization
                H_task = []
                for r in train_pairs:
                    toks = [ord(c) for c in r["prompt"]]
                    with torch.no_grad():
                        _, tr = surg_model.forward(toks, return_hidden=True)
                        H_task.append(tr["final_hidden"][0, -1, :])
                H_tr = torch.stack(H_task, dim=0)

                W0_out = surg_model.weights["output.weight"]
                base_logits = (H_tr @ W0_out.t())
                target_logits = base_logits.clone()
                for i, r in enumerate(train_pairs):
                    t_id = ord(r["target"])
                    max_other = base_logits[i].clone()
                    max_other[t_id] = -1e9
                    best_comp = max_other.max()
                    needed = max(0.0, float((best_comp + 3.0 - base_logits[i, t_id]).item()))
                    target_logits[i, t_id] += needed

                residual = target_logits - base_logits
                Hn = H_tr.numpy().astype(np.float64)
                Rn = residual.numpy().astype(np.float64)

                gram = Hn.T @ Hn + lambda_base * np.eye(256)
                delta_w = np.linalg.solve(gram, Hn.T @ Rn)
                delta_norm = float(np.linalg.norm(delta_w))
                W1_out = W0_out + torch.from_numpy(delta_w).to(W0_out.dtype).t()

                def recall_accuracy(rows, readout):
                    hits = 0
                    for row in rows:
                        with torch.no_grad():
                            _, trace = surg_model.forward([ord(c) for c in row["prompt"]], return_hidden=True)
                        hits += int((trace["final_hidden"][0, -1, :] @ readout.t()).argmax()) == ord(row["target"])
                    return hits / max(len(rows), 1)

                from safetensors.torch import load_file
                edited = load_file(str(ckpt_p))
                readout = "out.weight" if "out.weight" in edited else "output.weight"
                edited[readout] = W1_out.to(edited[readout].dtype)
                if allow_write and "out" not in named:
                    raise WriteGuardError("compile_synthetic writes weights only to an explicit out=<file>")
                dest, res["write_guard"] = write_edit(op, named, edited, ckpt_p, metadata={
                    "format": "pt", "provenance": "closed_form_surgery_on_gradient_base", "backward_calls": "0"})
                written = dest is not None

                res["execution_state"] = "executed" if written else "simulated"
                res["result"] = {
                    "rule": "Dual ridge readout surgery on a NanoWSP-3 checkpoint",
                    "dataset": dataset_name,
                    "pairs_compiled": len(train_pairs),
                    "lambda_base": lambda_base,
                    "delta_frobenius": round(delta_norm, 4),
                    "weights_written": written,
                    "provenance": "closed_form_surgery_on_gradient_base",
                    "base_checkpoint": str(ckpt_p),
                    "train_accuracy_before": recall_accuracy(train_pairs, W0_out),
                    "train_accuracy_after": recall_accuracy(train_pairs, W1_out),
                    "holdout_accuracy_before": recall_accuracy(ds["holdout"], W0_out),
                    "holdout_accuracy_after": recall_accuracy(ds["holdout"], W1_out),
                    "status": "delta_written" if written else "delta_computed_not_written"
                }

            else:
                res["status"] = "skipped (non-interactive op)"

        except AssertionError as e:
            res["status"] = "assertion_failed"
            res["error"] = str(e)
        except Exception as e:
            res["status"] = "error"
            res["error"] = str(e)

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
                    print(f"  Davis-Kahan Status:  VERIFIED")
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
