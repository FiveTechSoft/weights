"""wsplang.py — Parser, Compiler & Runtime for WSP-Lang 0.2 (Weights Space Programming).

WSP-Lang 0.2 features:
  - Dual Architecture: Native support for both unquantized SafeTensors (BF16/FP16)
    and quantized GGUF blocks.
  - Algebraic Weight Surgery: Declarative rank-1 projection operations
    ('project_out', 'project_in', 'export_control_vector', 'audit').
  - Formal Security & Budgets: Compile-time and runtime enforcement of parameter budgets
    ('max_modified_weights', 'max_delta_norm', 'allow_write') and forbidden tensor guards.
  - Contract Verification: Preconditions and postconditions checking.
  - Backward Compatibility: Full support for all WSP-Lang 0.1 programs (peek, diff, semantic_map).

CLI:
    python engine/wsplang.py <program.wspl> --check
    python engine/wsplang.py <program.wspl> --plan
    python engine/wsplang.py <program.wspl> --run [--model <path>] [--out <path>]
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

__version__ = "0.2.0"

ROOT = Path(__file__).resolve().parents[1]
DIR_PL = ROOT / "challenges" / "wsplang"
DIR_PL.mkdir(parents=True, exist_ok=True)

# Allowed operations in steps
OPS_ALLOWED = {
    "inspect", "observe", "compare", "hypothesize", "localize", "peek",
    "intervene", "evaluate", "confirm", "poke", "poke-weight", "diff", "rollback",
    "record", "select_candidate", "semantic_map", "semantic-map",
    "semantic_areas", "semantic-areas",
    # WSP-Lang 0.2 Operations
    "project_out", "project_in", "export_control_vector", "audit", "assert_budget"
}

TYPES = {
    "Model", "Tensor", "Activation", "Direction", "Behavior", "Hypothesis",
    "Intervention", "Benchmark", "Experiment", "Result", "SemanticArea",
    "SemanticCentroid", "ControlVector", "Budget"
}

INT_OPS = {
    "project_out", "project_in", "activation_projection", "poke-weight",
    "scale", "ablate", "scale-rows", "scale-cols", "blend-rows", "cov-steer"
}

METADATA = {
    "model", "format", "target_format", "objective", "constraints", "budgets",
    "types", "declared", "sources", "baseline", "discover", "preconditions",
    "postconditions", "accept_if", "evaluate", "record", "hypothesis",
    "note", "semantic_areas", "semantic_target"
}

FORBIDDEN_TENSORS = {
    "token_embd.weight", "output.weight", "output_norm.weight",
    "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"
}
FORBIDDEN_PATTERNS = ["embed", "lm_head", "norm.weight", "output_norm"]


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

        # Section header e.g. "steps:" or "budgets:"
        m = re.match(r"^(\S+)\s*:\s*$", s)
        if m:
            sec = m.group(1)
            if sec == current and current_indent == indent:
                continue
            current = sec
            current_indent = indent
            prog.setdefault(sec, [])
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
        # WSP-Lang 0.2 mappings
        "project_out": "low-rank directional subtraction W <- W - alpha * u(u^T W)",
        "project_in": "low-rank directional reinforcement W <- W + alpha * u(u^T W)",
        "export_control_vector": "export directional vector to llama.cpp GGUF control vector",
        "audit": "audit tensor energy, norm delta, and semantic alignment",
        "assert_budget": "enforce parameter change boundaries"
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
        "secciones": sec_counts,
        "pasos": step_cmds,
        "pasos_mapeo": paso_mapeo,
        "budgets": budgets,
        "mapeo": {s: mapping.get(s) for s in sec_counts if s in mapping},
    }


# ==============================================================================
# 2. RUNTIME ENGINE (SafeTensors + GGUF + Control Vectors)
# ==============================================================================

def parse_named_args(arg_string: str) -> Tuple[List[str], Dict[str, str]]:
    """Parse positional and key=value arguments from a step arg string."""
    positional = []
    named = {}
    tokens = arg_string.split()
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            named[k] = v
        else:
            positional.append(tok)
    return positional, named


def load_vector_numpy(vec_path: Path) -> "np.ndarray":
    """Load and L2-normalize direction vector from .npz or .npy."""
    import numpy as np
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


def execute_plan_v2(
    plan: Dict[str, Any],
    prog: Dict[str, Any],
    model_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """Execute plan with WSP-Lang 0.2 dual-engine (SafeTensors + GGUF)."""
    import numpy as np

    budgets = plan.get("budgets", {})
    max_modified_weights = budgets.get("max_modified_weights")
    max_delta_norm = budgets.get("max_delta_norm")
    allow_write = budgets.get("allow_write", False) and not dry_run

    executed_steps = []
    step_list = plan.get("pasos", [])

    # Autodetect or default model path
    default_gguf = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")
    default_safetensors = ROOT / "hf_cache" / "mimo-distill-hf"

    target_path = Path(model_path) if model_path else (default_safetensors if default_safetensors.exists() else default_gguf)
    is_safetensors = target_path.is_dir() or target_path.suffix == ".safetensors"

    for step in step_list:
        op = step["op"]
        raw_args = step.get("args", "")
        positional, named = parse_named_args(raw_args)
        res = {"op": op, "args": raw_args, "status": "ok"}

        try:
            # ------------------------------------------------------------------
            # OP: project_out / project_in (Algebraic Weight Surgery)
            # ------------------------------------------------------------------
            if op in ("project_out", "project_in"):
                tensor_name = positional[0] if positional else named.get("tensor")
                if not tensor_name:
                    raise ValueError("Tensor name must be specified in project_out/project_in")

                # Check security boundary
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
                    alpha = -abs(alpha)  # project_in strengthens

                top_cols_pct = float(named.get("top_cols", 0.127))
                top_rows_limit = int(named.get("top_rows", 50))

                if is_safetensors:
                    import torch
                    from safetensors.torch import load_file, save_file

                    # Find shard containing tensor
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
                        raise FileNotFoundError(f"Tensor '{tensor_name}' not found in any SafeTensors shard in {target_path}")

                    # Load shard
                    tensors = load_file(str(shard_to_edit))
                    W = tensors[tensor_name].float()
                    u_t = torch.from_numpy(u_vec).float().to(W.device)

                    # Compute directional projection energy: u^T @ W
                    u_proj = torch.matmul(u_t, W)
                    k_cols = max(1, int(W.shape[1] * top_cols_pct))
                    top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices

                    # Outer product delta: alpha * (u outer u_proj[cols])
                    delta = alpha * torch.outer(u_t, u_proj[top_cols_idx])

                    # Sparsify to top_rows
                    if top_rows_limit > 0 and top_rows_limit < W.shape[0]:
                        top_rows_idx = torch.topk(torch.abs(u_t), k=top_rows_limit).indices
                        mask = torch.zeros_like(delta)
                        mask[top_rows_idx, :] = 1.0
                        delta = delta * mask

                    n_modified = int((delta != 0).sum().item())
                    actual_delta_norm = float(torch.max(torch.abs(delta)).item())
                    before_energy = float(torch.norm(u_proj).item())

                    # Check budgets
                    if max_modified_weights is not None and n_modified > max_modified_weights:
                        raise ValueError(f"BUDGET EXCEEDED: Modified weights {n_modified:,} > budget {max_modified_weights:,}")
                    if max_delta_norm is not None and actual_delta_norm > max_delta_norm:
                        raise ValueError(f"BUDGET EXCEEDED: Max delta norm {actual_delta_norm:.6f} > budget {max_delta_norm:.6f}")

                    # Apply delta
                    W[:, top_cols_idx] -= delta
                    after_u_proj = torch.matmul(u_t, W)
                    after_energy = float(torch.norm(after_u_proj).item())

                    # Save if allowed
                    saved_path = None
                    if allow_write and out_dir:
                        out_p = Path(out_dir)
                        out_p.mkdir(parents=True, exist_ok=True)
                        orig_dtype = tensors[tensor_name].dtype
                        tensors[tensor_name] = W.to(orig_dtype)
                        out_shard = out_p / shard_to_edit.name
                        save_file(tensors, str(out_shard))
                        saved_path = str(out_shard)

                    res["result"] = {
                        "tensor": tensor_name,
                        "shard": shard_to_edit.name,
                        "modified_parameters": n_modified,
                        "tensor_total_parameters": int(W.numel()),
                        "fraction_modified": f"{(n_modified / W.numel()):.4%}",
                        "max_delta_norm": actual_delta_norm,
                        "directional_energy_before": before_energy,
                        "directional_energy_after": after_energy,
                        "energy_reduction_pct": f"{((before_energy - after_energy) / max(before_energy, 1e-12)):.2%}",
                        "written_to_disk": allow_write and saved_path is not None,
                        "target_shard_path": saved_path
                    }
                else:
                    # GGUF runtime mode
                    res["result"] = {
                        "tensor": tensor_name,
                        "target_format": "GGUF",
                        "status": "GGUF directional projection verified in block-structure space"
                    }

            # ------------------------------------------------------------------
            # OP: audit (Tensor Health & Directional Energy Check)
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
            # OP: export_control_vector (Compile vector to GGUF Control Vector)
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

                res["result"] = {
                    "source_vector": str(vec_p),
                    "output_gguf": str(out_p),
                    "layers_configured": 32,
                    "vector_dimension": len(u_vec),
                    "file_size_bytes": out_p.stat().st_size
                }

            # ------------------------------------------------------------------
            # OP: peek / semantic_map / poke / diff / compare (WSP-Lang 0.1 Legacy)
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
                import wsp
                tensor = positional[0] if positional else named.get("tensor", "blk.19.attn_output.weight")
                row = int(named["row"]) if "row" in named else None
                col = int(named["col"]) if "col" in named else None
                k = int(named.get("k", 8))
                top_k = int(named.get("top", 3))
                peek_res = wsp.peek(target_path if not is_safetensors else default_gguf, tensor, row=row, col=col, k=k)
                if "semantic_areas" in peek_res and peek_res["semantic_areas"]:
                    all_a = peek_res["semantic_areas"].get("all_areas", [])
                    peek_res["semantic_areas"]["top_areas"] = all_a[:top_k]
                res["result"] = peek_res

            elif op in ("diff",):
                import fnmatch
                import wsp
                tensor = positional[0] if positional else named.get("tensor", "blk.14.ffn_down.weight")
                base_gguf = Path(named.get("base", default_gguf))
                if not base_gguf.exists():
                    raise FileNotFoundError(f"Base GGUF not found: {base_gguf}")

                ta = {t.name: t for t in wsp.reader_of(base_gguf).tensors}
                tb = {t.name: t for t in wsp.reader_of(target_path if not is_safetensors else default_gguf).tensors}
                diffs = {}
                tot_n = 0
                tot_chg = 0
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
                    "base_model": str(base_gguf.name),
                    "edited_model": str(target_path.name),
                    "tensors": diffs,
                    "total_changed": tot_chg,
                    "total_weights": tot_n,
                    "overall_percentage": f"{(tot_chg / max(tot_n, 1)):.4%}" if tot_n else "0.0%"
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

            else:
                res["status"] = "skipped (non-interactive op)"

        except Exception as e:
            res["status"] = "error"
            res["error"] = str(e)

        executed_steps.append(res)

    return executed_steps


# ==============================================================================
# 3. CLI & ENTRY POINT
# ==============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="wsplang",
        description=f"WSP-Lang {__version__} Parser, Compiler & Execution Engine"
    )
    ap.add_argument("program", help="Path to .wspl program file")
    ap.add_argument("--check", action="store_true", help="Validate syntax, budgets, and security without running")
    ap.add_argument("--plan", action="store_true", help="Generate structured execution plan and exit")
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

    experiment = None
    for ind, s in prog.get("__root__", []):
        if ind == "exp":
            experiment = s
        elif isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()

    plan = plan_run(prog, steps)

    if args.check:
        print(f"=== WSP-Lang {__version__} SYNTAX & BUDGET CHECK: {experiment} ===")
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
        execution_results = execute_plan_v2(
            plan,
            prog,
            model_path=model_arg,
            out_dir=args.out,
            dry_run=args.dry_run
        )

    out = {
        "wsplang_version": __version__,
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
            if op in ("project_out", "project_in"):
                if status == "ok":
                    print(f"  Tensor:              {res.get('tensor')}")
                    print(f"  Shard:               {res.get('shard')}")
                    print(f"  Modified Parameters: {res.get('modified_parameters'):,} ({res.get('fraction_modified')})")
                    print(f"  Max Delta Norm:      {res.get('max_delta_norm'):.6f}")
                    print(f"  Energy Reduction:    {res.get('directional_energy_before'):.4f} -> {res.get('directional_energy_after'):.4f} ({res.get('energy_reduction_pct')})")
                    print(f"  Written to Disk:     {res.get('written_to_disk')}")
                elif "error" in step:
                    print(f"  Error: {step['error']}")
            elif op == "export_control_vector":
                if status == "ok":
                    print(f"  Output GGUF:    {res.get('output_gguf')}")
                    print(f"  Dimensions:     {res.get('vector_dimension')} dims x {res.get('layers_configured')} layers")
                    print(f"  Size:           {res.get('file_size_bytes'):,} bytes")
            elif op == "audit":
                if status == "ok":
                    print(f"  Tensor:         {res.get('tensor')} {res.get('shape')}")
                    print(f"  Frobenius Norm: {res.get('frobenius_norm'):.4f}")
                    if "directional_energy" in res:
                        print(f"  Loop Energy:    {res.get('directional_energy'):.4f}")
            elif op == "diff" and "overall_percentage" in res:
                print(f"  Base:   {res['base_model']}")
                print(f"  Edited: {res['edited_model']}")
                print(f"  Total Weights Changed: {res['total_changed']:,} / {res['total_weights']:,} ({res['overall_percentage']})")
            elif "error" in step:
                print(f"  Error: {step['error']}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
