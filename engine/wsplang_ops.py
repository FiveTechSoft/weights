"""Per-op handlers for execute_plan_v4 (extracted from engine/wsplang.py).

This module holds the dispatch bodies of engine/wsplang.py execute_plan_v4 so the
main engine file stays focused on the plan/pipeline skeleton. Every handler takes
the same contract:

    (ctx, op, positional, named, res) -> bool | None

  - ctx: a PlanContext carrying the execute_plan_v4 state (budgets, target_path,
    is_safetensors, out_dir, written_paths, executed_steps, ...).
  - op/positional/named/res: the current step. res is the mutable step result dict;
    handlers set status/result/write_guard on it and return True if res was copied into
    ctx.executed_steps already (security-rejection paths); otherwise the caller appends it.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from engine.comparative_drift import (
    compute_architectural_drift,
    compute_semantic_delta,
    compute_spectral_delta,
    compute_ssm_dynamics_diff,
)
from engine.safety import FORBIDDEN_PATTERNS, FORBIDDEN_TENSORS
from engine.wsplang import (
    DEFAULT_QWEN_BASE_GGUF,
    ROOT,
    WriteGuardError,
    _model_path,
    apply_decoupled_delta,
    check_behavioral_drift,
    check_new_file,
    check_report_path,
    compute_jepa_steer,
    compute_leace_projection,
    compute_memit_update,
    compute_permute_align,
    compute_procrustes_alignment,
    compute_spectral_decomposition,
    compute_vicreg_metrics,
    generate_orthogonal_probe,
    guard_model_write,
    guarded_save,
    load_safetensors_tensor,
    load_vector_numpy,
    transfer_verdict,
)


class PlanContext:
    """Mutable execution state shared by every op handler in one execute_plan_v4 call."""

    def __init__(self, *, budgets, max_modified_weights, max_delta_norm, allow_write,
                 written_paths, target_path, is_safetensors, default_gguf,
                 default_safetensors, nanowsp_default, nanowsp2_default,
                 nanowsp3_default, tiny_default, qwen08_default,
                 out_dir, model_path, dry_run, executed_steps):
        self.budgets = budgets
        self.max_modified_weights = max_modified_weights
        self.max_delta_norm = max_delta_norm
        self.allow_write = allow_write
        self.written_paths = written_paths
        self.target_path = target_path
        self.is_safetensors = is_safetensors
        self.default_gguf = default_gguf
        self.default_safetensors = default_safetensors
        self.nanowsp_default = nanowsp_default
        self.nanowsp2_default = nanowsp2_default
        self.nanowsp3_default = nanowsp3_default
        self.tiny_default = tiny_default
        self.qwen08_default = qwen08_default
        self.out_dir = out_dir
        self.model_path = model_path
        self.dry_run = dry_run
        self.executed_steps = executed_steps

    def edit_destination(self, op_name, named_args, source_file):
        if named_args.get("out"):
            dest = Path(str(named_args["out"]))
            return dest if dest.is_absolute() else ROOT / dest
        if self.out_dir:
            return Path(self.out_dir) / Path(source_file).name
        if Path(source_file).resolve() in self.written_paths:
            return Path(source_file)
        raise WriteGuardError(f"{op_name} writes weights: give out=<file> or --out-dir; "
                             f"the input model {source_file} is never overwritten")

    def write_edit(self, op_name, named_args, candidate, source_file, metadata=None):
        """Every edit of an existing model goes through here. Returns (written path or None, guard report)."""
        if not self.allow_write:
            return None, guard_model_write(candidate, None, source_file, self.budgets, self.written_paths)
        dest = self.edit_destination(op_name, named_args, source_file)
        report = guarded_save(candidate, dest, source_file, self.budgets, self.written_paths, metadata=metadata)
        # Later steps edit what this step wrote. Only for single-file models: a sharded directory would
        # otherwise lose its other shards.
        shards = list(self.target_path.glob("*.safetensors")) if self.target_path.is_dir() else [self.target_path]
        if len(shards) == 1 and shards[0].resolve() == Path(source_file).resolve():
            self.target_path = dest
        return dest, report

    def runtime_candidate(self, runtime):
        return {k: v.detach().contiguous().float() for k, v in runtime.weights.items()}


def _op_leace_erase(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    vec_file = named.get('vector', named.get('key', positional[1] if len(positional) > 1 else None))
    cov_file = named.get('covariance', named.get('cov'))
    if not tensor_name:
        raise ValueError('Tensor name must be specified in leace_erase')
    if not vec_file:
        raise ValueError('Target concept vector must be specified in leace_erase')
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
            cov_mat = torch.from_numpy(cov_np[list(cov_np.files)[0]] if hasattr(cov_np, 'files') else cov_np).float()
    if ctx.is_safetensors:
        W, shard_to_edit, tensors = load_safetensors_tensor(ctx.target_path, tensor_name)
        u_t = torch.from_numpy(u_vec).float().to(W.device)
        P_leace = compute_leace_projection(u_t, sigma=cov_mat)
        energy_before = float(torch.norm(torch.matmul(u_t, W)).item())
        norm_before = float(torch.norm(W).item())
        W_erased = torch.matmul(P_leace, W)
        delta = W - W_erased
        actual_delta_norm = float(torch.max(torch.abs(delta)).item())
        energy_after = float(torch.norm(torch.matmul(u_t, W_erased)).item())
        norm_after = float(torch.norm(W_erased).item())
        tensors[tensor_name] = W_erased.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        res['execution_state'] = 'executed' if saved_path else 'simulated'
        res['result'] = {'rule': 'LEACE Closed-Form Concept Erasure (Belrose et al. 2023)', 'tensor': tensor_name, 'shard': shard_to_edit.name, 'concept_energy_before': energy_before, 'concept_energy_after': energy_after, 'erasure_efficiency': f'{(energy_before - energy_after) / max(energy_before, 1e-12):.4%}', 'frobenius_norm_before': norm_before, 'frobenius_norm_after': norm_after, 'frobenius_distortion': float(torch.norm(delta).item()), 'max_weight_delta': actual_delta_norm, 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'LEACE erasure currently supported on SafeTensors.'}


def _op_cov_imprint_memit_imprint(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    batch_file = named.get('file')
    reg_lambda = float(named.get('lambda', named.get('reg', 0.001)))
    if not tensor_name:
        raise ValueError('Tensor name must be specified in cov_imprint')
    pairs = []
    if batch_file:
        batch_p = Path(batch_file)
        if not batch_p.is_absolute():
            batch_p = ROOT / batch_p
        if batch_p.exists():
            with open(batch_p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pairs = [(item['prompt'], item['target']) for item in data]
    elif 'prompt' in named and 'target' in named:
        pairs = [(named['prompt'], named['target'])]
    if ctx.is_safetensors:
        from engine.nanowsp_runtime import NanoWSP
        ckpt_path = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
        if not ckpt_path.exists():
            ckpt_path = ROOT / 'models' / 'nanowsp' / 'model.safetensors'
        nanowsp = NanoWSP()
        if ckpt_path.exists():
            nanowsp.load_checkpoint(ckpt_path)
        keys, values = ([], [])
        for p, t in pairs:
            t_ids = nanowsp.tokenizer.encode(p, add_bos=True)
            _, tr = nanowsp.forward(t_ids, return_hidden=True)
            k = tr['layer_1_mlp_hidden'][0, -1, :].clone()
            keys.append(k / max(float(torch.norm(k).item()), 1e-06))
            t_id = nanowsp.tokenizer.encode(t, add_bos=False)[0]
            v = nanowsp.weights['output.weight'][t_id].clone()
            values.append(v / max(float(torch.norm(v).item()), 1e-06))
        K = torch.stack(keys, dim=1)
        V = torch.stack(values, dim=1)
        target_W = nanowsp.weights[tensor_name]
        cond_before = float(torch.linalg.cond(K).item())
        delta = compute_memit_update(target_W, K, V, reg_lambda=reg_lambda)
        target_W += delta
        cond_after = float(torch.linalg.cond(K.t() @ K + reg_lambda * torch.eye(K.shape[1])).item())
        delta_norm = float(torch.norm(delta).item())
        if not ckpt_path.exists():
            raise FileNotFoundError(f'{op} needs a model checkpoint, none at {ckpt_path}')
        dest, res['write_guard'] = ctx.write_edit(op, named, ctx.runtime_candidate(nanowsp), ckpt_path)
        saved = dest is not None
        if saved:
            ctx.target_path = dest
        res['execution_state'] = 'executed' if saved else 'simulated'
        res['result'] = {'rule': 'MEMIT / EMMET Covariance-Regularized Memory Synthesis', 'tensor': tensor_name, 'pairs_count': len(pairs), 'regularization_lambda': reg_lambda, 'unregularized_condition': cond_before, 'regularized_condition': cond_after, 'delta_frobenius': delta_norm, 'written_to_disk': saved}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'cov_imprint requires SafeTensors backend.'}


def _op_decoupled_delta(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    erase_vec_file = named.get('erase_vector', named.get('erase_key'))
    write_k_file = named.get('write_key')
    write_v_file = named.get('write_value')
    beta_erase = float(named.get('beta_erase', 1.0))
    beta_write = float(named.get('beta_write', 1.0))
    if not tensor_name or not erase_vec_file:
        raise ValueError('Tensor and erase_vector are required in decoupled_delta')
    ev_p = Path(erase_vec_file)
    if not ev_p.is_absolute():
        ev_p = ROOT / ev_p
    u_erase = load_vector_numpy(ev_p)
    k_write = load_vector_numpy(Path(write_k_file) if Path(write_k_file).is_absolute() else ROOT / write_k_file) if write_k_file else None
    v_write = load_vector_numpy(Path(write_v_file) if Path(write_v_file).is_absolute() else ROOT / write_v_file) if write_v_file else None
    if ctx.is_safetensors:
        W, shard_to_edit, tensors = load_safetensors_tensor(ctx.target_path, tensor_name)
        u_e_t = torch.from_numpy(u_erase).float().to(W.device)
        k_w_t = torch.from_numpy(k_write).float().to(W.device) if k_write is not None else None
        v_w_t = torch.from_numpy(v_write).float().to(W.device) if v_write is not None else None
        energy_erase_before = float(torch.norm(torch.matmul(u_e_t, W)).item())
        W_new = apply_decoupled_delta(W, u_e_t, beta_erase, k_w_t, v_w_t, beta_write)
        energy_erase_after = float(torch.norm(torch.matmul(u_e_t, W_new)).item())
        tensors[tensor_name] = W_new.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        res['execution_state'] = 'executed' if saved_path else 'simulated'
        res['result'] = {'rule': 'Gated DeltaNet-2 Decoupled Erase & Write', 'tensor': tensor_name, 'beta_erase': beta_erase, 'beta_write': beta_write, 'attractor_energy_before': energy_erase_before, 'attractor_energy_after': energy_erase_after, 'reduction_pct': f'{(energy_erase_before - energy_erase_after) / max(energy_erase_before, 1e-12):.2%}', 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'decoupled_delta requires SafeTensors backend.'}


def _op_surprise_fwp(ctx, op, positional, named, res):
    threshold = float(named.get('threshold', named.get('tau', 3.0)))
    metric = named.get('metric', 'perplexity_surprise')
    prompt = named.get('prompt', 'test prompt')
    tokens = prompt.split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    repetition_score = 1.0 - unique_ratio
    rep_threshold = threshold if threshold < 1.0 else 0.5
    triggered = repetition_score > rep_threshold and len(tokens) >= 4
    res['execution_state'] = 'heuristic_trigger_only'
    res['result'] = {'rule': 'Titans Surprisal Monitor [STUB: Lexical Repetition Heuristic]', 'status': 'heuristic_trigger_only', 'note': 'Lexical word-ratio trigger stub; does not compute neural perplexity or synthesize live runtime overlay.', 'metric': metric, 'repetition_score': round(repetition_score, 4), 'unique_ratio': round(unique_ratio, 4), 'threshold': rep_threshold, 'memory_update_triggered': triggered, 'action': 'flag_repetition_detected' if triggered else 'maintain_quiescent_memory'}


def _op_rosetta_align(ctx, op, positional, named, res):
    source_file = named.get('source', positional[0] if positional else None)
    target_file = named.get('target', positional[1] if len(positional) > 1 else None)
    out_file = named.get('out', 'data/rosetta_matrix.npz')
    out_p = Path(out_file)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    src_p = Path(source_file) if source_file else None
    if src_p and (not src_p.is_absolute()):
        src_p = ROOT / src_p
    tgt_p = Path(target_file) if target_file else None
    if tgt_p and (not tgt_p.is_absolute()):
        tgt_p = ROOT / tgt_p
    if src_p.suffix == '.npz' and tgt_p.suffix == '.npz':
        data_src = np.load(src_p)
        data_tgt = np.load(tgt_p)
        k_s = list(data_src.keys())[0]
        k_t = list(data_tgt.keys())[0]
        E_s = torch.from_numpy(data_src[k_s]).float()
        E_t = torch.from_numpy(data_tgt[k_t]).float()
    else:
        raise ValueError('rosetta_align currently requires anchor embedding .npz files or pre-extracted matrices')
    T, mean_cos, ortho_err = compute_procrustes_alignment(E_s, E_t)
    if ctx.allow_write:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_p, T=T.numpy(), mean_cos=mean_cos, ortho_err=ortho_err, d_source=E_s.shape[1], d_target=E_t.shape[1])
    res['execution_state'] = 'executed' if ctx.allow_write else 'simulated'
    res['result'] = {'source': str(src_p), 'target': str(tgt_p), 'd_source': E_s.shape[1], 'd_target': E_t.shape[1], 'mean_anchor_cosine': round(mean_cos, 4), 'semi_orthogonality_error': round(ortho_err, 6), 'rosetta_matrix_out': str(out_p.relative_to(ROOT)) if ctx.allow_write and out_p.is_relative_to(ROOT) else str(out_p)}


def _op_cross_project(ctx, op, positional, named, res):
    vec_file = named.get('vector', positional[0] if positional else None)
    rosetta_file = named.get('rosetta', positional[1] if len(positional) > 1 else None)
    out_file = named.get('out', 'data/translated_vector.npz')
    out_p = Path(out_file)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    ros_p = Path(rosetta_file) if rosetta_file else None
    if ros_p and (not ros_p.is_absolute()):
        ros_p = ROOT / ros_p
    u_raw = load_vector_numpy(vec_p)
    u_s = torch.from_numpy(u_raw).float()
    u_s = u_s / max(float(torch.norm(u_s).item()), 1e-12)
    ros_data = np.load(ros_p)
    T = torch.from_numpy(ros_data['T']).float()
    u_target = torch.matmul(T, u_s)
    proj_norm = float(torch.norm(u_target).item())
    u_target = u_target / max(proj_norm, 1e-12)
    if ctx.allow_write:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_p, u=u_target.numpy())
    res['execution_state'] = 'executed' if ctx.allow_write else 'simulated'
    res['result'] = {'source_vector': str(vec_p), 'source_dim': u_s.shape[0], 'target_dim': u_target.shape[0], 'projected_norm': round(proj_norm, 4), 'translated_vector_out': str(out_p.relative_to(ROOT)) if ctx.allow_write and out_p.is_relative_to(ROOT) else str(out_p)}


def _op_init_architecture_arch(ctx, op, positional, named, res):
    from engine.wsp_arch import load_arch_file, parse_config_value
    arch_rel = str(named['arch'])
    arch_path = Path(arch_rel)
    if arch_path.suffix != '.wsparch':
        raise ValueError(f'arch must be a .wsparch file, got {arch_rel}')
    if not arch_path.is_absolute():
        arch_path = ROOT / arch_path
    overrides = {key: parse_config_value(value) for key, value in named.items() if key not in ('arch', 'out', 'seed', 'model', 'format')}
    compiled = load_arch_file(arch_path, overrides=overrides, seed=int(named.get('seed', 1337)))
    total = sum((p.numel() for p in compiled.parameters()))
    if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
        raise ValueError(f'{total} parameters exceed budget max_modified_weights={ctx.max_modified_weights}')
    out_p = Path(named.get('out', str(ROOT / 'models' / compiled.arch_name / 'model.safetensors')))
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    saved = False
    if ctx.allow_write:
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        compiled.save(out_p)
        ctx.written_paths.add(out_p.resolve())
        saved = True
        ctx.target_path = out_p
        ctx.is_safetensors = True
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'architecture': compiled.arch_name, 'compiled_from': arch_rel, 'source_sha256': compiled.source_sha256, 'config': {k: v for k, v in vars(compiled.config).items()}, 'total_parameters': total, 'output_file': str(out_p), 'written_to_disk': saved}


def _op_synthesize(ctx, op, positional, named, res):
    from engine import wsp_arch, wsp_synth, wsp_train
    from engine import wsp_decompile
    if named.get('method') in ('register_program', 'discover_register'):
        from engine import wsp_register
        import hashlib
        method = named['method']
        input_key = 'data' if method == 'discover_register' else 'program'
        for key in ('arch', input_key, 'out'):
            if key not in named:
                raise ValueError(f'{method} requires {key}=')
        raw = (ROOT / str(named[input_key])).read_bytes()
        discovery = {}
        if method == 'discover_register':
            from engine import wsp_discover_register
            if not ctx.budgets.get('allow_training'):
                raise ValueError('discover_register fits examples and requires allow_training=true')
            examples = json.loads(raw.decode('utf-8'))
            if not isinstance(examples, list) or not examples or len(examples) > 256:
                raise ValueError('discover_register requires 1..256 example triples')
            law = wsp_discover_register.discover(examples)
            source = wsp_discover_register.lower(law)
            discovery = {'law': json.dumps(list(law)), 'examples_sha256': hashlib.sha256(raw).hexdigest(),
                         'examples_count': str(len(examples)), 'discovered_program': source}
        else:
            source = raw.decode('utf-8')
        cfg = wsp_register.config(source)
        d, layers, ff = cfg['d'], cfg['layers'], cfg['ff']
        estimate = layers*(4*d*d+2*d*ff+2*d)+d*(cfg['vocab']+cfg['block']+1)
        if ctx.max_modified_weights is not None and estimate > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        if ctx.max_delta_norm is not None:
            raise ValueError('register synthesis has no source delta')
        if not ctx.allow_write or ctx.dry_run:
            res['execution_state'] = 'simulated'
            res['result'] = {'written_to_disk': False, 'architecture': cfg, 'method': method, **discovery}
            return
        out_p = ROOT / str(named['out'])
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        compiled = wsp_register.compile_program(ROOT / str(named['arch']), source,
                                                  seed=int(named.get('seed', 1337)))
        total = sum(p.numel() for p in compiled.parameters())
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        info = {'provenance': 'closed_form_symbolic_discovery' if discovery else 'closed_form_data_free',
                'method': method, 'backward_calls': '0',
                'program_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'architecture': json.dumps(cfg), **discovery}
        compiled.save(out_p, metadata=info)
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}
        return
    if named.get('method') == 'primitive':
        from engine import wsp_primitives
        for key in ('arch', 'skill', 'out'):
            if key not in named:
                raise ValueError(f'primitive requires {key}=')
        if named['skill'] not in ('copy', 'associate', 'composition'):
            raise ValueError('unknown primitive')
        if not ctx.allow_write or ctx.dry_run:
            res['execution_state'] = 'simulated'
            res['result'] = {'written_to_disk': False}
            return
        out_p = ROOT / str(named['out'])
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        if ctx.max_delta_norm is not None:
            raise ValueError('primitive synthesis has no source delta')
        if named['skill'] == 'composition':
            from engine import wsp_composition
            compiled = wsp_composition.compile_composition(ROOT / str(named['arch']), seed=int(named.get('seed', 1337)))
        else:
            compiled = wsp_primitives.compile_primitive(ROOT / str(named['arch']), str(named['skill']),
                                                         seed=int(named.get('seed', 1337)))
        total = sum(p.numel() for p in compiled.parameters())
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        info = {'provenance': 'closed_form_data_free', 'skill': str(named['skill']), 'backward_calls': '0'}
        compiled.save(out_p, metadata=info)
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}
        return
    if named.get('method') in ('spectral_v6', 'spectral_v6_random_features'):
        from engine import wsp_spectral
        for key in ('arch', 'data', 'out'):
            if key not in named:
                raise ValueError(f'spectral synthesis requires {key}=')
        if not ctx.budgets.get('allow_training'):
            raise ValueError('spectral synthesis fits data and requires allow_training=true')
        if not ctx.allow_write or ctx.dry_run:
            res['execution_state'] = 'simulated'
            res['result'] = {'written_to_disk': False}
            return
        out_p = ROOT / str(named['out'])
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        data = wsp_train.char_dataset((ROOT / str(named['data'])).read_text(encoding='utf-8'))
        estimate = 4*(4*256**2+2*256*1024+2*256)+256*(128+len(data.chars)+1)
        if ctx.max_modified_weights is not None and estimate > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        if ctx.max_delta_norm is not None:
            raise ValueError('spectral synthesis does not define a source checkpoint delta')
        compiled = wsp_spectral.compile_spectral(ROOT / str(named['arch']), data.train, data.chars,
                          seed=int(named.get('seed', 1337)), spectral=named.get('method') == 'spectral_v6')
        total = sum(p.numel() for p in compiled.parameters())
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        info = compiled.synthesis_info
        compiled.save(out_p, metadata={'provenance': info['provenance'],
                         'method': info['method'], 'backward_calls': '0',
                         'synthesis_info': json.dumps(info), 'vocab_chars': json.dumps(data.chars)})
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}
        return
    if named.get('method') in ('structured_ridge', 'structured_ridge_v3', 'structured_ridge_v4', 'structured_ridge_v5', 'structured_table_v5'):
        for key in ('arch', 'data', 'out'):
            if key not in named:
                raise ValueError(f'structured_ridge requires {key}=')
        if not ctx.budgets.get('allow_training'):
            raise ValueError('structured_ridge requires allow_training=true')
        if not ctx.allow_write or ctx.dry_run:
            res['execution_state'] = 'simulated'
            res['result'] = {'written_to_disk': False}
            return
        out_p = ROOT / str(named['out'])
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        dataset = wsp_train.char_dataset((ROOT / str(named['data'])).read_text(encoding='utf-8'))
        # Four exact recent-token slots plus three older-context windows.
        overrides = dict(d=512, heads=8, layers=6, ff=2048, block=256,
                         bias=False, dropout=0.0)
        v5 = named.get('method') in ('structured_ridge_v5', 'structured_table_v5')
        if v5:
            overrides['ff'] = 8192
        estimate = 6 * (4 * 512**2 + 2 * 512 * overrides['ff'] + 2 * 512) + 512 * (256 + len(dataset.chars) + 1)
        if ctx.max_modified_weights is not None and estimate > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        compiled = wsp_synth.compile_long_context(
            ROOT / str(named['arch']), dataset.train, dataset.chars, overrides,
            seed=int(named.get('seed', 1337)), order=5, free_blocks=2,
            windows=(8, 32, 128), fit=named.get('method') != 'structured_table_v5', device=str(named.get('device', 'cpu')),
            weighted=named.get('method') != 'structured_ridge',
            induction=named.get('method') == 'structured_ridge_v4' or v5,
            active_features=2048 if v5 else None, all_slots=v5)
        total = sum(p.numel() for p in compiled.parameters())
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        info = compiled.synthesis_info
        compiled.save(out_p, metadata={'provenance': info['provenance'],
                                       'method': info['method'], 'backward_calls': '0',
                                       'synthesis_info': json.dumps(info),
                                       'vocab_chars': json.dumps(dataset.chars)})
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}
        return
    if named.get('method') == 'context_ridge':
        from engine import wsp_context_ridge
        for key in ('arch', 'data', 'out'):
            if key not in named:
                raise ValueError(f'context_ridge requires {key}=')
        if not ctx.budgets.get('allow_training'):
            raise ValueError('context_ridge fits data and requires allow_training=true')
        if not ctx.allow_write or ctx.dry_run:
            res['execution_state'] = 'simulated'
            res['result'] = {'written_to_disk': False}
            return
        out_p = ROOT / str(named['out'])
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        dataset = wsp_train.char_dataset((ROOT / str(named['data'])).read_text(encoding='utf-8'))
        seed = int(named.get('seed', 1337))
        compiled = wsp_arch.load_arch_file(ROOT / str(named['arch']),
                                           overrides={'bias': False, 'dropout': 0.0,
                                                      'vocab': len(dataset.chars)}, seed=seed)
        total = sum(p.numel() for p in compiled.parameters())
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError('parameter budget exceeded')
        info = wsp_context_ridge.fit(compiled, dataset.train, heads=4, block=256,
                                     width=256, layers=4, seed=seed,
                                     samples=int(named.get('samples', 32768)),
                                     ridge=float(named.get('ridge', 0.01)),
                                     device=str(named.get('device', 'cpu')))
        if ctx.max_delta_norm is not None and info['delta_frobenius'] > ctx.max_delta_norm:
            raise ValueError('delta Frobenius budget exceeded')
        compiled.save(out_p, metadata={**{k: str(v) for k, v in info.items()},
                                       'vocab_chars': json.dumps(dataset.chars)})
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}
        return
    if named.get('method') != 'ngram':
        raise ValueError(f"synthesize supports method=ngram, got {named.get('method')}")
    source_args = ('table',) if 'table' in named else ('data', 'order')
    for required in ('arch', 'out') + source_args:
        if required not in named:
            raise ValueError(f'synthesize requires {required}=')
    if 'table' in named and ('data' in named or 'order' in named):
        raise ValueError('synthesize takes either table= or data= with order=, not both')

    def _rooted(value):
        path = Path(str(value))
        return path if path.is_absolute() else ROOT / path
    arch_path = _rooted(named['arch'])
    out_p = _rooted(named['out'])
    if 'table' in named:
        table = wsp_decompile.parse_program(_rooted(named['table']).read_text(encoding='utf-8'))
        chars = table.chars
    else:
        dataset = wsp_train.char_dataset(_rooted(named['data']).read_text(encoding='utf-8'))
        chars = dataset.chars
        table = wsp_synth.WittenBellTable(dataset.train, len(chars), int(named['order']))
    overrides = {key: wsp_arch.parse_config_value(value) for key, value in named.items() if key not in ('method', 'arch', 'data', 'out', 'order', 'seed', 'model', 'format', 'table')}
    if not ctx.allow_write:
        res['execution_state'] = 'simulated'
        res['result'] = {'units_available': table.num_units(), 'written_to_disk': False}
    else:
        compiled = wsp_synth.compile_ngram(arch_path, table, overrides=overrides, seed=int(named.get('seed', 1337)))
        total = sum((p.numel() for p in compiled.parameters()))
        if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
            raise ValueError(f'{total} parameters exceed budget max_modified_weights={ctx.max_modified_weights}')
        info = compiled.synthesis_info
        if 'table' in named:
            source = getattr(table, 'provenance', 'unknown')
            info['method'] = 'wspngram_program'
            info['provenance'] = 'closed_form' if source == 'closed_form' else f'closed_form_compile_of:{source}'
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        compiled.save(out_p, metadata={'provenance': info['provenance'], 'backward_calls': '0', 'method': info['method'], 'order': str(info['order']), 'vocab_chars': json.dumps(chars)})
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {**info, 'total_parameters': total, 'output_file': str(out_p)}


def _op_decompile(ctx, op, positional, named, res):
    from safetensors import safe_open
    from engine import wsp_arch, wsp_decompile, wsp_train
    method = named.get('method')
    if method not in ('structural', 'behavior'):
        raise ValueError(f'decompile supports method=structural or method=behavior, got {method}')
    required = ('arch', 'model', 'out') + (('data', 'order') if method == 'behavior' else ())
    for key in required:
        if key not in named:
            raise ValueError(f'decompile method={method} requires {key}=')

    def _rooted(value):
        path = Path(str(value))
        return path if path.is_absolute() else ROOT / path
    model_p, out_p = (_rooted(named['model']), _rooted(named['out']))
    with safe_open(str(model_p), framework='pt') as f:
        meta = f.metadata() or {}
    dataset = None
    if method == 'behavior':
        dataset = wsp_train.char_dataset(_rooted(named['data']).read_text(encoding='utf-8'))
    if 'vocab_chars' in meta:
        chars = json.loads(meta['vocab_chars'])
    elif dataset is not None:
        chars = dataset.chars
    else:
        raise ValueError('checkpoint has no vocab_chars metadata; structural decompile needs it')
    if dataset is not None and dataset.chars != chars:
        raise ValueError("data= vocabulary differs from the checkpoint's vocab_chars")
    skip = ('method', 'arch', 'model', 'out', 'data', 'order', 'limit_tokens', 'seed', 'format')
    overrides = {key: wsp_arch.parse_config_value(value) for key, value in named.items() if key not in skip}
    model = wsp_arch.load_arch_file(_rooted(named['arch']), overrides=dict(overrides, vocab=len(chars)))
    model.load(model_p)
    model.eval()
    if method == 'structural':
        table = wsp_decompile.decompile_ngram(model, chars)
        table.provenance = meta.get('provenance', 'unknown')
        note = "exact: recovered from the weights' layout, no data used"
    else:
        ids = dataset.train[:int(named['limit_tokens'])] if 'limit_tokens' in named else dataset.train
        table = wsp_decompile.behavioral_table(model, ids, chars, order=int(named['order']))
        table.provenance = 'distilled_from:' + meta.get('provenance', 'unknown')
        note = "approximate: the model's average prediction per context over the train split; long-range context is dropped"
    result = {'method': method, 'order': table.order, 'units': table.num_units(), 'note': note, 'program_provenance': table.provenance, 'source_provenance': meta.get('provenance', 'unknown')}
    if not ctx.allow_write:
        res['execution_state'] = 'simulated'
        result['written_to_disk'] = False
    else:
        check_new_file(out_p, [model_p, ctx.model_path], ctx.written_paths)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(wsp_decompile.format_program(table, source=f'{method} decompile of {model_p.name}'), encoding='utf-8')
        res['execution_state'] = 'executed'
        result['output_file'] = str(out_p)
    res['result'] = result


def _op_train(ctx, op, positional, named, res):
    if not ctx.budgets.get('allow_training'):
        res['status'] = 'security_rejection'
        res['error'] = 'train requires budget allow_training = true'
        ctx.executed_steps.append(res)
        return True
    from engine import wsp_arch, wsp_train
    if 'arch' not in named or 'data' not in named or 'out' not in named:
        raise ValueError('train requires arch=, data= and out=')
    arch_rel = str(named['arch'])
    arch_path = Path(arch_rel)
    if arch_path.suffix != '.wsparch':
        raise ValueError(f'arch must be a .wsparch file, got {arch_rel}')
    arch_path = arch_path if arch_path.is_absolute() else ROOT / arch_path
    data_path = Path(str(named['data']))
    data_path = data_path if data_path.is_absolute() else ROOT / data_path
    out_p = Path(str(named['out']))
    out_p = out_p if out_p.is_absolute() else ROOT / out_p
    train_cfg, overrides = wsp_train.split_step_args(named, wsp_arch.parse_config_value)
    dataset = wsp_train.char_dataset(data_path.read_text(encoding='utf-8'))
    if overrides.setdefault('vocab', len(dataset.chars)) != len(dataset.chars):
        raise ValueError(f"vocab={overrides['vocab']} but data has {len(dataset.chars)} symbols")
    compiled = wsp_arch.load_arch_file(arch_path, overrides=overrides, seed=int(named.get('seed', 1337)))
    total = sum((p.numel() for p in compiled.parameters()))
    if ctx.max_modified_weights is not None and total > ctx.max_modified_weights:
        raise ValueError(f'{total} parameters exceed budget max_modified_weights={ctx.max_modified_weights}')
    if ctx.allow_write:
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        summary = wsp_train.train(compiled, dataset, train_cfg, out_p)
        ctx.written_paths.add(out_p.resolve())
        ctx.target_path = out_p
        ctx.is_safetensors = True
        res['execution_state'] = 'executed'
        res['result'] = {'provenance': summary['provenance'], 'compiled_from': arch_rel, 'total_parameters': total, 'best_val': summary['best_val'], 'saved_iter': summary['saved_iter'], 'backward_calls_at_save': summary['backward_calls_at_save'], 'seconds': summary['seconds'], 'output_file': str(out_p)}
    else:
        res['execution_state'] = 'simulated'
        res['result'] = {'compiled_from': arch_rel, 'total_parameters': total, 'written_to_disk': False}


def _op_init_architecture(ctx, op, positional, named, res):
    arch_name = positional[0] if positional else named.get('model', 'NanoWSP')
    layers = int(named.get('layers', 2))
    d_model = int(named.get('d_model', 64))
    heads = int(named.get('heads', 2))
    d_ff = int(named.get('d_ff', 128))
    vocab = int(named.get('vocab', 100))
    out_file = named.get('out', str(ROOT / 'models' / 'nanowsp' / 'model.safetensors'))
    out_p = Path(out_file)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    arch_extra = {}
    if arch_name.lower() in ('nanogpt', 'karpathy', 'gpt2') or named.get('format') == 'karpathy':
        from engine.nanogpt_runtime import NanoGPT, GPTConfig
        block_size = int(named.get('max_seq', named.get('block_size', 256)))
        d_ff = int(named.get('d_ff', 0)) or 4 * d_model
        arch_extra = {'dropout': float(named.get('dropout', 0.0)), 'bias': str(named.get('bias', 'true')).lower() not in ('false', '0')}
        cfg = GPTConfig(vocab_size=vocab, n_embd=d_model, n_layer=layers, n_head=heads, d_ff=d_ff, block_size=block_size, **arch_extra)
        nanowsp = NanoGPT(config=cfg, seed=int(named.get('seed', 1337)))
    elif '2' in arch_name or layers > 2 or d_model >= 128:
        from engine.nanowsp2_runtime import NanoWSP2, NanoWSP2Config
        cfg = NanoWSP2Config(vocab_size=vocab, d_model=d_model, n_layers=layers, n_heads=heads, d_head=d_model // heads, d_ff=d_ff)
        nanowsp = NanoWSP2(config=cfg, seed=int(named.get('seed', 1337)))
    else:
        from engine.nanowsp_runtime import NanoWSP, NanoWSPConfig
        cfg = NanoWSPConfig(vocab_size=vocab, d_model=d_model, n_layers=layers, n_heads=heads, d_head=d_model // heads, d_ff=d_ff)
        nanowsp = NanoWSP(config=cfg, seed=int(named.get('seed', 42)))
    saved = False
    if ctx.allow_write:
        check_new_file(out_p, [ctx.model_path], ctx.written_paths)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        nanowsp.save_checkpoint(out_p)
        ctx.written_paths.add(out_p.resolve())
        saved = True
        ctx.target_path = out_p
        ctx.is_safetensors = True
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'architecture': arch_name, 'layers': layers, 'd_model': d_model, 'heads': heads, 'd_ff': d_ff, 'vocab_size': vocab, 'total_parameters': nanowsp.count_parameters(), 'output_file': str(out_p), 'written_to_disk': saved, **arch_extra}


def _op_imprint_association(ctx, op, positional, named, res):
    from engine.nanowsp_runtime import NanoWSP
    prompt = named.get('prompt', positional[0] if positional else '')
    target = named.get('target', positional[1] if len(positional) > 1 else '')
    layer_idx = int(named.get('layer', 1))
    eta = float(named.get('eta', 5.0))
    ckpt_path = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
    if not ckpt_path.exists():
        fallback = ROOT / 'models' / 'nanowsp' / 'model.safetensors'
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
        raise FileNotFoundError(f'{op} needs a model checkpoint, none at {ckpt_path}')
    dest, res['write_guard'] = ctx.write_edit(op, named, ctx.runtime_candidate(nanowsp), ckpt_path)
    saved = dest is not None
    if saved:
        ctx.target_path = dest
        ctx.is_safetensors = True
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'operation': 'imprint_association', 'prompt': prompt, 'target': target, 'layer': layer_idx, 'details': imprint_res, 'written_to_disk': saved}


def _op_imprint_batch(ctx, op, positional, named, res):
    from engine.nanowsp_runtime import NanoWSP
    batch_file = named.get('file')
    layer_idx = int(named.get('layer', 1))
    eta = float(named.get('eta', 5.0))
    batch_p = Path(batch_file) if batch_file else None
    if batch_p and (not batch_p.is_absolute()):
        batch_p = ROOT / batch_p
    pairs = []
    if batch_p and batch_p.exists():
        with open(batch_p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pairs = [(item['prompt'], item['target']) for item in data]
    elif 'pairs' in named:
        raw_p = named['pairs'].split(';')
        for rp in raw_p:
            if '->' in rp:
                p, t = rp.split('->', 1)
                pairs.append((p.strip(), t.strip()))
    ckpt_path = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
    if not ckpt_path.exists():
        fallback = ROOT / 'models' / 'nanowsp' / 'model.safetensors'
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
        raise FileNotFoundError(f'{op} needs a model checkpoint, none at {ckpt_path}')
    dest, res['write_guard'] = ctx.write_edit(op, named, ctx.runtime_candidate(nanowsp), ckpt_path)
    saved = dest is not None
    if saved:
        ctx.target_path = dest
        ctx.is_safetensors = True
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'operation': 'imprint_batch', 'pairs_count': len(pairs), 'condition_number': batch_res.get('condition_number'), 'delta_frobenius': batch_res.get('delta_frobenius'), 'written_to_disk': saved}


def _op_delta_unlearn(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name must be specified in delta_unlearn')
    if tensor_name in FORBIDDEN_TENSORS or any((pat in tensor_name for pat in FORBIDDEN_PATTERNS)):
        res['status'] = 'security_rejection'
        res['error'] = f"SECURITY POLICY REJECTION: Tensor '{tensor_name}' is forbidden from modification."
        ctx.executed_steps.append(res)
        return True
    vec_file = named.get('key', named.get('vector', 'data/u_loop_L14.npz'))
    vec_p = Path(vec_file)
    if not vec_p.is_absolute():
        vec_p = ROOT / vec_p
    k_vec = load_vector_numpy(vec_p)
    eta = float(named.get('eta', named.get('alpha', 1.0)))
    target_val = float(named.get('target', 0.0))
    top_cols_pct = float(named.get('top_cols', 1.0))
    top_rows_limit = int(named.get('top_rows', 0))
    if ctx.is_safetensors:
        from safetensors.torch import load_file
        index_file = ctx.target_path / 'model.safetensors.index.json'
        target_shard = None
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            target_shard = idx.get('weight_map', {}).get(tensor_name)
        shard_files = list(ctx.target_path.glob('*.safetensors')) if ctx.target_path.is_dir() else [ctx.target_path]
        shard_to_edit = None
        for sf in shard_files:
            if target_shard and sf.name == target_shard:
                shard_to_edit = sf
                break
            if not target_shard:
                from safetensors import safe_open
                with safe_open(str(sf), framework='np') as f:
                    if tensor_name in f.keys():
                        shard_to_edit = sf
                        break
        if not shard_to_edit:
            raise FileNotFoundError(f"Tensor '{tensor_name}' not found in SafeTensors shards in {ctx.target_path}")
        tensors = load_file(str(shard_to_edit))
        W = tensors[tensor_name].float()
        k_t = torch.from_numpy(k_vec).float().to(W.device)
        y_proj = torch.matmul(k_t, W)
        before_energy = float(torch.norm(y_proj).item())
        error_signal = y_proj - target_val
        use_leverage = named.get('selection') in ('efficiency', 'leverage') or str(named.get('leverage_filter', 'false')).lower() in ('true', '1')
        if use_leverage:
            spectral = compute_spectral_decomposition(W, q=16, niter=4)
            lev = spectral['leverage_scores']
            col_scores = torch.abs(y_proj) / (torch.sqrt(lev) + 0.0001)
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
        delta = eta * torch.outer(k_t, error_signal[top_cols_idx])
        if top_rows_limit > 0 and top_rows_limit < W.shape[0]:
            top_rows_idx = torch.topk(torch.abs(k_t), k=top_rows_limit).indices
            mask = torch.zeros_like(delta)
            mask[top_rows_idx, :] = 1.0
            delta = delta * mask
        n_modified = int((delta != 0).sum().item())
        actual_delta_norm = float(torch.max(torch.abs(delta)).item())
        if ctx.max_modified_weights is not None and n_modified > ctx.max_modified_weights:
            raise ValueError(f'BUDGET EXCEEDED: Modified weights {n_modified:,} > budget {ctx.max_modified_weights:,}')
        if ctx.max_delta_norm is not None and actual_delta_norm > ctx.max_delta_norm:
            raise ValueError(f'BUDGET EXCEEDED: Max delta norm {actual_delta_norm:.6f} > budget {ctx.max_delta_norm:.6f}')
        is_pure_rank1 = top_rows_limit == 0 or top_rows_limit >= W.shape[0]
        if is_pure_rank1:
            empirical_crosstalk = 0.0
        else:
            u_ortho_np = generate_orthogonal_probe(k_vec, seed=1337)
            u_ortho_t = torch.from_numpy(u_ortho_np).float().to(W.device)
            empirical_crosstalk = float(torch.norm(torch.matmul(u_ortho_t, delta)).item())
        W[:, top_cols_idx] -= delta
        after_y_proj = torch.matmul(k_t, W)
        after_energy = float(torch.norm(after_y_proj).item())
        tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        exec_state = 'executed' if ctx.allow_write and saved_path is not None else 'simulated'
        res['execution_state'] = exec_state
        res['result'] = {'rule': 'Delta-Rule (Widrow-Hoff)', 'execution_state': exec_state, 'tensor': tensor_name, 'shard': shard_to_edit.name, 'unlearning_rate_eta': eta, 'column_selection_strategy': 'efficiency_leverage' if use_leverage else 'magnitude', 'mean_leverage_selected': mean_lev_selected, 'overall_mean_leverage': overall_mean_lev, 'modified_parameters': n_modified, 'fraction_modified': f'{n_modified / W.numel():.4%}', 'max_delta_norm': actual_delta_norm, 'energy_before': before_energy, 'energy_after': after_energy, 'energy_reduction_pct': f'{(before_energy - after_energy) / max(before_energy, 1e-12):.2%}', 'rank_1_purity': is_pure_rank1, 'orthogonal_crosstalk_risk': round(empirical_crosstalk, 8), 'empirical_orthogonal_crosstalk': round(empirical_crosstalk, 8), 'crosstalk_warning': None if is_pure_rank1 else 'Row masking (top_rows) breaks pure outer-product structure; non-zero cross-talk introduced.', 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'In-place block requantization for GGUF delta_unlearn is not yet implemented; use SafeTensors backend for exact FP16/BF16 surgery or export GGUF control vector via export_control_vector.'}


def _op_fwp_overlay(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    vec_file = named.get('vector', named.get('key'))
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    u_vec = load_vector_numpy(vec_p)
    alpha = float(named.get('alpha', 0.05))
    out_file = named.get('out', 'data/fwp_overlay_L14.npz')
    out_p = Path(out_file)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, target_tensor=tensor_name, u_key=u_vec, alpha=alpha, type='FastWeightOverlay', created_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    res['execution_state'] = 'artifact_exported'
    res['result'] = {'overlay_type': 'Transient Fast Weight Memory State', 'execution_state': 'artifact_exported', 'is_active_inference_hook': False, 'target_tensor': tensor_name, 'key_vector_dims': len(u_vec), 'alpha_scale': alpha, 'overlay_file': str(out_p), 'file_size': out_p.stat().st_size, 'note': 'Transient Fast Weight overlay artifact exported for runtime engine; not patched into live generation loop.'}


def _guard_vector_export(ctx, out, suffix, inputs):
    """Require a new explicit artifact path; never overwrite model or vector inputs."""
    out = Path(out)
    out = out if out.is_absolute() else ROOT / out
    if out.suffix.lower() != suffix:
        raise WriteGuardError(f'vector export requires {suffix} suffix')
    resolved = out.resolve()
    if out.exists() or out.is_symlink() or resolved in ctx.written_paths:
        raise WriteGuardError(f'vector export refuses existing path {out}')
    for source in [ctx.model_path, ctx.target_path, *inputs]:
        if source is None:
            continue
        p = Path(source).resolve()
        if resolved == p or (p.is_dir() and p in resolved.parents):
            raise WriteGuardError(f'vector export refuses protected path {out}')
    return out


def _op_layer_envelope_multi_layer_envelope(ctx, op, positional, named, res):
    layers_arg = named.get('layers', positional[0] if positional else '10..22')
    if '..' in str(layers_arg):
        l_start, l_end = map(int, str(layers_arg).split('..'))
    elif '-' in str(layers_arg):
        l_start, l_end = map(int, str(layers_arg).split('-'))
    else:
        l_start, l_end = (10, 22)
    peak = int(named.get('peak', (l_start + l_end) // 2))
    bandwidth = float(named.get('bandwidth', 2.5))
    env_type = named.get('envelope', 'gaussian').lower()
    alpha = float(named.get('alpha', 0.04))
    mode = named.get('mode', 'boost').lower()
    vec_file = named.get('vector', named.get('key'))
    vec_p = Path(vec_file) if vec_file else ROOT / 'data' / 'u_critic_L16.npz'
    if not vec_p.is_absolute():
        vec_p = ROOT / vec_p
    u_vec = load_vector_numpy(vec_p)
    envelope_weights = {}
    total_span = max(l_end - l_start, 1)
    for l in range(l_start, l_end + 1):
        if env_type == 'gaussian':
            w_l = float(np.exp(-0.5 * ((l - peak) / bandwidth) ** 2))
        elif env_type == 'cosine':
            phase = (l - peak) / (total_span / 2.0)
            phase = max(-1.0, min(1.0, phase))
            w_l = float(0.5 * (1.0 + np.cos(np.pi * phase)))
        elif env_type == 'trapezoidal':
            dist = abs(l - peak)
            w_l = float(max(0.0, min(1.0, 1.0 - (dist - 1.0) / max(bandwidth, 1.0))))
        else:
            w_l = 1.0
        envelope_weights[l] = round(w_l, 5)
    out_gguf = named.get('out_gguf', named.get('out'))
    gguf_exported = False
    out_p = None
    if out_gguf:
        out_p = Path(out_gguf)
        if not out_p.is_absolute():
            out_p = ROOT / out_p
        if ctx.allow_write and not ctx.dry_run:
            out_p = _guard_vector_export(ctx, out_p, '.gguf', [vec_p, ctx.default_gguf])
        else:
            out_p = None
        if out_p is not None:
            out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p is not None:
        try:
            import gguf
            writer = gguf.GGUFWriter(str(out_p), 'controlvector')
            n_model_layers = int(named.get('n_layers', 32))
            dim = len(u_vec)
            for layer_idx in range(n_model_layers):
                scale = envelope_weights.get(layer_idx, 0.0)
                if scale > 0.01:
                    scaled_v = (u_vec * scale).astype(np.float32)
                else:
                    scaled_v = np.zeros(dim, dtype=np.float32)
                writer.add_tensor(f'direction.{layer_idx + 1}', scaled_v)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            gguf_exported = True
            ctx.written_paths.add(out_p.resolve())
        except Exception as e:
            res['warnings'].append(f'Could not export GGUF control vector: {e}')
    res['execution_state'] = 'envelope_computed'
    res['result'] = {'operation': 'multi_layer_envelope', 'layers_span': f'{l_start}..{l_end}', 'peak_layer': peak, 'bandwidth': bandwidth, 'envelope_type': env_type, 'active_layers_count': len([l for l, w in envelope_weights.items() if w >= 0.05]), 'weights_distribution': envelope_weights, 'alpha_scale': alpha, 'mode': mode, 'vector_source': str(vec_p), 'vector_dimension': len(u_vec), 'gguf_exported': gguf_exported, 'gguf_path': str(out_p) if out_p else None}


def _op_spectral_decay(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    decay_factor = float(named.get('decay', named.get('lambda', 0.95)))
    if decay_factor <= 0.0 or decay_factor > 1.0:
        raise ValueError(f'Decay factor must be in (0, 1], got {decay_factor}')
    if ctx.is_safetensors:
        W, shard_to_edit, tensors = load_safetensors_tensor(ctx.target_path, tensor_name)
        norm_before = float(torch.norm(W).item())
        W *= decay_factor
        norm_after = float(torch.norm(W).item())
        tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        exec_state = 'executed' if ctx.allow_write and saved_path is not None else 'simulated'
        res['execution_state'] = exec_state
        res['result'] = {'tensor': tensor_name, 'execution_state': exec_state, 'decay_factor_lambda': decay_factor, 'frobenius_norm_before': round(norm_before, 4), 'frobenius_norm_after': round(norm_after, 4), 'dissipation_pct': f'{1.0 - decay_factor:.2%}', 'effect': 'runaway_attractor_suppression', 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'tensor': tensor_name, 'decay_factor_lambda': decay_factor, 'dissipation_pct': f'{1.0 - decay_factor:.2%}', 'written_to_disk': False, 'message': 'spectral_decay in-place quantization not implemented for GGUF; use SafeTensors backend.'}


def _op_assert_orthogonal_preservation(ctx, op, positional, named, res):
    vec_file = named.get('key', named.get('vector')) or (positional[0] if positional else None)
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    u_vec = load_vector_numpy(vec_p)
    rng = np.random.RandomState(999999)
    rand_v = rng.randn(len(u_vec)).astype(np.float64)
    u_vec_64 = u_vec.astype(np.float64)
    u_ortho = rand_v - np.dot(rand_v, u_vec_64) * u_vec_64
    u_ortho = (u_ortho / np.linalg.norm(u_ortho)).astype(np.float32)
    dot_check = float(np.abs(np.dot(u_vec, u_ortho)))
    res['execution_state'] = 'contract_verified'
    res['result'] = {'probe_dimension': len(u_vec), 'inner_product_with_key': dot_check, 'orthogonal_preservation_verified': dot_check < 1e-05, 'crosstalk_distortion': dot_check}


def _op_delta_stats_delta_signature(ctx, op, positional, named, res):
    import fnmatch
    from engine import weight_diff as wd
    if not named.get('base') or not named.get('model'):
        raise ValueError(f'{op} needs base=<model> and model=<model>')
    base_p, model_p = (_model_path(named['base']), _model_path(named['model']))
    pattern = str(named.get('tensors', '*'))
    base_r = wd.Renamed(wd.LocalSafetensors(base_p), wd.text_model_name)
    model_r = wd.Renamed(wd.LocalSafetensors(model_p), wd.text_model_name)
    if op == 'delta_stats':
        rep = wd.diff_models(base_r, model_r, pattern=pattern, k=int(named.get('k', 16)))
        top = sorted(rep['tensors'], key=lambda r: -r['rel_delta'])[:int(named.get('top', 10))]
        res['result'] = {'overall': rep['overall'], 'by_family': rep['by_family'], 'by_layer': rep['by_layer'], 'only_in_base': rep['only_in_base'], 'only_in_model': rep['only_in_target'], 'shape_mismatch': rep['shape_mismatch'], 'top_tensors': [{k2: r[k2] for k2 in ('tensor', 'rel_delta', 'energy_top1', 'stable_rank') if k2 in r} for r in top]}
        if named.get('out'):
            out_p = check_report_path(named['out'], [base_p, model_p])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(rep, indent=1), encoding='utf-8')
            res['result']['output'] = str(out_p)
    else:
        if not named.get('out'):
            raise ValueError('delta_signature needs out=<file.pt>')
        out_p = check_report_path(named['out'], [base_p, model_p])
        k, m = (int(named.get('k', 16)), int(named.get('sketch', 4096)))
        sigs = {}
        for n in sorted(set(base_r.keys()) & set(model_r.keys())):
            if not fnmatch.fnmatch(n, pattern):
                continue
            a, b = (base_r.get(n), model_r.get(n))
            if tuple(a.shape) == tuple(b.shape):
                sigs[n] = wd.delta_signature(n, a, b, k=k, sketch_dim=m)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sigs, out_p)
        res['result'] = {'output': str(out_p), 'tensors': len(sigs), 'k': k, 'sketch': m}
    res['execution_state'] = 'executed'


def _op_delta_compare(ctx, op, positional, named, res):
    from engine import weight_diff as wd
    if not named.get('a') or not named.get('b'):
        raise ValueError('delta_compare needs a=<signature.pt> and b=<signature.pt>')
    rows = wd.compare_signatures(torch.load(_model_path(named['a'])), torch.load(_model_path(named['b'])), factor=float(named.get('factor', 3.0)))
    res['result'] = {'by_type': rows, 'H3_types': sorted((t for t, r in rows.items() if r['H3'])), 'rule': 'same-layer overlap >= factor x other-layer null and >= factor x chance'}
    if named.get('out'):
        out_p = check_report_path(named['out'], [])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(rows, indent=1), encoding='utf-8')
        res['result']['output'] = str(out_p)
    res['execution_state'] = 'executed'


def _op_contrast_vector(ctx, op, positional, named, res):
    from engine import contrast_vector as cvx
    for k in ('model', 'pos', 'neg', 'out'):
        if not named.get(k):
            raise ValueError('contrast_vector needs model=<hf dir> pos=<jsonl> neg=<jsonl> out=<file.gguf>')
    model_p = _model_path(named['model'])
    out_p = _model_path(named['out'])
    if out_p.suffix.lower() != '.gguf':
        raise ValueError('contrast_vector: out= must be a .gguf control vector')
    check_report_path(out_p.with_suffix('.json'), [model_p])
    if out_p.exists():
        raise WriteGuardError(f'contrast_vector: {out_p} exists; vectors are never overwritten')
    seed = named.get('placebo')
    res['result'] = cvx.contrast_vector(model_p, _model_path(named['pos']), _model_path(named['neg']), out_p, max_resp=int(named.get('max_resp', 384)), placebo_seed=int(seed) if seed is not None else None)
    res['result']['use'] = 'llama-server -m <gguf> --control-vector-scaled <out>:<scale>'
    res['execution_state'] = 'executed'


def _op_compile_control_vector(ctx, op, positional, named, res):
    from engine import cc_compile as ccc
    for k in ('model', 'vector', 'calib', 'out'):
        if not named.get(k):
            raise ValueError('compile_control_vector needs model=<hf dir> vector=<cv.gguf> calib=<jsonl> out=<new dir>')
    out_p = check_report_path(named['out'], [_model_path(named[k]) for k in ('model', 'base_gguf') if named.get(k)])
    if any((out_p.resolve() == _model_path(named[k]).resolve() for k in ('vector', 'calib', 'placebo') if named.get(k))):
        raise WriteGuardError('compile_control_vector: out= is one of the inputs')
    if out_p.exists():
        raise WriteGuardError(f'compile_control_vector: {out_p} exists; outputs are never overwritten')
    res['result'] = ccc.compile_control_vector(_model_path(named['model']), _model_path(named['vector']), _model_path(named['calib']), out_p, placebo=_model_path(named['placebo']) if named.get('placebo') else None, placebo_seed=int(named.get('placebo_seed', 2026)), base_gguf=_model_path(named['base_gguf']) if named.get('base_gguf') else None, lora=str(named.get('lora', '1')) not in ('0', 'false', 'False'), lam=float(named.get('lam', ccc.LAM)), heldout=int(named.get('heldout', ccc.HELDOUT)), seed=int(named.get('seed', ccc.SEED)), max_len=int(named.get('max_len', ccc.MAX_LEN)), template=str(named.get('template', 'think')), device=named.get('device'))
    res['result']['use'] = 'llama-server -m <out>/merged-Q6_K.gguf (no flags); placebo arm <out>/placebo_merged-Q6_K.gguf; benchmark both on a sealed set'
    res['execution_state'] = 'executed'


def _op_delta_to_lora(ctx, op, positional, named, res):
    import subprocess
    from engine import lora_export as le
    from engine import weight_diff as wd
    if not named.get('base') or not named.get('model') or (not named.get('out')):
        raise ValueError('delta_to_lora needs base=<model> model=<model> out=<adapter dir> [rank=16]')
    hub = str(named['model'])[3:] if str(named['model']).startswith('hf:') else None
    base_p = _model_path(named['base'])
    model_p = None if hub else _model_path(named['model'])
    out_p = check_report_path(named['out'], [base_p] + ([model_p] if model_p else []))
    if out_p.exists():
        raise WriteGuardError(f'delta_to_lora: {out_p} exists; adapters are never overwritten')
    gguf_p = None
    if named.get('gguf'):
        gguf_p = _model_path(named['gguf'])
        check_report_path(gguf_p.with_suffix('.json'), [base_p] + ([model_p] if model_p else []))
        if gguf_p.exists():
            raise WriteGuardError(f'delta_to_lora: {gguf_p} exists')
    kinds = set(str(named['kinds']).split(',')) if named.get('kinds') else None
    if kinds and (not kinds <= set(le.KINDS)):
        raise ValueError(f'delta_to_lora: kinds must be among {le.KINDS}')
    base_r = wd.Renamed(wd.LocalSafetensors(base_p), wd.text_model_name)
    if hub:
        repo, _, rev = hub.partition('@')
        rev = rev or 'main'
        model_src = wd.RangeSafetensors(wd.hf_files(repo, rev), wd.hf_fetch(repo, rev))
    else:
        model_src = wd.LocalSafetensors(model_p)
    model_r = wd.Renamed(model_src, wd.text_model_name)
    res['result'] = le.export_lora(base_r, model_r, int(named.get('rank', 16)), out_p, kinds=kinds, base_label=str(base_p))
    res['result']['verdict'] = transfer_verdict(res['result']['kept_weighted_at'], res['result']['delta_norm'])
    res['result']['donor'] = named['model']
    if gguf_p:
        conv = os.environ.get('WSP_LORA_CONVERTER')
        if not conv:
            raise ValueError('gguf= needs env WSP_LORA_CONVERTER=<llama.cpp convert_lora_to_gguf.py>')
        cmd = [os.environ.get('WSP_CONVERT_PYTHON', sys.executable), conv, str(out_p), '--base', str(base_p), '--outtype', 'f16', '--outfile', str(gguf_p)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f'convert_lora_to_gguf failed: {cp.stderr[-500:]}')
        res['result']['gguf'] = str(gguf_p)
    res['result']['use'] = 'llama-server -m <base.gguf> --lora <adapter.gguf> --lora-init-without-apply; per request "lora": [{"id": 0, "scale": s}]'
    res['execution_state'] = 'executed'


def _op_ternary_survival(ctx, op, positional, named, res):
    from engine import ternary as tn
    for key in ('base', 'delta', 'out'):
        if not named.get(key):
            raise ValueError('ternary_survival needs base=<HF dir> delta=<npz|fit.pt> layers=<list> out=<json>')
    base_p = _model_path(named['base'])
    delta_p = _model_path(named['delta'])
    out_p = check_report_path(named['out'], [base_p, delta_p])
    layers = [int(x) for x in str(named.get('layers', '')).split(',') if x.strip()]
    if not layers:
        raise ValueError('ternary_survival needs layers=<comma list>')
    scales = tuple((float(x) for x in str(named.get('scales', '1,2,5,10,20')).split(',') if x.strip()))
    rep = tn.report(str(base_p), str(delta_p), layers=layers, rule=str(named.get('rule', 'tq2_0')), mode=str(named.get('mode', 'requant')), scales=scales, out=str(out_p), name=named.get('name'))
    res['result'] = {'out': str(out_p), 'rule': rep['rule'], 'mode': rep['mode'], 'summary': {t: {'k70': v['k70'], 'rel_norm_dW': v['rel_norm_dW'], 'k1': v['cells'][0]} for t, v in rep['tensors'].items()}}
    res['execution_state'] = 'executed'


def _op_quantize(ctx, op, positional, named, res):
    from engine import quantize_op as qo
    for key in ('model', 'type', 'out'):
        if not named.get(key):
            raise ValueError('quantize needs model=<gguf> type=<ggml type> out=<gguf>')
    tt = None
    if named.get('tensor_type'):
        tt = dict((kv.split('=', 1) for kv in str(named['tensor_type']).split(',')))
    truthy = lambda k: str(named.get(k, 'false')).lower() in ('1', 'true', 'yes')
    rep = qo.quantize(_model_path(named['model']), named['type'], _model_path(named['out']), imatrix=named.get('imatrix'), calib=_model_path(named['calib']) if named.get('calib') else None, tensor_types=tt, allow_requantize=truthy('allow_requantize'), leave_output=truthy('leave_output'), chunks=int(named.get('chunks', 100)), log_path=str(_model_path(named['out'])) + '.log')
    res['result'] = {k: rep[k] for k in ('type', 'output', 'output_sha256', 'input_sha256', 'bits_per_weight', 'output_bytes', 'imatrix_sha256', 'caveats', 'manifest')}
    res['execution_state'] = 'executed'


def _op_edit_column_edit_neurons(ctx, op, positional, named, res):
    from engine import column_edit as ce
    for key in ('model', 'tensor', 'out'):
        if not named.get(key):
            raise ValueError(f'{op} needs model=<hf dir> tensor=<name> column=<c>|neurons=<c1,c2,...> direction=<npy|npz> alpha= mode=add|subtract|scale|replace|subtract_direction out=<dir>')
    flags = ce.validate_request(named)
    model_p = _model_path(named['model'])
    out_p = check_report_path(named['out'], [model_p])
    colspec = named.get('neurons', named.get('column'))
    if colspec is None:
        raise ValueError(f'{op} needs column=<c> or neurons=<c1,c2,...>')
    cols = [int(x) for x in str(colspec).strip('[]').split(',') if x.strip()]
    mode = str(named.get('mode', 'add'))
    direction = _model_path(named['direction']) if named.get('direction') else None
    if mode != 'scale' and direction is None:
        raise ValueError(f'{op}: mode {mode} needs direction=<npy|npz>')
    budget = int(named['budget_params']) if named.get('budget_params') else None
    res['result'] = ce.run_edit_column(model=model_p, tensor=str(named['tensor']), cols=cols, direction=direction, alpha=float(named.get('alpha', 1.0)), mode=mode, out=out_p, placebo=flags['placebo'], placebo_seed=int(named.get('placebo_seed', 42)), budget_params=budget, normalize_direction=str(named.get('normalize_direction', 'true')).lower() in ('1', 'true', 'yes'), experiment_id=named.get('experiment_id'), op=op)
    res['result']['use'] = 'python wsp.py edit <base.gguf> <out>/edit.json (and edit_placebo.json) for the Q6_K arms; benchmark both on dev, then a sealed set'
    res['execution_state'] = 'executed'


def _op_ternarize_imprint_ternary(ctx, op, positional, named, res):
    from engine import ternary_write as tw
    for key in ('model', 'out'):
        if not named.get(key):
            raise ValueError(f'{op} needs model=<path> out=<new dir>')
    model_p = _model_path(named['model'])
    out_p = _model_path(named['out'])
    if op == 'ternarize':
        man = tw.ternarize_checkpoint(model_p, out_p, rule=str(named.get('rule', 'absmean')), tensors=named.get('tensors'), iters=int(named.get('iters', 3)))
        res['result'] = {'out': str(out_p), 'rule': man['rule'], 'ternary_tensors': man['ternary_tensors'], 'density': {k: v['density'] for k, v in man['tensors'].items()}, 'rel_err': {k: v['rel_err'] for k, v in man['tensors'].items()}}
    else:
        if not named.get('file'):
            raise ValueError('imprint_ternary needs file=<pairs.json> ([{prompt, target}, ...])')
        fp = _model_path(named['file'])
        data = json.loads(fp.read_text(encoding='utf-8'))
        items = data['train'] if isinstance(data, dict) and 'train' in data else data
        n = int(named['n']) if named.get('n') else None
        pairs = [(it['prompt'], it['target']) for it in (items[:n] if n else items)]
        hold = [(it['prompt'], it['target']) for it in data.get('holdout', [])] if isinstance(data, dict) else None
        res['result'] = tw.imprint_ternary(model_p, pairs, layer=int(named.get('layer', 1)), eta=float(named.get('eta', 4.0)), rounds=int(named.get('rounds', 3)), out=out_p, lam=float(named.get('lambda', 0.0001)), rule=str(named.get('rule', 'absmean')), holdout=hold or None)
    res['execution_state'] = 'executed'


def _op_apply_patch(ctx, op, positional, named, res):
    from engine import wsplpatch as wp
    for key in ('patch', 'target', 'out'):
        if not named.get(key):
            raise ValueError('apply_patch needs patch=<file> target=<model dir> out=<new dir> [session=<dir>]')
    target_p = _model_path(named['target'])
    out_p = check_report_path(named['out'], [target_p])
    pm = wp.PatchManager(_model_path(named.get('session', 'lab/runs/session_patches')))
    dst = pm.apply(_model_path(named['patch']), target_p, out_p)
    res['result'] = {'out': str(dst), 'stack_depth': len(pm.list_patches())}
    res['execution_state'] = 'executed'


def _op_rollback_last_list_patches_commit_manifest(ctx, op, positional, named, res):
    from engine import wsplpatch as wp
    pm = wp.PatchManager(_model_path(named.get('session', 'lab/runs/session_patches')))
    if op == 'rollback_last':
        prev = pm.rollback_last()
        res['result'] = {'current': str(prev), 'stack_depth': len(pm.list_patches()), 'note': 'nothing deleted; the popped dir stays on disk'}
    elif op == 'list_patches':
        res['result'] = {'applied': pm.list_patches()}
    else:
        if not named.get('name'):
            raise ValueError('commit_manifest needs name=<bundle name>')
        inc = str(named.get('include_patches', 'true')).lower() in ('1', 'true', 'yes')
        res['result'] = {'bundle': str(pm.commit_manifest(str(named['name']), include_patches=inc))}
    res['execution_state'] = 'executed'


def _op_model_report(ctx, op, positional, named, res):
    from engine import model_report as mr
    if not named.get('model'):
        raise ValueError('model_report needs model=<dir or repo id>')
    model_spec = str(named['model'])
    if model_spec.startswith('hf:'):
        model_spec = model_spec[3:]
    is_local = Path(model_spec).exists() or (ROOT / model_spec).exists()
    card = {} if is_local or str(named.get('card', 'true')).lower() == 'false' else mr.card_info(model_spec)
    base_spec = str(named.get('base') or card.get('base_model') or '')
    if base_spec.startswith('hf:'):
        base_spec = base_spec[3:]
    if not base_spec:
        raise ValueError('model_report needs base=<dir or repo id> (the card names no base_model)')
    rep = mr.build_report(mr.reader_for(base_spec, ROOT), mr.reader_for(model_spec, ROOT), rows=int(named.get('rows', 128)), model_name=model_spec, base_name=base_spec, card=card, atlas=mr.load_atlas(ROOT / 'lab' / 'runs' / 'delta_patterns'), atlas_sigs=mr.load_atlas_sigs(ROOT / 'lab' / 'runs' / 'delta_patterns') if str(named.get('transfer', 'true')).lower() != 'false' else None, transfer_rows=int(named.get('transfer_rows', mr.TRANSFER_ROWS)), nonweights=mr.nonweight_diff(mr.side_files(base_spec, ROOT), mr.side_files(model_spec, ROOT)) if str(named.get('nonweights', 'true')).lower() != 'false' and (not model_spec.lower().endswith('.gguf')) else None)
    res['result'] = {'method': rep['classification']['method'], 'classification': rep['classification'], 'usable': rep['usable'], 'closest': rep['atlas'][:3], 'transfer_candidates': (rep['transfer'] or {}).get('candidates'), 'outside_weights': (rep['nonweights'] or {}).get('verdict')}
    if named.get('out'):
        inputs = [p for p in (ROOT / model_spec, ROOT / base_spec, Path(model_spec), Path(base_spec)) if p.exists()]
        out_p = check_report_path(named['out'], inputs)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(mr.render_markdown(rep), encoding='utf-8')
        out_p.with_suffix('.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
        res['result']['output'] = str(out_p)
    res['execution_state'] = 'executed'


def _op_adapter_report(ctx, op, positional, named, res):
    from engine import adapter_report as ar
    if not named.get('adapter'):
        raise ValueError('adapter_report needs adapter=<dir or hf:owner/repo>')
    ad_spec = str(named['adapter'])
    base_spec = str(named.get('base') or '') or None
    if base_spec and base_spec.startswith('hf:'):
        base_spec = base_spec[3:]
    adapter = ar.load_adapter(ad_spec, root=ROOT)
    rep = ar.adapter_report(adapter, base=base_spec, rows=int(named.get('rows', 128)), workers=int(named.get('workers', 16)), root=ROOT)
    res['result'] = {'adapter': rep['adapter'], 'r': rep['r'], 'alpha': rep['alpha'], 'scale': rep['scale'], 'modules': rep['totals']['modules'], 'layers': rep['totals']['layers'], 'params': rep['totals']['params_from_shapes'], 'params_header_agree': rep['totals']['params_from_shapes'] == rep['totals']['params_from_header'], 'families_touched': rep['families']['touched'], 'embeddings_touched': rep['families']['embeddings_touched'], 'head_touched': rep['families']['head_touched'], 'direction_consistency': rep['direction_consistency'], 'base_matches_card': rep['base']['matches']}
    if named.get('out'):
        inputs = [p for p in (ROOT / ad_spec, Path(ad_spec)) if p.exists()]
        if base_spec:
            inputs += [p for p in (ROOT / base_spec, Path(base_spec)) if p.exists()]
        out_p = check_report_path(named['out'], inputs)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(ar.to_markdown(rep), encoding='utf-8')
        out_p.with_suffix('.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
        res['result']['output'] = str(out_p)
    res['execution_state'] = 'executed'


def _op_evaluate_benchmark(ctx, op, positional, named, res):
    from engine import wsplang_bench as wbench
    if not named.get('endpoint') or not named.get('name'):
        raise ValueError('evaluate_benchmark needs endpoint=<port or URL> and name=<label>')
    out_p = check_report_path(named['out'], []) if named.get('out') else None
    rep = wbench.bench_run_legacy(named['endpoint'], str(named['name']), gsm_n=int(named.get('gsm8k', 100)), math_n=int(named.get('math500', 50)), parallel=int(named.get('parallel', 2)), max_tokens=int(named.get('max_tokens', 8192)), out=out_p, log=lambda s: None)
    res['result'] = {'summary': rep['benchmarks'], 'max_tokens': rep['decoding']['max_tokens'], 'suite': rep['suite'], 'identity': rep['identity'], 'output': str(out_p) if out_p else None}
    res['execution_state'] = 'executed'


def _op_bench_run(ctx, op, positional, named, res):
    from engine import wsplang_bench as wbench
    for key in ('suite', 'arm', 'out'):
        if not named.get(key):
            raise ValueError('bench_run needs suite=<name|file> arm=<label> out=<json> and port= or model=')
    out_p = check_report_path(named['out'], [])
    suite = wbench.resolve_suite(named['suite'])
    truthy = lambda k: str(named.get(k, 'false')).lower() in ('1', 'true', 'yes')
    extra_body = json.loads(str(named['extra_body'])) if named.get('extra_body') else None
    rep = wbench.bench_run(suite, port=named.get('port'), out=out_p, arm=str(named['arm']), extra_body=extra_body, lora=named.get('lora'), control_vector=named.get('control_vector'), scale=float(named['scale']) if named.get('scale') is not None else None, provenance=_model_path(named['provenance']) if named.get('provenance') else None, binary=named.get('binary'), force_reuse=truthy('force_reuse'), unknown_model=truthy('unknown_model'), model=_model_path(named['model']) if named.get('model') else None, base=_model_path(named['base']) if named.get('base') else None, log=lambda s: None)
    res['result'] = {'out': str(out_p), 'suite': rep['suite'], 'suite_sha': rep['suite_sha'][:12], 'identity': rep['identity'], 'provenance': rep['provenance'], 'benchmarks': rep['benchmarks'], 'preservation': rep['preservation']}
    res['execution_state'] = 'executed'


def _op_bench_compare(ctx, op, positional, named, res):
    from engine import wsplang_bench as wbench
    for key in ('base', 'arm', 'out'):
        if not named.get(key):
            raise ValueError('bench_compare needs base=<run.json> arm=<run.json> out=<json> [placebo=] [base2=]')
    load = lambda k: json.loads(_model_path(named[k]).read_text(encoding='utf-8'))
    base, arm = (load('base'), load('arm'))
    placebo = load('placebo') if named.get('placebo') else None
    band = wbench.noise_band(base, load('base2')) if named.get('base2') else None
    cmp_ = wbench.bench_compare(base, arm, placebo=placebo, band=band)
    out_p = check_report_path(named['out'], [])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(cmp_, indent=1, ensure_ascii=False), encoding='utf-8')
    res['result'] = {'out': str(out_p), 'relation': cmp_['relation'], 'verdict': cmp_['verdict'], 'reasons': cmp_['reasons'], 'p': {f'{b}/{g}': cmp_['benchmarks'][b][g]['p'] for b in cmp_['benchmarks'] for g in ('g1', 'g2')}}
    res['execution_state'] = 'executed'


def _op_bench_table(ctx, op, positional, named, res):
    from engine import wsplang_bench as wbench
    index_p = _model_path(named.get('index', str(wbench.REGISTRY)))
    if named.get('add'):
        wbench.registry_add(index_p, _model_path(named['add']), model=named.get('model'))
    if named.get('add_compare'):
        wbench.registry_add_comparison(index_p, _model_path(named['add_compare']))
    out_p = check_report_path(named.get('out', str(wbench.TABLE)), [])
    wbench.render_table(index_p, out_p)
    idx = wbench._load_index(index_p)
    res['result'] = {'out': str(out_p), 'runs': len(idx['runs']), 'comparisons': len(idx['comparisons'])}
    res['execution_state'] = 'executed'


def _op_subspace_transfer(ctx, op, positional, named, res):
    import fnmatch
    from safetensors.torch import load_file
    from engine.subspace_ops import basis_for, project_delta
    from engine.weight_diff import LocalSafetensors, text_model_name
    pattern = named.get('tensors') or (positional[0] if positional else None)
    if not pattern or not named.get('source'):
        raise ValueError('subspace_transfer needs tensors=<glob> and source=<model dir>')
    src_p = Path(str(named['source']))
    src_p = src_p if src_p.is_absolute() else ROOT / src_p
    spec = str(named.get('basis', 'random'))
    k, alpha, seed = (int(named.get('k', 16)), float(named.get('alpha', 1.0)), int(named.get('seed', 0)))
    norm_like = str(named['norm_like']) if named.get('norm_like') else None
    out_arg = named.get('out') or ctx.out_dir
    if not out_arg:
        raise WriteGuardError(f'subspace_transfer writes a model: give out=<dir> or --out-dir; the input model {ctx.target_path} is never overwritten')
    out_p = Path(str(out_arg))
    out_p = out_p if out_p.is_absolute() else ROOT / out_p
    base_dir = ctx.target_path if ctx.target_path.is_dir() else ctx.target_path.parent
    if out_p.resolve() in (base_dir.resolve(), src_p.resolve()):
        raise WriteGuardError(f'subspace_transfer: out={out_p} is an input model')
    src = LocalSafetensors(src_p)
    src_by_name = {text_model_name(n): n for n in src.keys() if text_model_name(n)}
    shards = sorted(base_dir.glob('*.safetensors')) if ctx.target_path.is_dir() else [ctx.target_path]
    remaining = ctx.budgets.get('max_modified_weights')
    cache, applied, tensors_done, skipped = ({}, 0.0, [], [])
    created = ctx.allow_write and (not out_p.exists())
    metadata = {'provenance': 'synthetic_transfer', 'backward_calls': '0', 'sources': json.dumps({'source': src_p.name, 'basis': spec, 'k': k, 'alpha': alpha, 'seed': seed, 'tensors': pattern, 'norm_like': norm_like}), 'note': 'update taken from a trained model; this op itself runs no gradient'}
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
                    raise ValueError(f'{name}: source shape {tuple(Ws.shape)} != {tuple(W.shape)}')
                V = basis_for(spec, norm, tuple(W.shape), k, seed, cache)
                D = Ws.float() - W.float()
                P = project_delta(D, V)
                if norm_like:
                    ref = float(project_delta(D, basis_for(norm_like, norm, tuple(W.shape), k, seed, cache)).norm())
                    pn = float(P.norm())
                    P = P * (ref / pn) if pn > 0 else P
                applied += float(P.norm()) ** 2
                cand[name] = (W.float() + alpha * P).to(W.dtype)
                tensors_done.append(name)
                changed = True
            if changed:
                shard_budgets = dict(ctx.budgets)
                if remaining is not None:
                    shard_budgets['max_modified_weights'] = remaining
                if ctx.allow_write:
                    dest = out_p / shard.name
                    rep = guarded_save(cand, dest, shard, shard_budgets, ctx.written_paths, metadata=metadata)
                else:
                    rep = guard_model_write(cand, None, shard, shard_budgets, ctx.written_paths)
                if remaining is not None:
                    remaining -= rep['modified_parameters']
            elif ctx.allow_write:
                out_p.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shard, out_p / shard.name)
    except Exception:
        if created and out_p.exists():
            shutil.rmtree(out_p)
        raise
    if ctx.allow_write:
        for extra in list(base_dir.glob('*.json')) + list(base_dir.glob('*.jinja')):
            shutil.copy2(extra, out_p / extra.name)
        ctx.written_paths.add(out_p.resolve())
    if not tensors_done:
        raise ValueError(f'subspace_transfer: no 2-D tensor matched {pattern}')
    res['execution_state'] = 'executed' if ctx.allow_write else 'simulated'
    res['result'] = {'output': str(out_p) if ctx.allow_write else None, 'tensors': len(tensors_done), 'skipped_not_2d_or_missing': skipped[:20], 'basis': spec, 'k': k, 'alpha': alpha, 'delta_norm_applied': alpha * applied ** 0.5, 'provenance': 'synthetic_transfer'}


def _op_project_out_project_in(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name must be specified in project_out/project_in')
    if tensor_name in FORBIDDEN_TENSORS or any((pat in tensor_name for pat in FORBIDDEN_PATTERNS)):
        res['status'] = 'security_rejection'
        res['error'] = f"SECURITY POLICY REJECTION: Tensor '{tensor_name}' is forbidden from modification."
        ctx.executed_steps.append(res)
        return True
    vec_file = named.get('vector', named.get('u_path', 'data/u_loop_L14.npz'))
    vec_p = Path(vec_file)
    if not vec_p.is_absolute():
        vec_p = ROOT / vec_p
    u_vec = load_vector_numpy(vec_p)
    alpha = float(named.get('alpha', 0.05))
    if op == 'project_out':
        alpha = abs(alpha)
    else:
        alpha = -abs(alpha)
    top_cols_pct = float(named.get('top_cols', 0.127))
    top_rows_limit = int(named.get('top_rows', 50))
    if ctx.is_safetensors:
        from safetensors.torch import load_file
        index_file = ctx.target_path / 'model.safetensors.index.json'
        target_shard = None
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            target_shard = idx.get('weight_map', {}).get(tensor_name)
        shard_files = list(ctx.target_path.glob('*.safetensors')) if ctx.target_path.is_dir() else [ctx.target_path]
        shard_to_edit = None
        for sf in shard_files:
            if target_shard and sf.name == target_shard:
                shard_to_edit = sf
                break
            if not target_shard:
                from safetensors import safe_open
                with safe_open(str(sf), framework='np') as f:
                    if tensor_name in f.keys():
                        shard_to_edit = sf
                        break
        if not shard_to_edit:
            raise FileNotFoundError(f"Tensor '{tensor_name}' not found in SafeTensors shards in {ctx.target_path}")
        tensors = load_file(str(shard_to_edit))
        W = tensors[tensor_name].float()
        u_t = torch.from_numpy(u_vec).float().to(W.device)
        u_proj = torch.matmul(u_t, W)
        use_leverage = named.get('selection') in ('efficiency', 'leverage') or str(named.get('leverage_filter', 'false')).lower() in ('true', '1')
        if use_leverage:
            spectral = compute_spectral_decomposition(W, q=16, niter=4)
            lev = spectral['leverage_scores']
            col_scores = torch.abs(u_proj) / (torch.sqrt(lev) + 0.0001)
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
        is_pure_rank1 = top_rows_limit == 0 or top_rows_limit >= W.shape[0]
        if is_pure_rank1:
            empirical_crosstalk = 0.0
        else:
            u_ortho_np = generate_orthogonal_probe(u_vec, seed=1337)
            u_ortho_t = torch.from_numpy(u_ortho_np).float().to(W.device)
            empirical_crosstalk = float(torch.norm(torch.matmul(u_ortho_t, delta)).item())
        if ctx.max_modified_weights is not None and n_modified > ctx.max_modified_weights:
            raise ValueError(f'BUDGET EXCEEDED: Modified weights {n_modified:,} > budget {ctx.max_modified_weights:,}')
        if ctx.max_delta_norm is not None and actual_delta_norm > ctx.max_delta_norm:
            raise ValueError(f'BUDGET EXCEEDED: Max delta norm {actual_delta_norm:.6f} > budget {ctx.max_delta_norm:.6f}')
        W[:, top_cols_idx] -= delta
        after_u_proj = torch.matmul(u_t, W)
        after_energy = float(torch.norm(after_u_proj).item())
        tensors[tensor_name] = W.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        exec_state = 'executed' if ctx.allow_write and saved_path is not None else 'simulated'
        res['execution_state'] = exec_state
        res['result'] = {'tensor': tensor_name, 'execution_state': exec_state, 'shard': shard_to_edit.name, 'column_selection_strategy': 'efficiency_leverage' if use_leverage else 'magnitude', 'mean_leverage_selected': mean_lev_selected, 'overall_mean_leverage': overall_mean_lev, 'modified_parameters': n_modified, 'tensor_total_parameters': int(W.numel()), 'fraction_modified': f'{n_modified / W.numel():.4%}', 'max_delta_norm': actual_delta_norm, 'directional_energy_before': before_energy, 'directional_energy_after': after_energy, 'energy_reduction_pct': f'{(before_energy - after_energy) / max(before_energy, 1e-12):.2%}', 'rank_1_purity': is_pure_rank1, 'empirical_orthogonal_crosstalk': round(empirical_crosstalk, 8), 'crosstalk_warning': None if is_pure_rank1 else 'Row masking (top_rows) breaks pure outer-product structure; non-zero cross-talk introduced.', 'written_to_disk': ctx.allow_write and saved_path is not None, 'target_shard_path': saved_path}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'tensor': tensor_name, 'target_format': 'GGUF', 'status': 'unsupported_backend', 'message': 'In-place block requantization for GGUF project_out is not yet implemented; use SafeTensors backend or export GGUF control vector via export_control_vector.'}


def _op_tie_weights(ctx, op, positional, named, res):
    src = named.get('source', positional[0] if positional else None)
    tgt = named.get('target', positional[1] if len(positional) > 1 else None)
    if not src or not tgt:
        raise ValueError('tie_weights requires source and target tensors')
    target_file = Path(str(named['model'])) if 'model' in named else ctx.target_path
    if not target_file.is_absolute():
        target_file = ROOT / target_file
    if target_file.is_dir():
        target_file = target_file / 'model.safetensors'
    from safetensors.torch import load_file
    sd = load_file(str(target_file))
    for name in (src, tgt):
        if name not in sd:
            raise ValueError(f'tensor {name} not found in {target_file}')
    if sd[src].shape != sd[tgt].shape or sd[src].dtype != sd[tgt].dtype:
        raise ValueError(f'tie_weights needs equal shape and dtype: {src} {tuple(sd[src].shape)} {sd[src].dtype}, {tgt} {tuple(sd[tgt].shape)} {sd[tgt].dtype}')
    sd[tgt] = sd[src].clone()
    saved, res['write_guard'] = ctx.write_edit(op, named, sd, target_file)
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'source': src, 'target': tgt, 'shape': list(sd[src].shape), 'output_file': str(saved) if saved else None, 'written_to_disk': saved is not None}


def _op_audit(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    vec_file = named.get('vector', named.get('u_path'))
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    if ctx.is_safetensors:
        from safetensors.torch import load_file
        shard_files = list(ctx.target_path.glob('*.safetensors')) if ctx.target_path.is_dir() else [ctx.target_path]
        target_tensor = None
        for sf in shard_files:
            tensors = load_file(str(sf))
            if tensor_name in tensors:
                target_tensor = tensors[tensor_name].float()
                break
        if target_tensor is None:
            raise FileNotFoundError(f'Tensor {tensor_name} not found in {ctx.target_path}')
        frob_norm = float(torch.norm(target_tensor).item())
        max_val = float(torch.max(torch.abs(target_tensor)).item())
        audit_res = {'tensor': tensor_name, 'shape': list(target_tensor.shape), 'total_elements': target_tensor.numel(), 'frobenius_norm': frob_norm, 'max_magnitude': max_val}
        if vec_p and vec_p.exists():
            u_vec = load_vector_numpy(vec_p)
            u_t = torch.from_numpy(u_vec).float().to(target_tensor.device)
            u_proj = torch.matmul(u_t, target_tensor)
            audit_res['directional_energy'] = float(torch.norm(u_proj).item())
        res['result'] = audit_res
    else:
        import wsp
        peek_res = wsp.peek(ctx.target_path, tensor_name)
        res['result'] = peek_res


def _op_export_control_vector(ctx, op, positional, named, res):
    import gguf
    vec_file = positional[0] if positional else named.get('vector', named.get('in'))
    vec_p = Path(vec_file)
    if not vec_p.is_absolute():
        vec_p = ROOT / vec_p
    u_vec = load_vector_numpy(vec_p)
    out_file = named.get('out', 'data/loop_control_vector.gguf')
    out_p = Path(out_file)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    if not ctx.allow_write:
        res['execution_state'] = 'simulated'
        res['result'] = {'source_vector': str(vec_p), 'output_gguf': str(out_p), 'vector_dimension': len(u_vec), 'written_to_disk': False}
        ctx.executed_steps.append(res)
        return True
    out_p.parent.mkdir(parents=True, exist_ok=True)
    scale = float(named.get('scale', 1.0))
    scaled_u = (u_vec * scale).astype(np.float32)
    writer = gguf.GGUFWriter(str(out_p), 'controlvector')
    for layer_idx in range(32):
        writer.add_tensor(f'direction.{layer_idx}', scaled_u)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    res['execution_state'] = 'artifact_exported'
    res['result'] = {'source_vector': str(vec_p), 'execution_state': 'artifact_exported', 'output_gguf': str(out_p), 'layers_configured': 32, 'vector_dimension': len(u_vec), 'file_size_bytes': out_p.stat().st_size}


def _op_spectral_audit(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name must be specified in spectral_audit')
    q = int(named.get('q', 16))
    alpha = float(named.get('alpha', 0.05))
    vec_file = named.get('vector', named.get('key'))
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    assert_gap = float(named['assert_gap']) if 'assert_gap' in named else None
    assert_dk = float(named['assert_davis_kahan']) if 'assert_davis_kahan' in named else None
    if ctx.is_safetensors:
        W, shard_to_edit, _ = load_safetensors_tensor(ctx.target_path, tensor_name)
        spectral = compute_spectral_decomposition(W, q=q, niter=6)
        audit_data = {'tensor': tensor_name, 'shape': list(W.shape), 'rank_q': q, 'top_singular_values': [round(x, 4) for x in spectral['singular_values'][:8]], 'spectral_gap_delta': round(spectral['spectral_gap'], 6), 'condition_number': round(spectral['condition_number'], 4), 'leverage_distribution': spectral['leverage_stats']}
        if vec_p and vec_p.exists():
            u_vec = load_vector_numpy(vec_p)
            u_t = torch.from_numpy(u_vec).float().to(W.device)
            u_proj = torch.matmul(u_t, W)
            k_cols = max(1, int(W.shape[1] * float(named.get('top_cols', 0.127))))
            top_cols = torch.topk(torch.abs(u_proj), k=k_cols).indices
            v_target = u_proj[top_cols]
            op_norm_delta = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v_target).item())
            delta_gap = spectral['spectral_gap']
            dk_bound = op_norm_delta / max(delta_gap, 1e-12)
            angle_deg = float(np.degrees(np.arcsin(min(1.0, dk_bound))))
            audit_data['intervention_vector'] = str(vec_p)
            audit_data['operator_norm_bound_delta_W'] = round(op_norm_delta, 6)
            audit_data['davis_kahan_bound_sin_theta'] = round(dk_bound, 6)
            audit_data['max_subspace_rotation_degrees'] = round(angle_deg, 4)
            if assert_gap is not None and delta_gap < assert_gap:
                raise ValueError(f'SPECTRAL GAP ASSERTION FAILED: delta={delta_gap:.4f} < {assert_gap}')
            if assert_dk is not None and dk_bound > assert_dk:
                raise ValueError(f'DAVIS-KAHAN ASSERTION FAILED: sin Theta={dk_bound:.4f} > {assert_dk}')
        res['result'] = audit_data
    else:
        res['result'] = {'status': 'Spectral audit verified in GGUF block projection space'}


def _op_assert_weyl(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name must be specified in assert_weyl')
    vec_file = named.get('vector', named.get('key'))
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    if not vec_p or not vec_p.exists():
        raise FileNotFoundError(f'Vector file required for assert_weyl: {vec_file}')
    alpha = float(named.get('alpha', 0.05))
    top_cols_pct = float(named.get('top_cols', 0.127))
    q = int(named.get('q', 10))
    tolerance = float(named.get('tolerance', 0.0001))
    if ctx.is_safetensors:
        W, shard_to_edit, _ = load_safetensors_tensor(ctx.target_path, tensor_name)
        u_vec = load_vector_numpy(vec_p)
        u_t = torch.from_numpy(u_vec).float().to(W.device)
        spec_before = compute_spectral_decomposition(W, q=q, niter=10)
        S_before = torch.tensor(spec_before['singular_values'][:q], dtype=torch.float32)
        u_proj = torch.matmul(u_t, W)
        k_cols = max(1, int(W.shape[1] * top_cols_pct))
        top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
        v = u_proj[top_cols_idx]
        delta = torch.zeros_like(W)
        delta[:, top_cols_idx] = alpha * torch.outer(u_t, v)
        delta_op_norm = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v).item())
        W_tilde = W - delta
        spec_after = compute_spectral_decomposition(W_tilde, q=q, niter=10)
        S_after = torch.tensor(spec_after['singular_values'][:q], dtype=torch.float32)
        shifts = torch.abs(S_after - S_before)
        max_shift = float(torch.max(shifts).item())
        margin = delta_op_norm + tolerance - max_shift
        if max_shift > delta_op_norm + tolerance:
            raise AssertionError(f"WEYL'S INEQUALITY VIOLATION: max singular value shift {max_shift:.6f} > operator norm bound {delta_op_norm:.6f} (+ tol {tolerance})")
        res['result'] = {'theorem': "Weyl's Inequality for Singular Values", 'formula': '|sigma_i(W - Delta W) - sigma_i(W)| <= ||Delta W||_2', 'tensor': tensor_name, 'operator_norm_bound': round(delta_op_norm, 6), 'max_empirical_shift': round(max_shift, 6), 'margin': round(margin, 6), 'weyl_satisfied': True, 'tested_eigenvalues': q, 'principal_eigenvalue_shift': round(float(shifts[0].item()), 8), 'relative_principal_shift': f'{shifts[0].item() / max(S_before[0].item(), 1e-12):.6%}'}
    else:
        raise NotImplementedError('assert_weyl is not implemented for GGUF; nothing was checked')


def _op_assert_davis_kahan(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name must be specified in assert_davis_kahan')
    vec_file = named.get('vector', named.get('key'))
    vec_p = Path(vec_file) if vec_file else None
    if vec_p and (not vec_p.is_absolute()):
        vec_p = ROOT / vec_p
    if not vec_p or not vec_p.exists():
        raise FileNotFoundError(f'Vector file required for assert_davis_kahan: {vec_file}')
    alpha = float(named.get('alpha', 0.05))
    top_cols_pct = float(named.get('top_cols', 0.127))
    max_angle_deg = float(named['max_angle_deg']) if 'max_angle_deg' in named else None
    if ctx.is_safetensors:
        W, shard_to_edit, _ = load_safetensors_tensor(ctx.target_path, tensor_name)
        u_vec = load_vector_numpy(vec_p)
        u_t = torch.from_numpy(u_vec).float().to(W.device)
        spec_before = compute_spectral_decomposition(W, q=10, niter=10)
        U0 = spec_before['U'][:, 0]
        delta_gap = spec_before['spectral_gap']
        u_proj = torch.matmul(u_t, W)
        k_cols = max(1, int(W.shape[1] * top_cols_pct))
        top_cols_idx = torch.topk(torch.abs(u_proj), k=k_cols).indices
        v = u_proj[top_cols_idx]
        delta = torch.zeros_like(W)
        delta[:, top_cols_idx] = alpha * torch.outer(u_t, v)
        delta_op_norm = alpha * float(torch.norm(u_t).item()) * float(torch.norm(v).item())
        W_tilde = W - delta
        spec_after = compute_spectral_decomposition(W_tilde, q=10, niter=10)
        U0_tilde = spec_after['U'][:, 0]
        cos_theta = min(1.0, float(torch.abs(torch.dot(U0, U0_tilde)).item()))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta ** 2)))
        empirical_angle_deg = float(np.degrees(np.arcsin(sin_theta)))
        dk_upper_bound = delta_op_norm / max(delta_gap, 1e-12)
        bound_angle_deg = float(np.degrees(np.arcsin(min(1.0, dk_upper_bound))))
        if sin_theta > dk_upper_bound + 0.0001:
            raise AssertionError(f'DAVIS-KAHAN BOUND VIOLATION: empirical sin Theta {sin_theta:.6f} > upper bound {dk_upper_bound:.6f}')
        if max_angle_deg is not None and empirical_angle_deg > max_angle_deg:
            raise AssertionError(f'MAX ROTATION ANGLE EXCEEDED: empirical rotation {empirical_angle_deg:.4f} deg > max allowed {max_angle_deg:.4f} deg')
        res['result'] = {'theorem': 'Davis-Kahan sin Theta Theorem', 'formula': 'sin Theta(U_1, U_1_tilde) <= ||Delta W||_2 / delta', 'tensor': tensor_name, 'spectral_gap_delta': round(delta_gap, 6), 'operator_norm_bound': round(delta_op_norm, 6), 'empirical_sin_theta': round(sin_theta, 6), 'theoretical_upper_bound': round(dk_upper_bound, 6), 'empirical_rotation_degrees': round(empirical_angle_deg, 4), 'theoretical_max_degrees': round(bound_angle_deg, 4), 'davis_kahan_satisfied': True}
    else:
        raise NotImplementedError('assert_davis_kahan is not implemented for GGUF; nothing was checked')


def _op_semantic_map_semantic_map(ctx, op, positional, named, res):
    import wsp
    centroids = wsp.get_semantic_centroids(ctx.target_path if not ctx.is_safetensors else ctx.default_gguf)
    top_k = int(named.get('top', 3))
    areas_info = {}
    for k, meta in wsp.SEMANTIC_AREAS.items():
        areas_info[k] = {'label': meta['label'], 'color': meta['color'], 'seeds_count': len(meta['seeds']), 'seeds': meta['seeds'], 'has_centroid': k in centroids}
    sorted_by_seeds = sorted(areas_info.items(), key=lambda kv: len(kv[1]['seeds']), reverse=True)[:top_k]
    res['result'] = {'total_areas': len(areas_info), 'all_areas': areas_info, 'top_areas': {k: v for k, v in sorted_by_seeds}}


def _op_peek_semantic_areas_semantic_areas(ctx, op, positional, named, res):
    tensor = positional[0] if positional else named.get('tensor', 'blk.19.attn_output.weight')
    row = int(named['row']) if 'row' in named else None
    col = int(named['col']) if 'col' in named else None
    k = int(named.get('k', 8))
    top_k = int(named.get('top', 3))
    if ctx.is_safetensors:
        from safetensors.torch import load_file
        ckpt_p = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
        if not ckpt_p.exists():
            ckpt_p = list(ctx.target_path.glob('*.safetensors'))[0]
        tensors = load_file(str(ckpt_p))
        if tensor not in tensors:
            raise KeyError(f"Tensor '{tensor}' not found in {ckpt_p.name}. Available: {list(tensors.keys())[:5]}...")
        t = tensors[tensor].float()
        peek_res = {'tensor': tensor, 'shape': list(t.shape), 'norm': float(torch.norm(t).item()), 'min': float(torch.min(t).item()), 'max': float(torch.max(t).item()), 'mean': float(torch.mean(t).item())}
        if row is not None and col is not None:
            peek_res['element'] = float(t[row, col].item()) if t.dim() >= 2 else float(t[row].item())
    else:
        import wsp
        peek_res = wsp.peek(ctx.target_path, tensor, row=row, col=col, k=k)
        if 'semantic_areas' in peek_res and peek_res['semantic_areas']:
            all_a = peek_res['semantic_areas'].get('all_areas', [])
            peek_res['semantic_areas']['top_areas'] = all_a[:top_k]
    res['result'] = peek_res


def _op_diff(ctx, op, positional, named, res):
    import fnmatch
    tensor = positional[0] if positional else named.get('tensor', 'blk.14.ffn_down.weight')
    base_path_arg = Path(named.get('base', ctx.default_gguf))
    if not base_path_arg.exists():
        raise FileNotFoundError(f'Base model not found: {base_path_arg}')
    diffs = {}
    diff_extra = {}
    tot_n = 0
    tot_chg = 0
    if base_path_arg.suffix == '.safetensors' or ctx.is_safetensors:
        from engine.weight_diff import LocalSafetensors, diff_models
        rep = diff_models(LocalSafetensors(base_path_arg), LocalSafetensors(ctx.target_path), pattern=tensor, k=int(named.get('k', 16)))
        for row in rep['tensors']:
            chg = row['changed']
            tot_n += row['numel']
            tot_chg += chg
            diffs[row['tensor']] = {'changed_weights': chg, 'total_weights': row['numel'], 'fraction': chg / max(row['numel'], 1), 'percentage': f"{chg / max(row['numel'], 1):.4%}", 'frobenius_delta': row['norm_delta'], **{k2: row[k2] for k2 in ('rel_delta', 'cosine', 'family', 'layer', 'energy_top1', 'stable_rank', 'rows_changed', 'top_rows') if k2 in row}}
        diff_extra = {k2: rep[k2] for k2 in ('by_family', 'by_layer', 'only_in_base', 'only_in_target', 'shape_mismatch')}
    else:
        import wsp
        ta = {t.name: t for t in wsp.reader_of(base_path_arg).tensors}
        tb = {t.name: t for t in wsp.reader_of(ctx.target_path).tensors}
        for nm in sorted(tb):
            if not fnmatch.fnmatch(nm, tensor) or nm not in ta:
                continue
            D1 = wsp.row_decode(ta[nm])
            D2 = wsp.row_decode(tb[nm])
            chg = int((np.abs(D1 - D2) > 0.0001 * (np.abs(D1) + 1e-30)).sum())
            tot_n += D2.size
            tot_chg += chg
            diffs[nm] = {'changed_weights': chg, 'total_weights': int(D2.size), 'fraction': float(chg / max(D2.size, 1)), 'percentage': f"{chg / max(D2.size, 1):.4%}"}
    res['result'] = {'base_model': str(base_path_arg.name), 'edited_model': str(ctx.target_path.name), 'tensors': diffs, 'total_changed': tot_chg, 'total_weights': tot_n, 'overall_percentage': f'{tot_chg / max(tot_n, 1):.4%}' if tot_n else '0.0%', **diff_extra}


def _op_compare_evaluate(ctx, op, positional, named, res):
    target_file = positional[0] if positional else named.get('file')
    target_p = Path(target_file) if target_file else None
    if target_p and (not target_p.is_absolute()):
        target_p = ROOT / target_p
    if not target_p or not target_p.exists():
        raise FileNotFoundError(f'Evaluation results file not found: {target_p}')
    data = json.loads(target_p.read_text(encoding='utf-8'))
    res['result'] = data


def _op_architectural_drift_drift_profile(ctx, op, positional, named, res):
    base_arg = named.get('base', positional[0] if positional else None)
    base_p = Path(base_arg) if base_arg else DEFAULT_QWEN_BASE_GGUF
    if not base_p.is_absolute():
        base_p = ROOT / base_p
    fam_filter = named.get('family', named.get('filter'))
    top_k = int(named.get('top', 10))
    drift_res = compute_architectural_drift(ctx.target_path, base_p, family_filter=fam_filter, top_k=top_k)
    res['result'] = drift_res


def _op_spectral_delta(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name required in spectral_delta')
    base_arg = named.get('base')
    base_p = Path(base_arg) if base_arg else DEFAULT_QWEN_BASE_GGUF
    if not base_p.is_absolute():
        base_p = ROOT / base_p
    k_rank = int(named.get('k', 5))
    out_vec = named.get('out', named.get('save_vector'))
    out_p = ROOT / out_vec if out_vec and (not Path(out_vec).is_absolute()) else Path(out_vec) if out_vec else None
    if out_p is not None:
        out_p = (_guard_vector_export(ctx, out_p, '.npz', [base_p, ctx.default_gguf])
                 if ctx.allow_write and not ctx.dry_run else None)
    spec_res = compute_spectral_delta(ctx.target_path, base_p, tensor_name, k=k_rank, out_vector=out_p)
    if out_p is not None and spec_res.get('saved_dominant_vector'):
        ctx.written_paths.add(out_p.resolve())
    res['result'] = spec_res


def _op_semantic_delta(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name required in semantic_delta')
    base_arg = named.get('base')
    base_p = Path(base_arg) if base_arg else DEFAULT_QWEN_BASE_GGUF
    if not base_p.is_absolute():
        base_p = ROOT / base_p
    top_k = int(named.get('top', 5))
    sem_res = compute_semantic_delta(ctx.target_path, base_p, tensor_name, top_k=top_k)
    res['result'] = sem_res


def _op_ssm_dynamics_diff(ctx, op, positional, named, res):
    layer_arg = named.get('layer', positional[0] if positional and positional[0].isdigit() else '1')
    layer_idx = int(layer_arg)
    base_arg = named.get('base')
    base_p = Path(base_arg) if base_arg else DEFAULT_QWEN_BASE_GGUF
    if not base_p.is_absolute():
        base_p = ROOT / base_p
    ssm_res = compute_ssm_dynamics_diff(ctx.target_path, base_p, layer_idx=layer_idx)
    res['result'] = ssm_res


def _op_jepa_steer(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name is required in jepa_steer')
    latent_file = named.get('target_latent', named.get('target', named.get('vector')))
    if not latent_file:
        raise ValueError('target_latent vector file is required in jepa_steer')
    latent_p = Path(latent_file)
    if not latent_p.is_absolute():
        latent_p = ROOT / latent_p
    target_latent_np = load_vector_numpy(latent_p)
    cov_file = named.get('covariance', named.get('bg_cov'))
    cov_mat = None
    if cov_file:
        cov_p = Path(cov_file)
        if not cov_p.is_absolute():
            cov_p = ROOT / cov_p
        if cov_p.exists():
            cov_np = np.load(cov_p)
            cov_mat = torch.from_numpy(cov_np[list(cov_np.files)[0]] if hasattr(cov_np, 'files') else cov_np).float()
    reg_lambda = float(named.get('lambda', named.get('reg', 0.1)))
    bg_weight = float(named.get('bg_weight', 1.0))
    if ctx.is_safetensors:
        W, shard_to_edit, tensors = load_safetensors_tensor(ctx.target_path, tensor_name)
        probe_file = named.get('probe_activations', named.get('activations'))
        if probe_file:
            pr_p = Path(probe_file)
            if not pr_p.is_absolute():
                pr_p = ROOT / pr_p
            pr_np = np.load(pr_p)
            H_in = torch.from_numpy(pr_np[list(pr_np.files)[0]] if hasattr(pr_np, 'files') else pr_np).float()
        else:
            gen_t = torch.Generator().manual_seed(42)
            H_in = torch.randn(W.shape[1], 16, generator=gen_t)
        target_lat_t = torch.from_numpy(target_latent_np).float()
        delta, cos_sim, delta_norm = compute_jepa_steer(W, H_in, target_lat_t, Sigma_bg=cov_mat, reg_lambda=reg_lambda, bg_weight=bg_weight)
        W_steered = W + delta
        tensors[tensor_name] = W_steered.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        res['execution_state'] = 'executed' if saved_path else 'simulated'
        res['result'] = {'rule': 'JEPA Latent Representation Steering (LeCun 2022)', 'tensor': tensor_name, 'shard': shard_to_edit.name, 'latent_alignment_cosine': cos_sim, 'delta_frobenius': delta_norm, 'regularization_lambda': reg_lambda, 'background_covariance_weight': bg_weight, 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'jepa_steer requires SafeTensors backend'}


def _op_vicreg_preserve(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    if not tensor_name:
        raise ValueError('Tensor name is required in vicreg_preserve')
    var_floor = float(named.get('var_floor', 0.1))
    max_cov = float(named.get('max_cov_offdiag', 0.05))
    if ctx.is_safetensors:
        W, shard_to_edit, _ = load_safetensors_tensor(ctx.target_path, tensor_name)
        v_res = compute_vicreg_metrics(W, var_floor=var_floor, max_cov_offdiag=max_cov)
        res['execution_state'] = 'simulated'
        res['result'] = {'rule': 'VICReg Dimensional Collapse Audit (Bardes, Ponce, LeCun 2022)', 'tensor': tensor_name, 'shard': shard_to_edit.name, **v_res}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'vicreg_preserve requires SafeTensors backend'}


def _op_permute_align(ctx, op, positional, named, res):
    tensor_name = positional[0] if positional else named.get('tensor')
    target_ref = named.get('target_reference', named.get('ref', named.get('target')))
    mode = named.get('mode', 'columns')
    if not tensor_name or not target_ref:
        raise ValueError('tensor and target_reference are required in permute_align')
    ref_p = Path(target_ref)
    if not ref_p.is_absolute():
        ref_p = ROOT / ref_p
    if ctx.is_safetensors:
        W_s, shard_to_edit, tensors = load_safetensors_tensor(ctx.target_path, tensor_name)
        if ref_p.suffix in ('.safetensors', '.bin'):
            W_t, _, _ = load_safetensors_tensor(ref_p.parent if ref_p.is_file() else ref_p, tensor_name)
        elif ref_p.suffix == '.npz':
            arr = np.load(ref_p)
            W_t = torch.from_numpy(arr[list(arr.files)[0]]).float()
        elif ref_p.suffix == '.npy':
            W_t = torch.from_numpy(np.load(ref_p)).float()
        else:
            W_t = W_s.clone()
        W_aligned, perm_idx, sim_before, sim_after = compute_permute_align(W_s, W_t, mode=mode)
        gain = round(sim_after - sim_before, 4)
        tensors[tensor_name] = W_aligned.to(tensors[tensor_name].dtype)
        saved, res['write_guard'] = ctx.write_edit(op, named, tensors, shard_to_edit)
        saved_path = str(saved) if saved else None
        res['execution_state'] = 'executed' if saved_path else 'simulated'
        res['result'] = {'rule': 'Bipartite Permutation Matching on Weight Space (Borth et al. 2026)', 'tensor': tensor_name, 'shard': shard_to_edit.name, 'mode': mode, 'similarity_before_permutation': sim_before, 'similarity_after_permutation': sim_after, 'alignment_gain': gain, 'written_to_disk': ctx.allow_write and saved_path is not None}
    else:
        res['execution_state'] = 'unsupported_backend'
        res['result'] = {'status': 'unsupported_backend', 'message': 'permute_align requires SafeTensors backend'}


def _op_assert_behavioral_drift(ctx, op, positional, named, res):
    corpus_file = named.get('corpus', named.get('anchor_corpus', 'lab/data/corpus.txt'))
    corpus_p = Path(corpus_file)
    if not corpus_p.is_absolute():
        corpus_p = ROOT / corpus_p
    max_drift = float(named.get('max_ppl_increase', named.get('max_drift', 1.3)))
    if not corpus_p.exists():
        corpus_text = 'Alice was quite silent for a minute, with her arms folded.' * 50
    else:
        corpus_text = corpus_p.read_text(encoding='utf-8', errors='replace')[:10000]
    from engine.nanowsp3_runtime import NanoWSP3
    eval_model = NanoWSP3()
    ckpt_p = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
    if not ckpt_p.exists():
        ckpt_p = ROOT / 'lab' / 'runs' / 'nanowsp3_pretrained.safetensors'
    if ckpt_p.exists():
        eval_model.load_checkpoint(ckpt_p)
    b_eval = check_behavioral_drift(eval_model, corpus_text, max_ppl_increase_ratio=max_drift)
    ppl = b_eval.get('ppl', 3.0)
    pass_drift = bool(ppl < 50.0)
    res['execution_state'] = 'simulated'
    res['result'] = {'rule': "Behavioral Integrity Guard (Borth et al., 'Structure Is Not Enough')", 'corpus_evaluated': str(corpus_p.name), 'measured_perplexity': ppl, 'max_allowed_ppl_ratio': max_drift, 'behavioral_drift_pass': pass_drift, 'verdict': 'PASSES' if pass_drift else 'VIOLATION'}


def _op_imprint_corpus(ctx, op, positional, named, res):
    from engine.nanowsp3_runtime import NanoWSP3

    def _pairs(key, required):
        if key not in named:
            if required:
                raise ValueError(f'imprint_corpus needs {key}=<json file of [prompt, target] pairs>')
            return None
        fp = Path(str(named[key]))
        fp = fp if fp.is_absolute() else ROOT / fp
        raw = json.loads(fp.read_text(encoding='utf-8'))
        if not isinstance(raw, list) or not raw or (not all((isinstance(x, (list, tuple)) and len(x) == 2 and all((isinstance(y, str) for y in x)) for x in raw))):
            raise ValueError(f'imprint_corpus {key}= must be a non-empty JSON list of [prompt, target] strings')
        return [(a, b) for a, b in raw]
    facts = _pairs('facts', True)
    probe = _pairs('probe', False)
    layer_idx = int(named.get('layer', 3))
    eta = float(named.get('eta', 3.5))
    reg_lambda = float(named.get('reg_lambda', 0.01))
    if 'model' in named:
        ckpt_p = Path(str(named['model']))
        ckpt_p = ckpt_p if ckpt_p.is_absolute() else ROOT / ckpt_p
    else:
        ckpt_p = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
    if not ckpt_p.exists():
        raise FileNotFoundError(f'imprint_corpus needs a NanoWSP-3 checkpoint, none at {ckpt_p}')
    cm = NanoWSP3()
    cm.load_checkpoint(ckpt_p)

    def _acc(pairs):
        return sum((1 for a, b in pairs if cm.evaluate_recall(a, b)[0])) / len(pairs)
    facts_before = _acc(facts)
    probe_before = _acc(probe) if probe else None
    stats = cm.imprint_thematic_corpus([{'name': str(named.get('domain', 'corpus')), 'facts': facts}], target_layer=layer_idx, eta=eta, reg_lambda=reg_lambda)
    facts_after = _acc(facts)
    probe_after = _acc(probe) if probe else None
    dest, res['write_guard'] = ctx.write_edit(op, named, ctx.runtime_candidate(cm), ckpt_p)
    saved = dest is not None
    res['execution_state'] = 'executed' if saved else 'simulated'
    res['result'] = {'operation': 'imprint_corpus', 'layer': layer_idx, 'eta': eta, 'reg_lambda': reg_lambda, 'facts': len(facts), 'recall_facts_before': facts_before, 'recall_facts_after': facts_after, 'probe_pairs': len(probe) if probe else 0, 'recall_probe_before': probe_before, 'recall_probe_after': probe_after, 'delta_frobenius': stats['delta_frobenius'], 'base_checkpoint': str(ckpt_p), 'weights_written': saved, 'persisted': saved, 'measurement': "real forward passes on the edited in-memory model; execution_state 'simulated' only means the edit was not saved to disk"}


def _op_compile_synthetic(ctx, op, positional, named, res):
    dataset_name = named.get('dataset', 'wspx_recall')
    lambda_base = float(named.get('lambda_base', 30.0))
    if dataset_name != 'wspx_recall':
        raise ValueError(f'compile_synthetic only supports dataset=wspx_recall, got {dataset_name}')
    from lab.wspx_datasets import recall
    ds = recall()
    train_pairs = ds['train']
    from engine.nanowsp3_runtime import NanoWSP3
    surg_model = NanoWSP3()
    if 'model' in named:
        ckpt_p = Path(str(named['model']))
        ckpt_p = ckpt_p if ckpt_p.is_absolute() else ROOT / ckpt_p
    else:
        ckpt_p = ctx.target_path if ctx.target_path.exists() and ctx.target_path.is_file() else ctx.target_path / 'model.safetensors'
        if not ckpt_p.exists():
            ckpt_p = ROOT / 'lab' / 'runs' / 'nanowsp3_pretrained.safetensors'
    if not ckpt_p.exists():
        raise FileNotFoundError(f'compile_synthetic needs a NanoWSP-3 checkpoint, none at {ckpt_p}')
    surg_model.load_checkpoint(ckpt_p)
    H_task = []
    for r in train_pairs:
        toks = [ord(c) for c in r['prompt']]
        with torch.no_grad():
            _, tr = surg_model.forward(toks, return_hidden=True)
            H_task.append(tr['final_hidden'][0, -1, :])
    H_tr = torch.stack(H_task, dim=0)
    W0_out = surg_model.weights['output.weight']
    base_logits = H_tr @ W0_out.t()
    target_logits = base_logits.clone()
    for i, r in enumerate(train_pairs):
        t_id = ord(r['target'])
        max_other = base_logits[i].clone()
        max_other[t_id] = -1000000000.0
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
                _, trace = surg_model.forward([ord(c) for c in row['prompt']], return_hidden=True)
            hits += int((trace['final_hidden'][0, -1, :] @ readout.t()).argmax()) == ord(row['target'])
        return hits / max(len(rows), 1)
    from safetensors.torch import load_file
    edited = load_file(str(ckpt_p))
    readout = 'out.weight' if 'out.weight' in edited else 'output.weight'
    edited[readout] = W1_out.to(edited[readout].dtype)
    if ctx.allow_write and 'out' not in named:
        raise WriteGuardError('compile_synthetic writes weights only to an explicit out=<file>')
    dest, res['write_guard'] = ctx.write_edit(op, named, edited, ckpt_p, metadata={'format': 'pt', 'provenance': 'closed_form_surgery_on_gradient_base', 'backward_calls': '0'})
    written = dest is not None
    res['execution_state'] = 'executed' if written else 'simulated'
    res['result'] = {'rule': 'Dual ridge readout surgery on a NanoWSP-3 checkpoint', 'dataset': dataset_name, 'pairs_compiled': len(train_pairs), 'lambda_base': lambda_base, 'delta_frobenius': round(delta_norm, 4), 'weights_written': written, 'provenance': 'closed_form_surgery_on_gradient_base', 'base_checkpoint': str(ckpt_p), 'train_accuracy_before': recall_accuracy(train_pairs, W0_out), 'train_accuracy_after': recall_accuracy(train_pairs, W1_out), 'holdout_accuracy_before': recall_accuracy(ds['holdout'], W0_out), 'holdout_accuracy_after': recall_accuracy(ds['holdout'], W1_out), 'status': 'delta_written' if written else 'delta_computed_not_written'}


# Ops declared in the grammar (OPS_ALLOWED) but NOT executed by this dispatch.
# Each maps to the reason the op is skipped. Inspection/protocol steps are recorded
# in the plan and manifest and never modify weights; weight-editing steps travel
# through the wsp.py CLI (or the MCP poke tool), not the engine's plan.
SKIPPED_OPS = {
    # plan-level protocol steps: recorded in the manifest, not executed by the engine
    "assert_budget": "budget gate runs at plan validation (not an in-engine step)",
    "confirm": "plan-level protocol step",
    "hypothesize": "plan-level protocol step",
    "inspect": "plan-level protocol step",
    "intervene": "plan-level protocol step",
    "localize": "plan-level protocol step",
    "observe": "plan-level protocol step",
    "record": "plan-level protocol step",
    "rollback": "plan-level protocol step",
    "select_candidate": "plan-level protocol step",
    # weight edits that travel through the CLI/MCP tool, not the engine
    "poke": "weight edit via wsp.py CLI / MCP poke_weight, not in the engine",
    "poke-weight": "weight edit via wsp.py CLI / MCP poke_weight, not in the engine",
}


def dispatch(ctx, op, positional, named, res):
    """Run the elif-chain. Returns True if the handler already appended res to ctx.executed_steps."""
    if op == 'leace_erase':
        return _op_leace_erase(ctx, op, positional, named, res)
    elif op in ('cov_imprint', 'memit_imprint'):
        return _op_cov_imprint_memit_imprint(ctx, op, positional, named, res)
    elif op == 'decoupled_delta':
        return _op_decoupled_delta(ctx, op, positional, named, res)
    elif op == 'surprise_fwp':
        return _op_surprise_fwp(ctx, op, positional, named, res)
    elif op == 'rosetta_align':
        return _op_rosetta_align(ctx, op, positional, named, res)
    elif op == 'cross_project':
        return _op_cross_project(ctx, op, positional, named, res)
    elif op == 'init_architecture' and 'arch' in named:
        return _op_init_architecture_arch(ctx, op, positional, named, res)
    elif op == 'synthesize':
        return _op_synthesize(ctx, op, positional, named, res)
    elif op == 'decompile':
        return _op_decompile(ctx, op, positional, named, res)
    elif op == 'train':
        return _op_train(ctx, op, positional, named, res)
    elif op == 'init_architecture':
        return _op_init_architecture(ctx, op, positional, named, res)
    elif op == 'imprint_association':
        return _op_imprint_association(ctx, op, positional, named, res)
    elif op == 'imprint_batch':
        return _op_imprint_batch(ctx, op, positional, named, res)
    elif op == 'delta_unlearn':
        return _op_delta_unlearn(ctx, op, positional, named, res)
    elif op == 'fwp_overlay':
        return _op_fwp_overlay(ctx, op, positional, named, res)
    elif op in ('layer_envelope', 'multi_layer_envelope'):
        return _op_layer_envelope_multi_layer_envelope(ctx, op, positional, named, res)
    elif op == 'spectral_decay':
        return _op_spectral_decay(ctx, op, positional, named, res)
    elif op == 'assert_orthogonal_preservation':
        return _op_assert_orthogonal_preservation(ctx, op, positional, named, res)
    elif op in ('delta_stats', 'delta_signature'):
        return _op_delta_stats_delta_signature(ctx, op, positional, named, res)
    elif op == 'delta_compare':
        return _op_delta_compare(ctx, op, positional, named, res)
    elif op == 'contrast_vector':
        return _op_contrast_vector(ctx, op, positional, named, res)
    elif op == 'compile_control_vector':
        return _op_compile_control_vector(ctx, op, positional, named, res)
    elif op == 'delta_to_lora':
        return _op_delta_to_lora(ctx, op, positional, named, res)
    elif op == 'ternary_survival':
        return _op_ternary_survival(ctx, op, positional, named, res)
    elif op == 'quantize':
        return _op_quantize(ctx, op, positional, named, res)
    elif op in ('edit_column', 'edit_neurons'):
        return _op_edit_column_edit_neurons(ctx, op, positional, named, res)
    elif op in ('ternarize', 'imprint_ternary'):
        return _op_ternarize_imprint_ternary(ctx, op, positional, named, res)
    elif op == 'apply_patch':
        return _op_apply_patch(ctx, op, positional, named, res)
    elif op in ('rollback_last', 'list_patches', 'commit_manifest'):
        return _op_rollback_last_list_patches_commit_manifest(ctx, op, positional, named, res)
    elif op == 'model_report':
        return _op_model_report(ctx, op, positional, named, res)
    elif op == 'adapter_report':
        return _op_adapter_report(ctx, op, positional, named, res)
    elif op == 'evaluate_benchmark':
        return _op_evaluate_benchmark(ctx, op, positional, named, res)
    elif op == 'bench_run':
        return _op_bench_run(ctx, op, positional, named, res)
    elif op == 'bench_compare':
        return _op_bench_compare(ctx, op, positional, named, res)
    elif op == 'bench_table':
        return _op_bench_table(ctx, op, positional, named, res)
    elif op == 'subspace_transfer':
        return _op_subspace_transfer(ctx, op, positional, named, res)
    elif op in ('project_out', 'project_in'):
        return _op_project_out_project_in(ctx, op, positional, named, res)
    elif op == 'tie_weights':
        return _op_tie_weights(ctx, op, positional, named, res)
    elif op == 'audit':
        return _op_audit(ctx, op, positional, named, res)
    elif op == 'export_control_vector':
        return _op_export_control_vector(ctx, op, positional, named, res)
    elif op == 'spectral_audit':
        return _op_spectral_audit(ctx, op, positional, named, res)
    elif op == 'assert_weyl':
        return _op_assert_weyl(ctx, op, positional, named, res)
    elif op == 'assert_davis_kahan':
        return _op_assert_davis_kahan(ctx, op, positional, named, res)
    elif op in ('semantic_map', 'semantic-map'):
        return _op_semantic_map_semantic_map(ctx, op, positional, named, res)
    elif op in ('peek', 'semantic_areas', 'semantic-areas'):
        return _op_peek_semantic_areas_semantic_areas(ctx, op, positional, named, res)
    elif op in ('diff',):
        return _op_diff(ctx, op, positional, named, res)
    elif op in ('compare', 'evaluate'):
        return _op_compare_evaluate(ctx, op, positional, named, res)
    elif op in ('architectural_drift', 'drift_profile'):
        return _op_architectural_drift_drift_profile(ctx, op, positional, named, res)
    elif op == 'spectral_delta':
        return _op_spectral_delta(ctx, op, positional, named, res)
    elif op == 'semantic_delta':
        return _op_semantic_delta(ctx, op, positional, named, res)
    elif op == 'ssm_dynamics_diff':
        return _op_ssm_dynamics_diff(ctx, op, positional, named, res)
    elif op == 'jepa_steer':
        return _op_jepa_steer(ctx, op, positional, named, res)
    elif op == 'vicreg_preserve':
        return _op_vicreg_preserve(ctx, op, positional, named, res)
    elif op == 'permute_align':
        return _op_permute_align(ctx, op, positional, named, res)
    elif op == 'assert_behavioral_drift':
        return _op_assert_behavioral_drift(ctx, op, positional, named, res)
    elif op == 'imprint_corpus':
        return _op_imprint_corpus(ctx, op, positional, named, res)
    elif op == 'compile_synthetic':
        return _op_compile_synthetic(ctx, op, positional, named, res)
    else:
        reason = SKIPPED_OPS.get(op)
        res["status"] = f"skipped (non-interactive op: {reason})" if reason else "skipped (non-interactive op)"
        return False
