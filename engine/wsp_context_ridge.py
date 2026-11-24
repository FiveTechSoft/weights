"""No-gradient contextual ridge synthesis for the declared NanoWSP-LM GPT."""
import torch
import torch.nn.functional as F

from engine.nanogpt_runtime import GPTConfig, NanoGPT


@torch.no_grad()
def fit(compiled, ids, *, heads, block, width, layers, ridge=0.01,
        samples=32768, device="cpu", seed=1337):
    """Fit last MLP projection to centered next-token logit targets.
    Frozen random contextual features; no donor, optimizer or backward.
    Final layernorm makes this an approximate residual target, not an exact logit fit.
    """
    if ridge <= 0 or samples <= 0 or block < 2 or len(ids) <= block:
        raise ValueError("invalid ridge, samples or corpus length")
    sd = compiled.export_state_dict()
    vocab = sd["transformer.wte.weight"].shape[0]
    cfg = GPTConfig(block_size=block, vocab_size=vocab, n_layer=layers,
                    n_head=heads, n_embd=width, bias=False, dropout=0.0)
    model = NanoGPT(cfg, seed).model
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    proj = model.transformer.h[-1].mlp.c_proj
    head = model.lm_head.weight.double()
    inverse = torch.linalg.pinv(head.T)
    # Row-vector target logits times pseudoinverse of the tied head transpose.
    gen = torch.Generator().manual_seed(seed)
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    gram = torch.zeros((proj.in_features, proj.in_features), dtype=torch.float64, device=device)
    cross = torch.zeros((proj.in_features, width), dtype=torch.float64, device=device)
    captured = []
    hook = proj.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        remaining = samples
        while remaining:
            batch = min(8, (remaining + block - 1) // block)
            starts = torch.randint(len(ids) - block, (batch,), generator=gen).to(device)
            positions = starts[:, None] + torch.arange(block, device=device)
            x, y = ids[positions], ids[positions + 1]
            logits, _ = model(x, y)
            n = min(remaining, batch * block)
            features = captured.pop().reshape(-1, proj.in_features)[:n].double()
            target = F.one_hot(y.reshape(-1)[:n], vocab).double()
            target = 3.0 * (target - 1.0 / vocab)
            # Match residual logit error in the local linear approximation.
            error = (target - logits.reshape(-1, vocab)[:n].double()) @ inverse
            gram += features.T @ features
            cross += features.T @ error
            remaining -= n
    finally:
        hook.remove()
    scale = gram.diag().mean().clamp_min(1e-12)
    gram.diagonal().add_(ridge * scale)
    delta = torch.linalg.solve(gram, cross).T
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite ridge solution")
    key = f"transformer.h.{layers-1}.mlp.c_proj.weight"
    sd[key] = sd[key] + delta.cpu().to(sd[key].dtype)
    compiled.load_state_dict(sd, strict=True)
    return {"method": "context_ridge", "provenance": "closed_form_data_fit",
            "backward_calls": 0, "samples": samples, "ridge": ridge,
            "delta_frobenius": float(delta.norm()), "tensor": key}
