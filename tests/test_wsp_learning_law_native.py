"""Native nanoGPT projection integration for the frozen-feature learning law."""
import copy
import torch
from engine.nanogpt_runtime import GPTConfig, KarpathyGPT
from engine.wsp_learning_law import discover, generate, apply


def test_native_mlp_projection_law():
    torch.manual_seed(824)
    def fixture(seed, width, hidden):
        torch.manual_seed(seed)
        model = KarpathyGPT(GPTConfig(block_size=8, vocab_size=16,
            n_layer=2, n_head=2, n_embd=width, d_ff=hidden,
            dropout=0., bias=False)).double().eval()
        block = model.transformer.h[0]
        tokens = torch.randint(16, (96, 8))
        with torch.no_grad():
            z = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(8))
            z = z + block.attn(block.ln_1(z))
            x = block.mlp.gelu(block.mlp.c_fc(block.ln_2(z))).reshape(-1, hidden)
            teacher = torch.randn(width, hidden, dtype=x.dtype) / hidden**.5
            y = x @ teacher.T
        return model, tokens, x, y

    def reference(w, x, y, steps):
        p = torch.nn.Parameter(w.detach().clone())
        opt = torch.optim.SGD([p], lr=.2)
        for _ in range(steps):
            opt.zero_grad()
            ((x @ p.T - y).square().sum() / (2 * len(x))).backward()
            opt.step()
        return p.detach()

    a, _, x, y = fixture(12, 16, 64)
    w = a.transformer.h[0].mlp.c_proj.weight.detach()
    law = discover(w, reference(w, x[:512], y[:512], 1), x[:512], y[:512])
    b, tokens, x, y = fixture(19, 24, 96)
    initial = {k: v.clone() for k, v in b.state_dict().items()}
    w = b.transformer.h[0].mlp.c_proj.weight
    candidate = apply(w, generate(w, law, x[:512], y[:512], steps=128))
    expected = reference(w, x[:512], y[:512], 128)
    torch.testing.assert_close(candidate, expected, atol=1e-10, rtol=1e-10)
    with torch.no_grad():
        control = copy.deepcopy(b)
        control.transformer.h[0].mlp.c_proj.weight.copy_(expected)
        w.copy_(candidate)
        pred = b(tokens[64:], tokens[64:])[0]
        ref_pred = control(tokens[64:], tokens[64:])[0]
        torch.testing.assert_close(pred, ref_pred, atol=1e-10, rtol=1e-10)
        for name, value in b.state_dict().items():
            if name != "transformer.h.0.mlp.c_proj.weight":
                assert torch.equal(value, initial[name]), name
        before = (x[512:] @ initial["transformer.h.0.mlp.c_proj.weight"].T - y[512:]).square().mean()
        after = (x[512:] @ w.T - y[512:]).square().mean()
        assert after < before
        print({"heldout_projection_mse_before": float(before),
               "heldout_projection_mse_after": float(after),
               "max_weight_error": float((candidate-expected).abs().max()),
               "max_native_logit_error": float((pred-ref_pred).abs().max())})
