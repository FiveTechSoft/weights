# Architecture language (`.wsparch`)

A `.wsparch` file describes a neural network as data. `engine/wsp_arch.py` compiles it to a PyTorch module without importing any model class, so the text is the architecture: edit a line and the network changes. `programs/arch/nanogpt.wsparch` is the reference example.

The language only defines architectures and initial weights. It does not train or synthesize learned weights.

## Use

From a WSPLang program:

```
step init_architecture arch=programs/arch/nanogpt.wsparch layers=4 d=128 seed=1337 out=models/x/model.safetensors
```

Every extra argument overrides a `config` value; unknown names are errors. The step fails if the parameter count exceeds `max_modified_weights`.

From Python:

```python
from engine import wsp_arch
model = wsp_arch.load_arch_file("programs/arch/nanogpt.wsparch", overrides={"layers": 4}, seed=1337)
logits, loss = model(idx, targets)
```

The module also provides `generate()`, `save()` and a strict `load()`. Checkpoints use upstream tensor names and record the source hash in their metadata.

## Structure

```
arch name:
    config:
        d = 384
        ff = 4 * d

    graph(idx):
        scope transformer:
            tok = embedding(wte, idx, vocab, d)
        logits = linear(lm_head, tok, d, vocab, false)
        output logits

    init:
        pass:
            embedding.weight = normal(0, 0.02)
            linear.weight = normal(0, 0.02)
        tie transformer.wte.weight = lm_head.weight
```

### `config`

`name = expression`, evaluated in order, so later values can derive from earlier ones and follow overrides. Expressions allow numbers, `true`, `false`, earlier names, `+ - * / //`, unary minus and `sqrt()`.

### `graph(input)`

| Statement | Meaning |
|---|---|
| `var = op(args)` | Run a primitive |
| `a, b = split(x, 2)` | Split the last dimension |
| `var = other` | Alias a variable |
| `scope path:` | Prefix tensor names inside the block |
| `repeat i in expression:` | Repeat a block; use `{i}` in scope paths |
| `output var` | Declare the logits (required) |

| Primitive | Arguments | Tensors declared |
|---|---|---|
| `embedding` | name, ids, count, dim | `name.weight` |
| `positions` | ids | none |
| `add` | a, b | none |
| `dropout` | x, p | none |
| `gelu` | x | none |
| `layernorm` | name, x, dim, bias | `name.weight`, optional `name.bias` |
| `linear` | name, x, in, out, bias | `name.weight`, optional `name.bias` |
| `split` | x, parts | none |
| `causal_attention` | q, k, v, heads, p | none |

An embedding fed by `positions` sets the maximum sequence length. Dimensions are checked at compile time, and errors report the source line.

### `init`

Runs top to bottom. Each `pass:` visits tensors in graph order and applies the first matching rule, drawing from the global RNG. A tensor no pass reaches is a compile error.

- **Selectors:** `embedding.weight`, `linear.weight`, `linear.bias`, `layernorm.weight`, `layernorm.bias`, or `match "glob"` on the full tensor name.
- **Functions:** `normal(mean, std)`, `kaiming_uniform(a)`, `fan_in_uniform()`, `zeros()`, `ones()`.
- **`tie a = b`:** makes `a` an alias of `b`.

## nanoGPT parity

`tests/test_wsp_arch.py` compares the compiled model against `lab/reference/karpathy_nanogpt/model.py` (commit `3adf61e`) for: same-seed W0 at several sizes, logits, loss, dropout stream in train mode, a 25-step AdamW trajectory, sampled generation and checkpoint interchange. It also mutates the source and requires each mutation to break parity.

`lab/verify_wsp_arch.py` repeats the full training control on real text and compares its gate report with the official model's.

To match upstream for a given seed, the first init pass declares what `nn.Linear` and `nn.Embedding` draw at construction. Those values are overwritten, but they advance the RNG. W0 parity therefore cannot pin everything in that pass:

- moving `tie` before the first pass gives identical weights, because the draws are the same;
- `normal_` and `uniform_` consume the RNG identically for sizes that are multiples of 16, so such configs cannot tell them apart. The mutation tests use `d=34` for that reason.

## Limits

- The loss is fixed to cross-entropy with `ignore_index=-1`.
- The primitive set covers GPT-style models. There is no RoPE, RMSNorm, or caller-defined module, so NanoWSP cannot be expressed yet.
- `causal_attention` is one primitive, not built from smaller operations.
- Execution is an interpreter over a flat instruction list, with no graph optimization or `torch.compile` support tested.
- `init_architecture` without `arch=` still uses the older fixed Python classes.

## Safety

Source text is parsed with `ast` and evaluated through an allowlist, so it cannot run Python. Repeat counts, instruction counts and total parameters are limited before tensors are allocated.
