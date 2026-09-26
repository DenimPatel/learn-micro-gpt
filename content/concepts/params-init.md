---
id: params-init
title: Parameters and their initialisation
chapter: machinery
order: 3
difficulty: 1
summary: "state_dict is nine named matrices built by one lambda. Every weight is a Gaussian draw with std 0.08, and params flattens all 4,192 of them into a single list because the optimizer does not care about structure."
math: |
  W \in \mathbb{R}^{n_{out} \times n_{in}},\qquad W_{ij} \sim \mathcal{N}(0,\; \sigma^2),\quad \sigma = 0.08
shapes:
  - { name: wte, shape: "[27, 16]" }
  - { name: wpe, shape: "[16, 16]" }
  - { name: lm_head, shape: "[27, 16]" }
  - { name: "layer0.attn_wq", shape: "[16, 16]" }
  - { name: "layer0.mlp_fc1", shape: "[64, 16]" }
  - { name: "layer0.mlp_fc2", shape: "[16, 64]" }
anchors:
  python: { selector: "section:Initialize the parameters" }
  c: { selector: "def:init_weights" }
related: [autograd-value, adam, embedding, lm-head]
prereqs: [autograd-value]
---

```python
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
```

## The five numbers

`n_embd = 16` is the model's width — the size of every vector it passes around.
`n_head = 4` splits that into four independent attention heads of width
`head_dim = 16 // 4 = 4`. `n_layer = 1` is the depth. `block_size = 16` is the
maximum context length.

Every other shape follows from those four. Change `n_embd` to 32 and the model
is still 199 lines; you have simply made it twice as wide.

## `state_dict` is the whole model

Three global matrices, then nine per layer:

| key | shape | what it does |
| --- | --- | --- |
| `wte` | [27, 16] | token embedding: id -> vector |
| `wpe` | [16, 16] | position embedding: position -> vector |
| `lm_head` | [27, 16] | final projection: vector -> logits |
| `layer{i}.attn_wq/wk/wv` | [16, 16] | attention queries, keys, values |
| `layer{i}.attn_wo` | [16, 16] | attention output projection |
| `layer{i}.mlp_fc1` | [64, 16] | MLP up-projection (4x) |
| `layer{i}.mlp_fc2` | [16, 64] | MLP down-projection |

The names matter more than they look. `gpt()` reaches into this dict with
`state_dict[f'layer{li}.attn_wq']` using a format string, which means adding a
layer is a matter of changing `n_layer` and nothing else. The dict *is* the
parameter namespace, and the C port has the identical names in the identical
flat arrays — which is what makes the two files comparable line by line.

## Why 0.08

`std=0.08` is a small, deliberate number. The consequences are visible in the
very first forward pass:

- Activations stay small, so nothing saturates. The initial logits have a
  standard deviation around 0.30, which makes the initial distribution nearly
  uniform — measured entropy 3.252 against 3.296 for a perfectly uniform guess.
  See [the tokenizer](../concepts/tokenizer.md) for those numbers.
- A 16-wide dot product of N(0, 0.08) values has standard deviation
  `0.08 * sqrt(16) = 0.32`. That is the same 0.30, and it is the reason the
  network starts as a near-uniform guess rather than a confidently wrong one.

The general rule — `std = 1/sqrt(n_in)`, i.e. 0.25 here — would be the
"correct" scaling for fan-in initialisation, and this file uses 0.08 instead.
It works. The reason is RMSNorm immediately after the embedding, which
renormalises the scale anyway, so the initial scale matters much less than it
would in an unnormalised network.

## The flatten

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
```

Three levels of unpacking, and the only reason Adam can be written as a single
loop over `range(len(params))`. From here on, structure is gone: parameter 2,047
is a weight in `mlp_fc1` and there is no way to tell from the list.

That is a genuine trade. A flat list makes the optimizer trivially simple and the
gradient bookkeeping trivial too. It also means nothing checks that a gradient
arrived at the right place — which is precisely the class of bug that the C
port's separate `g_wte`, `g_attn_wq`, ... arrays are immune to and this file is
not. The printed `num params: 4192` is the check that the shapes are what you
think they are.

:::callout "Count them yourself"
`27*16 + 16*16 + 27*16 + 4*(16*16*4) + 64*16 + 16*64` = `432 + 256 + 432 + 1024 + 1024 + 1024` =
**4192**. If the number printed does not match your arithmetic, a shape is wrong
and every downstream concept in this chapter is about to be confusing.
:::
