---
id: mlp-block
title: The MLP block
chapter: architecture
order: 6
difficulty: 2
summary: "Two matrix multiplies with a ReLU between them, wrapped in a residual connection. The MLP has no attention in it at all — it is the per-position computation, and it holds two thirds of the layer's parameters."
math: |
  \mathrm{MLP}(x) = W_2 \,\mathrm{ReLU}\!\big(W_1 x\big), \qquad W_1 \in \mathbb{R}^{4d \times d},\; W_2 \in \mathbb{R}^{d \times 4d}
shapes:
  - { name: x, shape: "[16]" }
  - { name: mlp hidden, shape: "[64] = 4 * n_embd" }
  - { name: mlp out, shape: "[16]" }
anchors:
  python: { selector: "def:gpt offset:[27, -3]" }
  c: { selector: "def:forward_pos offset:[59, -6]" }
related: [attention-block, linear, rmsnorm, params-init]
prereqs: [linear, rmsnorm]
---

```python
# 2) MLP block
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]
```

Seven lines. Structurally it is the *same shape* as the attention block that
precedes it — normalise, transform, add the residual — and that symmetry is the
point. The whole transformer layer is that pattern, twice.

## What is different from attention

**It is per-position.** There is no loop over `t`, no `keys`, no weights. Each
position's `[16]` vector goes in and a `[16]` vector comes out, independently of
every other position. In a batched implementation you would notice this as the
absence of any mixing across the sequence axis.

**It is where the parameters are.** At `n_embd = 16`:

```
mlp_fc1  [64, 16]  = 1024
mlp_fc2  [16, 64]  = 1024
                     -----
                     2048
```

against attention's `4 * 16 * 16 = 1024`. The MLP is two thirds of the layer,
and it holds the vast majority of what the network actually stores knowledge in.
Attention decides *where* to look; the MLP decides *what to do with what it
found*. In current practice, parameter count and most of the model's knowledge
live in these two matrices, and in a dozen more like them.

## Why 4x

`4 * n_embd` is a bottleneck-and-expand: compress to 16 dimensions? No — expand
to 64, do the nonlinearity there, and compress back to 16. The ReLU needs room
to be a nonlinearity. A 16→16→16 MLP with a ReLU in the middle is severely
capacity-limited: the function can be written, but it is far easier to fit with
an intermediate space to work in.

4 is a convention, not a derivation. It has been 4 in essentially every
transformer since GPT-2. It is one of the numbers you can change and immediately
see the parameter count move.

## ReLU, not GeLU

```python
x = [xi.relu() for xi in x]
```

`relu` is `max(0, x)` with derivative 1 above zero and 0 below. The file's header
comment names the substitution: "GeLU -> ReLU". GeLU is a smooth, slightly
better approximation to how neurons are believed to fire; ReLU is six times
cheaper and trains identically well at this scale. Another deliberate
simplification, another thing you would change for a real model.

The zero derivative below the threshold is worth noticing. A ReLU that is off
contributes exactly nothing to the gradient, so a unit can get permanently
stuck — though because the whole vector is rescaled by `rmsnorm` on the way in,
and weights are updated by Adam rather than plain SGD, this is much less of a
problem than it was in 2012. It is the reason the ecosystem moved to GELU/SiLU
and then, eventually, back to ReLU with better initialisation schemes.

## The residual add

```python
x = [a + b for a, b in zip(x, x_residual)]
```

Byte-for-byte the same line as the end of the attention block. The MLP's job is
to compute a *correction* to the residual stream, not to replace it. Setting
`mlp_fc2` to zero would make this block an identity function and the model would
still train — it would just be a shallower model. That property is a direct
consequence of the residual connection and is why these stacks are
trainable to great depth.

:::callout "The one-line experiment"
Replace `x = [xi.relu() for xi in x]` with `x = [xi * xi for xi in x]` (a ReLU
replacement, the poor man's squared activation) and rerun. The loss still falls.
Then replace the whole MLP body with `x = x_residual` — a no-op — and watch it
stall near `ln(27)`. The first says "the model is robust to architectural detail
at this scale". The second says "4,192 parameters are almost all in the
embeddings, the head, and attention". Both are worth seeing once.
:::
