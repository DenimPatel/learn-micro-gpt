---
id: rmsnorm
title: RMSNorm
chapter: architecture
order: 3
difficulty: 2
summary: "RMSNorm rescales a vector by the reciprocal of its root-mean-square, with a small epsilon in the denominator. Four lines, and it is the reason a 4-layer network can stack without its activations either vanishing or exploding."
math: |
  \mathrm{rms}(x) = \sqrt{\frac{1}{n}\sum_i x_i^2 + \varepsilon}, \qquad \tilde{x}_i = \frac{x_i}{\mathrm{rms}(x)}
shapes:
  - { name: x, shape: "[16]" }
  - { name: ms, shape: "float" }
  - { name: scale, shape: "float = (ms + 1e-5) ** -0.5" }
anchors:
  python: { selector: "def:rmsnorm" }
  c: { selector: "def:rmsnorm_fwd" }
related: [attention-block, mlp-block, gpt-forward, linear]
prereqs: [linear]
---

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

Mean square, then inverse square root, then multiply. The name says it all:
root-mean-square normalisation.

## RMSNorm vs LayerNorm

LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm only
divides by the root-mean-square. That is the entire difference, and it costs
LayerNorm about 30% of its arithmetic — one pass instead of two, and no
subtraction.

What it gives up is the guarantee that the output has zero mean. In a transformer
that guarantee turns out not to matter, which is the empirical finding that
made RMSNorm the default in LLaMA, Gemma, and most of what came after. This file
notes it in a comment: "layernorm -> rmsnorm".

## The epsilon, and where it is

```python
scale = (ms + 1e-5) ** -0.5
```

`1e-5` goes **inside** the square root, added to the mean square. It guards
`ms == 0`, i.e. an all-zero input, which would otherwise be a division by zero.
The C port is more careful still, and its comments are worth reading:

```c
/* Add epsilon BEFORE sqrt to avoid 0/NaN */
```

Same placement, same reason. Note that it is *not* `ms ** -0.5 + eps_adam` —
putting epsilon after the power would bias the scale rather than bound it.

## Why it is load-bearing

Every residual branch in this architecture is preceded by an `rmsnorm`:

```python
x = rmsnorm(x)                    # once, after the embedding
x_residual = x
x = rmsnorm(x)                    # before attention
... 
x = [a + b for a, b in zip(x, x_residual)]
```

That pattern is a pre-norm transformer, and it is the reason deep stacks are
trainable at all. The residual connection gives gradients a direct path from the
loss to the input, straight past every block, and the norm keeps the magnitude
of `x` roughly constant so that path's scale does not drift as the network gets
deeper. Norm first, add second. If you added the norm *after* the residual
add, you would break the identity path that made it work.

## The C port

`rmsnorm_fwd` in the C tab is 38 lines against Python's 4, and every one of them
is about the same mathematics:

```c
/* NEON sum of squares for N_EMBD=16 */
/* Fast inverse sqrt via NEON + Newton refinement */
```

Because `n_embd = 16` divides evenly by 4, the sum of squares is four
`float32x4` accumulations. And rather than calling `1.0f/sqrtf(x)` (two
functions, two roundings), it uses a hardware reciprocal-square-root estimate
followed by two Newton–Raphson steps, which converges to full float32 precision
in a handful of instructions.

The reason to care: **`rmsnorm` is on the critical path of every token, and it is
not the interesting part of the model.** Moving the uninteresting part into
vector instructions is exactly the trade this file is making. `rmInvSqrt` is not
mathematics, it is bookkeeping — and moving bookkeeping out of the way is how the
interesting part becomes readable.
