---
id: linear
title: Linear (a matrix multiply with a list comprehension)
chapter: architecture
order: 1
difficulty: 1
summary: "linear(x, w) is two lines: for every row of the weight matrix, sum the elementwise products with the input. It is the only place a weight matrix is used, and it is where most of the arithmetic in the model happens."
math: |
  y = W x, \qquad y_i = \sum_{j} W_{ij}\, x_j
shapes:
  - { name: x, shape: "[16]" }
  - { name: w, shape: "[16, 16]" }
  - { name: y, shape: "[16]" }
anchors:
  python: { selector: "def:linear" }
  c: { selector: "def:linear_fwd" }
related: [embedding, softmax, rmsnorm, gpt-forward]
prereqs: [params-init]
---

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

Two nested generator expressions. Read the inner one first: for a single output
row `wo`, pair up its entries with `x` and sum the products. The outer one does
that once per row. The result is `[16]` in, `[16]` out.

There is no bias term. The comment at the top of the architecture section calls
this out: "no biases, GeLU -> ReLU". Both are deliberate simplifications, and
both are things you would add back the moment you wanted this to be a real
model rather than a readable one.

## Where it is used

Seven call sites, and they are all of one of three shapes:

| call | input | weight | output |
| --- | --- | --- | --- |
| `q = linear(x, ...attn_wq)` | [16] | [16, 16] | [16] |
| `k = linear(x, ...attn_wk)` | [16] | [16, 16] | [16] |
| `v = linear(x, ...attn_wv)` | [16] | [16, 16] | [16] |
| `x = linear(x_attn, ...attn_wo)` | [16] | [16, 16] | [16] |
| `x = linear(x, ...mlp_fc1)` | [16] | [64, 16] | [64] |
| `x = linear(x, ...mlp_fc2)` | [64] | [16, 64] | [16] |
| `logits = linear(x, lm_head)` | [16] | [27, 16] | [27] |

That is the entire model. Everything else is normalisation, the attention loop,
and arithmetic on the results.

## Row-major, and why `wo` is the row

`for wo in w` iterates rows, and each row produces one output element. So `w` is
`[n_out, n_in]` — output rows, input columns. This is the same convention as the
`state_dict` shapes in [parameter init](../concepts/params-init.md)
(`mlp_fc1` is `[64, 16]`, not `[16, 64]`), and it is the transpose of the PyTorch
convention, which is `[n_in, n_out]`. Every C, Go, Rust, and TypeScript port in
this repository uses the same `[n_out, n_in]` layout, which is why their weight
arrays are the same length in the same order as the Python list-of-lists.

## What the C port adds

```c
/* Use Accelerate cblas for matrix-vector multiply: out = W * x */
```

`cblas_sgemv` does the identical sum, in hand-tuned BLAS, on a matrix that is
128-byte aligned so it can be loaded with four `float4`s at a time. On Apple
Silicon that is roughly two orders of magnitude faster than the Python loop, for
exactly the same floating-point operation.

This is the clearest illustration of the difference between this Python file and
"pure, dependency-free": the *algorithm* is two lines of `zip` and `sum`; the
*implementation* is a call into a library that has been optimised for thirty
years. On this machine — an M-series Mac — `make run-python` takes about 108
seconds for 1,000 steps and `make run-c` takes about 0.04, from the same
algorithm and the same dataset. The exact figures for a pinned runner are in
`benchmarks/results.json`; the methodology, and the reasons a number from your
laptop is not a leaderboard entry, are in `docs/BENCHMARKS.md`.

## The one subtlety

`sum(...)` starts at integer `0`, not at a float. The first addition is
`0 + w[0]*x[0]`, which is fine in Python because it promotes. It is worth
noting only because the same expression in a language with stricter typing —
or in a JIT that optimises the integer path — is a real bug, and this repository's
ports have all had to handle it explicitly.
