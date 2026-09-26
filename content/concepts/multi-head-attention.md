---
id: multi-head-attention
title: Multi-head attention
chapter: architecture
order: 5
difficulty: 3
summary: "Each head slices q, k, and v into a narrow slice, scores how well the query matches every key so far, softmaxes those scores into weights, and takes a weighted average of the values. Four heads of width 4 instead of one head of width 16."
math: |
  \mathrm{attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
shapes:
  - { name: q_h, shape: "[4] = q[h*4 : h*4+4]" }
  - { name: k_h, shape: "[4] per position t" }
  - { name: v_h, shape: "[4] per position t" }
  - { name: attn_logits, shape: "[T]" }
  - { name: attn_weights, shape: "[T], sums to 1" }
  - { name: head_out, shape: "[4]" }
  - { name: x_attn, shape: "[16] after 4 heads" }
anchors:
  python: { selector: "def:gpt offset:[15, -12]" }
  c: { selector: "def:forward_pos offset:[28, -39]" }
related: [attention-block, softmax, linear, embedding]
prereqs: [linear, embedding]
---

This is the concept the whole file exists to demonstrate. Twelve lines of Python,
and it is the reason the model can compose at all.

## The loop, unrolled

```python
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
    attn_weights = softmax(attn_logits)
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
    x_attn.extend(head_out)
```

Four steps, repeated four times.

## 1. Slicing

`hs = h * head_dim` is the only arithmetic separating the heads. With
`n_embd = 16` and `n_head = 4`, head 0 takes `q[0:4]`, head 1 takes `q[4:8]`, and
so on. The full-width `q` is **never used directly** — it exists only so that
`attn_wq` can be a single `[16, 16]` matrix. Slicing afterwards is equivalent to
four separate `[4, 16]` projections, and it is cheaper to write and to
implement, at the cost of a slightly more confusing mental model.

## 2. Scoring

```python
attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
```

For every position `t` seen so far, the dot product `q_h · k_h[t]`. Position
`t = pos_id` always scores highest, because `k_h[pos_id]` came from the same
projection of the same `x` as `q_h` — that is the residual stream talking to
itself. Everything else is a genuine comparison against earlier tokens.

**The division by `head_dim**0.5` is not optional.** A dot product of two
independent width-`d` vectors grows like `sqrt(d)`: with `d = 4` the raw scores
land around ±0.6, but at `d = 128` they would be around ±11, and softmax over
logits that spread that far is nearly a hard argmax. One head would receive
essentially all the weight and the other 127 dimensions would get no gradient.
Dividing by `sqrt(d)` makes the score distribution independent of `d`. The C
port precomputes it as a macro, because it is a constant:

```c
#define INV_SQRT_HD 0.5f /* 1/sqrt(HEAD_DIM=4) */
```

## 3. Softmax into weights

```python
attn_weights = softmax(attn_logits)
```

A distribution over positions. The causal mask is **absent** — and it does not
need to be, because `keys[li]` only ever contains positions `<= pos_id`. The
mask is structural, built into the signature. This is the single nicest idea in
the file: causality falls out of the loop structure instead of costing a
`masked_fill`.

## 4. The weighted average

```python
head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
```

Each output dimension is a convex combination of the value vectors, weighted by
how well that position matched the query. Because `attn_weights` sums to 1, this
is an *average*, not a sum — which is why the output magnitude does not grow
with sequence length.

## Why several heads

`n_head = 4` with `head_dim = 4` is not a capacity trick, it is a
*representational* one. Each head performs a different query function, so
different heads can specialise: one may learn to look one character back, another
three back, another at the `BOS` that marks the end. Concatenating them
(`x_attn.extend(head_out)`) gives one `[16]` vector containing four different
views.

The alternative — one head of width 16 — has the same parameter count and the
same FLOPs. What it cannot do is form a single soft attention over 16 dimensions
of one query. Heads make attention itself a multi-dimensional operation.

## Reading the traces

This is the one concept where a recorded trace beats any amount of prose, so
the Atlas inlines one. For step 0, position 3, head 0 you get a 4x4 matrix of
weights. Two things to look for:

- The **diagonal** entry (t = 3) is the largest, because of the self-match above.
- The weights **sum to 1 along rows**, and change a lot from step 0 to step 500,
  because the projection matrices are being trained and the queries are moving.

:::callout "Set n_head to 1 and watch"
`n_head = 1` makes `head_dim = 16` and the loss curve gets noticeably worse. Set
it to 16 and `head_dim = 1`, and attention collapses to "pick the single
dimension that matches best". The sweet spot is not derivable from first
principles — it is one of the few genuinely empirical numbers in the file, and it
is worth finding yourself rather than taking from us.
:::
