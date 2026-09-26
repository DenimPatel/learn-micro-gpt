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
the Atlas inlines one. Here is step 0, position 3, head 0 — one weight per
position the model can currently see:

:::trace kind=attention step=0 pos=3 head=0

Four columns, and they come out `[0.231, 0.279, 0.220, 0.270]`. Nearly uniform,
and almost nothing special: that is an untrained model.

The same view at step 200:

:::trace kind=attention step=200 pos=3 head=0

Now it is `[0.394, 0.236, 0.197, 0.174]`. Structured, and leaning hard on
position 0.

**That the diagonal is the largest number is not true, and that is the
interesting part.** Measured over the recorded trace, for head 0 at every
position with more than one candidate:

| step | diagonal is the argmax | mean weight on the diagonal | mean weight on the best earlier position |
| --- | --- | --- | --- |
| 0 | 2 of 4 | 0.389 | 0.373 |
| 25 | 0 of 4 | 0.245 | 0.617 |
| 100 | 0 of 6 | 0.242 | 0.386 |
| 500 | 0 of 4 | 0.292 | 0.503 |

Training *reduces* the weight on the current position and *raises* the weight on
the best earlier one. By step 25 the current position is not the argmax at all.

That is the opposite of what most people's first mental model of attention
predicts, and it makes sense once you think about what the model is doing. The
self-match is a mathematical artefact of `k` and `q` coming from the same
projection of the same vector. It carries no information, so the network is free
to down-weight it — and it does, because the tokens that *precede* the current
one are the ones that predict the next.

Three things to check in any trace you look at:

- **Every row sums to 1**, to floating-point precision. The structural
  invariant.
- **The row has exactly `pos + 1` entries**, not `block_size`. That is the causal
  mask — implemented by the shape of the `keys` list, not by a mask value
  anywhere. The C port does the same thing with `num_keys = pos_id + 1`.
- **Position 0 always gives `[1.0]`.** With one position visible there is
  nothing to choose between, so the softmax of a single logit is 1. If your trace
  shows anything else at position 0, something is wrong before you look at
  anything else.

:::trace kind=loss-curve

And here is the loss curve those attention weights came from, with the
per-step noise that makes it unreadable unless you smooth it:

:::callout "Set n_head to 1 and watch"
`n_head = 1` makes `head_dim = 16` and the loss curve gets noticeably worse. Set
it to 16 and `head_dim = 1`, and attention collapses to "pick the single
dimension that matches best". The sweet spot is not derivable from first
principles — it is one of the few genuinely empirical numbers in the file, and it
is worth finding yourself rather than taking from us.
:::
