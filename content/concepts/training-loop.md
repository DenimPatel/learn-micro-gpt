---
id: training-loop
title: The training loop
chapter: training
order: 3
difficulty: 2
summary: "One document, tokenized and framed with BOS, run through the model one position at a time, averaged into a loss, backpropagated, and updated with Adam. 1,000 times. The loop is where the loss becomes a number, and it is the only place in the file where anything is."
math: |
  \theta_{t+1} = \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, \qquad \eta_t = 0.01\big(1 - t/T\big)
shapes:
  - { name: tokens, shape: "[n+1], BOS + chars + BOS" }
  - { name: n, shape: "min(block_size, len(tokens) - 1)" }
  - { name: losses, shape: "[n]" }
anchors:
  python: { selector: "section:Repeat in sequence" }
  c: { selector: "def:main offset:[12, -48]" }
related: [adam, cross-entropy-loss, gpt-forward, data-loading]
prereqs: [cross-entropy-loss, adam]
---

```python
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        ...
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

## Five steps, in order

1. **Get a document.** `docs[step % len(docs)]` walks the shuffled list. With
   32,033 documents and 1,000 steps, this touches the first 1,000 of the shuffled
   list and never sees the other 31,033. That is fine for a demo and is *not*
   fine as a claim about the dataset.
2. **Frame it.** `[BOS] + chars + [BOS]`, then `n = min(block_size, len(tokens)-1)`.
   The `- 1` is because the last token is a target, not an input.
3. **Forward, position by position.** Reset the KV cache, then call `gpt` `n`
   times. See [the forward pass](../concepts/gpt-forward.md) for why this is one
   call per position rather than one call per document.
4. **Backward.** `loss.backward()` — one line that walks the entire graph. See
   [walking the graph backwards](../concepts/autograd-backward.md).
5. **Update.** Adam, with the linear decay, then reset `p.grad`.

## The graph is rebuilt from scratch, every step

Nothing is cached between iterations. `keys` and `values` are re-created, `gpt`
constructs new `Value` objects, the computation graph is built, walked, and
discarded. 1,000 times.

That is a design choice, and at `n_layer = 1` it is the right one — a cache would
be premature. It is also the dominant cost. The C port keeps
`saved_x_embed[BLOCK_SIZE]`, `saved_q[BLOCK_SIZE]`, `saved_attn_weights[BLOCK_SIZE][N_HEAD][BLOCK_SIZE]`
and dozens more as preallocated static arrays precisely so that it does not have
to rebuild them, and it processes *all positions* in one backward pass rather
than one at a time. Same algorithm; completely different bookkeeping.

## What this loop does not do

Every one of these is standard, and every one is absent:

- **No batching.** `n` here is *positions in one document*, not examples in a
  batch. The `1/n` in the loss is averaging over sequence positions.
- **No gradient accumulation, no clipping.**
- **No validation set.** There is no held-out data, so there is no way to tell
  overfitting from learning. Set `n_layer = 8` and the model gets worse on
  names it has seen; nothing in this loop would tell you.
- **No checkpointing, no resume.** If it dies at step 900, it starts again.
- **No early stopping, no LR schedule beyond the linear decay.**

## The output, and how to read it

`step 1000 / 1000 | loss 2.6497` on the reference. That number is *not* worse
than step 200's 2.3097; it is a different, harder name. See
[cross-entropy loss](../concepts/cross-entropy-loss.md) for the full measurement.

What is unambiguous is the trend. Measured on the reference over 1,000 steps:

| window | mean loss |
| --- | --- |
| steps 1–50 | 2.852 |
| steps 151–200 | 2.532 |
| steps 951–1000 | 2.323 |

And the whole curve, raw and smoothed, which is the only honest way to see
a trend in something this noisy:

:::trace kind=loss-curve

A drop of 0.53 nats — a factor of `exp(0.53) = 1.70` in likelihood, achieved by
a model with 4,192 parameters, on 1,000 documents, in 199 lines. The final
samples are `kamon`, `ann`, `karai`, `ana`. That is a real language model. It is
just a very small one, trained on very little data, by a loop that fits in a
screenshot.

:::callout "Change one number and rerun"
`num_steps = 10000` needs no other edit. The samples stop being real names
(`ann`, `kamon`) and start being name-*like* — plausible letter combinations
that are not in the dataset. That transition from memorisation to generation, in
a single integer, is the most direct demonstration of what "overfitting" means
that this repository contains. It also takes about 18 minutes.
:::
