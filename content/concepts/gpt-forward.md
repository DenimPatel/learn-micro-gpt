---
id: gpt-forward
title: The forward pass, end to end
chapter: architecture
order: 9
difficulty: 2
summary: "gpt() is a stateless function from one token id at one position to a vector of logits over the vocabulary. Given keys and values from earlier positions, it is the entire model. Given nothing else, it is 37 lines."
math: |
  \mathrm{gpt}(t, p, K, V) = W_{\mathrm{lm}}\Big(x + \mathrm{MLP}\big(\mathrm{RMSNorm}(x + \mathrm{Attn}(\mathrm{RMSNorm}(x)))\big)\Big)
shapes:
  - { name: keys[li], shape: "list of [16], one per position <= pos_id" }
  - { name: values[li], shape: "list of [16], same length" }
  - { name: return, shape: "[27] logits" }
anchors:
  python: { selector: "def:gpt" }
  c: { selector: "def:forward_pos" }
related: [attention-block, lm-head, embedding, cross-entropy-loss]
prereqs: [embedding, attention-block, mlp-block, lm-head]
---

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        ...

    logits = linear(x, state_dict['lm_head'])
    return logits
```

Read the signature first, because it explains almost everything else.

## Stateless, except for those two arguments

The file's own comment: *"a stateless function mapping token sequence and
parameters to logits over what comes next."* The parameters come from the
module-level `state_dict`, which is the only state. `gpt` mutates nothing.

The exception is `keys[li].append(k)` and `values[li].append(v)`, and those two
lines are why the signature has four arguments instead of two. Attention at
position 3 needs the keys and values from positions 0, 1, and 2. Rather than
recompute them, `gpt` appends its own to the caller's list and reads the rest
back.

That is the **KV cache**, spelled out in the least efficient way possible:

| | this file | a real implementation |
| --- | --- | --- |
| one call | one position | the whole sequence |
| K/V for earlier positions | Python list of `Value` | contiguous preallocated tensor |
| lifetime | rebuilt and thrown away every step | kept across steps |
| cost of position 3 | walk positions 0..3 again | attend to cached K/V |

During training the cache buys nothing — the whole graph is discarded at the end
of each step anyway. During inference it is the single most important
optimisation there is, because it turns generation from O(n²) into O(n). The
same 37 lines serve both, and the mechanism is the same four-argument signature.

:::callout "The cost of this design, measured"
The reference trains 1,000 steps in about 108 seconds on an M-series Mac, and the
C port does the same work in about 0.04 seconds. None of that 2,700x is the
arithmetic. It is: not rebuilding a Python graph 1,000 times, and not walking the
document one position at a time. See [linear](../concepts/linear.md) for the
split and `docs/BENCHMARKS.md` for the methodology.
:::

## The call sequence in training

```python
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
```

The cache is reset once per document, then `gpt` is called `n` times, and it
appends one key and one value each time. By the final call, `keys[0]` holds `n`
entries and the last head is attending over all of them.

Note `tokens[pos_id], tokens[pos_id + 1]` — input and target are the *same* list
offset by one. Every position is trained simultaneously, in the sense that a
single pass over the document supervises all `n` next-character predictions. It
just happens to happen one call at a time.

## What `n_layer = 1` hides

The `for li in range(n_layer)` loop runs exactly once. With `n_layer = 4` the
cache is a list of four lists, and each layer has its own `attn_wq`. The only
structural thing that changes is that `x` now flows through four blocks instead
of one, and that the keys/values for layer 1 are *not* the same objects as for
layer 0.

## The whole model, in order

If you have read this far, you have read the whole model. The 199 lines break down
as:

| lines | what |
| --- | --- |
| 13–20 | data |
| 22–26 | tokenizer |
| 28–71 | autograd |
| 73–89 | parameters |
| 93–143 | architecture — everything above |
| 145–183 | loss, Adam, training loop |
| 185–199 | inference |

There is nothing else. No dropout, no layer norm with learnable gains, no biases,
no weight tying, no batching, no checkpointing, no learning-rate schedule beyond
a linear decay, no early stopping. Every one of those is a real technique, and
every one of them is a decision that *this file declines to make*. Knowing what
was left out is as much of the lesson as what is here.
