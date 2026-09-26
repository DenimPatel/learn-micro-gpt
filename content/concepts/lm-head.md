---
id: lm-head
title: The language model head
chapter: architecture
order: 8
difficulty: 2
summary: "One final matrix multiply maps the 16-dimensional residual stream to 27 logits, one per token in the vocabulary. It is the only output the model has, and it is the same operation as every other linear layer."
math: |
  \mathrm{logits} = W_{\mathrm{lm}} x, \qquad W_{\mathrm{lm}} \in \mathbb{R}^{V \times d}
shapes:
  - { name: x, shape: "[16]" }
  - { name: lm_head, shape: "[27, 16]" }
  - { name: logits, shape: "[27]" }
anchors:
  python: { selector: "def:gpt offset:[35, 0]" }
  c: { selector: "def:forward_pos offset:[83, -1]" }
related: [linear, softmax, cross-entropy-loss, params-init]
prereqs: [linear, softmax]
---

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

Two lines, and they are the end of the forward pass. `linear` is the same
function used seven other times in this model; the only difference is the shape
of the weight matrix, `[27, 16]` rather than `[16, 16]`.

## Tied or not

Note that `wte` is `[27, 16]` and `lm_head` is *also* `[27, 16]` — same shape,
different table. They are not the same parameters. That is a design choice worth
naming, because the obvious alternative is to tie them: use `wte` as the output
projection too, so that "the vector for token 3" and "how much I like token 3"
are the same numbers.

Tying halves the parameter count and is what small models often do. Untying gives
the input and output embeddings room to specialise independently, and it is what
this file does. Either is defensible; what matters is that you know which one
you have, because untying them and then accidentally sharing a reference is a
very quiet bug.

## Logits, not probabilities

The function returns raw scores. It does not softmax. That is deliberate: the
softmax is applied *outside*, at two different call sites with different
purposes, and applying it inside would mean computing it three times — and
throwing away information at the point where the loss wants it.

The loss is `-probs[target].log()`, and it is much better conditioned to
receive logits. Every production framework makes the same choice.

## The vocabulary is the output space

`27` is `vocab_size`, and it is the width of the model's only output. Two
consequences:

- **Every prediction is a full distribution over 27 characters.** No beam, no
  sampling trick, no structured output. The model's entire hypothesis space at
  each step is 27 numbers.
- **The output layer is a big share of the parameters.** 432 of 4,192, about
  10%. At this scale that is a lot; at `d_model = 4096` and `V = 50,000` it
  would be 205 million parameters in one matrix, and a disproportionate amount of
  memory traffic. It is why vocab-sharding is a real technique.

## The C port's tiny addition

```c
linear_fwd(x, lm_head, saved_logits[pos_id], vocab_size, N_EMBD);
softmax_fwd(saved_logits[pos_id], saved_probs[pos_id], vocab_size);
```

The C version computes the softmax *immediately*, storing both `saved_logits` and
`saved_probs`. The Python version returns logits and lets the caller softmax.
Both are correct; the C one stores twice as much because it wants the
probabilities again in the backward pass, where the gradient of cross-entropy
with respect to the logits is simply `probs - onehot(target)`:

```c
/* dL/d(logits) = (probs - one_hot(target)) / n */
```

That is the whole backward pass for this layer, in one comment. The next
concept picks it up.
