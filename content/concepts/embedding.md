---
id: embedding
title: Token and position embeddings
chapter: architecture
order: 4
difficulty: 1
summary: "A token id indexes a learned vector, a position id indexes another learned vector, and the two are added. That sum is the model's entire representation of a token at that position, before anything else happens."
math: |
  x = \mathrm{wte}[t] + \mathrm{wpe}[p], \qquad t \in [0, V),\; p \in [0, T)
shapes:
  - { name: tok_emb, shape: "[16] = wte[token_id]" }
  - { name: pos_emb, shape: "[16] = wpe[pos_id]" }
  - { name: x, shape: "[16]" }
anchors:
  python: { selector: "def:gpt offset:[1, -32]" }
  c: { selector: "def:forward_pos offset:[3, -76]" }
related: [params-init, multi-head-attention, gpt-forward]
prereqs: [params-init]
---

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x)
```

Four lines, and this is where the token becomes a vector the rest of the model
can work with.

## Addition, not concatenation

The obvious alternative is `[tok_emb, pos_emb]` — a 32-wide vector, or a
32-wide parameter table. Addition keeps the width at `n_embd` for free, so every
downstream matrix keeps the same shape, and the two sources of information are
forced into the same space from the start.

The cost is that the model cannot afterwards recover "which part of this vector
was the character and which was the position." It has to learn a superposition.
For a model this small that is a real constraint, and it is the reason `wte` and
`wpe` are separate tables rather than one table indexed by `token_id * T +
pos_id`. Neither is obviously better; addition is simply smaller.

## Position is a learned absolute index

`wpe` has 16 rows and `pos_id` counts from 0. Position 5 is position 5, always —
there is no notion of distance, and no relative offset. So:

- The model can learn that position 0 tends to start a name, because that is
  where it always is.
- It cannot learn "the vowel two positions back" without attending to both
  positions separately. That is what attention is for.

This is the original Transformer design, and it is the reason context length is
baked in as a parameter count. Extending it is a research project; making it
relative (ALiBi, RoPE) is a different research project. Neither is in scope for
199 lines, and both are worth reading about once you have this working.

## rmsnorm immediately after

```python
x = rmsnorm(x)
```

Not redundant, and the reason is the residual connections inside the layer. The
block's input is added to its output, so the magnitude of `x` propagates
additively through the layer. Normalising once, on the way in, fixes the scale
that everything downstream inherits. Karpathy's newer revision of the file adds
exactly this comment: `# note: not redundant due to backward pass via the residual
connection`.

## One position at a time

Look at the signature: `gpt(token_id, pos_id, keys, values)`. It processes a
*single* position. The earlier positions are not re-computed — their keys and
values arrive in the `keys` and `values` lists, which the training loop resets
at the start of each document. That is the KV cache, written the long way.

For position 3 of `"[BOS] m a r y [BOS]"`, the forward pass runs 4 times, and by
the 4th call the `keys[0]` list holds 4 entries. The attention loop then has 4
keys to attend over instead of 1. Everything about the training loop's
`for pos_id in range(n)` follows from this signature — and so does its
embarrassing slowness, because each call re-walks the whole document rather than
processing the document as a batch.

:::callout "The thing to try"
Change `block_size` from 16 to 32 and add 16 rows to `wpe`. The model trains the
same way, on longer names. Now change it to 8 and watch the loss get *worse* —
you are truncating most names. `block_size` is a capacity decision disguised as
a hyperparameter, and this dataset makes that visible within a hundred steps.
:::
