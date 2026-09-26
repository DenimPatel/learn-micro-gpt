---
id: attention-block
title: The attention block
chapter: architecture
order: 7
difficulty: 3
summary: "The whole transformer layer is two sub-blocks in sequence, each of which is the same three steps: save the input, normalise it, transform it, and add the saved input back. The layer is that pattern applied twice."
math: |
  x \leftarrow x + \mathrm{Attn}\big(\mathrm{RMSNorm}(x)\big), \qquad
  x \leftarrow x + \mathrm{MLP}\big(\mathrm{RMSNorm}(x)\big)
shapes:
  - { name: x_residual, shape: "[16], saved before the norm" }
  - { name: "q, k, v", shape: "[16] each" }
  - { name: x_attn, shape: "[16] after concatenating 4 heads" }
anchors:
  python: { selector: "def:gpt offset:[7, -10]" }
  c: { selector: "def:forward_pos offset:[16, -6]" }
related: [multi-head-attention, mlp-block, rmsnorm, gpt-forward]
prereqs: [multi-head-attention, mlp-block, rmsnorm]
---

```python
for li in range(n_layer):
    # 1) Multi-head attention block
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, state_dict[f'layer{li}.attn_wq'])
    k = linear(x, state_dict[f'layer{li}.attn_wk'])
    v = linear(x, state_dict[f'layer{li}.attn_wv'])
    keys[li].append(k)
    values[li].append(v)
    x_attn = []
    for h in range(n_head):
        ...
    x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
    x = [a + b for a, b in zip(x, x_residual)]
    # 2) MLP block
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
    x = [xi.relu() for xi in x]
    x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
    x = [a + b for a, b in zip(x, x_residual)]
```

## The pattern

Both sub-blocks are four steps, in the same order:

| step | attention | MLP |
| --- | --- | --- |
| 1 | `x_residual = x` | `x_residual = x` |
| 2 | `x = rmsnorm(x)` | `x = rmsnorm(x)` |
| 3 | Q/K/V, heads, `attn_wo` | `mlp_fc1`, ReLU, `mlp_fc2` |
| 4 | `x = x + x_residual` | `x = x + x_residual` |

This is a **pre-norm** transformer, and the ordering is the whole design. The
norm goes *before* the transform, not after, which keeps the residual path
unobstructed: the identity from `x_residual` to the output of step 4 is exact,
so gradients reach early layers through a matrix of ones plus whatever the block
does. Post-norm architectures — the original 2017 Transformer, and this file's
own ancestor — put the norm on the output and force gradients through the
transform itself, which is why they were hard to train past a few layers.

The consequence is a specific, testable claim: **each block is at most a
correction to the residual stream.** Set `attn_wo` and `mlp_fc2` to zero and the
layer becomes the identity. The model still trains. It is just `n_layer` times
shallower.

## Q, K, V: three projections, one input

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
```

Three separate weight matrices, applied to the *same* `x`. This is the step that
is most often described as hand-wavy, so it is worth being concrete about what
the three names mean:

- **Query** — what this position is looking for.
- **Key** — what this position offers to be matched against.
- **Value** — what this position actually contributes if it is matched.

Q and K are only ever used in the dot product that produces `attn_logits`. V is
never compared to anything. So the split is real: a position can be *relevant*
(high `q·k`) without being *informative* (its `v` is unremarkable), and the
model learns those independently, which is exactly the flexibility that lets
different heads specialise.

## The output projection is not optional

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

`x_attn` is the concatenation of 4 independent heads. Without `attn_wo`, each
head's output would occupy a disjoint 4-wide slice of the residual stream and
could never exchange information with the others. The projection mixes them.
It is also, along with `mlp_fc2`, the matrix that makes the block a *learned*
correction rather than a fixed one.

Note what the C port does with it — it stores the result as
`saved_x_attn_out[pos_id]` and keeps it, because the backward pass needs to
reconstruct `x_attn` from the saved attention weights and values rather than
recomputing the whole head loop. The C file's comment says so:

```c
/* We need x_attn — reconstruct it from saved attention weights and values */
```

Storing activations, or recomputing them from what is stored, is the central
engineering decision in every autodiff implementation, and it is the thing that
does not exist at all in the Python version — which rebuilds the entire graph
from scratch every step and throws it away.

## `n_layer = 1`

One iteration of this loop. That means the whole model is: embed, normalise, one
attention block, one MLP, unembed. There is no second layer to read, and no
loop-invariant subtlety to worry about. When you set `n_layer = 4` and the file
starts working on more than one document at a time — or once you batch — the
per-position KV cache threading through `keys[li]` is the first thing that needs
restructuring. That is a real cost of this design, and it is the main reason
production implementations do not do it this way.

:::callout "The one-line experiment"
Set `n_layer = 4` and rerun. It works — that is the payoff of naming the weights
`layer{i}.*` with a format string. The loss gets a little better and the run
gets a little slower. Then set `n_layer = 8` and watch it get *worse* again: at
`n_embd = 16` there are not enough parameters per layer to justify the depth,
and overfitting a 32,000-word vocabulary starts to bite. Depth is a trade, and
199 lines does not tell you where the optimum is. Finding that is the exercise.
:::
