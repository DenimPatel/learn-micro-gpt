---
id: softmax
title: Softmax
chapter: architecture
order: 2
difficulty: 1
summary: "Softmax turns a vector of unnormalised scores into a probability distribution, by exponentiating, subtracting the max for stability, and dividing by the total. It is the only step where a list of 27 numbers becomes a distribution over 27 things."
math: |
  p_i = \frac{\exp(z_i - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)}
shapes:
  - { name: logits, shape: "[27]" }
  - { name: probs, shape: "[27]" }
  - { name: "sum(probs)", shape: "= 1.0" }
anchors:
  python: { selector: "def:softmax" }
  c: { selector: "def:softmax_fwd" }
related: [cross-entropy-loss, inference-sampling, linear]
prereqs: [linear]
---

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Four lines, and the middle one is the reason this file works at all.

## The max subtraction

`max_val` is not a numerical nicety, it is what makes the function usable.
`math.exp` overflows above about 709 and underflows to zero below about -746.
Logits from a freshly initialised network sit near zero, so a naive
`exp(z)/sum(exp(z))` would survive today — and then blow up the moment someone
raises the learning rate or trains for longer.

Subtracting the same constant from every logit leaves the result *exactly*
unchanged, because

```
exp(z_i - m) / sum(exp(z_j - m))  =  exp(z_i) / sum(exp(z_j))
```

for any `m`. It is the algebraically free trick: same function, guaranteed
finite inputs. Picking `m = max(z)` bounds every argument to `(-inf, 0]`, so
every exponential lands in `(0, 1]` and the total can never underflow to zero.

This is not an optimisation anyone would add later. It is a correctness fix, and
it is the sort of thing that separates a working softmax from one that works on
Tuesday.

## Two uses, and they are different

`softmax` is called in exactly two places, and the distinction matters:

- **In training**, on the raw logits from `lm_head`, to get the predicted
  distribution. Then `-probs[target_id].log()` is the loss. See
  [cross-entropy loss](../concepts/cross-entropy-loss.md).
- **In attention**, on the attention logits over `t` positions, to get the
  attention *weights*. Then those weights are used to average the values. See
  [multi-head attention](../concepts/multi-head-attention.md).

Same function, completely different job: in one case a distribution over the
vocabulary, in the other a distribution over positions.

## What it does not do

No temperature, no top-k, no masking. It is a plain softmax. Temperature appears
later, in [inference](../concepts/inference-sampling.md), and it is implemented
by dividing the logits *before* calling this function — a nice illustration that
softmax itself needs no notion of temperature at all.

:::callout "The invariant to check"
`sum(softmax(x))` is 1.0, always, to floating-point precision. It is worth
adding that assertion the first time you run this — the C port's
finite-difference tests do exactly that for its `softmax_fwd`, and it is the
cheapest possible check that a normalisation function is normalising. If the sum
is 1.0 and the output has a `NaN` in it, you have found a divide-by-zero that
`max_val` was supposed to prevent.
:::
