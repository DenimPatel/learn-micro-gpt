---
id: cross-entropy-loss
title: Cross-entropy loss
chapter: training
order: 1
difficulty: 2
summary: "The loss is the negative log of the probability the model assigned to the character that actually came next, averaged over the document. It is the definition of surprise, and it is why the number the script prints goes down."
math: |
  \mathcal{L} = -\frac{1}{n}\sum_{i=0}^{n-1} \log p_\theta\big(x_{i+1} \mid x_{\le i}\big)
shapes:
  - { name: probs, shape: "[27]" }
  - { name: loss_t, shape: "float" }
  - { name: loss, shape: "float, mean of n values" }
anchors:
  python: { selector: "def:gpt offset:[55, 25]" }
  c: { selector: "def:forward_pos offset:[85, -1]" }
related: [softmax, lm-head, adam, training-loop]
prereqs: [softmax, lm-head]
---

```python
for pos_id in range(n):
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_t = -probs[target_id].log()
    losses.append(loss_t)
loss = (1 / n) * sum(losses)
```

## Surprise, per character

`-probs[target_id].log()` is `-log p`, the negative log-likelihood. Read it as
*how surprised were you that the next character was this one*.

| the model's probability | loss |
| --- | --- |
| 1.0 (certain) | 0.0 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 1/27 ≈ 0.037 (guessing) | 3.30 |
| 0.0 | infinity |

The units are **nats**, and the `log` is natural. This is what makes the scale
interpretable: a loss of 3.30 literally means "as surprised as a uniform guess
over 27 characters", and a loss of 2.0 means "about 7x more likely than
uniform on the character you got". Because the log is natural, the differences
are the log-ratios, so the loss is a sum of surprisal — which is why it can be
averaged over positions at all.

## Why `-log p` and not something else

Three reasons, and the third is the important one:

1. **It is a proper scoring rule.** Assigning probability 1 to the right answer
   and 0 elsewhere gives loss 0. Any hedging is penalised. A model cannot win by
   being confidently wrong in a way that happens not to be scored.
2. **It is the negative log-likelihood**, so minimising it is maximum likelihood
   estimation. No derivation required.
3. **Its gradient is trivial.** `d(-log p)/d(logits) = probs - onehot(target)`.
   That is the entire backward pass for this layer, and the C port states it as
   its own comment: `/* dL/d(logits) = (probs - one_hot(target)) / n */`. This
   is why the loss is applied to logits and not to probabilities — the softmax
   Jacobian folds into this one line.

## The average, and why it is a mean not a sum

`loss = (1 / n) * sum(losses)`. Summing would make the loss grow with document
length, so a long name would dominate the gradient and a short one would be
invisible. The `1/n` makes the scale independent of `n`, which is what lets the
learning rate mean the same thing for every document.

## The number is noisy, and that is not a bug

`loss_t` is computed for one position of one document, and the printed `loss` is
the mean over the positions of *that one document*. The training loop then moves
to a completely different document. So the printed value is a single sample from
a distribution whose spread is about ±0.4.

Measured on the reference, over 1,000 steps:

| statistic | value |
| --- | --- |
| mean | 2.452 |
| standard deviation | 0.392 |
| minimum | 1.578 (step 537) |
| maximum | 3.907 |
| **step-to-step moves that increase** | **500 of 999** |

Half the steps go *up*. This is the first thing that breaks a naive expectation
about training, and it is entirely correct behaviour: `"xander"` is a hard name
and `"ana"` is an easy one, and the model is being asked to do both in
alternating steps.

:::callout "A gate that would have failed the reference"
An early draft of this repository's parity specification required the loss to be
*strictly decreasing*, and within ±10% of the reference at single steps 50 and
200. The reference itself violates the first condition 500 times, and its
per-step standard deviation is 16% of its mean — so a ±10% band on a raw
single-step value is inside the noise. Both were wrong, and both were wrong in a
way that would only have been discovered by running them. The gate now requires
a *downward trend* in windowed means, and a ±10% band on a **smoothed** loss,
where the smoothing removes the document-to-document variance. The evidence and
the arithmetic are in `docs/BENCHMARKS.md`.
:::

## The reading you should take away

`step 1000 | loss 2.6497` does not mean the model got worse in the last step. It
means the model scored 2.65 on that particular name. The last-step number is the
least informative thing in the file, and it is the only one printed. Real
implementations print a running mean, and this repository's tooling compares
smoothed curves for the same reason.
