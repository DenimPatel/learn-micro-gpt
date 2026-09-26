---
id: adam
title: Adam
chapter: training
order: 2
difficulty: 2
summary: "Adam keeps two running averages per parameter — the mean of recent gradients and the mean of recent squared gradients — and takes a step whose size is the ratio. The result is an update of roughly the learning rate, per parameter, regardless of gradient magnitude."
math: |
  m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2
shapes:
  - { name: m, shape: "[4192], first moment" }
  - { name: v, shape: "[4192], second moment" }
  - { name: beta1, shape: "0.85" }
  - { name: beta2, shape: "0.99" }
anchors:
  python: { selector: "comment:Adam optimizer update offset:[0, 8]" }
  c: { selector: "def:adam_update" }
related: [cross-entropy-loss, autograd-backward, params-init, training-loop]
prereqs: [cross-entropy-loss, autograd-backward]
---

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
...
    lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

Eight lines, and the whole reason training 4,192 parameters with one document at
a time works at all.

## The intuition

Plain gradient descent steps every parameter by the same *absolute* amount. A
parameter with a consistently large gradient would keep moving far, and one with
a consistently small gradient would barely move — even if the ratio between them
was meaningful.

Adam divides the gradient by an estimate of its own scale:

```
update = m_hat / (sqrt(v_hat) + eps)
```

`m_hat` is the gradient's recent direction; `sqrt(v_hat)` is the recent typical
magnitude. Their ratio is dimensionless and *self-normalising*, so every
parameter moves by about `lr_t` per step. That is the entire trick, and it is
why Adam tolerates wildly different gradient scales across a network without any
per-layer tuning.

## The two corrections that are easy to miss

```python
m_hat = m[i] / (1 - beta1 ** (step + 1))
v_hat = v[i] / (1 - beta2 ** (step + 1))
```

Both `m` and `v` start at zero, so on step 0 they are biased towards it: `m[0]`
is only `(1-beta1) * g`, a biased estimate of the true mean. Dividing by
`1 - beta^t` is the standard bias correction, and it is what makes the *first*
step the same size as every other step. Without it, training would spend its
first few dozen steps taking vanishingly small steps — a bug that looks exactly
like "the learning rate is too low".

`1 - beta1**0` is exactly 0, and `m[0]/0` would divide by zero. Because the
loop uses `step + 1`, the exponent starts at 1, and the denominator is never 0.
The `+ 1` is load-bearing.

## The betas are unusual, and deliberately so

`beta1 = 0.85` (not the usual 0.9) and `beta2 = 0.99` (not 0.999). The effective
window of `v` is roughly `1/(1-beta2)` = 100 steps, not 1,000. Over a 1,000-step
run that is a tenth of the whole training.

Karpathy's choice, and it is not obviously better or worse than the defaults —
it is tuned for *this* run length. It is also a good illustration that these are
hyperparameters like any other, and that the "standard" values are themselves a
tuning decision someone made for a different run.

## `eps_adam` and the order of operations

```python
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

`eps` goes *outside* the square root, added to `sqrt(v_hat)`, which is the
correct placement for a variance floor. `sqrt(v_hat + eps)` would bias the
denominator upward and slow every step slightly.

It exists for one specific failure: if a parameter's gradient is exactly zero for
a long time, `v_hat` approaches 0, and `m_hat / 0` is infinity. With
`eps = 1e-8` the denominator bottoms out at `1e-8` and the step is bounded. The
C port is explicit about the same hazard in a comment where it moves the epsilon
around for a fast reciprocal-sqrt:

```c
/* Add epsilon BEFORE sqrt to avoid 0/NaN */
```

## The learning rate schedule

```python
lr_t = learning_rate * (1 - step / num_steps)
```

Linear decay to zero over the run. At step 999, `lr_t` is `1e-5` — effectively
frozen. The effect is a deliberate annealing: explore early with big steps, then
settle into a minimum rather than bouncing around it.

`0.01` is a large learning rate for a 4,192-parameter model. It works because
there is one document per step and the whole model fits comfortably in memory,
so a big step costs nothing. The same `0.01` on a 175M-parameter model with batch
size 512 would diverge.

## The line that looks redundant

```python
p.grad = 0
```

It is not redundant. `backward()` *accumulates* into `.grad` with `+=`, and
nothing else ever resets it. Delete this line and parameter 0 accumulates every
gradient it has ever seen, forever. The loss curve will still fall, initially —
it is a small, plausible-looking bug, which is what makes it worth the comment.

:::callout "What the C port adds"
`adam_update` in the C tab is 58 lines against Python's 8. It replaces
`v_hat ** 0.5` with a NEON reciprocal-square-root and Newton refinement, hoists
the `1/(1-beta^t)` corrections out of the parameter loop into two scalars, and
then updates 4,192 float32s in tight 128-byte-aligned loops. The mathematics is
identical, to the last bit of the formula. The C file's own comments note the
fused form: `p -= lr * (m*inv_b1c) / (sqrt(v*inv_b2c + eps))`. It is
reciprocal-multiply rather than divide, which is the same operation with one
fewer rounding step.
:::
