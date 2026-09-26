---
id: inference-sampling
title: Inference and sampling
chapter: inference
order: 1
difficulty: 2
summary: "The same gpt() call, run autoregressively, feeding each sample back in as the next token. One extra number — temperature — divides the logits before the softmax, and that single division is the difference between memorising and inventing."
math: |
  p(x) = \mathrm{softmax}\!\left(\frac{\mathrm{logits}(x)}{T}\right), \qquad T \in (0, 1]
shapes:
  - { name: sample, shape: "list[str], grows to block_size" }
  - { name: temperature, shape: "float = 0.5" }
  - { name: probs, shape: "[27]" }
anchors:
  python: { selector: "section:Inference:" }
  c: { selector: "def:forward_inference" }
related: [softmax, gpt-forward, training-loop, lm-head]
prereqs: [gpt-forward, training-loop]
---

```python
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

## The only new idea

The training loop calls `gpt` once per position, with the *real* next character.
This loop calls it once per position, with the *model's own* previous output.
Everything else — `gpt`, the KV cache, `softmax` — is identical.

That is autoregressive generation, and the reason it terminates is a two-line
detail:

```python
if token_id == BOS:
    break
```

The model learned, during training, that `BOS` follows the last character of
every name. Sampling `BOS` is therefore the model's way of saying "I'm done", and
the loop honours it. This is the payoff of the trailing `BOS` in
[the tokenizer](../concepts/tokenizer.md): the stop condition is not special-cased
machinery, it is a token the model was trained to emit.

## Temperature, the one knob

```python
probs = softmax([l / temperature for l in logits])
```

`T = 0.5` sharpens the distribution. Dividing logits by `T < 1` multiplies every
logit gap by 2, which makes the largest logit relatively more dominant. `T = 1`
is the plain model distribution. `T > 1` flattens it, and `T -> 0` becomes
`argmax`.

The implementation detail worth noticing: **the division happens outside
softmax, on the logits.** `softmax` itself has no notion of temperature at all —
it is the same four lines as everywhere else in the file, and it will be handed
temperature-scaled logits in exactly one place. This is the standard way to do
it, and the alternative (adding a `temperature` parameter to `softmax`) would
push a sampling concern into a layer that does not need to know about it.

The samples at `T = 0.5`, straight from the reference:

```output
kamon   ann    karai  jaire  vialan  karia  yeran  anna   areli  kaina
konna   keylen liole  alerin earan   lenne  kana   lara   alela  anton
```

Real names and invented ones, in roughly equal measure. `anna` and `kamon` are
in the dataset. `ann`, `jaire`, `yeran` are not. Lower `T` to 0.2 and you get
mostly real names back; raise it to 1.5 and you get `qzx` and `wfj`. The
reference's comment says the range is `(0, 1]` — and the upper end is where it
stops being a name generator at all.

## `random.choices`

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

Weighted sampling, not argmax. `argmax` would return the same name every single
time — the model's single most confident path — and the 20 samples would be
identical. This is the whole reason the output looks varied.

`[p.data for p in probs]` unwraps the `Value` objects back to floats. Sampling
needs a real number, and it is the one place in this file where the autograd
wrapper has to be taken off before use.

## The KV cache earns its keep here

`keys, values` are reset once per *sample* and then grow across 16 positions. If
they were reset per position too, the model could not see what it had already
generated, and every sample would be 16 independent characters. This is the
contrast with the training loop that makes the four-argument `gpt()` signature
click: the same mechanism, same code, but here it is load-bearing.

## What a real implementation adds

- **A stop condition beyond `BOS`** — newline, a max length, an end-of-document
  token from a proper tokenizer.
- **Top-k or top-p filtering** before sampling. Mentioned in
  [the tokenizer](../concepts/tokenizer.md) as the standard alternative to
  temperature: it forbids the tail outright instead of just making it unlikely.
  Greedy decoding plus top-k is a very strong default.
- **Batching across samples.** 20 samples one at a time is 20x more
  `gpt` calls than needed.
- **A real cache.** Here `keys[0]` is a Python list of lists of `Value` objects,
  rebuilt per sample. A production cache is a contiguous preallocated tensor that
  survives across steps and batches. This is the single biggest difference
  between this file and a real serving stack, and it is *entirely* the same
  algorithm.

:::callout "Try it, in this order"
1. `temperature = 0.2`, rerun. Names get more real. Nothing else changes.
2. Replace `random.choices(...)` with `[max(range(vocab_size), key=lambda i: probs[i].data)]`. All 20 samples become identical. That is greedy decoding, and it is why sampling exists.
3. `temperature = 0.0`. You get a division by zero. `T -> 0` is `argmax`, but you have to implement that as a branch, which is a nice illustration that "0 is the low-temperature limit" is a limit and not a value.
:::
