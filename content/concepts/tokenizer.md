---
id: tokenizer
title: The tokenizer
chapter: data
order: 2
difficulty: 1
summary: "Three lines turn arbitrary strings into integers and back again: the sorted set of every character in the dataset becomes the vocabulary, and one extra id is reserved for end-of-word."
math: |
  V = \mathrm{sort}\!\big(\mathrm{set}\big(\textstyle\bigoplus_{d \in \mathrm{docs}} d\big)\big), \qquad \mathrm{vocab\_size} = |V| + 1
shapes:
  - { name: uchars, shape: "[26]" }
  - { name: BOS, shape: "int = 26" }
  - { name: vocab_size, shape: "int = 27" }
anchors:
  python: { selector: "section:Let there be a Tokenizer" }
  c: { selector: "def:load_data offset:[27, -9]" }
related: [data-loading, cross-entropy-loss, inference-sampling]
prereqs: [data-loading]
---

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

That is the whole tokenizer. Every character that appears anywhere in the dataset
becomes a token id, ordered by character code so the mapping is deterministic
across machines and Python versions. On the names dataset that is 26 lowercase
letters, so `vocab_size` is 27.

## Why the +1

`BOS` is a token that is not a character. It means *beginning of sequence*, and
this file uses it for two things: it pads the start of a name, and it terminates
it. `"mary"` becomes:

```
[BOS] m a r y [BOS]
```

The trailing `BOS` is the whole trick, and it is worth pausing on. It turns a
closed-ended task into an open-ended one. During training, the model only ever
predicts the next character, but the sample ends with `BOS` — so the model
learns that `BOS` is a legitimate thing to emit, and therefore learns *where
names stop*. Without it, the model has never seen a document end, and at
inference time it will happily generate forever.

This is the cheapest possible document-boundary marker, and it costs one row in
the embedding table.

## The mapping, in both directions

Forward, encoding a string:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

Backward, decoding ids to a string:

```python
sample.append(uchars[token_id])
```

`uchars.index(ch)` is a linear scan, so encoding a character is O(26). Irrelevant
at this scale, and the kind of thing you replace with an array lookup the moment
it shows up in a profile. Notice there is no dictionary anywhere — the list *is*
the lookup table, and `index` is the inverse map.

## What the vocabulary size costs you

`vocab_size` is the width of the final output layer, and the width of the token
embedding table. It shows up in the parameter count:

```
27 * 16  (wte)      = 432
27 * 16  (lm_head)  = 432
16 * 16  (wpe)      = 256
```

4,192 parameters in total for the whole model. Character-level tokenisation is
what makes that possible. A word-level vocabulary on this dataset would be
~32,000 entries, and the embedding and head alone would dwarf everything else —
for a model with no capacity to learn word structure anyway.

:::callout "Character-level is a real tradeoff, not a simplification"
You get a tiny model, a tiny vocabulary, and no unknown-word problem. You pay for
it with sequence length: a name is ~7 tokens here, so `block_size = 16` is
already cutting longer names in half. Bigger models, and this repo's `shakespeare.txt`,
move to word-level or BPE precisely because of that. The idea is identical; only
the size of the alphabet changes.
:::

## The `ln(27)` ceiling

`vocab_size` sets the scale of the whole problem. A model that has learned
nothing guesses uniformly and scores

```
-log(1/27) = 3.2958
```

The reference's **first step prints 3.3660** — just above that. Nothing is wrong.
Freshly initialised logits are small (standard deviation ≈ 0.30), so the initial
distribution is nearly uniform: measured over the first 25 documents, its mean
entropy is **3.252**, a hair *sharper* than a perfect uniform guess, and its mean
window loss is **3.283**, a hair *below* it. The 3.3660 is one document (`yuheng`,
7 positions) that happened to be slightly harder than average.

That is the whole story of the loss curve in miniature, and worth internalising
before the next chapter: **the number you print is a sample, not an estimate.**
Any claim about training progress needs a window behind it.
