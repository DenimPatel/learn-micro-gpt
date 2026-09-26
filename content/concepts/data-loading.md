---
id: data-loading
title: Loading the dataset
chapter: data
order: 1
difficulty: 1
summary: "The model never sees a file. It sees a Python list of strings, one per document, and it trains on one of them per step. Getting to that list is four lines and contains one decision that matters: the file is committed here, not downloaded."
math: |
  \mathrm{docs} = \big[\, l.\mathrm{strip}() \;\big|\; l \in \texttt{input.txt},\; l \neq \texttt{""} \,\big]
shapes:
  - { name: docs, shape: "[32033]" }
  - { name: doc, shape: "str" }
anchors:
  python: { selector: "section:Let there be an input dataset" }
  c: { selector: "def:load_data" }
related: [tokenizer, training-loop]
prereqs: []
---

Everything downstream is a list of strings. Getting there is four lines, and one
of them is a trap.

## The four lines

```python
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

That is a name dataset: 32,033 of them, one per line. The comprehension strips
whitespace, splits on newlines, and drops empties — so a trailing newline in the
file does not produce a phantom empty document.

Then `random.shuffle(docs)`. The seed on line 11 makes this reproducible, which
matters more than it looks: `docs[step % len(docs)]` walks the list in order, so
*the shuffle is the only thing deciding which name step 0 sees*. Without it, the
model trains on the first 1000 names in the file and only ever sees names
starting with 'a'. That is a real bug, and it is silent — the loss still falls.

:::callout "The shuffle is not optional"
Without it you are training on an alphabet, not on a language. The loss curve
looks fine. The samples look terrible. This is the first place in 199 lines where
something can be quietly, invisibly wrong.
:::

## The trap: `input.txt` from the current directory

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/...'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Two problems, both of which this repository had to fix:

1. **It resolves from the CWD, not from the script's directory.** Run it from
   anywhere else and you get a fresh download — or, in a read-only directory, a
   crash. `make run` works around this by copying `data/input.txt` into a
   throwaway `.run/` directory and executing there, so the reference file itself
   never changes.
2. **The URL is mutable** (`refs/heads/master`, not a commit sha). The bytes
   behind it can change at any time, and the recorded loss numbers in
   `traces/` would silently stop describing the same experiment.

Both are why `data/input.txt` is *committed* — 228 KB, byte-identical to
`karpathy/makemore@988aa59`, sha256 pinned in `docs/PROVENANCE.md`. With the file
present, the `os.path.exists` check short-circuits, nothing is downloaded, and
the run is hermetic. It is also what lets the browser run the reference offline
later, by pre-writing the same file into Pyodide's virtual filesystem.

## Why one document per step

The training loop takes `docs[step % len(docs)]` — a single name, a single
batch, no batching at all. That is the most extreme possible choice, and it has a
consequence you will notice immediately when you look at the loss: **each printed
loss is the loss on that one name**, not a running average.

So the curve is not smooth. It bounces between roughly 1.6 and 3.9, and it is
*supposed* to. Half of its 999 step-to-step movements are upwards. Any tooling
that expects a monotonically falling curve — including the parity gate this
repository originally specified — has misunderstood what it is looking at. See
[Cross-entropy loss](../concepts/cross-entropy-loss.md) for how we measure
progress properly instead.

## Swapping the dataset

`data/shakespeare.txt` is the same idea at a different scale: 40,000 lines of
real English. Change the filename, and the tokenizer — which is derived, not
configured — grows to cover the new alphabet automatically. The model is
untouched. That is the payoff of character-level tokenisation, and it is the
first experiment worth running yourself.
