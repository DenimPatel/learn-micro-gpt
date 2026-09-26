# How to read it

Two ways in, and they are genuinely different.

- **You want to understand the program.** Read the 199 lines, in order, with the
  concepts open beside them. Start at
  [Loading the dataset](content/concepts/data-loading.md) and finish at
  [Inference and sampling](content/concepts/inference-sampling.md).
- **You want to understand one idea.** Use the graph, or the chapter list, and
  jump.

Either way, start by running it. A loss curve you watched fall is worth more
than any amount of prose about one.

---

## Just run it

```sh
make run
```

That trains the reference for 1,000 steps — about two minutes on a laptop CPU —
and prints 20 samples at the end. It executes the file in a scratch directory
with `data/input.txt` copied in, so `reference/microgpt.py` itself is never
modified. See [why that matters](#why-the-reference-runs-in-a-scratch-directory).

Prefer to run it in a browser? The atlas has a playground that runs the same file
in Pyodide, with 200 steps so a tab does not freeze for a minute.

## Then read the file

```sh
less reference/microgpt.py
```

It is 199 lines. Here is the shape of it, so you know what you are looking at:

| lines | what | concepts |
| --- | --- | --- |
| 1–6 | the docstring | — |
| 8–11 | imports, seed | — |
| 13–20 | the dataset | [data-loading](content/concepts/data-loading.md) |
| 22–26 | the tokenizer | [tokenizer](content/concepts/tokenizer.md) |
| 29–71 | autograd | [a number that remembers](content/concepts/autograd-value.md), [walking the graph backwards](content/concepts/autograd-backward.md) |
| 73–89 | the parameters | [parameters and their initialisation](content/concepts/params-init.md) |
| 91–143 | the architecture | [linear](content/concepts/linear.md), [softmax](content/concepts/softmax.md), [RMSNorm](content/concepts/rmsnorm.md), [embedding](content/concepts/embedding.md), [multi-head attention](content/concepts/multi-head-attention.md), [the MLP block](content/concepts/mlp-block.md), [the attention block](content/concepts/attention-block.md), [the head](content/concepts/lm-head.md), [the forward pass](content/concepts/gpt-forward.md) |
| 145–148 | Adam | [Adam](content/concepts/adam.md) |
| 150–183 | training | [cross-entropy](content/concepts/cross-entropy-loss.md), [the training loop](content/concepts/training-loop.md) |
| 185–199 | inference | [inference and sampling](content/concepts/inference-sampling.md) |

Every range in that table is the range the corresponding concept's own selector
resolves to, checked by `tools/tests/test_content.py` — so this table cannot drift
away from what the atlas actually shows. The one place it is narrower than the
section is autograd: the `# Let there be Autograd` comment is on line 28, and the
`class Value` definition the concept anchors starts on 29.

The line numbers are stable because the file is byte-pinned: CI checks its
sha256, and `docs/PROVENANCE.md` records it. If you have a version whose
concepts' line ranges do not match, yours is not the vendored one.

## Reading order, and why it is that order

The order is the prerequisite order, and it is enforced rather than merely
suggested: each concept declares its own `prereqs`, `tools/validate_concepts.py`
rejects a cycle, and the site draws the graph from exactly that field.

**Data.** You cannot understand a model that trains on characters until you know
where the characters come from and how they became integers.

**Machinery.** Autograd, the parameters, and the reverse pass. This is the only
genuinely subtle part of the file, and it is 44 lines.

**Architecture.** The longest chapter and the one worth slowing down in. Three
tiny building blocks, one attention loop, and a head. Everything here is
straightforward *once* the machinery is understood, and incomprehensible before.

**Training.** Turning the right answer into a number worth minimising, and
walking the gradient back.

**Inference.** The same code path, fed its own output.

## Things worth doing yourself

The concepts point at the experiments. These are the ones that teach something
per keystroke, in rough order of value.

1. **`n_head = 1`**, then `n_head = 16`. `head_dim` goes 4 → 16 → 1, and the loss
   curve tells you the sweet spot is in the middle. Nothing in the file tells you
   where.
2. **`block_size = 8`.** Most names are now truncated, and the loss gets worse.
   A hyperparameter that is really a capacity decision.
3. **`num_steps = 10000`.** Takes about eighteen minutes. The samples stop being
   names in the dataset and become name-*like* — the transition from
   memorisation to generation, in one integer.
4. **`n_layer = 4`, then `8`.** Works, then gets worse. At 16 dimensions wide
   there is not enough per layer to justify the depth.
5. **`learning_rate` 0.01 → 0.1.** Watch it diverge. Then find the edge.
6. **Delete `docs = [...].` shuffle and re-run.** The loss still falls. The
   samples get much worse. The most quietly destructive edit in the file.
7. **Remove the `p.grad = 0` line.** The loss curve still looks fine for a
   while. This is what a missing line looks like.
8. **Delete the `x * 16/4**0.5` in the attention logits.** Everything still
   works at this width, and breaks as soon as `n_embd` grows.

## Why the reference runs in a scratch directory

`make run` copies `data/input.txt` into a throwaway `.run/` and executes the
reference there. That is not tidiness — it is because the reference does this:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/...'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Two problems, and both are unfixable without editing a byte-pinned file:

- It resolves `input.txt` from the **current working directory**, not the script's
  directory. Run it from anywhere else and you get a fresh download — or, in a
  read-only directory, a crash.
- The URL points at `refs/heads/master`, a **mutable branch**. The bytes behind
  it can change at any time, which would silently invalidate every loss number
  recorded in this repository.

Committing `data/input.txt` short-circuits the check: the file is present, so
nothing is downloaded and the run is exactly the recorded one. The bug is
unreachable through the Makefile. It is still reachable by running
`python3 reference/microgpt.py` from a directory with no `input.txt`, and that is
issue 3 in [`docs/KNOWN-ISSUES.md`](KNOWN-ISSUES.md).

The alternative — editing the reference to load the dataset properly — would
mean the file is no longer the vendored one, and every recorded number, every
concept's prose, and every trace in this repository would describe code that no
longer exists upstream.

## If you only read one concept

[Multi-head attention](content/concepts/multi-head-attention.md). It is the
concept the file exists to demonstrate, it is the one the recorded traces say
something surprising about, and it is the one where the intuition most people
start with turns out to be wrong.
