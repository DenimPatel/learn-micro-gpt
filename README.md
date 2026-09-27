# learn-microgpt

A guided, visual reading of Andrej Karpathy's
[`microgpt.py`](reference/microgpt.py) — 199 lines that implement a complete
GPT: character-level tokenizer, autograd, a transformer layer, Adam, and
autoregressive sampling. It trains on 32,033 names in about two minutes on
laptop CPU, and the samples at the end are real words.

This repository is an *atlas* of that file. Eighteen concepts, each anchored to
a resolved line range, each with the tensor shapes spelled out, each cross-linked
to the same algorithm in four other languages, and each quoting numbers that came
out of a real run.

**[Open the atlas](docs/HOW-TO-READ.md)** ·
**[Run the 199 lines](docs/HOW-TO-READ.md#just-run-it)** ·
**[What is known to be wrong](docs/KNOWN-ISSUES.md)** ·
**[Let a model try to improve the Rust port](docs/AUTORESEARCH.md)**

> **The research track.** `make autoresearch-rust` lets a model rewrite the Rust
> port while a harness measures every attempt, keeps the ones that win on either
> loss or speed without regressing the other, and commits all of them — the
> failures included. The frozen track it is measured against is sha256-pinned and
> cannot move, and every candidate is finite-difference checked before it is
> allowed to compete, because a loss number cannot tell a correct gradient from a
> wrong one. Start at [`docs/AUTORESEARCH.md`](docs/AUTORESEARCH.md).

---

## Just run it

You do not need to read anything else first. Clone, then:

```sh
make run          # train the 199-line reference, 1,000 steps, ~2 minutes
make dev          # the atlas in a browser at localhost:5173
```

`make` on its own prints everything else. Every target checks its own
prerequisites and tells you what to install if something is missing.

## The three things this repository is actually for

**1. Every number is measured, and regenerable.**

The loss curve, the attention weights, the speed comparison, the per-parameter
gradient checks — all from real runs, committed to the repository, regenerated
by `make trace` and `make bench`, and re-checked by CI. Nothing here is a
plausible-looking invention. Where a number is quoted in prose, a test asserts it
against the recording.

```sh
make trace        # re-record traces/ from the real reference
make trace-check  # regenerate and fail on any diff
make bench        # re-measure every track on this machine
```

**2. Nothing rots silently.**

The previous version of this visualizer hardcoded line numbers: the
multi-head-attention block was `attn-heads: [122, 131]`. That rots invisibly —
edit the reference and every range points at the wrong code while the site keeps
rendering, just wrongly, teaching the wrong thing with total confidence.

So concepts declare **AST selectors**, not line numbers, and they are resolved
per language at build time:

```yaml
anchors:
  python: { selector: "def:gpt offset:[15, -12]" }     # -> lines 122-131
  c:     { selector: "comment:Multi-head attention" }  # -> lines 446-465
```

A selector that resolves to nothing is a build failure naming the concept and
the selector. Editing the reference moves the highlights with it.

**3. What is wrong is written down.**

The C port's hand-written backward pass does not correctly propagate the key and
value gradients of earlier positions back to their embeddings. Only the output
head's gradient is right. Measured as a directional derivative over all 4,192
parameters, its gradient comes out at **−0.12×** the true value, where a correct
gradient is 1.0×.

**Its loss curve still tracks the reference to within 7%, and it still trains.**
Adam divides by an estimate of the gradient's own magnitude, so a gradient that
is wrong by a factor barely moves the step.

That is the most instructive thing in the repository, and it is why
[`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md) exists. A loss curve is evidence
that something learned, not evidence that the thing that learned was the
gradient — and the only thing that can tell those apart is a finite difference.

## Layout

```
reference/     microgpt.py, byte-pinned. And an annotated fork for reading.
implementations/
  c/           the parity track (micro config) and the scaled config, for speed
  go/ rust/ typescript/   the same algorithm, a small autograd tape each
content/       18 concepts + a schema + the curriculum spine. The source of truth.
tools/         stdlib-only Python: anchors, validation, traces, parity, benchmarks.
traces/        recorded runs, committed. Every number on the site comes from here.
benchmarks/    measured costs, with the machine recorded.
docs/          how to read it, how it is built, what is known to be wrong.
visualizer/    the atlas. Vite + React + Tailwind + Shiki + KaTeX + dagre.
```

## Attributing everything

Nothing here is ours, and all of it is attributed.

- **`reference/microgpt.py`** — Andrej Karpathy, from his *Intro to Large
  Language Models* talk (November 2023), reproduced byte-for-byte and pinned by
  sha256. Karpathy has published no explicit license for it; it is reproduced
  here for educational commentary with full attribution and no modification.
- **The C ports** descend from
  [`vixhal-baraiya/microgpt-c`](https://github.com/vixhal-baraiya/microgpt-c),
  MIT, © 2026 Vishal (TheVixhal). Our modifications are ours, and MIT.
- **`data/input.txt`** is `karpathy/makemore@988aa59`'s `names.txt`,
  byte-identical.
- This repository's own code and docs are MIT.

Digests, upstream commits, and the full list are in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md); the narrative is in
[`CREDITS.md`](CREDITS.md).

## Verifying it yourself

```sh
make verify-provenance   # every vendored file still matches its sha256
make validate            # every concept's selectors resolve; schema; graph; fences
make test                # 107 Python tests, stdlib unittest, no dependencies
make c-test              # three C build configurations agree; gradient checks
make parity              # all five tracks within the statistical band
make check               # everything CI runs
```

The `tools/` directory has **no dependencies at all**. Not "minimal" — none. The
ethos of the reference is that the whole algorithm fits in one file with nothing
but a standard library, and the tooling that checks it should be auditable
without a package manager. `tools/yamlite.py` is a strict YAML subset written
for the purpose, and it has a test suite.

## Where to go next

| If you want to | Go to |
| --- | --- |
| read the file properly | [`docs/HOW-TO-READ.md`](docs/HOW-TO-READ.md) |
| understand the code layout | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| know what is broken | [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md) |
| see what a model did to the Rust track | [`docs/AUTORESEARCH.md`](docs/AUTORESEARCH.md), or `#/research` on the site |
| add a sixth language | [`docs/ADDING-A-LANGUAGE.md`](docs/ADDING-A-LANGUAGE.md) |
| understand the numbers | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| write a concept | `content/concepts/*.md` and `content/schema.json` |
| add a widget | [`docs/TRACE-FORMAT.md`](docs/TRACE-FORMAT.md) |

## The reference, in full, for reference

```python
"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
@karpathy
"""
```

199 lines, 9,164 bytes, sha256
`4e0c321442692d5a2ccf0328c5e472cd7efa27fa2c8a243438e783054f4eb6ae`. MIT for
everything else in this repository.
