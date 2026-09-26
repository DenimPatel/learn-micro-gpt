# `reference/`

## `microgpt.py` — the reference

Andrej Karpathy's 199 lines. **Byte-for-byte as vendored, and byte-pinned.**

- sha256 `4e0c321442692d5a2ccf0328c5e472cd7efa27fa2c8a243438e783054f4eb6ae`
- 9,164 bytes, no trailing newline
- origin: <https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95>
  (published with the *Intro to Large Language Models* talk, November 2023)

`make verify-provenance` fails if these bytes move. That is not bureaucracy:
every number in this repository is a claim about this file, and the concepts'
anchors are resolved against it.

**It is an earlier revision of that gist, not its current `HEAD`.** Same
algorithm; the differences are comment wording, the `docs = [...]` comprehension,
the dataset URL, and a trailing `\r` on the progress print. We pinned the bytes
we actually had rather than quietly swapping in the newer text, because every
recorded loss, every attention weight, and every benchmark in `traces/` and
`benchmarks/` was produced from *these* bytes.

Karpathy has published no explicit license for it. It is reproduced here for
educational commentary, with the original docstring and `@karpathy` signature
intact and no modification. If you are the author and would prefer different
terms, please open an issue and we will change them.

## `microgpt-annotated.py` — a fork, and a reading aid

A heavily commented expansion of the same program: 507 lines against 199, with
explanations interleaved. Useful for reading, and **deliberately excluded from
everything**:

- **Not the reference track.** `tools/trace.py` never runs it; the concepts'
  anchors never point at it; the parity gate never measures it.
- **Not a target for the browser.** The visualizer bundles it only if something
  asks, and nothing does.

The reason is structural: the annotations sit *between* the code lines, so every
line number differs from `microgpt.py`. A concept anchored to a line range would
point at the right code in the wrong file, which is the exact failure the
selector system exists to prevent. If you want a concept to reference the
annotated fork, it needs its own anchor, not a reused one.

**`tools/tests/test_content.py` asserts this.** Every concept's anchor is resolved
against `reference/microgpt.py`, and every untagged code fence is checked against
that resolution.
