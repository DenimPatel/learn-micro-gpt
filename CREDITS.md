# Credits

Everything here is someone else's work, or ours on top of theirs. Digests,
generated detail, and the exact pinned-vs-derived rules live in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md), which is generated from
[`tools/provenance.py`](tools/provenance.py). This file is the human-readable
version.

---

## The subject: `microgpt.py`

**Andrej Karpathy** — [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

199 lines. No dependencies beyond the Python standard library. It is the whole
algorithm, and everything else is efficiency. Published alongside Karpathy's
*Intro to Large Language Models* talk, November 2023.

`reference/microgpt.py` is that file, vendored and **byte-for-byte unmodified**.
It is the reference track for this repository: the thing learners read, the
thing the browser runs, the thing every other port is measured against, and the
thing whose sha256 CI pins. The original docstring header, including the
`@karpathy` signature, is intact.

**On the exact bytes.** The vendored copy is an *earlier revision* of that gist,
not its current `HEAD`. The algorithm is identical; what differs is prose and
plumbing — comment wording, the `docs = [...]` comprehension, the dataset URL,
and a trailing `\r` on the progress print. We pinned what we actually had rather
than quietly swapping in the newer text, because the pinned bytes are what every
recorded loss number in this repository was produced from. If the reference ever
moves, `tools/check_provenance.py` fails, and the traces and benchmarks have to
be regenerated with it.

**License.** Karpathy has not put an explicit license on this file. It is
reproduced here for educational commentary, with full attribution and no
modification. If you are the author and would prefer different terms, please open
an issue and we will change them.

---

## The C ports

**Vishal (TheVixhal)** — [`vixhal-baraiya/microgpt-c`](https://github.com/vixhal-baraiya/microgpt-c) — **MIT**

`Copyright (c) 2026 Vishal (TheVixhal)`. Full text reproduced at
[`THIRD_PARTY_LICENSES/microgpt-c-LICENSE`](THIRD_PARTY_LICENSES/microgpt-c-LICENSE).

Two files in `implementations/c/` descend from this line of work:

| File | Config | Role |
| --- | --- | --- |
| `microgpt.c` | `n_embd 16`, `n_head 4`, `n_layer 1`, `block 16` | the **parity** track — same shape as the Python reference, so losses are comparable |
| `microgpt-scaled.c` | `n_embd 256`, `n_head 8`, `n_layer 4`, `block 256` | the **benchmark** track — the config that shows why the speed gap matters |

Our local revisions do not match any published upstream commit byte-for-byte.
We added Apple `Accelerate`/NEON paths with a portable scalar fallback, made the
build flags overridable so the code compiles on Linux CI, capped document length
for the parity config, and taught the parity build to emit the JSONL trace
format. Those modifications are ours and are MIT; the lineage and the upstream
copyright remain theirs.

Prior art in other languages, credited because learners will find them and it is
better that they find them from here:

- **JavaScript** — [`xenova/microgpt.js`](https://github.com/xenova/microgpt.js) (Hugging Face)
- **Go** — [`prasad83/go-microgpt`](https://github.com/prasad83/go-microgpt)

---

## Datasets

**Andrej Karpathy** — [`karpathy/makemore`](https://github.com/karpathy/makemore) — MIT

`data/input.txt` is `names.txt` at commit `988aa59`: 32,033 names, byte-identical
to upstream (including the missing trailing newline). It is committed rather
than downloaded, so `reference/microgpt.py` never touches the network — its
`os.path.exists('input.txt')` check finds our copy and short-circuits. That is
also what makes the browser able to run the reference offline.

**William Shakespeare** (public domain) — via [`karpathy/char-rnn`](https://github.com/karpathy/char-rnn)

`data/shakespeare.txt` is the canonical "Tiny Shakespeare" character-level
corpus: 40,000 lines, ~1.1 MB. No v1 track uses it. It is committed because the
first thing anyone does after finishing the names walkthrough is ask "what if I
trained it on real text?", and the answer should not be "go download something
and possibly break the reference".

---

## Ours

Everything else — `tools/`, `content/`, `visualizer/`, `docs/`, the CI workflows,
the trace and benchmark harnesses — is MIT, in [`LICENSE`](LICENSE).
