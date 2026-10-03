# Contributing

This repository is a *reading* of a 199-line file, and almost everything in it is
a claim about something that was measured. That single fact decides most of what
a good change looks like here: a change is not "done" when it builds, it is done
when the numbers it touches have been regenerated and the prose that quotes them
has been re-read.

`README.md` describes the product. This file describes the process. If you only
read one section, read [The green loop](#the-green-loop); if you are touching
`reference/`, `traces/` or `benchmarks/`, read
[Read-only and write-targets](#read-only-and-write-targets) first.

---

## The green loop

Everything is reachable from `make`. Bare `make` prints the full target list.

```sh
make lint        # every linter this repository has
make test        # the Python suite (stdlib unittest, no dependencies)
make validate    # selectors, schema, graph, fences, generated files resolve
make check       # everything CI runs, in the order CI runs it
```

`make check` is the honest answer to "is this green?" — it chains `lint`, `test`,
`validate`, `test-web`, `c-test`, `test-rust`, `test-ts`, `test-go`, `parity`,
`build`, and finally `autoresearch verify`. If `make check` passes on your
machine, CI should pass too.

A few practical notes:

- **`make` tells you what is missing.** Every target checks its own
  prerequisites and says what to install, rather than failing with a shell error.
- **`tools/` has no dependencies at all** — not "minimal", none. Do not add one.
  If you need YAML, extend the strict subset in `tools/yamlite.py`, which has its
  own tests. This is a deliberate constraint, not an oversight; see
  `docs/ARCHITECTURE.md`.
- **`make fmt` edits files in place.** Run it, then review the diff — it is the
  one target whose whole job is to write.
- **First run of `make check` is slow** (it builds four language tracks and the
  site). Subsequent runs are cached.

Before opening a pull request, `make check` should be green on your branch.

## Read-only and write-targets

The repository holds two kinds of file, and confusing them is the most common way
to make a change that looks correct and is not.

### Byte-pinned — do not edit

`reference/microgpt.py` is reproduced **byte-for-byte** from Karpathy's talk and
pinned by sha256. It is not ours to modify, and every number in the repository is
a claim about those exact bytes. The annotated fork for reading lives beside it;
that is the file to touch if you are adding commentary.

If you ever *do* change a pinned file, the pin is not the problem — the fix is
the four-part sequence in `docs/TRACE-FORMAT.md` § *Regenerating after a change
to the reference*: update the digest in `tools/provenance.py`, bump `PIN_EPOCH`,
regenerate the traces and the benchmarks, and re-read the concepts that quote
numbers. `make verify-provenance` will tell you when this is needed:

```sh
make verify-provenance   # every vendored file still matches its sha256
```

### Generated — regenerate, never hand-edit

These are outputs. A hand-edit will be overwritten, and worse, will disagree with
the thing that produces it:

| Directory / file | Produced by | Note |
| --- | --- | --- |
| `visualizer/src/data/generated/*.json` | `make validate` | anchors, content, index |
| `traces/` | `make trace` | `make trace-check` regenerates and fails on drift |
| `benchmarks/results.json` | `make bench` | machine-specific; record the machine |
| `docs/PROVENANCE.md` | `make provenance` | re-renders from pins already in source |

`make validate` regenerates the generated JSON as part of its work; that is
expected, and it is why those files are not things you edit by hand.

### Safe to edit

`content/`, `tools/`, `implementations/`, `visualizer/src/`, `docs/`, and this
file. Within those, prefer *extending* what is there over adding a parallel
mechanism — the reference is the model for how to write code here.

## Prose carries claims; keep them true

A change to a concept can invalidate a number without failing any build. The
tests pin *some* of these — `tools/tests/test_content.py` connects a quoted
number to the recording that produced it — but they will not catch a claim that
moved slightly, and those are exactly the ones worth a second look.

So when you touch `content/`, say in the pull request **which concepts and which
sections** you meant to change. That is the fastest way to catch a prose change
that quietly made an older claim false.

Two rules that CI *does* enforce, and that are easy to get wrong:

- **An untagged code fence is a verbatim excerpt of the concept's own anchor
  range.** Not "similar to" — verbatim. Illustrative arithmetic, a diagram,
  program output or a cross-reference declares an info string instead. CI checks
  the untagged ones.
- **A `:::term` must resolve to an existing glossary entry.** `make validate`
  fails otherwise.

`docs/ADDING-A-CONCEPT.md` is the recipe for adding a concept;
`docs/ADDING-A-LANGUAGE.md` is the contract (and definition-of-done checklist)
for adding a language track.

## What CI checks, and what it cannot

CI is strong where a machine can be certain and deliberately silent where it
cannot. Knowing which side you are on is part of the job:

| Checked | Not checked |
| --- | --- |
| every concept's selectors resolve | whether the prose is *good* |
| schema conformance, graph acyclicity, glossary completeness | whether a quoted number is still *right* after a small change |
| untagged fences are verbatim excerpts of their anchor | whether a reader will understand it |
| the reference's sha256, and the generated docs against it | — |
| the traces regenerate identically | — |
| every track builds and is within the parity band | **whether a track's gradient is correct** |

**The parity gate cannot catch a mis-scaled gradient.** Adam divides by an
estimate of the gradient's own magnitude, so a gradient that is wrong by a
constant factor barely moves the loss. The C port once carried a gradient wrong
by a factor of `-0.12` — wrong sign, an order of magnitude out — and its loss
curve still passed. That is why every track needs a directional-derivative check
(requirement 5 in `docs/ADDING-A-LANGUAGE.md`), why `docs/KNOWN-ISSUES.md`
exists, and why the frozen Rust track's 6.32%-high gradient is written down
rather than fixed. If you touch a backward pass, look at the finite-difference
check first.

## Style

- **Python in `tools/`:** standard library only. Do not add a dependency.
- **Comments explain *why*.** The reference is commented exhaustively and well,
  and that is the model. A comment restating the next line is noise; a comment
  recording a bug already found and fixed twice is the most valuable thing in the
  file.
- **Do not reformat code you are not changing.** The reference is pinned, and the
  ports are read more often than they are edited. A drive-by reformat buries the
  change that matters.

## Pull request checklist

Copy this into your description if the template does not already prompt you:

- [ ] `make check` is green on this branch.
- [ ] I said what I changed **and why**.
- [ ] If I touched `content/`: I named the concepts and sections, and the
      numbers I quote still match the recordings.
- [ ] If I touched `reference/`: I updated the pin, bumped `PIN_EPOCH`,
      regenerated traces and benchmarks, and re-read the concepts that quote
      numbers.
- [ ] If I touched a backward pass: I looked at the directional-derivative check.
- [ ] I regenerated any generated file rather than editing it.
- [ ] I ran `make fmt`.

The pull request template (`.github/pull_request_template.md`) expands the first
three into sentences. `CODEOWNERS` routes review; the two paths that get a
closer look are the ones where a wrong change is **silently** wrong
(`reference/`, `traces/`) rather than loudly wrong.

## Attribution

Nothing here is ours, and all of it is attributed. If you add a vendored file,
record its upstream commit and digest and add it to the pins, so
`make verify-provenance` can hold it. The narrative is in `CREDITS.md`; the list
is in `docs/PROVENANCE.md`. This repository's own code and docs are MIT.
