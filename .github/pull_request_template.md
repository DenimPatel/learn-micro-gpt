# Pull requests

Thanks for looking at this. A note on what a good one looks like here, so the
template is not busywork.

## The three things that need a sentence each

**What you changed and why.** Usually obvious, occasionally not — the *why* is
the part worth reading.

**If you touched `content/`, which concepts and which sections.** A prose change
can invalidate a claim without failing a build, and the fastest way to catch that
is for the author to say which prose they meant to change.

**If you touched `reference/`, what it invalidates.** That file is byte-pinned
and every number in this repository is a claim about it. See
`docs/TRACE-FORMAT.md` § "Regenerating after a change to the reference": update
the pin, bump `PIN_EPOCH`, regenerate the traces and the benchmarks, and re-read
the concepts that quote numbers. The tests will catch the ones that moved a lot;
they will not catch the ones that moved slightly, and those are worth a look.

## What CI will check, and what it cannot

| Checked | Not checked |
| --- | --- |
| every concept's selectors resolve | whether the prose is *good* |
| schema conformance, graph acyclicity, glossary completeness | whether a quoted number is still *right* after a small change |
| untagged code fences are verbatim excerpts of their own anchor | whether a reader will understand it |
| the reference's sha256, and the generated docs against it | — |
| the traces regenerate identically | — |
| every track builds, and is within the parity band | **whether a track's gradient is correct** — see below |
| the site builds with no absolute fetches, and works from a subpath | — |
| 107 Python tests, 18 unit, 18 e2e | — |

**The parity gate cannot catch a mis-scaled gradient**, because Adam divides by
an estimate of the gradient's own magnitude: the C port's gradient is wrong by a
factor of −0.12 and its loss curve still passes. That is why requirement 5 in
`docs/ADDING-A-LANGUAGE.md` asks every track for a directional-derivative check,
and why `docs/KNOWN-ISSUES.md` exists. If you touch a backward pass, that is the
check to look at first.

## Style

- **Python in `tools/`:** the standard library only, and the reason is in
  `docs/ARCHITECTURE.md`. Do not add a dependency. If you need YAML or a test
  runner, the honest move is to extend the strict subset that already exists.
- **Comments explain *why*.** The reference is commented exhaustively and well,
  and that is the model. A comment restating the next line is noise; a comment
  recording a bug that was already found and fixed twice is the most valuable
  thing in the file.
- **A concept's untagged code fence claims to be a verbatim excerpt of its own
  anchor range.** Anything else — illustrative arithmetic, a diagram, program
  output, a deliberate cross-reference — declares an info string. CI checks the
  untagged ones.
- **Do not reformat code you are not changing.** The reference is byte-pinned and
  must stay that way; the ports are read far more often than they are edited.

## Checks

```sh
make check          # everything CI runs
make validate       # selectors, schema, graph, fences, generated files
make test           # the Python suite
make c-test         # three C build configurations agree; gradient checks
make parity         # all five tracks within the band
make lint && make fmt
```

If you are adding a language track, `docs/ADDING-A-LANGUAGE.md` has a
definition-of-done checklist.
