# Architecture

How this repository is put together, and — more usefully — the decisions that
were not obvious, why they were made, and what they cost.

---

## The dependency graph

```
content/concepts/*.md          authored prose: frontmatter + :::  directives
        │
        ├── tools/selectors.py       AST selectors → line ranges, per language
        │        │
        │        └── tools/gen_anchors.py  → visualizer/src/data/generated/anchors.json
        │
        └── tools/concepts.py        body → typed blocks
                 │
                 └── tools/render_content.py → visualizer/src/data/generated/content.json
                                                   visualizer/src/data/generated/index.json

reference/microgpt.py ──?raw──┐
implementations/*/*      ──?raw──┼── visualizer/src/data/sources.ts ──→ the app
traces/**/*.jsonl        ──?raw──┤
benchmarks/results.json  ──?raw──┤
autoresearch/results.json ──json─┘
```

The two Python entry points are `make validate` (which runs
`gen_anchors.py`, then `validate_concepts.py`, then `render_content.py`). Nothing
in `visualizer/` reads `content/` at runtime. The content is compiled to JSON
before the site is built, and the site reads only JSON.

## Four decisions, and what each one bought

### 1. AST selectors instead of line numbers

The previous visualizer hardcoded `attn-heads: [122, 131]`. That rots
*silently*: edit the reference and every range points at the wrong code while the
site keeps rendering, just wrongly, teaching the wrong thing with total
confidence. Nothing fails.

Concepts now declare selectors:

```yaml
anchors:
  python: { selector: "def:gpt offset:[15, -12]" }
  c:     { selector: "comment:Multi-head attention" }
```

The grammar is four forms — `def:`, `assign:`, `comment:`, `section:` — plus a
relative `offset:[a, b]`. `offset` is relative on purpose: it lets a concept point
at a region *inside* a function without either duplicating the function's span or
hardcoding a number that drifts the moment the function above it grows a comment.
`section:` exists because three concepts describe whole regions, and a numeric
offset to reach the end of one is exactly the hardcoded line number this design
exists to remove.

**Verification:** running the resolver over `reference/microgpt.py` reproduces
all 19 line ranges of the old `mapping.js` exactly. That file was spot-checked
against the reference by hand and all 19 nodes were correct, so it is the one
piece of inherited ground truth, and it is now an assertion
(`tools/tests/test_selectors.py::TestLegacyMappingParity`) rather than a memory.

**Cost:** the C-family resolver is a documented heuristic, because the standard
library has no C parser. It does real brace-matched definitions and file-scope
declarations, and it deliberately *ignores* declarations inside function bodies —
with a brace-matching scan, "the first statement that binds `x`" would mean
something different in every function that has one, and a concept that silently
jumped between them would be worse than one that said "use `def:` with an
offset". Python gets the real `ast` module.

### 2. Everything imported at build time, nothing fetched at runtime

The previous visualizer did `fetch('/microgpt.py')` with a `../microgpt.py`
fallback. The absolute URL ignores whatever subpath GitHub Pages served from; the
relative one breaks on any route at a different depth. Both are gone as a
*category*: every source is imported with Vite's `?raw`, resolved by the bundler
and inlined into JavaScript. There is no URL left to get wrong.

The imports are *dynamic* — `() => import('... ?raw')` — so the five language
sources (about 152 KB together) land in their own chunks rather than the initial
payload. That does not reintroduce a runtime fetch. It takes the initial route
from 213 kB to 180 kB gzipped.

**CI enforces it:** a grep for `fetch('/` in the built bundle fails the build. The
same regex was checked against a planted sample to confirm it still matches.

**Cost:** the source files must exist at build time, so a partial checkout cannot
build the site. That is fine — `make validate` reports it.

### 3. Hash routing, and a 404 fallback anyway

A static host with no rewrite config 404s every history-routing deep link, and a
404 in a teaching site is a broken lesson rather than a cosmetic problem. The
hash needs no server configuration and behaves identically at a domain root, under
a subpath, and from `file://`.

The uglier URLs (`#/learn/softmax`) are the price. Given the choice between
prettier URLs that 404 on GitHub Pages and working ones, this takes the working
ones. A `dist/404.html` copy is written at build time, so switching to history
routing later is a one-line change if the repo ever moves to a host with rewrite
support.

**And the test that proves it:** the e2e suite runs against a production build
served from `/learn-micro-gpt/` by a 60-line static server that 404s anything
outside that prefix. `vite preview` strips the base path rather than serving under
it, so it would have passed while the site was broken in production.

### 4. Standard-library-only tooling

`tools/` has **no dependencies at all**. The reference's whole ethos is that the
algorithm fits in one file with nothing but a standard library, and the tooling
that checks it should be auditable without a package manager.

The cost is `tools/yamlite.py`: a strict YAML subset, ~250 lines, because PyYAML is
off the table. It supports exactly what the concept frontmatter uses, has an
explicit nesting depth limit, and refuses everything else loudly — a permissive
frontmatter parser is how a content repository ends up with a concept whose
`shapes` silently became a string, rendering a table of one row forever.

---

## The content pipeline

Concept bodies use a small directive syntax, and `tools/concepts.py` owns it:

| directive | form | meaning |
| --- | --- | --- |
| `:::callout` / `:::note` | block, until `:::` | a boxed aside, optionally titled |
| `:::term` | one line | link a glossary term; validation fails if undefined |
| `:::shape` | one line | inline one shape-table entry |
| `:::trace` | one line | inline a widget backed by a committed trace |
| `:::lang` | block, until the next one | a language range |

Every directive validates. A near-miss is an error, not prose: a stray keystroke
once produced `::::trace`, which failed the regex and rendered as a paragraph of
colons — silently, with no build failure. There is an explicit check for that
now, and a unit test.

**The fence convention** matters more than it looks. An *untagged* code fence
claims to be a verbatim excerpt of the concept's own anchor range, and
`tools/tests/test_content.py` checks every line of it against the range the
selector resolves to. Anything that is not a verbatim excerpt — illustrative
arithmetic, a diagram, program output, a deliberate cross-reference to another
part of the file — declares an info string instead.

A concept quoting code outside the range it highlights is the worst failure this
repository can have: the prose is right, the highlight is wrong, and both render
with total confidence. The convention makes it answerable by looking at the
fence.

## How the site is put together

```
src/
  data/sources.ts     build-time imports, generated JSON, trace parsing, EMA
  data/types.ts        hand-written contract for the generated shapes
  content/markdown.tsx marked token walk + KaTeX; no dangerouslySetInnerHTML
  content/highlight.ts Shiki, loaded lazily, five grammars
  app/router.ts        40-line hash router
  app/state.ts         theme and reading progress
  components/          CodePanel, widgets, ConceptPage, AtlasGraph, RunPlayground
  worker/              Pyodide, in a Web Worker, from a pinned CDN
```

**Markdown is walked as a token tree**, not parsed to an HTML string. Every
element is created by React and inspectable, and there is no
`dangerouslySetInnerHTML` anywhere except KaTeX's own output from a frontmatter
literal. The reasoning: "the content is trusted" is exactly the assumption that
rots, and a fork with a bad edit should render broken rather than execute.

**Highlighting cannot split a token.** The old site ran Prism and then split its
output on newlines, one line at a time — and the reference opens with a
multi-line docstring, which is a *single* token, so the very first code block
rendered mangled. Here Shiki's HTML goes into a `<pre>` verbatim and the line
numbers live in a separate gutter column. A test asserts no panel has unbalanced
`<span>` counts or a line starting with a closing tag.

**Shiki is imported finely.** `shiki/core` plus five grammars, two themes, and
the JavaScript regex engine. The obvious `import('shiki')` bundled every
language Shiki ships: the first build emitted 785 kB of C++ and 790 kB of Emacs
Lisp for a site that shows five, plus a 622 kB oniguruma wasm chunk the JS engine
does not need.

**KaTeX is 77 kB gzipped of the initial 180.** Lazy-loading it would cut that,
at the cost of a flash of unstyled formulas on the three concepts that have any.
Not worth it; recorded here so the decision is not re-litigated every time the
bundle size is looked at.

## What CI runs, and why in that order

| job | gates | why it is where it is |
| --- | --- | --- |
| `provenance` | everything | the reference is byte-pinned, and every number is a claim about those bytes |
| `content` | tracks, web | selectors, schema, graph, fences, tests, trace drift |
| `tracks` | deploy | all five implementations build, agree, and pass the gradient checks |
| `web` | deploy | lint, typecheck, unit, build, no-absolute-fetch, e2e from a subpath |
| `deploy` | — | default branch only, environment-protected, non-cancelling |

The ordering is the point. Content validation cannot meaningfully run before the
provenance check: validating the prose about a set of numbers while the subject of
those numbers has moved is theatre.

`tracks` runs parity at 200 steps rather than 1,000. The gate compares smoothed
losses at checkpoints and a trend over 50-step windows, and 200 steps is enough
for both to be meaningful; the full run is a `workflow_dispatch` input and the
nightly job. Timings are regenerated weekly into a PR rather than committed on
every push, because a benchmark change is a thing a human should look at.

## The research track, and the three things it is fenced off from

`autoresearch/` is a second Rust crate that a model rewrites unsupervised, a
harness that measures it, and a ledger. `docs/AUTORESEARCH.md` explains how it
works; this section is about why it is shaped the way it is, because all three
constraints are ones that would look arbitrary without the reasoning.

**The frozen track is pinned, not merely documented.** `implementations/rust/`
gets sha256 entries in `tools/provenance.py`, which the `provenance` CI job
enforces and which every deploy depends on. This is the mechanism behind the
claim "the first version is preserved": a claim in a README is worth nothing, a
claim with a failing build behind it is worth something. The flag is `derived=True`
rather than pinned, because the file is our adaptation of Karpathy's Python and
not bytes fetched from anywhere &mdash; the comparison happens either way, and the
entry's `note` says explicitly that for this file "derived" means the opposite of
what it usually means.

**The parity gate was not modified.** The obvious alternative was to make
`tools/parity.py`'s symmetric ±10% band one-sided so a better Rust track could
pass it. That would have been a mistake twice over: `test_trace_parity.py` pins
`parity.BAND == 0.10` and defends the symmetric band on purpose, and more
importantly the gate's job is to show that five languages compute the same
function. A track that is deliberately diverging is no longer evidence of that,
so the candidate is not gated by it at all. It has its own gate &mdash; a
digest-pinned finite-difference probe plus a two-axis keep/discard rule &mdash; and
that separation is what lets the parity job, and every number recorded against it,
stay exactly as they were.

**The candidate lives outside `implementations/`.** Not for tidiness.
`tools/selectors.py` resolves concept anchors per language and
`anchors.json` records each source's line count; a rewritten
`implementations/rust/src/lib.rs` would eventually break a resolved range, and
`gen_anchors.py` records that file's size for any language a concept declares an
anchor in. Putting the tuned crate at `autoresearch/candidate/` means no existing
tool can mistake it for a track, and `autoresearch/candidate/src/lib.rs` is *not*
a member of the `Language` union &mdash; the site's `LOADERS` map is keyed by that
type, and adding a sixth key to it would make the type lie by claiming all six are
cross-linked tracks.

**What CI added.** Three steps in the `tracks` job, which already owns the language
tracks and gates `deploy`: the candidate builds and is clippy-clean at the same bar
as the frozen one; `python3 -m tools.autoresearch verify` re-checks the committed
results against the committed source; and `results.json` is regenerated and
diffed. The `verify` step's checks are ordered by how much they can actually
establish. The source digest comparison is exact and free. The gradient probe and
the loss re-measurement are the same checks a candidate is held to, run on a
different machine, so the loss is checked against the tolerance rather than for
bit-equality &mdash; `f32` contraction under a different LLVM means two honest
runs of the same code agree to about the third significant figure, which
`docs/BENCHMARKS.md` already says. The speed axis is not checked at all, and
`verify` prints that it is not, because a check that quietly cannot pass is
indistinguishable from one that quietly did.

## Things that would be reasonable to change

- **History routing**, if the repo ever moves to a host with rewrite support.
  The `404.html` fallback is already emitted.
- **Lazy KaTeX**, if the bundle budget tightens. Three concepts have
  display maths.
- **More trace widgets.** The format supports what a 199-line model produces; the
  selection settings are in `tools/trace.py` and documented in
  `docs/TRACE-FORMAT.md`.
- **A real fix for the C backward pass.** `docs/KNOWN-ISSUES.md` issue 1 says
  what is wrong, what the fix would be, and which test asks to be promoted when
  it lands.
