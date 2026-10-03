# Adding a concept

A concept is the unit of teaching in this repo. One file in
`content/concepts/`, one Markdown page on the site, one row in the graph.

There is a [guide for adding a language track](ADDING-A-LANGUAGE.md) because the
tests and the parity gate make a track a *claim*. Adding a concept has the same
shape of contract — the goal is the same: nothing you write should be able to rot
silently. This document is what you have to satisfy for that to be true.

If you would rather read a finished example than a checklist, read
[`content/concepts/tokenizer.md`](../content/concepts/tokenizer.md) top to
bottom and come back when something is unclear. It uses a frontmatter block and
a `:::callout`, and it is short.

---

## 0. The loop you will work in

```sh
make validate      # schema + anchors + selector resolution (regenerates derived data)
python3 -m unittest tools.tests.test_content   # the prose/code rules
```

`make validate` is the schema-and-anchors contract. It is *not* the whole
contract: the prose and unanchored-fence rules live in
`python3 -m unittest tools.tests.test_content`, which `make validate` does not
run. Run both. `make validate` is fast and names the file and line when it
fails; the unittest is slower but catches the writing rules. The single most
common cause of a concept losing half its body is a missing `:::` closer, and
`make validate` catches exactly that.

Nothing below is optional, but only the first section is *mechanical*. The rest
is the part that decides whether the page teaches.

---

## 1. The frontmatter

Every concept is YAML frontmatter between `---` fences, then a Markdown body.

```yaml
---
id: tokenizer
title: The tokenizer
chapter: data
order: 2
difficulty: 1
summary: "Three lines turn arbitrary strings into integers and back again: the sorted set of every character in the dataset becomes the vocabulary, and one extra id is reserved for end-of-word."
math: |
  V = \mathrm{sort}\!\big(\mathrm{set}\big(\textstyle\bigoplus_{d \in \mathrm{docs}} d\big)\big)
shapes:
  - { name: uchars, shape: "[26]" }
anchors:
  python: { selector: "section:Let there be a Tokenizer" }
  c: { selector: "def:load_data offset:[27, -9]" }
related: [data-loading, cross-entropy-loss, inference-sampling]
prereqs: [data-loading]
---
```

The authoritative schema is [`content/schema.json`](../content/schema.json); it
is read by `tools/validate_concepts.py`. Every field below is checked against it,
and the frontmatter parser is [`tools/yamlite.py`](../tools/yamlite.py), a
**strict subset** of YAML — no anchors, no multi-document, no flow style beyond
the inline maps and lists shown here. If a mapping looks odd and `make validate`
accepts it, it is because `yamlite` is deliberately small.

| field | required | what it is, and the rule that bites |
| --- | --- | --- |
| `id` | ✅ | Filename stem, and must **equal** it. Lowercase slug: `^[a-z0-9]+(-[a-z0-9]+)*$`. Globally unique. |
| `title` | ✅ | Human title, 3–80 chars. |
| `chapter` | ✅ | One of `data`, `machinery`, `architecture`, `training`, `inference`. Must exist in `content/index.json`. |
| `order` | ✅ | Position within the chapter, 1-based. Must **match the position in `index.json`'s list** — the curriculum and the graph are required to agree. This means the value is whatever slot you *append* to; you do not get to pick a "nice" number, and you must not reuse a number that is taken. If `index.json` already lists four concepts, yours is `5` — see §6. |
| `summary` | ✅ | One or two sentences, 20–400 chars. This is the graph node label and the meta description, so it has to work out of context. Write it last, once you know what the page says. |
| `anchors` | ✅ | Where this concept lives in each language. `python` is required; the others are optional. See §3. |
| `related` | ✅ | Concept ids shown as "see also". A reader may visit these in any order. No cycles are implied — `related` is a hint, not an order. |
| `prereqs` | ✅ | Concept ids that should be read first. Drives the chapter nav and the graph edges. **Cycles are rejected.** |
| `difficulty` | — | `1` skimmable, `2` the main event (default), `3` will make you think. |
| `math` | — | KaTeX/LaTeX for one display block. No `$` delimiters — the app supplies them. |
| `shapes` | — | Tensor shapes at this hyperparameter setting, as `{ name, shape }`. `shape` is a free string like `"[T, 4]"`, and it must be true, not a reassurance. |
| `traces` | — | Recorded-run widgets declared in frontmatter. The inline `:::trace` form in §4 is more common. |

An empty `related` or `prereqs` is written `[]`. A prereq that names a concept
that does not exist fails `make validate` — that is the point, a "see also" to
nowhere is worse than no link.

---

## 2. The body, and the one rule that catches everyone

The body is Markdown. The app renders it, and it adds three things the Markdown
does not hint at: **directives** (`:::` lines), **language tabs**, and
**anchored code fences**.

### Untagged fences are a promise

A fenced code block with **no info string** claims to be a *verbatim excerpt of
this concept's own anchor range*:

````md
```
uchars = sorted(set(''.join(docs)))
```
````

The fence above is checked line by line against the lines your `anchors`
selector resolves to. If a quoted line is not in that range, the test fails —
because prose that is right and a highlight that is wrong is the worst failure
this repo can have: both are rendered with total confidence.

The rule in practice:

- Code **from your anchor range** → leave the fence untagged (```` ``` ````).
  Only an untagged fence is checked against your range; that is what makes the
  highlight trustworthy.
- Code **from elsewhere in the file**, quoted for comparison → tag it
  ```` ```python ````. A tag (any tag) means "this is not the anchored region",
  so a ```` ```python ```` fence is *not* range-checked. Do not tag a fence you
  mean as verbatim; tagging it silently removes it from the check.
- Arithmetic, a diagram or a table → ```` ```text ````.
- A captured run's output → ```` ```output ````.
- An elision you are quoting around → use `...` on its own line. Normalisation
  strips trailing comments and trailing commas, so you can quote a source line
  without its `# comment`, but you cannot paraphrase it.

`tools/tag_fences.py` will classify untagged fences for you if you are unsure,
but it only *reports*: read its answer, do not trust it blindly.

### A body has to actually teach

`test_every_body_has_prose` requires more than **500 characters of prose** —
text that is not code. A page that is mostly a code fence and a one-line
summary fails this gate; the walkthrough for this guide tripped it at 314
characters. The prose is the part that explains *why*, so write it before you
trim the fences.

### Headings end callouts

A `##` heading inside an open `:::callout` is an error, not a silent boundary.
See below for why.

---

## 3. Anchors: how a concept points at code

Never write a line number. Write a **selector**, and `tools/gen_anchors.py`
resolves it against the real source at build time.

| selector | means |
| --- | --- |
| `def:<name>` | the body of the function/method named `<name>` |
| `assign:<name>` | the smallest statement or declaration that binds `<name>` |
| `comment:<exact prefix>` | the single line whose comment starts with `<prefix>` |
| `section:<exact prefix>` | that comment through to the next section boundary |
| `offset:[a, b]` | deltas applied *after* resolution |

```yaml
anchors:
  python: { selector: "def:gpt offset:[15, -10]" }
  c:      { selector: "def:forward_pos offset:[16, -6]" }
```

`offset` is **relative, not absolute**, on purpose: `def:gpt` on the reference is
lines 107–143, and `offset:[15, -10]` narrows that to 122–133 without anyone
writing 122. `a` is added to the resolved **start**; `b` is added to the
resolved **end**. So:

- A **negative `b`** trims inward from the end. `[15, -10]` means "start 15 lines
  after the top, end 10 lines before the bottom".
- A **positive `a`** moves the start down.
- The offset may be written inline in the selector (shown here) or as a sibling
  `offset: [a, b]` key; both are accepted.

Offsets are validated, not best-effort. An offset that pushes the start before
line 1, pushes it past the end of the file, or inverts the span (end before
start) is a hard error with the concept id and the offset in the message. The
**one** silent case is an oversized *end* offset, which is clamped to the last
line — that is the documented `clamp` contract, so use it deliberately.

`section:` exists because five concepts describe a whole chapter-sized region,
and a numeric offset to reach the end of one would be the hardcoded line number
this design exists to remove.

A concept fails `make validate` if it names a language with no source file. If
the language track does not exist yet, drop the anchor; do not point at a file
that is not there.

---

## 4. Directives

A directive is a line beginning with `:::`. There are seven. Each either applies
to a single line (inline) or opens a block that **must be closed with a bare
`:::` on its own line**.

| directive | kind | syntax | notes |
| --- | --- | --- | --- |
| `:::trace` | inline | `:::trace kind=loss-curve` `:::trace kind=attention step=0 pos=3 head=0` | Inlines a committed widget. `kind` is one of `loss-curve`, `attention`, `softmax`, `shapes`, `tokens`, `samples`; other args are `step`, `pos`, `head`, `top`, `lang`, `config`. See [TRACE-FORMAT.md](TRACE-FORMAT.md). Args are range-checked against the recorded run, so a widget that would render empty fails the build. |
| `:::term` | inline | `:::term BOS` | Links a glossary term. **Fails if the term is not in `content/index.json`'s `glossary.terms`.** You may therefore have to add the definition there as part of adding the concept. |
| `:::shape` | inline | `:::shape uchars` | Shows one `shapes` entry by name. |
| `:::callout` | block | `:::callout "Optional title"` … `:::` | A boxed aside. The first argument, if double-quoted, is the title. |
| `:::note` | block | `:::note` … `:::` | A quieter boxed aside. |
| `:::lang` | block | `:::lang config=python` … next `:::lang` or end of body | Starts a language-tab range. A `:::lang` range is closed by the **next `:::lang`**, not by a bare `:::`. |

Three rules fall out of this, and each one is a real bug that shipped once:

1. **Close your blocks.** A `:::callout` or `:::note` without its closing `:::`
   is the most common concept bug. When it happens, the parser would otherwise
   swallow everything to the next bare `:::`, silently deleting a heading.
2. **An inline directive cannot be the body of a block.** If you see
   `:::trace` / `:::term` / `:::shape` inside an open callout, the block is
   missing its closer — the parser says so instead of swallowing the widget.
3. **A `##` heading cannot appear inside an open block.** Same reason.

A near-miss like `::::trace` (too many colons) is rejected rather than rendered
as a paragraph of colons. Unknown directive names are rejected with the list of
known ones.

---

## 5. Register the concept

A concept file is invisible until it is listed in
[`content/index.json`](../content/index.json). Add its id to its chapter's
`concepts` array, at the position matching its `order`. `make validate` checks
that the frontmatter `order` and the `index.json` position agree — the
curriculum and the graph are not allowed to disagree.

If your concept uses `:::term`, add each term to `glossary.terms` in the same
file. The value is the definition; the key is the exact string in the directive.

Running `make validate` also regenerates the site's derived data
(`visualizer/src/data/generated/anchors.json`, `index.json`, `content.json`).
Those are checked in, so they will show up as modified — commit them with the
concept. You do not hand-edit them.

### Link it from `docs/HOW-TO-READ.md`

[`docs/HOW-TO-READ.md`](HOW-TO-READ.md) is the file-by-file map of the
implementation, and `tools/tests/test_content.py` requires that **every concept
links to it and every row there resolves to a real concept**. Add a row for
yours, in the same `| start–end | what | concepts |` shape as its neighbours:

```md
| 78–78 | head dim | [Head dim](content/concepts/head-dim.md) |
```

Two rules the test enforces, both easy to get wrong:

- The row's link set and the corpus must match **exactly** — a row that links a
  concept the corpus lacks (or a concept with no row) fails. If you fold your
  concept into an existing row, put its link *inside that row's* cell; do not
  append a stray link to a neighbouring row, which corrupts both.
- If your row cites a single concept, the `start–end` numbers must be **exactly**
  the selector's resolved span (`make validate` prints it). A row that spans
  several concepts is exempt from the span check, because the span is not one
  concept's span.

### Bump the corpus size guard

`tools/tests/test_content.py` contains a `test_there_are_eighteen_concepts`
count guard — a deliberate tripwire so that adding or removing a concept is a
conscious act, not an accident. When you add one, raise the expected count in
that test (and its name's number) in the same commit. This is the one step that
exists purely to make you stop and notice.

---

## 6. A worked example, end to end

Small, but it exercises every gate. Suppose you want a concept for the
`head_dim` arithmetic — "attention heads are just a split of one wider vector".

**1. Create the file.** `content/concepts/head-dim.md`:

```md
---
id: head-dim
title: Heads are a split, not a copy
chapter: architecture
order: 10
difficulty: 2
summary: "n_head × head_dim is n_embd: the attention heads are a reshape of one wide vector, not four separate models, and that is why the projection is a single matmul."
anchors:
  python: { selector: "assign:head_dim" }
related: [multi-head-attention]
prereqs: [multi-head-attention]
---

`head_dim` is not a second dimension the model learns. It is `n_embd / n_head`,
fixed by the shape of a single projection:

```
head_dim = n_embd // n_head
```

That is why the attention block does one matmul and then reshapes, rather than
running four small matmuls. The heads share nothing but the reshape: the same
weights produce all of them, and the only thing that distinguishes head `h` from
head `h+1` is which slice of the output it reads. When you hear "multi-head
attention", it is tempting to picture four separate attention mechanisms whose
results are concatenated. The arithmetic says otherwise. There is one projection
of width `n_embd`, and `head_dim` is the width each head gets when that vector
is cut into `n_head` equal pieces. Change `n_head` and you are not adding a
model, you are changing the grain of a reshape.
```

**2. Pick the anchor.** `assign:head_dim` resolves to line `78` of
`reference/microgpt.py`, the statement that binds `head_dim`. The quoted
`head_dim = n_embd // n_head` is a verbatim line inside that range, so its
fence is left **untagged** (bare ```` ``` ````) and the test checks it against
the range. `make validate` prints the span it resolved (`78–78`); keep it for
step 3.

**3. Register it.** In `content/index.json`, append `"head-dim"` to
`architecture`'s `concepts` array — *at the end*, because test and graph both
require `order` to equal the position in that array. The chapter already lists
nine concepts (`linear` … `gpt-forward`), so the new position is `10` and the
frontmatter says `order: 10`. Reusing `4` because it "looks right" fails with
`must agree and be 1..N`: the value is the append slot, not a free choice.

**4. Link it from `docs/HOW-TO-READ.md`.** Add a row whose span is exactly the
selector's `78–78`, and whose cell links this file:

```md
| 78–78 | head dim | [Head dim](content/concepts/head-dim.md) |
```

**5. Bump the corpus guard.** Raise the expected count (and the number in its
name) in `test_there_are_eighteen_concepts`.

**6. Validate.**

```sh
make validate
python3 -m unittest tools.tests.test_content
```

Both pass. The page is live at `#/learn/head-dim`, appears in the architecture
chapter nav after `gpt-forward`, and has a graph node.

**7. If a gate fails, read the message.** It names the file, line and rule. The
common ones:

| message | fix |
| --- | --- |
| `:::trace` widget … `(step, pos, head)` not in the trace | quote a combination the recorded run actually has; see TRACE-FORMAT.md |
| `:::term X` is not defined in index.json's glossary | add `X` to `glossary.terms` |
| quoted line … is not in the anchor range | fix the fence tag, the quote, or the `offset` |
| `:::callout` block is never closed with `:::` | add the bare `:::` |
| `order` … does not match index.json | make the frontmatter `order` and the `index.json` position agree |
| `49 not greater than 500` | write more explanation; the body needs > 500 characters that are not code |
| HOW-TO-READ links … but the corpus has … | link your concept from `docs/HOW-TO-READ.md` (or fold it into a row that already exists; do not append a stray link to a neighbour) |

---

## 7. Definition of done

A concept is done when:

1. `make validate` exits 0.
2. `python3 -m unittest tools.tests.test_content` passes.
3. Every **untagged** code fence quotes only lines inside your anchor range.
   (A tagged fence — `python`, `text`, `output` — is not checked, so tag an
   excerpt you do *not* mean as verbatim.)
4. The body has more than 500 characters of prose (text, not code).
5. The concept is linked from `docs/HOW-TO-READ.md`, and the corpus-size guard
   in `test_content.py` has been bumped by one.
6. The `summary` reads as a standalone sentence — it is what a reader sees in
   the graph and in a search result.
7. Every `related` and `prereqs` id names a concept that exists.

If all seven hold, the page is as safe to trust as the loader claims.
