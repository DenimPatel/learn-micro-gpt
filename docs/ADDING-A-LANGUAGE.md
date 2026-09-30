# Adding a language track

A contract, not a guide. The tests and the parity gate are what make a new track
a *claim*; this document is what you have to satisfy for the claim to be true.

There is prior art in other languages — `xenova/microgpt.js`,
`prasad83/go-microgpt` — and reading one is a reasonable way to start. Crediting
whatever you build on is a requirement, not a nicety; see `CREDITS.md`.

---

## The contract

Six requirements. Four of them are mechanical, and the fifth is the one that
matters.

### 1. Same hyperparameters, declared not assumed

```
n_embd 16   n_head 4   n_layer 1   block_size 16   head_dim 4   vocab_size 27
```

`n_embd` and `head_dim` are related by definition. Derive one from the other in
your code rather than writing both.

The parity gate does not read your config from a file. It runs your program and
parses its output, so **the run must cover at least 200 steps** for the trend
check to mean anything.

### 2. Print the same line, every step

```
step <n+1> / <total> | loss <loss to 4 dp>
```

That single format is what `tools/parity.py` parses. A track that prints only
the first five steps and then every hundredth looks correct in a terminal and
yields a 7-point "curve" to a machine — which is exactly what Go and TypeScript
did before they grew a `--trace` output.

### 3. A trace output, or every step printed

`--trace <path>` writes JSONL, one object per line:

```json
{"type":"meta","lang":"<name>","steps":1000,"seed":42}
{"type":"step","step":0,"loss":3.3660}
...
{"type":"sample","step":1000,"text":"kamon"}
```

`tools/parity.py` reads loss values from this when the track declares
`parse: "trace"` in its runner. See `TrackRunner` in that file.

### 4. Take the dataset path as a flag, and read nothing else

```sh
your-track --input ../../data/input.txt --steps 1000 --seed 42
```

**Do not** resolve the dataset from the current working directory, and **never**
download it. The Python reference does both, which is issue 3 in
`docs/KNOWN-ISSUES.md`; do not copy it. The parity runner passes `--input`
explicitly and runs you in a scratch directory, so a track that reads
`input.txt` from the CWD will work by accident locally and be a source of
confusion in CI.

### 5. Verify the gradient, or do not claim parity

This is the requirement that matters.

A loss curve is evidence that something learned. It is **not** evidence that the
thing that learned was the gradient. The C port is the measurement that
established this, and it is worth knowing what happened to it: its
hand-written backward pass *was* wrong by a factor of −0.12, and **its loss
curve still tracked the reference to within 7% and it still trained** — because
Adam divides by an estimate of the gradient's own magnitude, so a gradient that
is wrong by a factor barely moves the step. The parity gate passed the whole
time. That number is the reason every candidate in this repository goes through
a finite-difference probe before it is allowed to compete, and it is why
requirement 5 exists rather than the loss-band check.

The C port is now fixed, at 1.03 — inside the float32 noise floor of the
difference itself — and it is the one track here that satisfies this
requirement. It is worth being precise about what that costs, because it is the
part a hand-written backward pass does not give you: nothing in the code says
whether it is right. `docs/KNOWN-ISSUES.md` issue 1 has the measurements, the
two bugs, and the number the test now asserts.

So before your track is allowed to say "parity", it needs a
**directional derivative check**: with a fixed pseudo-random direction `d` over
the whole parameter vector, confirm

```
<gradient, d>  ==  (L(w + h*d) - L(w - h*d)) / 2h
```

to within a few percent, at two step sizes.

Two traps, both of which cost real time here:

- **Per-parameter finite differences mostly do not work.** A float32 loss of ~3.2
  has an absolute resolution of about 4e-7, so a central difference cannot
  resolve a gradient below roughly 2e-4 at `h = 1e-3`. Most attention and MLP
  gradients at initialisation are 1e-5 to 1e-2. Check the *aggregate*, where
  both sides are O(1).
- **Fix the direction.** Generate it from your own PRNG or a fixed LCG, not from
  whatever your program happens to have consumed. It is checked on every run, so
  it has to be the same direction every time.

The cheapest way to be correct by construction, and the route the Go, Rust, and
TypeScript tracks took: **reimplement the reference's autograd tape** rather than
hand-writing a backward pass. A tape computes the chain rule; a hand-written one
is somebody remembering it. The cost is that you then need a small arena or
`RefCell`, and your code will not be as direct as the Python's. The benefit is
that the check above passes the first time.

### 6. Ship a test that can fail

The `test_gradients.c` shipped here replaced an older file of that name that
printed a hardcoded number and returned 0. It could not fail, which is worse than
having no test: it looked like evidence.

At minimum:

- softmax normalises, and survives a logit of 1e30
- the loss is finite and positive at initialisation
- attention rows sum to 1 and are never longer than the causal window
- a gradient check that would catch a wrong derivative

## Wiring it in

### 1. The source

Put it at `implementations/<name>/`, with a top-level comment explaining two
things: **why your port looks like the reference** (it should — the reference's
model code is unremarkable and the interesting part is autograd) and **what your
language made you do differently** (no operator overloading; GC; borrow checker;
`valueOf`; whatever applies). That second part is the actual content of a port.

If it needs a dependency, justify it. Three tracks here have none, and the
stated reason is that a port needing a BLAS binding would be measuring the
binding.

### 2. The selector targets

The anchor resolver needs to find your language's functions. For a new language,
add a resolver in `tools/selectors.py` following the `c_like_spans` pattern: a
function-definition regex, logical-statement joining for multi-line signatures,
and brace matching. Add it to `SOURCES` and to `TRACK_LANGUAGES`.

Then add `anchors.<name>` to at least one concept and check it resolves:

```sh
make validate          # an unresolvable selector is a build failure
```

Concepts without an anchor in your language are fine — they render without your
tab. Concepts *with* one that does not resolve are a build failure, which is the
point.

### 3. The parity runner

In `tools/parity.py`, add a `TrackRunner`:

```python
TrackRunner(
    lang="<name>",
    config="micro",
    source=REPO_ROOT / "implementations/<name>/...",
    run=["<name>", "--input", "{input}", "--steps", "{steps}", "--trace", "{trace}"],
    tools=("<name>",),        # tool *names*, resolved against PATH
    cwd=REPO_ROOT / "implementations/<name>",  # if you need your module root
    parse="trace",            # or "stdout" if you print every step
)
```

`{input}`, `{steps}` and `{trace}` are substituted; the run happens in a scratch
directory with `data/input.txt` copied in, unless `cwd` says otherwise.

### 4. The Makefile and CI

Add a build target, a `run-<name>` target, and the track to `make parity` and the
`tracks` job. In CI, set the toolchain up in that job — a track whose toolchain
is missing is *skipped* locally with a message, and silently skipping in CI is
how a track rots into not being tested at all.

### 5. The Makefile help

`make help` lists `run-<name>` for every other track. Add yours. The help text is
the first thing anyone reads.

### 6. Provenance

If your port derives from someone else's, add an entry to
`tools/provenance.py` and re-run `make provenance`. Set `derived=True` for an
adapted file: its digest is recorded but not pinned, because you will change it.
`CREDITS.md` and `docs/PROVENANCE.md` are generated from that table, and CI fails
if they are stale.

## The parity gate, and what it will and will not catch

It checks two things:

1. **Trend.** The mean over the last 50 steps must beat the mean over the first
   50 by at least 10%. The reference achieves 18.5%.
2. **Smoothed agreement.** An EMA (alpha 0.05) at fixed checkpoints, within ±10%
   of the reference's smoothed value. Measured Python-vs-C: +3.7%, −1.2%, +4.3%,
   +2.2%, +7.0% at steps 50/100/200/500/1000.

And it explicitly *cannot* catch a mis-scaled gradient, because Adam is nearly
scale-invariant per parameter. That is what requirement 5 is for, and it is why
that requirement exists rather than the loss-band check.

The original gate specification was "loss strictly decreasing, and within ±10%
at single steps 50 and 200". Neither half is satisfiable by the reference itself:
it rises on 500 of its 999 steps, and its per-step standard deviation is 16% of
its mean. `tools/parity.py` and `docs/BENCHMARKS.md` have the arithmetic, and two
tests assert the measurements so the original version does not get reintroduced.

## Definition of done

- [ ] `make run-<name>` works from a clean checkout
- [ ] `--input`, `--steps`, `--seed`, `--trace` all accepted
- [ ] the per-step line matches the format above, or the trace carries the losses
- [ ] a directional-derivative gradient check that **can fail**, and passes
- [ ] unit tests that can fail
- [ ] at least one concept anchored, resolving
- [ ] in the parity gate, and passing it
- [ ] in the `tracks` CI job, with its toolchain installed there
- [ ] in `make help`
- [ ] provenance recorded, if anything is derived
- [ ] `make check` green
