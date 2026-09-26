# Trace format

Every number the site shows comes from a committed trace. This is what that
means, and what a trace is allowed to contain.

## Why they are committed

A widget that computes its own numbers is a second, unverifiable implementation
of the model. The alternative — the site recomputing a loss curve to draw a
chart — would mean the chart could disagree with the prose next to it, and
nothing would notice.

So `tools/trace.py` records the numbers, the JSONL is committed, and the app
only draws. When a trace is regenerated and the numbers move, the tests fail and
the prose that quotes them gets reviewed.

```sh
make trace         # regenerate traces/
make trace-check   # regenerate and fail on any diff
```

`trace-check` is in CI. It is what stops a committed trace from quietly ceasing
to describe the code it came from.

## Layout

```
traces/<lang>/<config>/
  meta.json      one object: how the run was made, and what it cost
  steps.jsonl    the loss curve, one line per step
  attn.jsonl     attention weights, at selected (step, pos, head)
  probs.jsonl    the top-k next characters, at selected (step, pos)
  samples.jsonl  20 sampled names after training
  stdout.txt     the raw output the above were parsed from -- gitignored
```

`stdout.txt` is build output. `steps.jsonl` and `samples.jsonl` are exactly what
it was parsed into, and committing both would give the drift check two things to
disagree about.

## How a run is made

```sh
python3 -m tools.trace --lang python --config micro --steps 1000
```

1. `data/input.txt` is copied into a throwaway directory.
2. The **real** `reference/microgpt.py` is executed there, unmodified. Not an
   import, not a re-implementation — the file, run as a file, the way a learner
   would run it.
3. Two things are captured. The **loss curve and the samples** are parsed from
   stdout, which is exact and needs no cleverness. The **attention weights and
   probabilities** need instrumentation, so the AST is rewritten in memory and
   the file on disk is never touched.

### The instrumentation, in four lines

`tools/trace.py` injects a call after any assignment to a watched name:

```python
attn_weights = softmax(attn_logits)
_emit('attn_weights', attn_weights)      # <- the only thing added
```

The counters — which step, which position, which head — are read out of the
**calling frame** rather than passed in. That matters because they are not all
module globals: `step` and `pos_id` are, since the training loop runs at module
level, but `h` is a local of `gpt()`. Reading the caller's locals gets all three
at any nesting depth, and keeps working if somebody indents the code differently.

It is verified not to change the run:

```sh
MICROGPT_SLOW_TESTS=1 python3 -m unittest tools.tests.test_trace_parity
```

re-runs the reference under instrumentation and asserts the loss curve and all
20 samples are **digit for digit identical** to the committed trace. If a rewrite
ever started consuming randomness before initialisation, this is what would
catch it.

## The selection

Recorded, not arbitrary. These are the settings in `tools/trace.py`:

| setting | value | why |
| --- | --- | --- |
| `positions` | 0, 1, 3, 7, 15 | early, middle, late, and the diagonal — what makes the causal triangle visible |
| `heads` | 0, 1 | two of four; head 0 is where the finding lives, head 1 shows it is not a fluke |
| `steps` | 0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000 | untrained, just-trained, and the parity checkpoints |
| `top_k` | 8 | enough to see the model's shape, few enough to read |

Recording *all* positions and heads would be megabytes of noise nobody reads. The
resulting trace is 87 KiB, against a 2 MB per-track budget that CI enforces.

## The format

One JSON object per line, no trailing commas, no wrapper. `meta.json` is the only
pretty-printed file, because it is the one a human reads.

```jsonl
{"type":"step","step":0,"loss":3.366}
{"type":"attn","step":0,"pos":3,"head":0,"weights":[0.231,0.279,0.22,0.27],"sum":1.0}
{"type":"probs","step":200,"pos":0,"top":[{"token":"a","p":0.0412}]}
{"type":"sample","step":1000,"text":"kamon"}
```

**`attn.weights` is a flat list, not a matrix.** The query is a single position
and the only axis is position, so a 2D array would be inventing a dimension the
model does not have. Its length is always `pos + 1` — that *is* the causal mask,
structurally, with no mask value anywhere. The C port expresses the same thing as
`num_keys = pos_id + 1`.

## What the trace says that prose cannot

The most useful thing in it: **the diagonal is not the argmax**.

Measured over the recorded trace, for head 0 at every position with more than one
candidate:

| step | diagonal is the argmax | mean weight on the diagonal | mean weight on the best earlier position |
| --- | --- | --- | --- |
| 0 | 2 of 4 | 0.389 | 0.373 |
| 25 | 0 of 4 | 0.245 | 0.617 |
| 100 | 0 of 6 | 0.242 | 0.386 |
| 500 | 0 of 4 | 0.292 | 0.503 |

Training *reduces* the weight on the current position and *raises* the weight on
the best earlier one. Almost every reader's first model of attention says the
opposite, and the recorded run says otherwise from step 25 onward.

Those four numbers are quoted in
[`multi-head-attention.md`](../content/concepts/multi-head-attention.md) and
pinned by `tools/tests/test_trace_parity.py`, so regenerating the trace with
different settings fails the tests rather than quietly invalidating the prose.

## Adding a widget

1. Add a `kind` to `BlockView` in `visualizer/src/components/ConceptPage.tsx`
   and a component in `visualizer/src/components/widgets.tsx`.
2. Add the kind to the `enum` in `content/schema.json`.
3. Reference it from a concept: `:::trace kind=<name> step=0 pos=3 head=0`.

Validation rejects an unknown kind, with a message that says an unknown kind
renders nothing — which is exactly how this failed once.

## Regenerating after a change to the reference

The reference is byte-pinned, so a change to it is a deliberate act, and it
invalidates the trace:

1. Update the pin in `tools/provenance.py`.
2. Bump `PIN_EPOCH` in `tools/check_provenance.py`.
3. `make trace && make trace-check && make bench`.
4. Re-read every concept that quotes a number. The tests will fail on the ones
   that moved; the ones that moved *slightly* will not, and those are the ones
   worth looking at.
5. Say so in the pull request. Every recorded number describes the old bytes.
