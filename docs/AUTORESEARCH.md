# The autoresearch loop

A model rewrites the Rust track. A harness measures it. A rule decides whether the
rewrite is kept. Every attempt &mdash; kept, discarded, and crashed &mdash; is
committed, and the GitHub Page draws the result.

```sh
make autoresearch-rust EXPERIMENTS=5
```

This descends from [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
(MIT, © Karpathy). Everything that differs here is listed under **Divergences**
below, with the reason.

---

## What it does, and what it will not do

**It will not touch the frozen track.** `implementations/rust/src/lib.rs`,
`Cargo.toml` and `src/main.rs` are pinned by sha256 in `tools/provenance.py`, and
the `provenance` CI job &mdash; which every deploy depends on &mdash; fails if a
byte moves. The tuned code lives at `autoresearch/candidate/src/lib.rs`. The site
renders the frozen one, 18 concepts cross-link to it, and its own doc comment
claims a parameter count; a model rewriting it would make the site teach something
false with total confidence, which is the failure mode the rest of this repository
exists to prevent.

**It will not publish a number it cannot back.** `results.json` is regenerated
from the ledger by CI and any diff fails the build. The candidate's source digest
is recorded with every run, and `--verify-only` re-checks it.

**It will not let a broken gradient win on loss.** See below. This is the whole
point of the gradient probe.

**It will not decide.** `verdict()` is a pure function of measured medians. The
model proposes an edit and never states an outcome.

## The gradient probe, and why it is the important part

`docs/KNOWN-ISSUES.md` issue 1: the C port's hand-written backward pass has a
directional derivative that is **−0.12×** the true value, and **its loss curve
still tracks the reference to within 7%, and it still trains**. Adam divides by an
estimate of the gradient's own magnitude, so a gradient that is uniformly wrong
barely changes the step.

**A loss number cannot tell a correct gradient from a wrong one.** Only a finite
difference can. So every candidate goes through
`autoresearch/candidate/tests/gradient_check.rs` before it is allowed to compete
on loss at all: the directional derivative of the loss over all parameters, against
a central difference at two step sizes, where a correct gradient reads 1.0.

That probe is written by this repository, not by the loop, and its sha256 is
verified before every experiment. A candidate cannot ship a weakened probe next to
a weakened gradient, because the two would agree with each other.

**It found something on its first run.** The Rust port's `rmsnorm` reads tensor
*values* rather than handles, so the whole normalisation sits outside the tape and
its gradient is **6.32% high**, stably, across four orders of magnitude of step
size. That is issue 5, and it is measured rather than argued. The loop's first
experiment put `rmsnorm` on the tape; the ratio went from **1.0632 to 0.9999**,
and the experiment was then *discarded* because it cost 4.8% throughput for a 0.1%
loss improvement. Both halves of that sentence are the system working.

The band is 0.5–2.0 rather than something tight around 1.0, and the probe's module
docs say why at length: the honest answer for a correct tape *in this crate* is
1.063. What catches a broken tape is the factor, not the precision &mdash; −0.12, 0,
and any sign flip all fail by 4× or more. The check that does the real work is the
**stability** one: a ratio that does not move when the step size moves is a
property of the tape rather than of the difference operator.

## The objective: two axes, not one

A candidate is **kept** if either

- the loss falls by at least **2%** while speed stays within **5%** of the session
  baseline, **or**
- speed rises by at least **10%** while the loss stays within **5%**,

and it learned (windowed loss fell by at least 10% overall). A tie is a discard.

The four numbers are not the same number four times, and the reason is worth
stating because the first version of this wrote it wrong:

- The two **"at least"** values are noise floors. The reference's per-step loss
  has a standard deviation of 0.392 on a mean of 2.4517 &mdash; 16% noise, because
  each step is a different document. Over a 50-step window the standard error
  falls to about 0.055, roughly 2.2% of the mean, so 2% is about one standard
  error and 5% is about two. Loss gets small thresholds because a fixed seed
  makes it **deterministic**: the PRNG is xoshiro256++ with SplitMix64 seeding, so
  a given candidate produces a given loss curve or it does not.
- The two **"within"** values are regression limits, and they are much larger on
  the speed axis because wall clock moves with whatever else the machine is doing.
  10% is an unambiguous speedup; 2% is not.

There is no ordering between the two kinds, and there should not be: a 1% *gain* is
inside the noise and cannot be trusted, while a 5% *regression* is two standard
errors and can be. `tools/tests/test_autoresearch.py` asserts that they stay
distinct and that no arrangement of the other axis rescues a 10% loss regression.

## Why the baseline is re-measured every session

`benchmarks/results.json` says the Rust track runs at 53 steps/sec. This
repository's CI runner is `ubuntu-24.04`; the benchmark was taken on an Apple M1.
On a modern laptop the same code measures **95 steps/sec**.

A speed ratio against a number from another machine is a number about that machine.
So the first thing a session does is rebuild the frozen track and run it three
times, and everything after that is compared to *that*. `autoresearch/baseline.json`
records the result along with the runner identity, and the page shows the runner
next to the numbers, because a speed figure without its machine is not evidence.

**This means speed numbers are only comparable within one session.** The page says
so, `results.json` carries it as a caveat, and `benchmarks/results.json` carries
the same objection for the same reason.

## What each run costs

| | |
| --- | --- |
| loss axis | one 1,000-step run, ~10 s |
| speed axis | 3 runs of the same, ~32 s |
| build + clippy + probe | ~10 s after the first |
| model call | 1&ndash;2 min at `reasoning_effort: max` |
| **per experiment** | **roughly 1&ndash;2 minutes** |

The model is priced at 0 prompt and 0 completion as of the verification date in
`autoresearch/model.json`, so a session costs tokens rather than money. The run
record still logs token usage, because a free model today is not a promise about
tomorrow.

## Before your first run

```sh
export OPENROUTER_API_KEY=...   # must be visible to the shell you run make from
make autoresearch-rust EXPERIMENTS=1 PUSH=0   # one experiment, committed locally
```

`PUSH=0` matters for a first run. The loop pushes one commit per experiment, and
CI runs `provenance`, `content`, `tracks` and `web` &mdash; about 15 minutes of
wall clock each. Twenty experiments is twenty CI cycles. And `ci.yml` sets
`concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }`, so a
burst of pushes **cancels the earlier runs**: the last push is what deploys, and
intermediate experiments are never validated at all. `PUSH_DELAY=90` spaces them.

The loop also refuses to start if there are uncommitted changes to tracked files
outside `autoresearch/`, or if you are not on `--branch` (default `main`). It only
ever stages `autoresearch/`, so it cannot sweep up your work, but it would rather
not measure a tree you did not mean.

## What you are reading afterwards

- `#/research` &mdash; the ledger, the Pareto scatter, the per-experiment gains,
  the gradient ratios, and the patch from the frozen track to the current best.
- `autoresearch/results.tsv` &mdash; the ledger, one row per experiment.
- `autoresearch/runs/0001.json` &mdash; one experiment in full, including the
  complete per-step loss curve and the model's token usage.
- `autoresearch/program.md` &mdash; **the agent brief, and the one part of this
  system meant to be edited by a human.** It is what the model reads as its system
  prompt. Changing the objective means changing it.

## The files

| path | what it is |
| --- | --- |
| `tools/autoresearch.py` | the harness: measure, decide, record, commit |
| `autoresearch/program.md` | the brief. The human's lever. |
| `autoresearch/model.json` | pinned model id and reasoning parameter, with how it was verified |
| `autoresearch/candidate/` | the tuned crate. The only thing the model edits. |
| `autoresearch/candidate/tests/gradient_check.rs` | the finite-difference gate. Ours, digest-pinned. |
| `autoresearch/results.tsv` | the ledger, append-only, tab-separated |
| `autoresearch/results.json` | generated; the only file the site reads |
| `autoresearch/baseline.json` | this session's comparator, and the machine it came from |
| `autoresearch/runs/*.json` | one experiment each, in full |
| `autoresearch/diffs/*.patch` | what each experiment proposed, and the cumulative diff |
| `autoresearch/logs/` | raw model responses. Gitignored. |

## Divergences from `karpathy/autoresearch`

| autoresearch | here | why |
| --- | --- | --- |
| 5-minute wall-clock budget, metric `val_bpb` | fixed 1,000 steps at seed 42; loss = last-50-step windowed mean, speed = steps/sec | no GPU, and a time budget makes runs incomparable across machines. Fixed steps make the loss axis deterministic; a per-session re-measured baseline makes the speed axis self-relative. |
| the model edits `train.py` on a branch, branch advanced on keep | `autoresearch/candidate/src/lib.rs`, committed on **every** experiment, keep or discard | failures have to be published or they get retried |
| the **model** reads `val_bpb` and decides keep/discard | `tools/autoresearch.py` decides, from measured medians | an LLM told "keep it if the number went down" drifts toward keeping its own bad ideas within about ten experiments, and then the log stops being evidence of anything |
| no gradient check | every candidate finite-difference checked, digest-pinned, before it competes | issue 1: the C port's gradient is wrong by −8× and its loss curve still looks fine |
| no held-out validation split | none | 1,000 documents drawn from 32,033, one per step: overfitting is not a failure mode here, and the frozen baseline cannot be made to hold anything out without editing it. The gradient probe is the anti-gaming gate instead. |
| `results.tsv` deliberately left untracked | committed | the point is to publish the negative results |
| single scalar objective | Pareto over loss and speed | "lower loss" alone is how a repository quietly ships a slower model it is proud of |

## What is deliberately not here

- **Other tracks.** `--track` takes `rust` and rejects anything else loudly.
- **A held-out split** or any other change of metric.
- **Concept anchors into the candidate.** The atlas describes the reference
  algorithm, not the tuned variant, and `anchors.json` must keep describing the
  code the concepts quote.
- **Cross-talk with the parity gate.** It is untouched. The candidate is not a
  parity track and is not gated by one &mdash; its gate is the gradient probe plus
  the two-axis rule.
- **Agent memory across sessions** beyond the ledger and the current best. The
  ledger is handed to the model in the prompt; anything more persistent is
  `program.md`, and putting it there is a human's decision.
