# Benchmarks, and how to read them

<!-- Generated from benchmarks/results.json by `make bench`; the table below is
     checked by tools/tests/test_content.py. -->

Measured on **Apple M1** (Darwin 25.1.0, 8 cores,
8.0 GB), 1000 steps at the micro config, on
2026-09-26T19:45:35Z.

| track | steps/sec | vs Python | built as |
| --- | --- | --- | --- |
| c | 2,343 | 271.48x | native (NEON + Accelerate) |
| typescript | 56 | 6.54x | tsx |
| rust | 53 | 6.17x | release |
| go | 41 | 4.79x | go run |
| python | 9 | 1.0x | reference |

## Read this before you read a number

**A number from your laptop is not a leaderboard entry.** These are wall-clock
times on one specific machine, and the only honest comparisons are *ratios
between tracks measured in the same run*. A 2023 laptop and a CI runner differ by
more than any algorithmic difference below.

What the ratios are good for is showing **where the time goes**, and that is the
actual lesson.

## Three things this table makes concrete

**1. Autograd's cost is the Python file's dominant cost, not the arithmetic.**

Rust and Go run the *identical* tape-based algorithm in about 6x the speed, with
no special libraries. A compiled language and an interpreted one land within a
few percent of each other. So the roughly 15,000 `Value` objects allocated and
freed per step *are* the bottleneck, and the C port is fast because it allocates
nothing and does the same sums in registers.

**2. The data structure is a much stronger lever than the language.**

Rust is a systems language; TypeScript here runs under `tsx`. They share almost
nothing, and they are within a few percent of each other. What they share is a
tape, rather than a fresh object per operation. "Which language" is a much weaker
question than "what is allocated per operation".

**3. The fast path is worth about 45x over the same C code, scalar.**

`implementations/c/test_equivalence.sh` builds the C port three ways and checks
they produce the same loss curve:

| configuration | where | speed |
| --- | --- | --- |
| Accelerate + NEON | Apple Silicon | fastest |
| scalar BLAS + NEON | Apple Silicon, no framework | intermediate |
| scalar BLAS + scalar SIMD | anywhere, incl. Linux CI | about 45x slower than the first |

The three agree to within **0.11%**, which is float32 rounding rather than a
semantic difference: `cblas_sgemv` and a naive loop sum a 16-element dot product
in different orders, the last bits differ, and 1,000 steps of a chaotic training
process amplify that. All of the C port's speed is NEON, Accelerate, and not
allocating -- none of it a different algorithm.

## What is deliberately not here

- **No `scaled` config row.** `microgpt-scaled.c` is n_embd 256 / 4 layers /
  block 256 against 16 / 1 / 16 -- roughly two orders of magnitude more work.
  Reporting it next to the others would compare two different problems. Run
  `make run-scaled` to see it.
- **No peak RSS.** Meaningful numbers need `/usr/bin/time -l` or a profiler, and
  a number produced by a best-effort path is worse than an absent one.
- **No "vs PyTorch" row.** This repository has no PyTorch dependency on purpose,
  and a number from a machine without it, measured today, would be fiction.

## The measurement, precisely

Wall-clock time for the whole training loop, including the reference's per-step printing. No warm-up run and no repetition: a single measurement is honest about being a single measurement. The Python figure is dominated by output volume rather than arithmetic, so the Python-vs-others ratio should be read as an upper bound on the gap.

And the caveats the tool records:

- Ratios between rows are meaningful; absolute seconds are machine-specific.
- The Python row includes printing 1,000 lines, which the C and Rust rows also pay and the trace-based runs do not.
- Go and TypeScript are run without tuning flags on purpose. With -O flags, a Go binary instead of `go run`, or esbuild instead of tsx, those rows would improve substantially.
- The `scaled` config is deliberately absent: it is a different model, so comparing it here would compare two different problems.

## Reproducing it

```sh
make bench       # measures every track, records the machine, writes results.json
```

The file is committed so the site can show it without running anything. A weekly
scheduled job regenerates it and opens a PR, because a benchmark change is a
thing a human should look at: "the benchmark went down by 3%" and "the benchmark
is now measuring a different thing" look identical in a diff.

## The related question: is it the *same* algorithm?

Speed is the easy half. The parity gate checks the other half, and its design is
worth stating because the obvious version does not work.

The gate compares an exponentially-smoothed loss at fixed checkpoints, and
separately requires the windowed trend to improve by at least 10%. The obvious
version -- "loss strictly decreasing, within +/-10% of the reference at steps 50
and 200" -- **cannot pass, by its own reference**:

| measured on the reference, 1,000 steps | value |
| --- | --- |
| mean | 2.4517 |
| standard deviation | 0.3920 |
| minimum (step 537) | 1.5782 |
| maximum | 3.9066 |
| **step-to-step moves that increase** | **500 of 999** |
| best-to-worst spread | 95% of the mean |

Half the steps go *up*, because the training loop reads
`doc = docs[step % len(docs)]` -- one document per step, no batching -- so the
printed loss is a single sample, not an estimate of anything. And a +/-10% band on
a raw single-step value is inside the reference's own noise floor.

Bit-exact parity was never available either: the reference seeds Python's
Mersenne Twister while every other track uses xoshiro256++, and CPython uses
float64 where C and Rust use float32. So the question is not "do the numbers
match" but "do the runs learn the same thing", and that is what the gate asks.

Measured, all four tracks at step 200: C +4.3%, Go -7.1%, Rust +1.5%,
TypeScript +0.0%, against a +/-10% band. Trends +10.3% to +18.8%, against the
reference's +18.5%.

**And what the gate cannot catch:** a mis-scaled gradient. Adam divides by an
estimate of the gradient's own magnitude, so a gradient that is wrong by a factor
barely moves the step. The C port's gradient is wrong by a factor of -0.12 and
its loss curve still passes. That is why every track needs a
directional-derivative gradient check as well -- see
[`docs/ADDING-A-LANGUAGE.md`](ADDING-A-LANGUAGE.md) requirement 5 and
[`docs/KNOWN-ISSUES.md`](KNOWN-ISSUES.md) issue 1.
