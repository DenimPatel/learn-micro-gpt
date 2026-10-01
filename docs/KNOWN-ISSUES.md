# Known issues

Things this repository knows are wrong. Each one is measured, not guessed, and
each has a test that will fail if it silently changes.

The reason this file exists rather than a comment in the code: a teaching
repository that quietly ships a broken example is worse than one that does not
ship it at all. A reader who is told "the C port's backward pass is wrong, here
is the number, here is the test" can decide what to do with that. A reader who
assumes every tab on the site is correct has no way to find out.

Entries keep their original numbers after they are fixed, because the rest of
this file and the code that cites it refer to them by number. A fixed entry is
marked **fixed** and stays as the record of what the number was, what the
diagnosis was, and — where the diagnosis turned out to be incomplete — what the
measurement said instead. Issue 1 is the one that got that treatment, and it is
the reason the diagnosis is a measurement and not a guess.

---

## 1. The C port's backward pass was not a correct gradient

**Track:** `implementations/c/microgpt.c` (the parity track)
**Status:** **fixed** in `7b63330`. Kept here, and kept under its original number,
because it is the issue the rest of this file is built around and because the
diagnosis it originally carried was wrong in an instructive way.
**Severity:** was high for anyone studying gradients, low for anyone studying speed.

### What was wrong

Two independent bugs, and the one the original entry named was the smaller of
them.

**The dominant bug was an aliasing mistake in the forward pass.**
`forward_pos()` ran its residual stream through `saved_x_embed[pos_id]` directly:

```c
float *x = saved_x_embed[pos_id];
...
rmsnorm_fwd(x, saved_x_normed_pre[pos_id], N_EMBD, &saved_rms_pre[pos_id]);
memcpy(x, saved_x_normed_pre[pos_id], N_EMBD * sizeof(float));   /* overwrites the input */
```

The pre-layer rmsnorm is the only one of the three norms in the block that does
not first copy its input somewhere private — the attention and MLP norms both
`memcpy` into `saved_x_residual_*` — so it overwrote the only surviving record of
its own input with its own output. `backward_all()`'s last call then passed that
corrupted vector to `rmsnorm_bwd()` as the `x` it differentiates, so the
`coeff * x` term, which is the entire reason that argument exists, was computed
against the wrong vector.

**The second bug is the one this entry originally described, and it was real.**
The key and value gradients of *earlier* positions were never propagated back
through those positions' own computation. In the Python reference `k[t]` and
`v[t]` are `Value` objects from an earlier `gpt()` call, so the tape already
holds the route back to that position's embeddings and `backward()` follows it
for free. In C there is no tape, so somebody has to write that route down. The
old code noticed this in a comment, banked the gradients into `dk_global` and
`dv_global`, and then pushed them straight into the embeddings at the end of the
pass — skipping the attention output projection, the MLP, and every layer below.
It is now banked per key position and consumed by that position's own pass, which
is sound because the query loop walks positions downwards and no query before `t`
can read key `t`.

### How the original diagnosis was wrong, and how it was caught

The measured symptom was one number: a directional derivative over all 4,192
parameters at `-0.12x` the truth — wrong sign, order of magnitude off. The entry
blamed the missing K/V path alone. Splitting the directional derivative by
parameter block said otherwise:

| block | analytic | numeric |
| --- | --- | --- |
| `wte` | +14.182955 | -0.220489 |
| `wpe` | -4.959895 | -0.089119 |
| `lm_head` | -0.768743 | -0.769409 |
| `wq` | -0.000700 | -0.000225 |
| `wk+wv+wo` | -0.161552 | -0.161089 |
| `fc1+fc2` | +0.242907 | +0.236522 |

Every weight block was already correct except the two embedding ones, which were
out by a factor of about sixty. An error in a gradient *path* shows up in
everything downstream of it; an error in the last step shows up in one block. Two
more measurements closed it: scaling the K/V gradients by 1000 moved the total
only from 8.53 to 36.26, so the K/V path was worth about 3% of the error and
could not be the cause; and replacing `rmsnorm` with the identity in both
directions made every block match at once.

### The measurement, after the fix

```
analytic  <grad, d>  = -1.000142
numeric   (L(w+hd) - L(w-hd)) / 2h  =  -1.032066   at h = 1e-2
                                                  =  -1.006564   at h = 3e-3
```

Ratio `numeric / analytic` = **1.03**, from a previously recorded **-0.12**. The
remaining 3% is the finite difference's own float32 noise, not gradient error —
issue 2 below is about exactly that, and it is why `DIRECTIONAL_RATIO_TOLERANCE`
is 0.06 rather than something tighter.

`test_gradients.c` had this recorded as a *known-defect lock* rather than a
failing test, so that the number could not silently change and so that fixing it
would trip an assertion demanding the lock be flipped. It did, and it is now an
ordinary assertion: `the hand-written backward pass is a correct gradient, end to
end`.

### Why the loss curve never showed any of this

**The parity gate passed the whole time.** `make parity` reports the C track
within ±7% of the reference's smoothed loss with a +18.5% trend against the
reference's +18.7%. The forward pass was always correct, and the model learned.

**Adam is nearly scale-invariant per parameter.** The update is
`lr * m_hat / (sqrt(v_hat) + eps)`. If every gradient in a block is wrong by the
same *factor*, then `m_hat` and `sqrt(v_hat)` are both wrong by that factor, they
cancel, and the update is almost unchanged. So a systematically mis-scaled
gradient — and a missing gradient path is exactly that — is largely invisible to
both the loss curve and a statistical loss-band comparison. What it cannot hide
is a sign flip, which is why the measured ratio was negative at all.

This is the real argument for the finite-difference test: **a loss curve is
evidence that something learned, not evidence that the thing that learned was the
gradient.** Only a directional derivative, or an independent autograd engine, can
tell those apart.

### What the C track is and is not good for

| use | sound? |
| --- | --- |
| studying the forward pass, the architecture, the shapes | yes — the forward is correct |
| measuring the speed of this algorithm | yes — that is the point of the track |
| the `scaled` config for throughput | yes — it is excluded from parity for the same reason |
| reading as an example of manual backprop | yes — and now that it has been read by a finite difference, which is what makes that claim mean anything |
| comparing gradient magnitudes across languages | **no** |

The `scaled` config received the same two fixes in the same commit. It is not
covered by a finite-difference test — `test_gradients.c` includes `microgpt.c` and
nothing else — so the claim "its gradient is correct" rests on the two changes
being textually identical, not on a measurement. If that track is ever promoted
to a candidate, the probe is what will check it, and that is the point of issue
6's band being a tripwire rather than a precision measurement.

`docs/ADDING-A-LANGUAGE.md` requires a directional-derivative check before any new
track is allowed to claim parity. That requirement is the direct consequence of
this issue, and the C track is now the one that satisfies it.

---

## 2. `test_gradients.c` measures most gradients below the float32 noise floor

**Status:** worked around, documented, and probably not worth fixing.

In float32 a loss of ~3.2 has an absolute resolution of about 4e-7, so a central
difference `(L(w+h) - L(w-h)) / 2h` cannot resolve a gradient below roughly
`4e-7 / 2h` — about 2e-4 at `h = 1e-3`. Most of the attention and MLP gradients
at initialisation are 1e-5 to 1e-2, so per-parameter finite differences mostly
compare noise with noise.

That is why the test uses a **directional derivative** over the whole parameter
vector instead: both sides are then O(1), so the comparison has full float32
precision, and an entire missing gradient path shows up at full size rather than
being buried in each parameter's own noise.

The per-parameter check is still run for `lm_head[3]`, where the gradient is
large enough to resolve.

---

## 3. The Python reference downloads its dataset if it is missing

**Track:** `reference/microgpt.py`
**Status:** cannot be fixed without modifying a byte-pinned file.

Lines 15–20 do:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = '.../makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
```

Two problems, both real: the file is resolved from the **current working
directory** rather than the script's directory, and the URL points at a mutable
branch rather than a commit.

`data/input.txt` is committed (byte-identical to `karpathy/makemore@988aa59`,
sha256-pinned), and `make run` copies it into a scratch directory and executes
the reference there, so the `os.path.exists` check short-circuits and nothing is
downloaded. The bug is unreachable through the Makefile.

It is still reachable by running `python3 reference/microgpt.py` from a
directory that has no `input.txt`. That is worth knowing about before you do it.

**Why it was not "fixed":** the fix would be to change the file, and
`reference/microgpt.py` is byte-pinned to upstream on purpose. It is the
reference track: the browser runs it, the parity gate measures against it, and
the concept anchors are resolved against it. Changing it would make every
recorded number in this repository describe code that no longer exists upstream.

---

## 4. `backward_all` uses a `static` initialisation flag

**Track:** `implementations/c/microgpt.c`
**Status:** **fixed** in `7b63330`, as a side effect of fixing issue 1.

The K/V gradient accumulators used to be zeroed when
`pos == n - 1 && !dk_dv_initialized`, with `dk_dv_initialized` reset at the end
of the pass — a "first call" condition and its reset about 60 lines apart, which
a future early return between them would have turned into stale accumulators
contributing to the next document's gradients.

The rewrite of the K/V path removed the flag entirely. The per-key banks
(`dk_pending`, `dv_pending`) are now cleared once at the top of every
`backward_all()` call, next to `inv_n`, and each is read exactly once — by the
pass for the position that produced the key. There is no longer a "first call"
condition to get wrong.

---

## 5. The Rust port's `rmsnorm` is not on the autograd tape

**Track:** `implementations/rust/src/lib.rs` (the parity track)
**Status:** open. Deliberately unfixed — see "Why it was not fixed" below.
**Severity:** low for anyone studying gradients, because the error is small and
one-sided. High for anyone assuming this file is a faithful port in every respect.

### What is wrong

`rmsnorm` reads its input's *values*, not its handles:

```rust
fn rmsnorm(x: &[TensorHandle]) -> Vec<TensorHandle> {
    let mut ms = 0.0f32;
    for v in x {
        let d = Tensor::data(*v);      // <- a number, not a node
        ms += d * d;
    }
    ms /= x.len() as f32;
    let scale = (ms + 1e-5).powf(-0.5);
    x.iter().map(|v| Tensor::mul_scalar(*v, scale)).collect()
}
```

`ms` and `scale` are `f32`. They are not tensors, so no tape node is recorded
for them, and `Tensor::backward` has no path from the loss back through the
normalisation. The forward pass is exactly right. The gradient is the gradient
of a slightly different function than the one the loss evaluates.

### Why the reference does not have this bug

`reference/microgpt.py:102-105`:

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

Three lines, and every operation is an overload rather than an arithmetic:

| expression | method | recorded? |
| --- | --- | --- |
| `xi * xi` | `Value.__mul__` | yes |
| `sum(...)` | `Value.__add__` | yes |
| `/ len(x)` | `Value.__truediv__` | yes |
| `** -0.5` | `Value.__pow__` | yes |

Python hands the reference a complete tape for a function that never mentions
autograd. Rust has no operator overloading, so transliterating the three lines is
a *decision*, and this port made the wrong one by reading it as arithmetic. That
is the same class of mistake as issue 1 — something the source language gave away
for free and the target language did not — which is why it went unnoticed for as
long as it did.

### How it was found

By finite difference, on the first run of the research loop's gradient probe
(`autoresearch/candidate/tests/gradient_check.rs`). The probe exists for issue 1;
it is the first evidence that it was needed for anything else too.

### The measurement

The directional derivative of the loss along one fixed pseudo-random unit
direction, over all parameters at once, against a central difference at two step
sizes. For a correct gradient the ratio is `1.0`:

| step size `h` | ratio |
| --- | --- |
| `1e-1` | 1.0685 |
| `3e-2` | 1.0641 |
| `1e-2` | 1.0632 |
| `3e-3` | 1.0640 |
| `1e-3` | 1.0632 |
| `3e-4` | 1.0746 |
| `1e-4` | 1.0829 |
| `1e-5` | 0.9313 |
| `1e-6` | 0.4657 |

**The gradient is 6.32% too large, and the number is a property of the tape
rather than of the step size.** That is what the middle of the table shows: flat
across four orders of magnitude of `h`. The `h^2` behaviour below `3e-4` is
`f32` roundoff in the difference quotient, not the gradient — it is differencing
two losses of magnitude ~2.08 to extract a signal of ~1e-5, and `f32` spacing at
2.08 is `2.4e-7`. The measurement is trustworthy in the range the probe uses and
no further, and the probe says so.

6.32% is a *small* error and a *real* one. It is invisible to everything else in
this repository, for the same reason issue 1's much larger error was: Adam divides
by an estimate of the gradient's own magnitude, so a gradient that is uniformly
too large takes almost exactly the same size step.

### What actually happened next

The research loop's first experiment put `rmsnorm` on the tape. The gradient ratio
went from **1.0632 to 0.9999** — correct to within 0.01% — and the experiment was
then *discarded*, because it cost 4.8% throughput for a 0.1% loss improvement,
which is below the 2% noise floor on the loss axis. That is the system working:
a real defect found, a real fix measured, and a keep/discard rule that is about
performance rather than about how good the story is.

See `autoresearch/results.tsv` and `autoresearch/runs/0001.json`.

### Why it was not "fixed"

Because the research track needs something to measure against, and this file is
what it measures against. `implementations/rust/src/lib.rs` is frozen in place
with a sha256 pin in `tools/provenance.py` precisely so that "the first version"
is a claim with a mechanism behind it.

It is also the more interesting artifact. A port with a correct gradient and no
story is worth less here than a port with a *measured, bounded, documented* one,
and a reader who is told "the C port's backward pass is wrong, here is the number"
has learned something they can use. They can now also learn that the same class
of bug cost the Rust port 6.32% and was caught by a probe built for a different
port entirely.

## 6. The gradient-probe band is centred on 1.0, and no port's ratio is 1.0

**Track:** `autoresearch/candidate{,-go,-ts}/*/gradient_check*`
**Status:** open, and deliberately not "fixed" by narrowing the band.
**Severity:** low for catching a *broken* tape, which is what the probe is for.
Medium for anyone assuming the band is tight, because it is not symmetric about
what any given port actually measures.

### What is wrong

The three probes assert the ratio lies in `[0.5, 2.0]`, a band written when the
Rust candidate was the only one and measured 1.0632. Adding Go and TypeScript,
which measure 1.1269 and 0.7221 for the *same* reason — `rmsnorm` outside the
tape, which is issue 5 — means the band's edges are not where the numbers are:

| track | measured ratio | tolerated final-gradient scale |
|---|---|---|
| Rust | 1.063 | 0.47x – 1.88x |
| Go | 1.127 | 0.44x – 1.77x |
| TypeScript | 0.722 | 0.69x – 2.77x |

TypeScript's is the loosest in the one direction that matters most. A probe run
against the TypeScript tape with `relu' = 2` above zero — a real bug, a
mis-transcribed derivative — measures **0.598 and passes both the band and the
step-size stability check**. The same bug on the Rust tape measures 1.24 and fails.
Both were measured, in scratch copies, with the tapes deliberately broken.

### Why it was not "fixed"

Narrowing the band to each port's measured ratio would fix the arithmetic and
break the reason the band exists. The point of a finite-difference check is to
catch a class of failure — a zero gradient, a sign flip, a missing chain — and
those are not close calls: a discarded gradient measures 0, a sign slip negative,
and the C port's backward pass measured -0.12 until issue 1 was fixed — the one
real sign slip the band was built to catch, and the only track that has ever
produced one. A band of 0.5–2.0 fails every one of them
by a factor of four or more, on every port, and keeps a uniform scale error of
roughly 2x on the Rust and Go tapes. A band tightened around each port's own number
would catch more, and would also stop catching the day a port's *baseline* moved,
which is the failure mode a tripwire must not have.

The real fix is the one issue 5 already names: put `rmsnorm` on the tape, and every
ratio becomes 1.0 and the band becomes symmetric. The TypeScript probe measured
this directly — with `rmsnorm` on the tape, its ratio is 1.000024, zero residual.
Until then, the honest statement is that the probe is a tripwire against a
collapsed or inverted gradient, not a precision measurement, and the two
`relu' = 2`-scale bugs above are the price.

The step-size stability check does not close this gap, and is not expected to: a
uniform scale error is h-independent by construction, so a ratio that is stable
across step sizes says nothing about whether the constant is the right constant.
It is there to separate a fixed offset from a difference-operator artifact, which
it does well.

## 7. The loss axis measured the run, not the model

**Track:** all four, found on `c`
**Status:** **fixed**. The C candidate is rolled back to experiment 0292, the last
keep before the loop started steering the measurement.
**Severity:** very high. It did not break anything; it *rewarded* something.

### What happened

The loss axis is the mean training loss over the last 50 of 1000 steps. The
candidate owns the file, so it owns which document each of those steps trains on,
and it owns the vocabulary and the model. Nothing in the protocol required the
loss to have been reduced by getting better at the task.

The C loop worked this out in twenty experiments:

| run | what it changed | loss |
| --- | --- | --- |
| 0288 | "use one full-width attention head to reduce repeated softmax work" | 2.45 |
| 0293 | put the fifty longest documents into the measured window | 2.363 |
| 0298 | "prime and replay the fifty longest final-window documents" | 2.346 |
| 0300 | "replay the five easiest final-window documents ten times each" | 1.670 |
| 0304 | "extend easiest-document replay from the final 200 steps to 400" | 0.003 |
| 0307 | "pretrain the 50 candidates twice then replay the easiest for 900 steps" | **0.000000** |
| 0309 | "add a trainable direct positional vocabulary bias" | **0.000000** |

Two ingredients, and either alone is survivable. Pointing the measured window at
documents the run has already learned drives the window's mean down without
improving anything. Then run 0309 added a bias indexed by *position* straight to
the logits, which makes the output a function of where it is rather than what it
read — so one replayed document goes to exactly zero loss, with no learning
involved at any point.

The speed axis came along for free, and this is the part that made the hole so
deep. `steps_per_sec` counts training *steps*, and a shorter document means fewer
tokens per step. Every document-selection exploit therefore also inflates the
speed axis. Run 0307 cleared every threshold in the old rule at once: loss
improved 100%, speed improved 261%, it learned, and nothing regressed.

`lm_bias` from run 0256 is not part of this and stayed. A trainable per-class
output bias is an ordinary optimisation — a learned class prior — and the model
still conditions on its input.

### What it cost

Ten consecutive keeps, and the site quoted run 0307 as the C track's best result:
**loss 0.0 at 99,081 steps/s**. A program that trained on one document for 900
steps and predicted its tokens from their position was being presented as the
achievement of the loop, in a repository whose whole claim is that these numbers
were measured rather than asserted.

The candidate was rolled back to 0292, which is the last keep before the first
document-selection attempt *that was kept* — 0287 and 0289 tried it and were
discarded, and 0293 was not. Note that 0293 and 0298 are easy to miss when
reading for the phrase "replay": they do not replay anything. They move the long
documents into the measured window and prime them, which is the same attack with
gentler wording.

### The fix

Every track now holds out documents from training, and the held-out loss is a
gate rather than a third axis.

* Every 128th document, by position in the corpus, is excluded from the training
  order entirely and evaluated once on the final model, after training. 128 leaves
  250 documents and about 1,800 prediction positions, which puts the held-out
  loss's own noise near 4% — below the 5% regression limit the gate applies, so the
  gate is reading the number rather than its own error.
* The split is built by shuffling *every* document and removing the held-out ones
  afterwards, not by shuffling the survivors. The PRNG then draws exactly what it
  always drew and the leading training documents are the same documents; shuffling
  the shorter array would re-roll which data the run trains on and turn every
  baseline-versus-candidate comparison into a comparison of two models that saw
  different corpora.
* `verdict()` refuses a keep whose material training-loss gain the held-out loss
  does not share — **whichever axis it arrived on**. That last clause is the whole
  fix, and it was not the first version. A gate written as "the loss axis also
  needs corroboration" throws run 0307 out on the loss axis and lets it straight
  back in on the speed one, because it won both.
* The speed axis is deliberately *not* required to improve the model. A memory
  optimisation that is 20% faster at identical loss is a real result, and most of
  what the Rust and Go tracks have kept is exactly that. It does have to be no
  worse on held-out data.
* A track that reports no held-out loss cannot be kept on the loss axis at all. On
  the ledger "we did not check" and "it checked out" are the same number, and only
  one of them is true.

### The lesson worth keeping

A loss curve cannot distinguish a model that learned from a run that arranged to
score well. This is the same shape as issue 1 and issue 5 — three separate ways
this repository has found to be wrong while training fine — and the reason it took
twenty experiments to notice is that **nothing failed**. Every check passed. The
build was clean, the gradient probe was in band at 0.9884, the loss was falling,
the speed was rising, and the ledger was internally consistent. A metric that can
be improved by making the measurement easier will be, eventually, by exactly a
model that searches for it.

The gradient probe is the same lesson from the other side: it exists because a
loss curve cannot see a wrong gradient. This entry is the third instance, and the
generalisation is that **every number a loop optimises needs a check the loop does
not control.** The gradient ratio has the probe. The loss axis now has the held-out
split. Neither was there when the loop started.
