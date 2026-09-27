# Known issues

Things this repository knows are wrong. Each one is measured, not guessed, and
each has a test that will fail if it silently changes.

The reason this file exists rather than a comment in the code: a teaching
repository that quietly ships a broken example is worse than one that does not
ship it at all. A reader who is told "the C port's backward pass is wrong, here
is the number, here is the test" can decide what to do with that. A reader who
assumes every tab on the site is correct has no way to find out.

---

## 1. The C port's backward pass is not a correct gradient

**Track:** `implementations/c/microgpt.c` (the parity track)
**Status:** open. Locked by `implementations/c/test_gradients.c`.
**Severity:** high for anyone studying gradients, low for anyone studying speed.

### What is wrong

`backward_all()` does not correctly propagate the key and value gradients of
*earlier* positions back through those positions' own computation to their
embeddings. In the Python reference this happens for free: `k[t]` and `v[t]`
are `Value` objects produced by earlier calls to `gpt()`, so the graph already
contains the path, and `backward()` follows it. In C there is no graph, so
somebody has to write that path down — and it is not written down correctly.

Only the output head's gradient is right.

### How it was found

By finite differences, which is exactly what the deleted `test_gradients.c` used
to *claim* to do and did not. It prints a hardcoded number and returns 0; it
could not fail.

### The measurement

A directional derivative over all 4,192 parameters at once, against a fixed
pseudo-random direction `d`:

```
analytic  <grad, d>  = +8.534968
numeric   (L(w+hd) - L(w-hd)) / 2h  =  -1.032066   at h = 1e-2
                                                  =  -1.006564   at h = 3e-3
```

For a correct gradient that ratio is `1.000`. It is `-0.12`: the wrong sign, and
an order of magnitude off. Stable across two step sizes, so it is not a
measurement artifact.

Per-parameter, the picture is consistent — analytic vs central difference on the
document `mary`:

| parameter | analytic | numeric |
| --- | --- | --- |
| `lm_head[3]` | 0.001064 | 0.001073 |
| `wte[0]` | wrong | -0.2639 |
| `wpe[1]` | wrong | -0.0682 |
| `attn_wq[0][0]` | wrong | 0.000119 |

### Why the loss curve does not show it

This is the part worth understanding, because it is a general lesson and not a
defence of the bug.

**The parity gate passes.** `make parity` reports the C track within ±7% of the
reference's smoothed loss, with a +17.6% trend against the reference's +18.5%.
The forward pass is correct, and the model learns.

**Adam is nearly scale-invariant per parameter.** The update is
`lr * m_hat / (sqrt(v_hat) + eps)`. If every gradient in a block is wrong by the
same *factor*, then `m_hat` and `sqrt(v_hat)` are both wrong by that factor,
they cancel, and the update is almost unchanged. So a systematically
mis-scaled gradient — and a missing gradient path is exactly that — is largely
invisible to both the loss curve and a statistical loss-band comparison.

This is the real argument for the finite-difference test the plan asked for and
the deleted one did not perform: **a loss curve is evidence that something
learned, not evidence that the thing that learned was the gradient.** Only a
directional derivative, or an independent autograd engine, can tell those apart.

### What the C track is and is not good for

| use | sound? |
| --- | --- |
| studying the forward pass, the architecture, the shapes | yes — the forward is correct |
| measuring the speed of this algorithm | yes — that is the point of the track |
| the `scaled` config for throughput | yes — it is excluded from parity for the same reason |
| reading as an example of manual backprop | **no** — it is an example of manual backprop with a bug in it |
| comparing gradient magnitudes across languages | **no** |

`docs/ADDING-A-LANGUAGE.md` therefore requires a directional-derivative check
before any new track is allowed to claim parity. That requirement is the direct
consequence of this issue.

### Fixing it

The fix is to push `dk_global[t]` and `dv_global[t]` back through *position t's*
`attn_wk`/`attn_wv` inputs, its pre-attention `rmsnorm`, and then its residual
add — and, critically, through the residual connection into the MLP block as
well, not only into the embedding. The scaffolding is already there
(`dk_global`, `dv_global`, the saved `x_normed_attn[t]` and `x_embed[t]`
buffers); the arithmetic along the path is what needs auditing.

When it is fixed, `test_gradients.c` will fail on
`the known gradient defect has not changed`, which is the prompt to set
`KNOWN_DIRECTIONAL_RATIO` to 1.0, delete the `KNOWN DEFECT` block, and delete
this section.

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
**Status:** benign, but fragile.

The K/V gradient accumulators are zeroed when `pos == n - 1 && !dk_dv_initialized`,
and `dk_dv_initialized` is reset at the end of the pass. It works, but the
"first call" condition and the reset live ~60 lines apart, and a future edit that
returns early between them would leave stale accumulators contributing to the
next document's gradients.

Not a bug today. Noted because it is the kind of thing that becomes one
silently, and because the C track is explicitly not a model of good practice —
see issue 1.

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
