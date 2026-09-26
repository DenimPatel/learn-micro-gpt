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
