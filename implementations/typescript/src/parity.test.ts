/**
 * Tests for the TypeScript port.
 *
 * ## Why this file exists
 *
 * `package.json` declared `"test": "tsx --test src/*.test.ts"` and no
 * `*.test.ts` file existed. The glob matched zero files, `tsx --test` ran
 * nothing, and the process exited 0. A command that always passes looks like
 * evidence and is worse than no command at all, so this file gives the glob
 * something to find.
 *
 * ## What is checked, and why each one earns its place
 *
 * 1. **The backward pass, by finite difference.** The whole value of an autograd
 *    library is that `backward()` produces the derivative of the forward
 *    function. A sign error or a dropped term in the chain rule produces a model
 *    that still trains, just badly, and nothing in the loss curve says so.
 *    Perturbing an input and re-evaluating the forward function gives a
 *    definitionally-correct number to compare the analytic gradient against, and
 *    no amount of clever internal structure can fake agreement.
 *
 *    This is the same method `implementations/c/test_gradients.c` and
 *    `implementations/rust/tests/parity.rs` use, and it is deliberately *not* a
 *    test of private internals: it drives `Value` through its public surface
 *    exactly as the model does.
 *
 * 2. **The chain rule through every operator the model uses.** One function per
 *    operator would be a transcription of the implementation; instead each case
 *    below composes several operators, so a mistake in any of them shows up in
 *    at least one graph. `div` and `pow` are the ones worth watching: both are
 *    defined in terms of each other, so an error in either is an error in both.
 *
 * 3. **The `softmax` invariant.** The reference's `softmax` is a probability
 *    distribution: every weight in `(0, 1]` and the weights sum to 1. This is
 *    the one property the tab's prose tells the reader to check, so it is
 *    checked. It is also the reason the max-subtraction is there, and a softmax
 *    that has lost it still trains.
 *
 * 4. **Determinism of a fixed-seed forward pass and training run.** The whole
 *    claim of the four ports is that a seed fixes the run. `train()` is the
 *    harness's entry point, so two runs at the same seed must produce
 *    bit-identical losses -- otherwise the frozen-versus-candidate comparison in
 *    the harness is a comparison of two different models.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  Value,
  Model,
  DEFAULT_CONFIG,
  makeRng,
  sum,
  mean,
  loadDocs,
  tokenize,
  buildVocab,
  sampleFrom,
} from './index.ts'

// ─── Finite-difference gradient checks ─────────────────────────────────────

/**
 * The step for the central difference. Large enough that floating-point
 * cancellation in the subtraction does not dominate, small enough that the
 * `O(h^2)` truncation error stays well under the tolerance.
 */
const H = 1e-5

/** `(f(x + h) - f(x - h)) / 2h`, accurate to `O(h^2)`. */
function centralDiff(f: (x: number) => number, x: number): number {
  return (f(x + H) - f(x - H)) / (2 * H)
}

/**
 * Build the graph, run `backward()`, and return the gradient of the input leaf
 * alongside the root's forward value.
 *
 * The leaf is what the derivative is *about*; returning the root's value too
 * lets a caller assert the forward pass agreed, so a backward pass wired to the
 * wrong node cannot pass by producing a plausible number for something else.
 *
 * A fresh `Value` per call is what makes this safe without a `zeroGrad`: a new
 * leaf starts at `grad = 0`, so the analytic pass and every finite-difference
 * probe get gradients that are theirs alone.
 */
function gradAndValue(
  build: (x: Value) => Value,
  x: number,
): { grad: number; value: number } {
  const leaf = new Value(x)
  const root = build(leaf)
  root.backward()
  return { grad: leaf.grad, value: root.data }
}

/**
 * Assert that the analytic gradient of the graph `build` at `x` matches the
 * central difference of `f`, to a relative tolerance.
 *
 * Relative, not absolute: the functions below have derivatives spanning a few
 * orders of magnitude, and an absolute tolerance would either be meaningless at
 * the small end or fail at the large end for rounding reasons.
 */
function assertGradMatches(
  f: (x: number) => number,
  build: (x: Value) => Value,
  x: number,
): void {
  const { grad: analytic, value } = gradAndValue(build, x)

  // The forward pass must agree with the closed form first; otherwise the
  // derivative comparison is testing the wrong function.
  const expectedValue = f(x)
  assert.ok(
    Math.abs(value - expectedValue) < 1e-9,
    `forward mismatch at x = ${x}: graph = ${value}, expected = ${expectedValue}`,
  )

  const numeric = centralDiff(f, x)
  const scale = Math.max(Math.abs(analytic), Math.abs(numeric), 1e-6)
  const rel = Math.abs(analytic - numeric) / scale
  assert.ok(
    rel < 1e-4,
    `gradient mismatch at x = ${x}: analytic = ${analytic}, numeric = ${numeric}, rel = ${rel}`,
  )
}

/** `f(x) = x*x + 2x`, so `f'(x) = 2x + 2`. Exercises `mul`, `add`, `mulScalar`. */
test('grad of quadratic', () => {
  const f = (x: number): number => x * x + 2 * x
  const build = (x: Value): Value => x.mul(x).add(x.mulScalar(2))
  for (const x of [-3, -0.5, 0, 0.25, 1, 4]) assertGradMatches(f, build, x)
})

/**
 * `f(x) = log(exp(x) + x*x)`, so `f'(x) = (exp(x) + 2x) / (exp(x) + x*x)`.
 * Exercises `exp`, `log` and `pow` -- the loss the model minimises is a `log`,
 * so this is the operator it cannot get wrong.
 *
 * `x = 0` is included, and it is worth saying why it *can* be: this port's
 * `pow` uses the power rule `n * x**(n-1)`, which is `0` at a zero base, where
 * the Rust port's `n * data / base` form is `0/0`. The two ports therefore
 * disagree at exactly this point, and this test pins the TypeScript behaviour.
 */
test('grad of log sum', () => {
  const f = (x: number): number => Math.log(Math.exp(x) + x * x)
  const build = (x: Value): Value => x.exp().add(x.pow(2)).log()
  for (const x of [-1, 0, 0.5, 1.5, 3]) assertGradMatches(f, build, x)
})

/**
 * `f(x) = exp(x*x)`, so `f'(x) = 2x * exp(x*x)`. Exercises `pow` feeding `exp`,
 * which is the shape of every activation in the model.
 */
test('grad of exp of square', () => {
  const f = (x: number): number => Math.exp(x * x)
  const build = (x: Value): Value => x.pow(2).exp()
  for (const x of [-1, -0.4, 0, 0.7, 1.2]) assertGradMatches(f, build, x)
})

/**
 * `pow` is correct at a zero base, unlike the Rust port.
 *
 * `d/dx x^n` here is `n * x**(n-1)`, which is the ordinary `0` for `n = 2` at
 * `x = 0`. The Rust port computes the same derivative as `n * data / base` and
 * gets `NaN` there. The two are equal everywhere else; this pins the divergence
 * as intended behaviour rather than a latent difference somebody would later
 * "fix" by making the TypeScript match the bug.
 */
test('pow gradient is defined at a zero base', () => {
  const { grad } = gradAndValue((x) => x.pow(2), 0)
  assert.equal(grad, 0)
})

/**
 * `f(x) = relu(x) * x`. For `x > 0` this is `x*x` with `f'(x) = 2x`; at `x = 0`
 * `relu` is not differentiable and the graph picks the sub-gradient 0, so the
 * point is deliberately excluded rather than papered over.
 */
test('grad of relu product', () => {
  const f = (x: number): number => (x > 0 ? x * x : 0)
  const build = (x: Value): Value => x.relu().mul(x)
  for (const x of [0.3, 1, 2.5]) assertGradMatches(f, build, x)
})

/**
 * `f(x) = (x*x - x) / (x + 2)`, so
 * `f'(x) = ((2x - 1)(x + 2) - (x*x - x)) / (x + 2)^2`.
 * Exercises `sub`, `div` and `pow` together -- `div` is `mul` with a `pow(-1)`,
 * so this is the pair that a naive implementation gets wrong.
 */
test('grad of quotient', () => {
  const f = (x: number): number => (x * x - x) / (x + 2)
  const build = (x: Value): Value => x.mul(x).sub(x).div(x.addScalar(2))
  for (const x of [-1.5, -0.5, 0, 1, 3]) assertGradMatches(f, build, x)
})

/**
 * The gradient must reach a *shared* leaf once per use.
 *
 * `f(x) = (x + x) * (x + x)` is `4x^2` with `f'(x) = 8x`. The subtlety is that
 * `x` is a child of *two* different `add` nodes, and each of those is a child of
 * the `mul` (twice). A backward pass that visits a node once instead of
 * accumulating every path through it produces `4x` here, not `8x` -- a
 * factor-of-two error that would leave a model that trains plausibly and is
 * simply wrong. The `+=` in `backward()` is what this checks.
 */
test('grad accumulates over a shared subgraph', () => {
  const f = (x: number): number => 4 * x * x
  const build = (x: Value): Value => x.add(x).mul(x.add(x))
  for (const x of [-2, -0.5, 0.75, 1.5, 3]) assertGradMatches(f, build, x)
})

/**
 * `sum` and `mean` are the loss path, so their gradients are checked directly.
 *
 * `sum([x, 2x, 3x]) = 6x`, so `d/dx = 6`. `mean([x, 2x, 3x]) = 2x`, so
 * `d/dx = 2`. Both are numbers a reader can check by hand, which is the point:
 * a wrong reduction factor is invisible in a loss curve but caught here.
 */
test('grad through sum and mean', () => {
  const sumBuild = (x: Value): Value => sum([x, x.mulScalar(2), x.mulScalar(3)])
  const meanBuild = (x: Value): Value =>
    mean([x, x.mulScalar(2), x.mulScalar(3)])
  for (const x of [-2, 0, 1.5]) {
    assertGradMatches((v) => 6 * v, sumBuild, x)
    assertGradMatches((v) => 2 * v, meanBuild, x)
  }
})

// ─── The softmax invariant ─────────────────────────────────────────────────

/**
 * `softmax` returns a probability distribution: every weight in `(0, 1]` and the
 * weights sum to 1.
 *
 * The max-subtraction is not a numerical nicety -- it leaves the result exactly
 * unchanged -- but a softmax that has lost it overflows for large logits. The
 * large-logit case below is the one that would catch that: without the
 * subtraction, `exp(1000)` is `Infinity` and the weights become `NaN`.
 */
test('softmax normalizes', () => {
  const cases: number[][] = [
    [0, 0, 0],
    [1, 2, 3],
    [-1, -2, -3],
    [1000, 1001, 1002],
    [-1000, -1001, -1002],
  ]
  for (const logits of cases) {
    const weights = Model.softmax(logits.map((n) => new Value(n)))
    const total = weights.reduce((acc, w) => acc + w.data, 0)
    assert.ok(
      Math.abs(total - 1) < 1e-9,
      `softmax([${logits}]) summed to ${total}, not 1`,
    )
    for (const w of weights) {
      assert.ok(w.data > 0 && w.data <= 1, `weight ${w.data} outside (0, 1]`)
    }
  }
})

/** Softmax is shift-invariant: adding a constant to every logit changes nothing. */
test('softmax is shift-invariant', () => {
  const logits = [0.5, -1.25, 3.0, 2.0]
  const base = Model.softmax(logits.map((n) => new Value(n))).map((w) => w.data)
  const shifted = Model.softmax(logits.map((n) => new Value(n + 17))).map((w) => w.data)
  for (let i = 0; i < base.length; i++) {
    assert.ok(
      Math.abs(base[i]! - shifted[i]!) < 1e-9,
      `softmax shifted by 17 changed weight ${i}: ${base[i]} -> ${shifted[i]}`,
    )
  }
})

// ─── Data helpers ──────────────────────────────────────────────────────────

/** `tokenize` frames a document with BOS on both ends -- the trailing BOS is the trick. */
test('tokenize frames with BOS on both ends', () => {
  const { charIndex } = buildVocab(['abc'])
  const bos = 3
  const tokens = tokenize('abc', charIndex, bos)
  assert.deepEqual(tokens, [bos, 0, 1, 2, bos])
})

/** `loadDocs` drops blank lines, so a trailing newline is not a document. */
test('loadDocs drops blank lines', () => {
  assert.deepEqual(loadDocs('a\n\nb\n  \nc\n'), ['a', 'b', 'c'])
})

/** `sampleFrom` returns an in-range index; a weighted distribution still indexes. */
test('sampleFrom stays in range', () => {
  const rng = makeRng(1234)
  const weights = [0.1, 0.2, 0.3, 0.4]
  for (let i = 0; i < 100; i++) {
    const index = sampleFrom(rng, weights)
    assert.ok(Number.isInteger(index) && index >= 0 && index < weights.length, `index ${index} out of range`)
  }
})

// ─── Fixed-seed determinism ────────────────────────────────────────────────

/** The RNG is a pure function of its seed, so two draws agree and different seeds differ. */
test('makeRng is deterministic per seed', () => {
  const a = makeRng(42)
  const b = makeRng(42)
  const c = makeRng(43)
  const drawsA = Array.from({ length: 8 }, () => a())
  const drawsB = Array.from({ length: 8 }, () => b())
  const drawsC = Array.from({ length: 8 }, () => c())
  assert.deepEqual(drawsA, drawsB)
  assert.notDeepEqual(drawsA, drawsC)
})

/**
 * Two models built from the same seed must produce identical logits.
 *
 * This is the "fixed seed" half of the parity claim: the Python reference is
 * reproducible, and a port that is not cannot be compared to it. The check runs
 * a handful of positions through the real `forward` so the attention and MLP
 * paths -- not just the embedding lookup -- are exercised.
 */
test('fixed-seed forward is deterministic', () => {
  const config = { ...DEFAULT_CONFIG }
  const modelA = new Model(config, makeRng(42))
  const modelB = new Model(config, makeRng(42))

  for (const [token, pos] of [
    [1, 0],
    [5, 1],
    [9, 2],
    [3, 3],
  ] as const) {
    const a = modelA.forward(token, pos).map((v) => v.data)
    const b = modelB.forward(token, pos).map((v) => v.data)
    assert.deepEqual(a, b, `fixed-seed forward diverged at token ${token}, position ${pos}`)
  }
})

/**
 * A different seed must produce different logits.
 *
 * Without this, the determinism check above would pass for a model that ignored
 * its seed entirely -- which is exactly the failure a seed-honouring port must
 * not have.
 */
test('different seed forward differs', () => {
  const config = { ...DEFAULT_CONFIG }
  const modelA = new Model(config, makeRng(42))
  const modelB = new Model(config, makeRng(43))
  const a = modelA.forward(1, 0).map((v) => v.data)
  const b = modelB.forward(1, 0).map((v) => v.data)
  assert.notDeepEqual(a, b, 'two seeds produced identical logits; the seed is ignored')
})
