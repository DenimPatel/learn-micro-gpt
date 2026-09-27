/**
 * A finite-difference check on this port's autograd tape.
 *
 * ## What it measures
 *
 * The directional derivative of the loss along one fixed direction `d`, taken over
 * *all* parameters at once, compared against a central difference. For a correct
 * gradient the ratio is 1.0.
 *
 * Every model is built from the same seed, so all of them hold identical
 * parameters: one for the analytic pass, and two per step size, for `w + h*d` and
 * `w - h*d`. A fresh model per evaluation is not tidiness -- `Model.forward`
 * appends to the KV cache, so a second forward sequence on the same model would
 * attend to the previous one's keys and quietly measure something else.
 *
 * ## What it is allowed to touch
 *
 * Only what `index.ts` already exports: `Model`, `Model.softmax`,
 * `Model.forward`, `model.params()`, and the public `Value` surface (`data`,
 * `grad`, `add`, `divScalar`, `neg`, `log`). Nothing is added to the track to make
 * this file easier to write, because the whole value of the check is that a
 * candidate cannot weaken it by making itself convenient. If something it needs
 * is unreachable, that is a finding to report, not an accessor to add.
 *
 * (The Rust probe at `autoresearch/candidate/tests/gradient_check.rs` has to
 * reimplement softmax inline, because `Model::softmax` is private there. Here it
 * is public and frozen, so the probe uses the real one -- which is the stricter
 * test, since the real softmax is part of what is being checked.)
 *
 * ## The ratio is not 1.0 on a correct tape, and that is not a bug in this file
 *
 * `reference/microgpt.py` differentiates through `rmsnorm`, because Python hands
 * it the tape for free. A transliteration to a language without operator
 * overloading has to notice that, and this port's `Model.rmsnorm` does not: it
 * computes `ms` from `v.data` as a plain number
 * (`implementations/typescript/src/index.ts:269-275`), so the whole normalisation
 * sits *outside* the tape while the resulting `scale` is applied on it.
 *
 * So the analytic derivative is the gradient of a slightly different function
 * than the one the loss evaluates, and the two disagree by a fixed amount. The
 * band below is wide on purpose -- it is a tripwire, not a precision measurement.
 * The C port's broken backward pass measures -0.12, a zero gradient measures 0,
 * a sign slip measures negative; a factor-of-four band catches all of them. The
 * stability check is what does the real work: it says the number is a property of
 * the tape rather than of the step size.
 *
 * ## Run as
 *
 * ```text
 * npx tsx src/probe.ts
 * ```
 *
 * and it prints one machine-readable line, which the harness parses with a
 * regex, recording the ratio at every step size so drift stays visible on the
 * site even when the band passes.
 */

import { Model, makeRng } from './index.ts'
import type { Config, Value } from './index.ts'

/**
 * Same seed discipline as the model, so the parameters are identical on every
 * evaluation and the direction is fixed across runs.
 *
 * The Rust probe's seeds are `u64` literals above `Number.MAX_SAFE_INTEGER` and
 * `makeRng` takes a `number`, so transliterating them would round silently. These
 * are exact integers instead; what the seeds buy is reproducibility, not agreement
 * with the Rust tape's particular parameters, and a probe is only meaningful
 * against the tape in front of it.
 */
const MODEL_SEED = 0x5eed_1234
const DIRECTION_SEED = 0xabcd_0002

/**
 * Fixed step sizes for the central difference. Two, not one, and that is the
 * whole reason the measurement is trustworthy: the analytic derivative is exact,
 * so any error here is the difference operator's, and it splits into a truncation
 * term that falls as `h^2` and a roundoff term that grows as `1/h`. One step size
 * cannot tell the two apart. Two can -- if the ratio moved much when `h` moved,
 * the number would be an artifact of the step size instead of a property of the
 * tape.
 *
 * `1e-2` and `3e-3` are the coarsest pair that stays clear of f32 roundoff: past
 * about `1e-4` the central difference is differencing two losses of magnitude
 * ~2.08 to extract a signal of ~1e-5, and f32 spacing at 2.08 is 2.4e-7. This
 * port is float64, so its noise floor is much lower, but the step sizes are kept
 * identical to the frozen probe's so the two numbers are comparable.
 *
 * The labels are written out rather than formatted from the floats, so the line
 * the harness parses cannot drift with a locale or a formatting change.
 */
const STEPS: readonly { h: number; label: string }[] = [
  { h: 1e-2, label: 'h=1e-2' },
  { h: 3e-3, label: 'h=3e-3' },
]

/**
 * How far the analytic and numeric derivatives may disagree. See the module docs:
 * this is a band around a tape that is deliberately, and knowingly, not quite
 * correct, and it is not to be narrowed to make a number pass.
 */
const RATIO_MIN = 0.5
const RATIO_MAX = 2.0

/**
 * The ratio must not move when the step size moves. Truncation error falls as
 * `h^2` and roundoff grows as `1/h`, so a ratio that is stable across a 3.3x
 * change in `h` is neither. Anything wrong with the tape is a fixed offset and
 * survives that, and this is what separates "fixed offset" from "step size
 * artifact".
 */
const STEP_STABILITY = 0.05

/**
 * A deliberately tiny configuration. The check is about the tape's arithmetic,
 * not about the model, and this has to stay cheap enough to run on every
 * experiment.
 */
function probeConfig(): Config {
  return {
    n_embd: 8,
    n_head: 2,
    n_layer: 1,
    block_size: 8,
    head_dim: 4,
    vocab_size: 8,
    num_steps: 1,
  }
}

/**
 * A fixed token sequence, so the check needs no dataset file and cannot be
 * affected by one going missing. Every id is inside `vocab_size: 8`.
 */
const TOKENS = [0, 1, 2, 3, 4, 5] as const

/** A model with the probe's parameters, identically, on every machine and run. */
function fresh(): Model {
  return new Model(probeConfig(), makeRng(MODEL_SEED))
}

/**
 * The mean cross-entropy over the fixed sequence, rooted on the tape so the caller
 * can `backward()` it. Building the model is the caller's job, so the caller can
 * perturb the parameters before any forward pass runs.
 */
function lossValue(model: Model): Value {
  const n = TOKENS.length - 1
  const perPosition: Value[] = []
  for (let pos = 0; pos < n; pos++) {
    const logits = model.forward(TOKENS[pos]!, pos)
    const probs = Model.softmax(logits)
    // Log first, then negate: `-probs[target].log()`. The order matters, and the
    // other order gives `log(-p)` and therefore NaN immediately.
    perPosition.push(probs[TOKENS[pos + 1]!]!.log().neg())
  }
  let total = perPosition[0]!
  for (let i = 1; i < perPosition.length; i++) total = total.add(perPosition[i]!)
  return total.divScalar(perPosition.length)
}

/** `lossValue(model).data`, for the numeric side, which never needs the tape. */
function lossOf(model: Model): number {
  return lossValue(model).data
}

/** `x.toFixed(6)`, except that a non-finite number prints as itself and not as `NaN`/`` --`` nonsense. */
function f6(x: number): string {
  return Number.isFinite(x) ? x.toFixed(6) : String(x)
}

const failures: string[] = []

// The direction: one pseudo-random vector over all parameters, uniform in
// [-1, 1) and normalised to unit length, from its own seed so it is identical on
// every machine and every run.
const directionRng = makeRng(DIRECTION_SEED)
const paramCount = fresh().params().length
const raw: number[] = []
let squaredNorm = 0
for (let i = 0; i < paramCount; i++) {
  const d = directionRng() * 2 - 1
  raw.push(d)
  squaredNorm += d * d
}
const norm = Math.sqrt(squaredNorm)
const direction = raw.map((d) => d / norm)

// Analytic: one backward pass on a fresh model, then the dot product with the
// direction.
const analyticModel = fresh()
const analyticParams = analyticModel.params()
lossValue(analyticModel).backward()
let analytic = 0
for (let i = 0; i < analyticParams.length; i++) {
  analytic += analyticParams[i]!.grad * direction[i]!
}

// Numeric: for each step size, two more fresh models perturbed in opposite
// directions.
const reported: string[] = []
const ratios: number[] = []
for (const { h, label } of STEPS) {
  const plus = fresh()
  const plusParams = plus.params()
  for (let i = 0; i < plusParams.length; i++) plusParams[i]!.data += h * direction[i]!
  const lossPlus = lossOf(plus)

  const minus = fresh()
  const minusParams = minus.params()
  for (let i = 0; i < minusParams.length; i++) minusParams[i]!.data -= h * direction[i]!
  const lossMinus = lossOf(minus)

  const numeric = (lossPlus - lossMinus) / (2 * h)
  const ratio = analytic / numeric
  ratios.push(ratio)
  reported.push(`${label} ratio=${f6(ratio)} numeric=${f6(numeric)}`)

  if (!Number.isFinite(ratio)) {
    failures.push(
      `the ratio at ${label} is not finite (${String(ratio)}); the loss is diverging ` +
        `or a gradient is not a number`,
    )
  } else if (ratio < RATIO_MIN || ratio > RATIO_MAX) {
    failures.push(
      `directional derivative ratio is ${ratio.toFixed(4)} at ${label}, outside ` +
        `[${RATIO_MIN}, ${RATIO_MAX}]. Analytic ${analytic.toFixed(6)}. This is the ` +
        `shape of docs/KNOWN-ISSUES.md issue 1, where the same measurement is -0.12 ` +
        `for a backward pass that is wrong and whose loss curve still looks fine.`,
    )
  }
}

if (ratios.length === 2) {
  const spread = Math.abs(ratios[0]! - ratios[1]!)
  if (!(spread <= STEP_STABILITY)) {
    failures.push(
      `the ratio moved ${spread.toFixed(4)} when the step size moved from 1e-2 to ` +
        `3e-3 (${(ratios[0] ?? Number.NaN).toFixed(4)} then ` +
        `${(ratios[1] ?? Number.NaN).toFixed(4)}), more than ${STEP_STABILITY}, so it ` +
        `is an artifact of the difference operator rather than a property of the ` +
        `tape. Widen the step sizes, or the loss is not smooth enough at this scale ` +
        `for a central difference to mean anything.`,
    )
  }
}

process.stdout.write(
  `GRADCHECK analytic=${f6(analytic)} params=${analyticParams.length} ratios=${reported.join(' | ')}\n`,
)

// `exitCode` rather than `process.exit()`, so the line above is flushed before the
// process goes away. The harness reads that line whether the probe passes or
// fails, and a truncated line is worse than a non-zero exit.
if (failures.length > 0) {
  for (const failure of failures) process.stderr.write(`gradcheck: ${failure}\n`)
  process.exitCode = 1
}
