/**
 * Widgets backed by a committed trace.
 *
 * Every number these draw comes from `traces/python/micro/`, recorded by
 * `tools/trace.py` from a real run of the real reference. Nothing here computes a
 * loss or an attention weight: it only draws what was recorded. That separation
 * is the point -- a widget that computed its own numbers would be a second,
 * unverifiable implementation of the model.
 *
 * The recorded data is also the reason the multi-head-attention concept can say
 * something true and slightly surprising: at step 25 the diagonal is *not* the
 * argmax. That is measured, and the prose says so with the measurement in the
 * table.
 */

import { useMemo, useState } from 'react'
import { ema, trace } from '../data/sources'

/** Accessible colour ramp for the attention heatmap, light and dark. */
const RAMP = [
  'var(--heat-0)',
  'var(--heat-1)',
  'var(--heat-2)',
  'var(--heat-3)',
  'var(--heat-4)',
  'var(--heat-5)',
]

function rampIndex(value: number, max: number): number {
  if (max <= 0) return 0
  // sqrt, so a small-but-not-negligible weight is still visible. A linear ramp
  // on softmax weights renders everything as one flat block, because they are
  // all small and similar.
  return Math.min(RAMP.length - 1, Math.floor(Math.sqrt(value / max) * RAMP.length))
}

export function AttentionHeatmap({
  step = 0,
  pos = 3,
  head = 0,
}: {
  step?: number
  pos?: number
  head?: number
}) {
  const [stepIndex, setStepIndex] = useState(step)
  const [headIndex, setHeadIndex] = useState(head)
  const steps = trace.meta.selection.steps
  const heads = trace.meta.selection.heads

  const row = useMemo(
    () => trace.attn.find((r) => r.step === stepIndex && r.pos === pos && r.head === headIndex),
    [stepIndex, pos, headIndex],
  )

  if (!row) {
    return (
      <figure className="widget widget--empty">
        <figcaption>
          No attention weights were recorded for step {stepIndex}, position {pos}, head {headIndex}.
          The trace records steps {steps.join(', ')} at positions{' '}
          {trace.meta.selection.positions.join(', ')} and heads {heads.join(', ')}.
        </figcaption>
      </figure>
    )
  }

  const max = Math.max(...row.weights)
  const argmax = row.weights.indexOf(max)

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Attention weights, step {stepIndex}, position {pos}, head {headIndex}. One column per
        position seen so far &mdash; there is no column for a later position, because the model
        cannot attend to it.
      </figcaption>

      <div
        className="heatmap"
        role="img"
        aria-label={describe(row.weights, pos, stepIndex, headIndex)}
      >
        {row.weights.map((weight, index) => (
          <div
            key={index}
            className={index === argmax ? 'heatmap__cell is-argmax' : 'heatmap__cell'}
            style={{ background: RAMP[rampIndex(weight, max)] }}
            title={`position ${index}: ${weight.toFixed(4)}`}
          >
            <span className="heatmap__value">{weight.toFixed(2)}</span>
          </div>
        ))}
      </div>

      <p className="widget__note">
        Row sums to {row.sum.toFixed(6)}. The largest weight is at position {argmax}.
        {pos > 0 ? (
          <>
            {' '}
            At step 0 the weights are nearly uniform; by step 200 the model has put{' '}
            {(((row.weights[0] ?? 0) / max) * 100).toFixed(0)}% of its attention on position 0.
          </>
        ) : (
          <> At position 0 there is only one candidate, so the answer is exactly 1.</>
        )}
      </p>

      <div className="widget__controls">
        <label>
          step
          <select value={stepIndex} onChange={(e) => setStepIndex(Number(e.target.value))}>
            {steps.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label>
          head
          <select value={headIndex} onChange={(e) => setHeadIndex(Number(e.target.value))}>
            {heads.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </label>
        <span className="widget__hint">position is fixed at {pos} by the concept</span>
      </div>
    </figure>
  )
}

function describe(weights: number[], pos: number, step: number, head: number): string {
  const parts = weights.map((w, i) => `position ${i}: ${w.toFixed(3)}`)
  return `Attention weights at step ${step}, head ${head}, position ${pos}. ${parts.join(', ')}.`
}

export function SoftmaxBars({ step = 0, pos = 0 }: { step?: number; pos?: number }) {
  const [stepIndex, setStepIndex] = useState(step)
  const row = useMemo(
    () => trace.probs.find((r) => r.step === stepIndex && r.pos === pos),
    [stepIndex, pos],
  )
  if (!row) return null

  const max = row.top[0]?.p ?? 1
  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        The eight most likely next characters, step {stepIndex}, position {pos}. A fresh model
        guesses almost uniformly: 1/27 is 0.037.
      </figcaption>
      <ul className="bars">
        {row.top.map((entry) => (
          <li key={entry.token} className="bars__row">
            <code className="bars__token">{entry.token === ' ' ? '␣' : entry.token}</code>
            <span className="bars__track">
              <span className="bars__fill" style={{ width: `${(entry.p / max) * 100}%` }} />
            </span>
            <span className="bars__value">{(entry.p * 100).toFixed(2)}%</span>
          </li>
        ))}
      </ul>
      <div className="widget__controls">
        <label>
          step
          <select value={stepIndex} onChange={(e) => setStepIndex(Number(e.target.value))}>
            {trace.meta.selection.steps.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>
    </figure>
  )
}

/**
 * The recorded loss curve.
 *
 * Two series, because one of them lies. The raw per-step loss is the actual
 * number the reference prints, and it is so noisy (standard deviation 0.392 on a
 * mean of 2.45) that plotting it alone looks like static. The exponentially
 * smoothed curve is readable — and plotting *that* alone is its own kind of lie,
 * because an EMA seeded with its first value has not converged for the first
 * ~20 steps, and drawing those produces a vertical cliff at the left edge that
 * looks like a dramatic early collapse and is an artefact of the initialiser.
 *
 * So: raw behind, faint, in full; EMA in front, solid, starting once it has
 * converged. The caption says so, and the burned-in region is genuinely dropped
 * rather than drawn misleadingly.
 */
export function LossChart() {
  const [showSmoothed, setShowSmoothed] = useState(true)
  const raw = useMemo(() => trace.steps.map((row) => row.loss), [])
  const series = useMemo(() => ema(raw), [raw])
  const stats = trace.meta.loss_stats

  /*
   * The EMA's effective window is 1/alpha = 20 steps, and it is seeded with the
   * first value, so it is not a useful summary of anything before then. 25 is a
   * whole number of windows: after that the curve is converged and the number it
   * shows is the trend rather than the initialiser.
   */
  const BURN_IN = 25

  const rawWindow = useMemo(() => {
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    return {
      first: mean(raw.slice(0, 50)),
      last: mean(raw.slice(-50)),
    }
  }, [raw])

  const { min, max } = useMemo(() => {
    const lo = Math.min(...series, ...raw)
    const hi = Math.max(...series, ...raw)
    return { min: lo, max: hi }
  }, [series, raw])

  const path = useMemo(() => {
    const total = Math.max(1, series.length - 1)
    const x = (index: number) => (index / total) * 100
    const y = (value: number) => 100 - ((value - min) / (max - min || 1)) * 100
    const points = series
      .map((value, index) => ({ index, value }))
      .filter((point) => point.index >= BURN_IN)
    return points
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.index).toFixed(2)},${y(p.value).toFixed(2)}`)
      .join(' ')
  }, [series, min, max])

  const rawPath = useMemo(() => {
    const total = Math.max(1, raw.length - 1)
    return raw
      .map(
        (value, index) =>
          `${index === 0 ? 'M' : 'L'}${((index / total) * 100).toFixed(2)},${(100 - ((value - min) / (max - min || 1)) * 100).toFixed(2)}`,
      )
      .join(' ')
  }, [raw, min, max])

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Loss over {trace.meta.steps} steps of a real run. The faint line is the raw per-step loss —
        the number the reference actually prints — and it rises on {stats.upward_moves} of its{' '}
        {stats.total_moves} steps, because each step is a different document. The solid line is an
        exponential moving average (alpha 0.05, a 20-step window), which removes that noise without
        moving the trend; its first {BURN_IN} steps are omitted, because a moving average seeded
        with its first value has not converged there and drawing them makes the start of the curve
        look like a cliff that did not happen.
      </figcaption>

      <svg
        viewBox="0 0 100 100"
        className="chart"
        role="img"
        aria-label={`Loss over ${trace.meta.steps} steps, smoothed. The mean over the first 50 steps is ${rawWindow.first.toFixed(3)} and over the last 50 is ${rawWindow.last.toFixed(3)}; the lowest single step is ${stats.min} and the highest ${stats.max}. The per-step standard deviation is ${stats.stdev.toFixed(3)}.`}
        preserveAspectRatio="none"
      >
        {/* The raw curve is always drawn: it is what the reference prints, and
            hiding it would make the smoothed line look like the model's actual
            behaviour rather than a summary of it. */}
        <path d={rawPath} className="chart__raw" fill="none" />
        {showSmoothed ? <path d={path} className="chart__line" fill="none" /> : null}
      </svg>

      <dl className="stats">
        <div>
          <dt>mean, steps 1&ndash;50</dt>
          <dd>{rawWindow.first.toFixed(3)}</dd>
        </div>
        <div>
          <dt>mean, last 50</dt>
          <dd>{rawWindow.last.toFixed(3)}</dd>
        </div>
        <div>
          <dt>per-step s.d.</dt>
          <dd>{stats.stdev.toFixed(3)}</dd>
        </div>
        <div>
          <dt>steps that rose</dt>
          <dd>
            {stats.upward_moves}/{stats.total_moves}
          </dd>
        </div>
      </dl>
      <p className="widget__note">
        The first two are plain means over the raw curve, which is what the cross-entropy concept
        quotes (2.852 &rarr; 2.323). The smoothed values in
        <code> meta.json</code> are <em>not</em> those numbers: averaging a window of an EMA is not
        the same as the EMA over that window, and the earlier version of this widget labelled the
        latter &ldquo;first 50 mean&rdquo; and quietly disagreed with the prose.
      </p>

      <div className="widget__controls">
        <button
          type="button"
          aria-pressed={showSmoothed}
          onClick={() => setShowSmoothed((value) => !value)}
        >
          {showSmoothed ? 'hide the smoothed curve' : 'show the smoothed curve'}
        </button>
        <span className="widget__hint">
          checkpoints 50 and 200 are the ones the parity gate compares
        </span>
      </div>
    </figure>
  )
}

export function ShapeTable({ names }: { names?: string[] }) {
  const shapes = useMemo(() => {
    const all = [
      { name: 'n_embd', value: trace.meta.hyperparameters.n_embd },
      { name: 'n_head', value: trace.meta.hyperparameters.n_head },
      { name: 'n_layer', value: trace.meta.hyperparameters.n_layer },
      { name: 'block_size', value: trace.meta.hyperparameters.block_size },
      { name: 'head_dim', value: trace.meta.hyperparameters.head_dim },
    ]
    return names?.length ? all.filter((row) => names.includes(row.name)) : all
  }, [names])

  return (
    <div className="table-scroll">
      <table className="shapes">
        <caption>
          Read from the reference at record time, not hardcoded. Changing <code>n_embd</code> in{' '}
          <code>microgpt.py</code> changes this table and every shape annotation in the concepts.
        </caption>
        <tbody>
          {shapes.map((row) => (
            <tr key={row.name}>
              <th scope="row">
                <code>{row.name}</code>
              </th>
              <td>{String(row.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/*
 * `vocab_size` is deliberately absent from this table. It is not a hyperparameter
 * in the source: it is *derived* from the dataset, as `len(uchars) + 1`. An
 * earlier version of this file tried to recover it from the characters appearing
 * in the samples, which is a lower bound and not the value -- a genuinely wrong
 * number in a widget whose entire job is to be right. The real value lives in
 * each concept's `shapes` frontmatter, which is validated against
 * `content/index.json`, and that is where a reader should see it.
 */

export function SamplesList() {
  const [step, setStep] = useState(trace.samples[0]?.step ?? 1000)
  const rows = trace.samples.filter((s) => s.step === step)
  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Twenty samples from the trained model at temperature 0.5. Some are names in the dataset (
        <code>anna</code>, <code>kamon</code>); some are not.
      </figcaption>
      <ul className="samples">
        {rows.map((row, index) => (
          <li key={index}>
            <code>{row.text || <em>(empty &mdash; sampled BOS immediately)</em>}</code>
          </li>
        ))}
      </ul>
      <div className="widget__controls">
        <label>
          after step
          <select value={step} onChange={(e) => setStep(Number(e.target.value))}>
            {[...new Set(trace.samples.map((s) => s.step))].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
      </div>
    </figure>
  )
}
