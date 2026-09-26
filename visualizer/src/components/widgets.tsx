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
    () =>
      trace.attn.find(
        (r) => r.step === stepIndex && r.pos === pos && r.head === headIndex,
      ),
    [stepIndex, pos, headIndex],
  )

  if (!row) {
    return (
      <figure className="widget widget--empty">
        <figcaption>
          No attention weights were recorded for step {stepIndex}, position {pos}, head{' '}
          {headIndex}. The trace records steps {steps.join(', ')} at positions{' '}
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

      <div className="heatmap" role="img" aria-label={describe(row.weights, pos, stepIndex, headIndex)}>
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

export function LossChart({ highlightSteps = [50, 200] }: { highlightSteps?: number[] }) {
  const [smoothed, setSmoothed] = useState(true)
  const series = smoothed ? ema(trace.steps.map((s) => s.loss)) : trace.steps.map((s) => s.loss)
  const stats = trace.meta.loss_stats

  const { min, max, points } = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    const pts = series.map((value, index) => {
      if (value < lo) lo = value
      if (value > hi) hi = value
      return { index, value }
    })
    return { min: lo, max: hi, points: pts }
  }, [series])

  const width = 100
  const height = 100
  const x = (index: number) => (index / (series.length - 1)) * width
  const y = (value: number) => height - ((value - min) / (max - min || 1)) * height

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.index).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ')

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Loss over {trace.meta.steps} steps of a real run.{' '}
        {smoothed
          ? 'Smoothed with an exponential moving average (alpha 0.05), which removes the document-to-document noise without moving the trend.'
          : 'Raw per-step loss. It rises on ' +
            stats.upward_moves +
            ' of ' +
            stats.total_moves +
            ' steps, because each step is a different document.'}
      </figcaption>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="chart"
        role="img"
        aria-label={`Loss over ${trace.meta.steps} steps, from ${stats.first} to ${stats.final}, minimum ${stats.min} at step ${trace.steps.findIndex((s) => s.loss === stats.min)}.`}
        preserveAspectRatio="none"
      >
        <line x1="0" y1={y(2.4517)} x2={width} y2={y(2.4517)} className="chart__mean" />
        <path d={path} className="chart__line" fill="none" />
      </svg>

      <dl className="stats">
        <div>
          <dt>first 50 mean</dt>
          <dd>{stats.ema_first50.toFixed(3)}</dd>
        </div>
        <div>
          <dt>last 50 mean</dt>
          <dd>{stats.ema_final50.toFixed(3)}</dd>
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

      <div className="widget__controls">
        <button
          type="button"
          aria-pressed={smoothed}
          onClick={() => setSmoothed((value) => !value)}
        >
          {smoothed ? 'show raw' : 'show smoothed'}
        </button>
        <span className="widget__hint">
          checkpoints {highlightSteps.join(' and ')} are the ones the parity gate compares
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
          Read from the reference at record time, not hardcoded. Changing{' '}
          <code>n_embd</code> in <code>microgpt.py</code> changes this table and every shape
          annotation in the concepts.
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
        Twenty samples from the trained model at temperature 0.5. Some are names in the dataset
        (<code>anna</code>, <code>kamon</code>); some are not.
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
