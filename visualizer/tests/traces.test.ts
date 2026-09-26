import { describe, expect, it } from 'vitest'
import { ema, trace } from '../src/data/sources'

/**
 * The committed trace, as the browser sees it.
 *
 * The point of these tests is that the numbers on the site are the numbers
 * `tools/trace.py` recorded, and that a widget cannot quietly draw something the
 * trace does not say. If a trace is regenerated with different selection
 * settings, these fail and the prose that quotes the numbers gets reviewed.
 */
describe('the committed trace', () => {
  it('records the reference sha256 that the content index pins', async () => {
    const index = (await import('../src/data/generated/index.json')).default
    expect(trace.meta.reference.sha256).toBe(index.reference.sha256)
  })

  it('records the dataset sha256 that is actually committed', async () => {
    const expected = '0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d'
    expect(trace.meta.dataset.sha256).toBe(expected)
  })

  it('has one loss value per step, with no gaps', () => {
    expect(trace.steps).toHaveLength(trace.meta.steps)
    expect(trace.steps.map((row) => row.step)).toEqual(
      Array.from({ length: trace.steps.length }, (_, i) => i),
    )
  })

  it('has every attention row sum to 1', () => {
    expect(trace.attn.length).toBeGreaterThan(0)
    for (const row of trace.attn) {
      expect(Math.abs(row.weights.reduce((a, b) => a + b, 0) - row.sum)).toBeLessThan(1e-4)
      expect(Math.abs(row.sum - 1)).toBeLessThan(1e-4)
    }
  })

  it('has no attention row longer than the causal window', () => {
    // The claim in multi-head-attention.md is that the mask is structural, not a
    // mask value. If a row ever exceeded pos+1, that claim would be false.
    for (const row of trace.attn) expect(row.weights).toHaveLength(row.pos + 1)
  })

  it('records the noise the loss-chart prose quotes', () => {
    // cross-entropy-loss.md says 500 of 999 steps rise, and a standard deviation
    // of 0.392 on a mean of 2.452. If a regeneration changes those, the prose is
    // wrong and this is where it should be noticed.
    expect(trace.meta.loss_stats.upward_moves).toBe(500)
    expect(trace.meta.loss_stats.total_moves).toBe(999)
    expect(trace.meta.loss_stats.stdev).toBeCloseTo(0.392, 3)
    expect(trace.meta.loss_stats.mean).toBeCloseTo(2.4517, 3)
  })

  it('records the diagonal-not-argmax finding the concept reports', () => {
    // The multi-head-attention concept's table claims 0 of 4 at step 25 and 0 of 6
    // at step 100. Those counts are the whole reason the concept says something
    // surprising, so they are pinned here.
    const atStep = (step: number) =>
      trace.attn.filter((r) => r.step === step && r.pos > 0 && r.head === 0)
    for (const [step, expected] of [
      [25, 0],
      [100, 0],
    ] as Array<[number, number]>) {
      const rows = atStep(step)
      const argmax = rows.filter(
        (r) => r.weights.indexOf(Math.max(...r.weights)) === r.weights.length - 1,
      ).length
      expect(argmax, `step ${step}: ${argmax} of ${rows.length}`).toBe(expected)
    }
  })
})

describe('ema', () => {
  it('preserves length and stays inside the input range', () => {
    const values = [3.4, 3.2, 1.1, 2.9, 2.0]
    const smoothed = ema(values, 0.05)
    expect(smoothed).toHaveLength(values.length)
    for (const value of smoothed) {
      expect(value).toBeGreaterThanOrEqual(Math.min(...values) - 1e-9)
      expect(value).toBeLessThanOrEqual(Math.max(...values) + 1e-9)
    }
  })

  it('removes step-to-step noise while keeping the trend', () => {
    const noisy = trace.steps.map((row) => row.loss)
    const smoothed = ema(noisy, 0.05)
    const sd = (xs: number[]) => {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length
      return Math.sqrt(xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length)
    }
    const jitter = (xs: number[]) => sd(xs.slice(1).map((x, i) => x - xs[i]!))
    expect(jitter(smoothed)).toBeLessThan(jitter(noisy) / 3)
    // `noUncheckedIndexedAccess` is on, so the last element is `T | undefined`.
    const last = smoothed.at(-1)!
    const first = smoothed[0]!
    expect(last).toBeLessThan(first)
  })

  it('matches tools/trace.py on the recorded value', () => {
    // meta.loss_stats.ema_final50 was computed by the Python tool; the browser
    // must agree, or the chart and the documented number disagree.
    const smoothed = ema(trace.steps.map((row) => row.loss), trace.meta.loss_stats.ema_alpha)
    const last50 = smoothed.slice(-50)
    const mean = last50.reduce((a, b) => a + b, 0) / last50.length
    expect(mean).toBeCloseTo(trace.meta.loss_stats.ema_final50, 5)
  })
})
