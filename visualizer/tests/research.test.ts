import { describe, expect, it } from 'vitest'
import { bestVsBaselinePatch, research } from '../src/data/sources'
import {
  bestRow,
  diffCounts,
  experiments,
  formatGain,
  formatRatio,
  isFrontier,
  linearScale,
  logScale,
  lossTicks,
  measuredRows,
  ratioErrorPercent,
  shortDigest,
  speedTicks,
  statusCount,
} from '../src/data/research'

/**
 * The research page's arithmetic.
 *
 * `traces.test.ts` exists because the loss curve on the home page and the prose in
 * the concepts quote the same number, and a reader comparing them should find
 * them equal. That is the argument here too, in a different form: a chart and a
 * table on the same page, both derived from `results.json`, should never be able
 * to disagree. So these assert on the derived values, not on rendered markup --
 * `e2e/research.spec.ts` covers that the page renders them at all.
 *
 * Nothing here renders React, and that is deliberate: the unit tests are
 * `environment: 'node'` because they cover pure functions. A test that needs a
 * DOM is a Playwright test, which runs against a real build.
 */

describe('the ledger reached the browser intact', () => {
  it('has a baseline row and it is the first', () => {
    expect(research.runs.length).toBeGreaterThan(0)
    expect(research.runs.at(0)?.status).toBe('baseline')
  })

  it('has no duplicates and no gaps in the run ids', () => {
    const ids = research.runs.map((row) => row.run_id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(ids).toEqual(ids.map((_, index) => String(index).padStart(4, '0')))
  })

  it('carries the protocol the prose quotes', () => {
    expect(research.protocol.steps).toBe(1000)
    expect(research.protocol.seed).toBe(42)
    expect(research.protocol.repeats).toBeGreaterThanOrEqual(3)
    expect(research.protocol.trend_window).toBe(50)
  })

  it('carries the four thresholds and the rule that uses them', () => {
    const t = research.thresholds
    expect(t.loss_material).toBeCloseTo(0.02, 6)
    expect(t.loss_tolerance).toBeCloseTo(0.05, 6)
    expect(t.speed_material).toBeCloseTo(0.1, 6)
    expect(t.speed_tolerance).toBeCloseTo(0.05, 6)
    expect(t.rule).toContain('keep if')
  })

  it('ships caveats, because a table of numbers without them is a lie', () => {
    expect(research.caveats.length).toBeGreaterThanOrEqual(3)
    expect(research.caveats.join(' ')).toContain('benchmarks/results.json')
  })

  it('records the digests the page links a reader to', () => {
    expect(research.provenance_of.baseline_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(research.provenance_of.candidate_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(research.provenance_of.gradient_probe_sha256).toMatch(/^[0-9a-f]{64}$/)
  })
})

describe('measured rows exclude crashes', () => {
  it('never plots a crash, whose zeros mean absence rather than a fast model', () => {
    for (const row of measuredRows()) {
      expect(row.status).not.toBe('crash')
      expect(row.loss).toBeGreaterThan(0)
      expect(row.steps_per_sec).toBeGreaterThan(0)
    }
  })

  it('counts statuses consistently with the rows themselves', () => {
    expect(research.counts.keep).toBe(statusCount('keep'))
    expect(research.counts.discard).toBe(statusCount('discard'))
    expect(research.counts.crash).toBe(statusCount('crash'))
    expect(research.counts.experiments).toBe(experiments().length)
  })

  it('excludes the baseline from the experiment count', () => {
    expect(experiments().some((row) => row.status === 'baseline')).toBe(false)
    expect(experiments().length).toBe(research.runs.length - 1)
  })
})

describe('the best run', () => {
  it('is the lowest-loss keep, or null when nothing has been kept', () => {
    const keeps = research.runs.filter((row) => row.status === 'keep')
    const best = bestRow()
    if (keeps.length === 0) {
      expect(best).toBeNull()
      return
    }
    expect(best).not.toBeNull()
    const lowest = Math.min(...keeps.map((row) => row.loss))
    expect(best?.loss).toBeCloseTo(lowest, 6)
  })

  it('is never a crash or a discard', () => {
    const best = bestRow()
    if (best) {
      expect(best.status).toBe('keep')
      expect(best.loss).toBeGreaterThan(0)
    }
  })
})

describe('the Pareto frontier', () => {
  it('contains only rows that exist and were measured', () => {
    const ids = new Set(research.runs.map((row) => row.run_id))
    for (const id of research.pareto) {
      expect(ids.has(id), `${id} is on the frontier but is not in the ledger`).toBe(true)
      const row = research.runs.find((r) => r.run_id === id)
      expect(row?.status === 'crash').toBe(false)
    }
  })

  it('has no point dominated by another point on it', () => {
    const rows = new Map(research.runs.map((row) => [row.run_id, row]))
    const frontier = research.pareto.map((id) => rows.get(id)!)
    for (const a of frontier) {
      for (const b of frontier) {
        if (a === b) continue
        const dominates =
          b.loss <= a.loss &&
          b.steps_per_sec >= a.steps_per_sec &&
          (b.loss < a.loss || b.steps_per_sec > a.steps_per_sec)
        expect(dominates, `${b.run_id} is on the frontier but dominates ${a.run_id}`).toBe(false)
      }
    }
  })

  it('always contains the baseline, which nothing dominated can replace', () => {
    // Only true while there is at most one measured candidate; the seed state.
    // Stated rather than assumed so that if it ever fails it fails loudly here
    // instead of quietly rendering a chart with no reference point.
    if (measuredRows().length === 1) {
      expect(research.pareto).toContain(research.runs.at(0)?.run_id)
    }
  })

  it('agrees with the isFrontier helper the chart uses', () => {
    for (const row of research.runs) {
      expect(isFrontier(row.run_id)).toBe(research.pareto.includes(row.run_id))
    }
  })
})

describe('scales', () => {
  it('maps the minimum to the left edge and the maximum to the right', () => {
    const values = [2.1, 2.4, 2.7]
    const scale = linearScale(values)
    expect(scale.at(Math.min(...values))).toBeLessThan(scale.at(Math.max(...values)))
    // Padded, so no point lands exactly on the frame.
    expect(scale.at(Math.min(...values))).toBeGreaterThan(0)
    expect(scale.at(Math.max(...values))).toBeLessThan(100)
  })

  it('handles a single point without dividing by zero', () => {
    const scale = linearScale([2.4])
    expect(Number.isFinite(scale.at(2.4))).toBe(true)
  })

  it('handles an empty series', () => {
    const scale = linearScale([])
    expect(Number.isFinite(scale.at(1))).toBe(true)
    expect(logScale([]).at(1)).toBeGreaterThanOrEqual(0)
  })

  it('is monotone on the log scale, which is why it is used for speed', () => {
    const scale = logScale([10, 20, 40, 80, 160])
    let previous = -Infinity
    for (const value of [10, 20, 40, 80, 160]) {
      const position = scale.at(value)
      expect(position).toBeGreaterThan(previous)
      previous = position
    }
  })

  it('spreads an octave of speed wider than a decade, as a log axis should', () => {
    const scale = logScale([10, 100])
    const decade = scale.at(100) - scale.at(10)
    expect(decade).toBeGreaterThan(50)
  })

  it('produces speed ticks on a single 1/2/5 step, all inside the range', () => {
    const ticks = speedTicks(logScale([40, 400]))
    expect(ticks.length).toBeGreaterThanOrEqual(2)
    expect(ticks.length).toBeLessThanOrEqual(6)

    // The step, recovered from the ticks, is what has to be a 1/2/5 decade
    // multiple. The *ticks* need not be: 300 is a fine tick on a step of 100, and
    // asserting a mantissa on every tick rejects it. The first version of this
    // test did exactly that, and also produced 34.0 for this range.
    const first = ticks[0] as number
    const second = ticks[1] as number
    const step = second - first
    for (let i = 2; i < ticks.length; i += 1) {
      expect((ticks[i] as number) - (ticks[i - 1] as number)).toBeCloseTo(step, 6)
    }
    const mantissa = step / 10 ** Math.floor(Math.log10(step))
    expect([1, 2, 5]).toContain(Number(mantissa.toFixed(6)))
    for (const tick of ticks) {
      expect(tick).toBeGreaterThanOrEqual(40)
      expect(tick).toBeLessThanOrEqual(400)
    }
  })

  it('spaces values that differ by a constant ratio evenly on the log scale', () => {
    // This is the property that makes a log axis the right one for speed: the
    // quantities being compared -- "3x the reference", "10% slower" -- are ratios,
    // so equal ratios have to get equal distances.
    const scale = logScale([50, 100, 200, 400])
    const first = scale.at(100) - scale.at(50)
    const second = scale.at(200) - scale.at(100)
    const third = scale.at(400) - scale.at(200)
    expect([first, second, third].every((v) => v > 0)).toBe(true)
    expect(second).toBeCloseTo(first, 6)
    expect(third).toBeCloseTo(first, 6)
  })

  it('produces evenly spaced loss ticks inside the range', () => {
    // A 1/2/5 mantissa is a *log* axis convention. On a linear loss axis 2.40
    // is a perfectly good tick, and asserting a mantissa here would be asserting
    // a rule that does not apply.
    const ticks = lossTicks(linearScale([2.4, 2.45]))
    expect(ticks.length).toBeGreaterThanOrEqual(2)
    expect(ticks.length).toBeLessThanOrEqual(6)
    const step = (ticks[1] as number) - (ticks[0] as number)
    for (let i = 1; i < ticks.length; i += 1) {
      expect((ticks[i] as number) - (ticks[i - 1] as number)).toBeCloseTo(step, 9)
    }
    for (const tick of ticks) {
      expect(tick).toBeGreaterThanOrEqual(2.39)
      expect(tick).toBeLessThanOrEqual(2.46)
    }
  })

  it('produces no ticks when the log range collapses, rather than thousands', () => {
    // One identical speed is the degenerate case a log scale can actually see.
    expect(speedTicks(logScale([50, 50]))).toEqual([])
  })

  it('still gives a single measured loss a usable axis', () => {
    // Not a degenerate case: `linearScale` pads a lone point by 10% of its
    // magnitude, because a one-point scatter with no axis is not a chart. Pinned
    // because the two scales pad differently and that difference should be a
    // decision rather than an accident.
    const ticks = lossTicks(linearScale([2.4]))
    expect(ticks.length).toBeGreaterThanOrEqual(2)
    expect(ticks.every((tick) => tick > 2.3 && tick < 2.5)).toBe(true)
  })
})

describe('formatting', () => {
  it('signs every gain, so a reader never has to guess the direction', () => {
    expect(formatGain(0.031)).toBe('+3.1%')
    expect(formatGain(-0.048)).toBe('−4.8%')
    expect(formatGain(0)).toBe('±0.0%')
  })

  it('shows a dash for an unmeasured ratio rather than 0.0000', () => {
    expect(formatRatio(0)).toBe('—')
    expect(formatRatio(1.06316)).toBe('1.0632')
  })

  it('reports the gradient error as a signed distance from a correct gradient', () => {
    expect(ratioErrorPercent(1.06316)).toBeCloseTo(6.316, 2)
    expect(ratioErrorPercent(0.9999)).toBeCloseTo(-0.01, 2)
  })

  it('shortens a digest to a table cell', () => {
    expect(shortDigest('a'.repeat(64))).toBe('a'.repeat(12))
    expect(shortDigest('')).toBe('—')
  })
})

describe('the diff tally', () => {
  it('ignores the --- and +++ file headers', () => {
    const patch = [
      '--- a/implementations/rust/src/lib.rs',
      '+++ b/autoresearch/candidate/src/lib.rs',
      '@@ -1,3 +1,3 @@',
      ' context',
      '-removed',
      '+added',
    ].join('\n')
    expect(diffCounts(patch)).toEqual({ added: 1, removed: 1 })
  })

  it('does not count a removed line whose text starts with dashes as a header', () => {
    // `---- not a header` is `---`-shaped, and the first version of this counted
    // it as a file header and reported zero changes. Hunk state is the only way
    // to tell the two apart, because the shape is genuinely ambiguous.
    const patch = ['--- a/x.rs', '+++ b/x.rs', '@@ -1,2 +1,2 @@', '---- not a header'].join('\n')
    expect(diffCounts(patch)).toEqual({ added: 0, removed: 1 })
  })

  it('ignores +++ and --- outside a hunk', () => {
    expect(diffCounts('--- a/x.rs\n+++ b/x.rs\n')).toEqual({ added: 0, removed: 0 })
  })

  it('tallies the seed patch as the one method it is', () => {
    // The candidate began as the frozen track plus `Tensor::set_data`, so the
    // best-vs-baseline patch cannot have grown before any experiment has been
    // kept. If this ever fails, a discarded experiment leaked into the candidate.
    const counts = diffCounts(bestVsBaselinePatch)
    expect(counts.added).toBeGreaterThan(0)
    expect(bestVsBaselinePatch).toContain('set_data')
    expect(bestVsBaselinePatch).not.toContain('fn rmsnorm')
  })
})
