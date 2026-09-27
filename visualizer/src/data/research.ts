/**
 * The research track's derived data: the arithmetic the `#/research` page draws.
 *
 * Kept out of the component on purpose. Every number this site shows is
 * measured, and the property that makes that worth trusting is that the same
 * function produced the chart, the table and the test's expectation. A scale
 * computed inline in JSX cannot be asserted on without rendering it, and a scale
 * that is asserted on by reading the JSX is not asserted on at all.
 *
 * Nothing here invents a measurement. `research.runs` is the ledger, as rendered
 * by `tools/autoresearch.py`; this file only selects, ranges and formats it.
 */

import { research } from './sources'
import type { ResearchRow, ResearchVerdict } from './types'

/** Rows with a real measurement. A `crash` has none, and its zeros are absence. */
export function measuredRows(): ResearchRow[] {
  return research.runs.filter((row) => row.status !== 'crash' && row.loss > 0)
}

/** Everything the loop actually tried, baseline excluded. */
export function experiments(): ResearchRow[] {
  return research.runs.filter((row) => row.status !== 'baseline')
}

export function statusCount(status: ResearchVerdict): number {
  return research.runs.filter((row) => row.status === status).length
}

export function bestRow(): ResearchRow | null {
  return research.best
}

/**
 * The run ids on the frontier, as recorded.
 *
 * `tools/autoresearch.py` computes this and `results.json` carries it, rather
 * than this file recomputing it. Two implementations of "is this dominated" is
 * two chances to disagree, and the disagreement would be invisible.
 */
export function frontierIds(): string[] {
  return research.pareto
}

export function isFrontier(runId: string): boolean {
  return research.pareto.includes(runId)
}

// --- scales -----------------------------------------------------------------

export interface Scale {
  /** A value in data units to a position in `0..100`. */
  at(value: number): number
  min: number
  max: number
}

/**
 * A linear scale over the measured values, padded so points do not sit on the
 * frame. Loss is linear because it moves by percent: a few percent is the whole
 * interesting range, and a log axis would exaggerate differences that are
 * inside the noise.
 */
export function linearScale(values: number[], pad = 0.08): Scale {
  if (values.length === 0) {
    return { at: () => 50, min: 0, max: 1 }
  }
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo || Math.max(Math.abs(hi) * 0.1, 1e-9)
  const min = lo - span * pad
  const max = hi + span * pad
  const width = max - min || 1
  return { at: (value) => ((value - min) / width) * 100, min, max }
}

/**
 * A logarithmic scale for speed.
 *
 * Steps per second is the one quantity here that is routinely compared as a
 * *ratio* — "this is 3x the reference" — and a linear axis makes a 3x
 * difference look like a small nudge while a 1.05x difference looks like a
 * cliff. `docs/BENCHMARKS.md` already carries the same objection about
 * `relative_to_python`, which spans 1x to 271x across the five tracks.
 *
 * The reference point is the baseline rather than zero, because speed has no
 * meaningful zero and a zero anchor on a log scale is not a number.
 */
export function logScale(values: number[]): Scale {
  const positive = values.filter((value) => value > 0)
  if (positive.length === 0) {
    return { at: () => 50, min: 0, max: 1 }
  }
  const lo = Math.min(...positive)
  const hi = Math.max(...positive)
  const logLo = Math.log10(lo)
  const logHi = Math.log10(hi)
  // Half a decade of padding, so the extreme points are not on the frame.
  const min = logLo - (logHi - logLo) * 0.08
  const max = logHi + (logHi - logLo) * 0.08
  const width = max - min || 1
  return {
    at: (value) => ((Math.log10(Math.max(value, 1e-9)) - min) / width) * 100,
    min: 10 ** min,
    max: 10 ** max,
  }
}

/**
 * Tick values inside a scale's range, on the 1/2/5 sequence.
 *
 * Found by searching for the coarsest step that still yields at most `count`
 * ticks, rather than by computing a step and hoping. The earlier version derived
 * the step from a decade argument and then incremented the *value* by it, which
 * produced 447 ticks for a range that wanted 4 — and, for the speed axis, ticks
 * at 34.0, which is a measurement and not a tick.
 *
 * Value-space 1/2/5 is the right grid for both axes here: on the log axis it
 * gives 50/100/200/400 rather than 10^2 and 10^2.5, which are technically even
 * spacing and much harder to read.
 *
 * The search counts before it builds. The first version of this walked from
 * 10^-8 upwards and materialised the array at each step, so a range of 33..480
 * tried to allocate 45 billion entries and threw `RangeError` — 14 seconds before
 * it did. The count is a subtraction; the array is the expensive part, and it is
 * only built once, for the chosen step.
 */
function niceTicks(lo: number, hi: number, count: number): number[] {
  if (!(hi > lo)) return []
  const tally = (step: number): number => Math.floor(hi / step) - Math.ceil(lo / step) + 1
  const build = (step: number): number[] => {
    const ticks: number[] = []
    for (let value = Math.ceil(lo / step) * step; value <= hi; value += step) {
      ticks.push(Number(value.toFixed(6)))
    }
    return ticks
  }
  let coarsestFitting = 0
  for (let exponent = -8; exponent <= 15; exponent += 1) {
    for (const multiplier of [1, 2, 5]) {
      const step = multiplier * 10 ** exponent
      if (step <= 0) continue
      const tallyAtStep = tally(step)
      if (tallyAtStep < 2) break
      if (tallyAtStep <= count) return build(step)
      coarsestFitting = step
    }
  }
  if (coarsestFitting > 0) return build(coarsestFitting)
  return build((hi - lo) / Math.max(count, 1))
}

/** Ticks for the speed axis, in steps per second. */
export function speedTicks(scale: Scale, count = 4): number[] {
  const lo = Math.max(scale.min, 1e-9)
  if (!(lo > 0) || !(scale.max > 0) || scale.max <= lo) return []
  return niceTicks(lo, scale.max, count)
}

/** Ticks for the loss axis, in nats. */
export function lossTicks(scale: Scale, count = 4): number[] {
  return niceTicks(scale.min, scale.max, count)
}

// --- formatting -------------------------------------------------------------

/** A signed percentage with one decimal, and an explicit sign even at zero. */
export function formatGain(value: number): string {
  const sign = value > 0 ? '+' : value < 0 ? '−' : '±'
  return `${sign}${Math.abs(value * 100).toFixed(1)}%`
}

export function formatLoss(value: number): string {
  return value.toFixed(4)
}

export function formatSpeed(value: number): string {
  return value.toFixed(1)
}

/**
 * A gradient ratio, with the number that matters beside it.
 *
 * `1.0` is a correct gradient. The frozen track measures 1.063, which is not a
 * fault in the measurement but a real property of the tape -- `rmsnorm` is not on
 * it, which is `docs/KNOWN-ISSUES.md` issue 5. So the function does not colour a
 * ratio by distance from 1.0; it reports the ratio and lets the note carry the
 * meaning.
 */
export function formatRatio(value: number): string {
  return value === 0 ? '—' : value.toFixed(4)
}

/** The distance of a ratio from a correct gradient, for the page's prose. */
export function ratioErrorPercent(value: number): number {
  return (value - 1) * 100
}

/**
 * Counts added and removed in a unified diff, ignoring the `---`/`+++` headers.
 *
 * Tracked with hunk state rather than by prefix, because the prefix alone is
 * ambiguous: a removed line whose *text* begins with two dashes is `---`-shaped,
 * and the first version of this silently counted it as a header. `git apply`
 * resolves the ambiguity by position, so this does too: `@@` opens a hunk, and
 * from there every `+` and `-` is a change until the next `@@`.
 *
 * `difflib` produced the text, so this is not a diff algorithm -- it is a tally of
 * what the loop recorded, and the page shows it so the size of a change is
 * visible before anyone opens the patch.
 */
export function diffCounts(patch: string): { added: number; removed: number } {
  let added = 0
  let removed = 0
  let inHunk = false
  for (const line of patch.split('\n')) {
    if (line.startsWith('@@')) {
      inHunk = true
      continue
    }
    if (!inHunk) continue
    if (line.startsWith('+')) added += 1
    else if (line.startsWith('-')) removed += 1
  }
  return { added, removed }
}

/** Shorten a sha256 to something a table cell can hold. */
export function shortDigest(digest: string): string {
  return digest ? digest.slice(0, 12) : '—'
}
