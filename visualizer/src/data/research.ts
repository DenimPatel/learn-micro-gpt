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
import type { ResearchRow, ResearchTrack, ResearchVerdict } from './types'

/**
 * The tracks the loop can be asked to optimise, in the order the page lists them.
 *
 * Read from `results.json` rather than repeated here, because the generator is
 * the thing that decides which tracks exist: a track added to the harness and
 * missed in this array would be a track the site silently refuses to show.
 */
export function tracks(): ResearchTrack[] {
  return Object.keys(research.tracks) as ResearchTrack[]
}

/** A row's track, defaulted. Rows predating the column are all Rust. */
export function rowTrack(row: ResearchRow): ResearchTrack {
  return row.track ?? 'rust'
}

/** Every row for one track. The first filter everything else applies. */
export function trackRows(track: ResearchTrack): ResearchRow[] {
  return research.runs.filter((row) => rowTrack(row) === track)
}

/** Rows with a real measurement. A `crash` has none, and its zeros are absence. */
export function measuredRows(track: ResearchTrack = 'rust'): ResearchRow[] {
  return trackRows(track).filter((row) => row.status !== 'crash' && row.loss > 0)
}

/** Everything the loop actually tried on one track, baseline excluded. */
export function experiments(track: ResearchTrack = 'rust'): ResearchRow[] {
  return trackRows(track).filter((row) => row.status !== 'baseline')
}

export function statusCount(status: ResearchVerdict, track: ResearchTrack = 'rust'): number {
  return trackRows(track).filter((row) => row.status === status).length
}

/**
 * The best kept candidate for one track, or null when nothing has been kept.
 *
 * Null rather than a number borrowed from another track, because a "best" across
 * languages would be a comparison between a Go build and a Rust build measured
 * against different comparators -- two numbers about different machines' worth of
 * work, ranked against each other.
 */
export function bestRow(track: ResearchTrack = 'rust'): ResearchRow | null {
  return research.bests[track] ?? null
}

/** A track's frozen comparator as this session measured it. */
export function baselineRow(track: ResearchTrack = 'rust'): ResearchRow | null {
  return research.baselines[track] ?? null
}

/**
 * The run ids on one track's frontier, as recorded.
 *
 * `tools/autoresearch.py` computes this and `results.json` carries it, rather
 * than this file recomputing it. Two implementations of "is this dominated" is
 * two chances to disagree, and the disagreement would be invisible.
 */
export function frontierIds(track: ResearchTrack = 'rust'): string[] {
  return research.pareto[track] ?? []
}

export function isFrontier(runId: string, track: ResearchTrack = 'rust'): boolean {
  return frontierIds(track).includes(runId)
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

// --- across tracks -----------------------------------------------------------

/**
 * One measured run, as a multiple of its own track's frozen comparator.
 *
 * `lossRatio` and `speedRatio` are both 1.0 at that track's session baseline, so
 * the two numbers are "how many times its own starting point", which is the only
 * form in which a Go steps-per-second and a C steps-per-second can share an axis.
 */
export interface RelativeRun {
  row: ResearchRow
  track: ResearchTrack
  language: string
  lossRatio: number
  speedRatio: number
  onFrontier: boolean
}

/** A track that can be placed on a ratio axis at all, with its runs. */
export interface PlottableTrack {
  track: ResearchTrack
  language: string
  runs: RelativeRun[]
  /** How many of the runs are on this track's own frontier. */
  frontierCount: number
  /** The lowest loss ratio any kept or discarded run reached. 1.0 if none beat it. */
  bestLossRatio: number
  /** The highest speed ratio any run reached. 1.0 if none beat it. */
  bestSpeedRatio: number
  keepCount: number
}

/**
 * Every track's runs, normalised against that track's own session baseline.
 *
 * ## Why this exists, and why the ratios are the only honest version of it
 *
 * This page says in three separate places that a steps-per-second on one track
 * is not a steps-per-second on another, and it means it: the C port runs at
 * 27,000 steps per second and the TypeScript port at 245, which is 110x, and
 * the two numbers are not 110x apart in any comparable sense -- they are the
 * rate of two different implementations of the same algorithm on the same
 * machine, each measured against its own frozen comparator. Plotting them on one
 * axis would produce a picture that is *visually* true (the C cloud really is
 * higher) and *substantively* false (nothing about the C port being faster says
 * the loop did better on C). It is the same mistake as ranking a Go build
 * against a Rust build measured against different comparators, which is why
 * `bestRow` refuses to borrow a number from another track.
 *
 * Dividing each run by its own baseline removes the part that cannot be compared
 * and keeps the part that can: how far the loop moved that track, as a multiple
 * of where that track started. After it, all four comparators sit at exactly
 * (1.0, 1.0) -- the same point, by construction, which is the clearest possible
 * statement that the axes are relative. A reader asking "how well does the system
 * optimise, in each language" is asking exactly that question, and the answer is
 * a ratio.
 *
 * ## Why a track can be missing
 *
 * A track with no baseline has nothing to be a ratio *of*, and a track with no
 * baseline and no runs has nothing to plot at all. Both are dropped rather than
 * drawn at zero: a run plotted against a missing baseline would sit at infinity,
 * and drawing it at 0.0 would be a fabrication. The caller is expected to report
 * which tracks it left out, because a silently shorter chart reads as "that
 * language has not been tried" when the real reason may be that its baseline has
 * not been seeded.
 */
export function relativeRuns(): PlottableTrack[] {
  const placed: PlottableTrack[] = []
  for (const track of tracks()) {
    const baseline = baselineRow(track)
    if (!baseline || !(baseline.loss > 0) || !(baseline.steps_per_sec > 0)) continue
    const frontier = new Set(frontierIds(track))
    const language = research.tracks[track]?.language ?? track
    const runs: RelativeRun[] = measuredRows(track)
      .filter((row) => row.steps_per_sec > 0)
      .map((row) => ({
        row,
        track,
        language,
        lossRatio: row.loss / baseline.loss,
        speedRatio: row.steps_per_sec / baseline.steps_per_sec,
        onFrontier: frontier.has(row.run_id),
      }))
    placed.push({
      track,
      language,
      runs,
      frontierCount: runs.filter((run) => run.onFrontier).length,
      bestLossRatio: runs.reduce((best, run) => Math.min(best, run.lossRatio), 1),
      bestSpeedRatio: runs.reduce((best, run) => Math.max(best, run.speedRatio), 1),
      keepCount: runs.filter((run) => run.row.status === 'keep').length,
    })
  }
  return placed
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
 * `1.0` is a correct gradient, and no port measures it: each tape computes
 * `rmsnorm` outside the tape, so each measures a fixed ratio above 1 -- 1.063 on
 * the Rust candidate as it stood, 1.127 for the frozen Go build, 0.722 for
 * TypeScript. That is not a fault in the measurement but a real property of the
 * tape, and it is `docs/KNOWN-ISSUES.md` issue 5. So the function does not colour
 * a ratio by distance from 1.0, and does not quote a single expected value: the
 * number belongs to whichever port produced it, and a reader comparing two tracks
 * needs each track's own, not a constant that is right for one of them.
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
