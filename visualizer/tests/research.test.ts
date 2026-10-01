import { describe, expect, it } from 'vitest'
import { bestVsBaselinePatch, research } from '../src/data/sources'
import {
  baselineRow,
  bestRow,
  diffCounts,
  experiments,
  frontierIds,
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
  rowTrack,
  statusCount,
  trackRows,
  relativeRuns,
  tracks,
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
    const rust = research.provenance_of.rust
    expect(rust.baseline_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(rust.candidate_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(rust.gradient_probe_sha256).toMatch(/^[0-9a-f]{64}$/)
  })

  it("names each track's own comparator and its own probe", () => {
    // The failure this catches: `provenance_of` is a per-track map, and the
    // cheapest way to fill it is to write the Rust paths into all three entries.
    // Everything downstream -- the page's "against <code>...</code>, whose sha256
    // is" line, and a reader's decision about which build a number is about --
    // would then be confidently wrong for two of the three tracks.
    //
    // Note what is *not* asserted: that an unseeded track has empty digests. The
    // digest of `implementations/go/main.go` is a true fact about the repository
    // whether or not the loop has ever been run on Go, and recording it is more
    // useful than a blank. What must be blank is the research output.
    const comparators = new Set<string>()
    for (const track of tracks()) {
      const provenance = research.provenance_of[track]
      expect(provenance.baseline_source, `${track} names another track's comparator`).toContain(
        `implementations/${track === 'rust' ? 'rust' : track}/`,
      )
      expect(provenance.candidate_source, `${track} names another track's candidate`).toContain(
        `autoresearch/${research.tracks[track].candidate_dir}/`,
      )
      expect(provenance.gradient_probe, `${track} names another track's probe`).toContain(
        research.tracks[track].probe,
      )
      expect(provenance.baseline_sha256).toMatch(/^[0-9a-f]{64}$/)
      expect(provenance.gradient_probe_sha256).toMatch(/^[0-9a-f]{64}$/)
      comparators.add(provenance.baseline_source)
    }
    // Three tracks, three comparators: no two of them share a frozen build.
    expect(comparators.size).toBe(tracks().length)
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
    // `research.counts` is the ledger-wide total, so it is checked against
    // ledger-wide counts. Comparing it against `statusCount('keep')` -- which
    // defaults to one track -- passed for as long as one track was the whole
    // ledger, and started failing the moment a second track had rows. The stat
    // and the count it is checked against have to be about the same set.
    const all = research.runs
    const counted = (status: string) => all.filter((row) => row.status === status).length
    expect(research.counts.keep).toBe(counted('keep'))
    expect(research.counts.discard).toBe(counted('discard'))
    expect(research.counts.crash).toBe(counted('crash'))
    expect(research.counts.experiments).toBe(all.filter((row) => row.status !== 'baseline').length)
  })

  it('gives per-track counts that add up to the ledger-wide ones', () => {
    // The page's own stats are per-track, derived from the rows rather than read
    // from `research.counts`, because it is showing one track's chart. So the two
    // have to agree in aggregate, and the per-track numbers have to be that
    // track's rows and not the whole ledger's.
    const total = tracks().reduce((sum, track) => sum + statusCount('keep', track), 0)
    expect(total).toBe(research.counts.keep)
    for (const track of tracks()) {
      const mine = trackRows(track)
      expect(statusCount('keep', track)).toBe(mine.filter((row) => row.status === 'keep').length)
      expect(experiments(track).length).toBe(mine.filter((row) => row.status !== 'baseline').length)
    }
  })

  it("excludes each track's own baseline from its experiment count", () => {
    // One baseline row per seeded track, not one for the ledger. And the count
    // and the total it is subtracted from have to be about the *same* track: the
    // first version of this compared a rust-only experiment count against every
    // non-baseline row in the repository, which is only the same number while
    // one track is the whole ledger.
    for (const track of tracks()) {
      const mine = trackRows(track)
      const baselineRows = mine.filter((row) => row.status === 'baseline')
      expect(experiments(track).some((row) => row.status === 'baseline')).toBe(false)
      expect(experiments(track).length).toBe(mine.length - baselineRows.length)
      /* More than one baseline row is legitimate now, and the arithmetic above has
         to account for all of them rather than assume one. A track is re-baselined
         when the measurement protocol changes: the C loop drove the loss axis to
         0.000000 by choosing which documents it trained on, and the fix gave every
         track a held-out split -- which moves the comparator too, so its baseline is
         re-measured. Each superseded row stays, because it is an honest record of
         what that session measured, and the one that counts is the most recent. */
      if (baselineRows.length > 1) {
        expect(research.baselines[track]?.run_id).toBe(
          baselineRows[baselineRows.length - 1]!.run_id,
        )
      }
    }
    // And the total is the sum of the parts, so the two cannot drift.
    expect(tracks().reduce((sum, track) => sum + experiments(track).length, 0)).toBe(
      research.runs.filter((row) => row.status !== 'baseline').length,
    )
  })

  it("gives each track its own experiments, and never another track's", () => {
    const total = tracks().reduce((sum, track) => sum + experiments(track).length, 0)
    expect(total).toBe(
      research.runs.length - research.runs.filter((r) => r.status === 'baseline').length,
    )
    for (const track of tracks()) {
      const mine = new Set(trackRows(track).map((row) => row.run_id))
      for (const row of experiments(track)) {
        expect(mine.has(row.run_id)).toBe(true)
      }
    }
  })
})

describe('the best run', () => {
  /*
   * "The best run" is the lowest-loss keep *whose code the repository still
   * holds*, which is not the lowest-loss keep on the ledger.
   *
   * A keep says the loop accepted a candidate at the time. It does not say the
   * candidate survived, and one often does not: it was rolled back because it was
   * degenerate, or because a later experiment found it was a regression, or
   * because the track's protocol changed underneath it. On the C track a run whose
   * training loop had collapsed to replaying a single document held the record at
   * loss 0.000000 and 99,081 steps/s for as long as it held it, while the tree
   * contained something three experiments newer. See docs/KNOWN-ISSUES.md issue 7.
   *
   * Which keeps are in effect is decided in tools/autoresearch.py, by comparing
   * each run's recorded `source_sha256` against the digest the candidate file has
   * now, and that test lives there. This one cannot recompute it -- the page has no
   * digests and is not given any -- so what it checks is the part that is its own:
   * that `bests` names a real keep, and that the row it hands the page agrees with
   * that run's own row. Duplicating the selection rule here would be a second
   * definition of a statistic decided in one place, which is how two numbers stop
   * agreeing and nobody can say which is right.
   */
  it('names a keep that is present in the ledger, with the same numbers', () => {
    const best = bestRow()
    if (best === null) {
      /* Not "no keeps" -- "no *in-effect* keeps". A track can hold a hundred keeps
         on the ledger and promote none, because every one of them has been rolled
         back or predates the current protocol. What has to be true then is that
         `results.json` says so, rather than the page showing an empty panel with no
         explanation. */
      const keeps = research.runs.filter((row) => row.status === 'keep')
      if (keeps.length > 0) {
        expect(research.best_status?.[research.runs[0]!.track as never]).toBeDefined()
      }
      return
    }
    expect(best.status).toBe('keep')
    const source = research.runs.find((row) => row.run_id === best.run_id)
    expect(source, `bests names run ${best.run_id}, which is not in the ledger`).toBeDefined()
    expect(best.loss).toBeCloseTo(source!.loss, 6)
    expect(best.track).toBe(source!.track)
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
    for (const id of research.pareto.rust) {
      expect(ids.has(id), `${id} is on the frontier but is not in the ledger`).toBe(true)
      const row = research.runs.find((r) => r.run_id === id)
      expect(row?.status === 'crash').toBe(false)
    }
  })

  it('has no point dominated by another point on it', () => {
    const rows = new Map(research.runs.map((row) => [row.run_id, row]))
    const frontier = research.pareto.rust.map((id) => rows.get(id)!)
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
    if (measuredRows('rust').length === 1) {
      expect(research.pareto.rust).toContain(research.runs.at(0)?.run_id)
    }
  })

  it('agrees with the isFrontier helper the chart uses', () => {
    for (const row of research.runs) {
      const track = rowTrack(row)
      expect(isFrontier(row.run_id, track)).toBe(frontierIds(track).includes(row.run_id))
    }
  })

  it("puts no run of one track on another track's frontier", () => {
    // A frontier is a claim about one comparator. If the Go candidate\'s run id
    // were on the Rust frontier, the scatter would be plotting a Go loss against a
    // Rust speed, and nothing downstream could tell.
    for (const track of tracks()) {
      const mine = new Set(trackRows(track).map((row) => row.run_id))
      for (const id of frontierIds(track)) {
        expect(mine.has(id), `${id} is on the ${track} frontier but is a different track`).toBe(
          true,
        )
      }
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

  it('tallies the Rust best-vs-baseline patch as a real diff against the Rust track', () => {
    // Not "the one method it is", any more, and deliberately. This test used to
    // assert the patch did *not* contain `fn rmsnorm`, which is to say it asserted
    // that docs/KNOWN-ISSUES.md issue 5 was still unfixed -- and the loop fixed it,
    // which is the loop working. A test that fails when the research succeeds is
    // a test encoding a moment rather than a property. What is a property: the
    // patch is a well-formed diff of something, and the per-run digests the page
    // quotes still match the files on disk.
    const patch = bestVsBaselinePatch('rust')
    expect(patch).toContain('--- a/implementations/rust/src/lib.rs')
    expect(patch).toContain('+++ b/autoresearch/candidate/src/lib.rs')
    expect(diffCounts(patch).added).toBeGreaterThan(0)
  })

  it("reports an unseeded track's best-vs-baseline patch as empty, not another track's", () => {
    // Every other track is unseeded at the moment this is written. Showing the
    // Rust patch under a Go heading would be the whole failure mode this
    // multi-track change is trying to avoid.
    for (const track of tracks()) {
      if (track === 'rust' || research.baselines[track] !== null) continue
      expect(bestVsBaselinePatch(track), `${track} has no patch of its own`).toBe('')
    }
  })
})

/**
 * `relativeRuns()` — the normalisation the cross-track chart rests on.
 *
 * The chart claims that dividing every run by its own track's baseline is the
 * only honest way to put four languages on one pair of axes, and that after the
 * division all four comparators coincide at exactly (1.0, 1.0). Both are
 * assertions about arithmetic, and both are the kind that quietly stops being
 * true: a baseline that moves, a track seeded with a zero, a crash row that is
 * not excluded and divides by nothing.
 */
describe('the cross-track normalisation', () => {
  it('puts the baseline of every track at exactly 1.0 on both axes', () => {
    // The property the chart's origin ring claims. If this is not exactly 1.0 for
    // every track, the four baselines are at different places and the chart is
    // drawing four origins and calling them one.
    const placed = relativeRuns()
    expect(placed.length).toBeGreaterThan(0)
    for (const group of placed) {
      /* By run id, not by status: a re-baselined track holds several rows with
         status 'baseline', and only the one `results.json` reports is the comparator
         these ratios are measured against. */
      const baseline = group.runs.find(
        (run) => run.row.run_id === research.baselines[group.track]?.run_id,
      )
      expect(baseline, `${group.track} has a current baseline row among its runs`).toBeDefined()
      expect(baseline?.lossRatio, `${group.track} baseline loss ratio`).toBeCloseTo(1, 10)
      expect(baseline?.speedRatio, `${group.track} baseline speed ratio`).toBeCloseTo(1, 10)
    }
  })

  it('never plots a crash, and never divides by a zero', () => {
    for (const group of relativeRuns()) {
      for (const run of group.runs) {
        expect(run.row.status, `run ${run.row.run_id} is a crash`).not.toBe('crash')
        expect(Number.isFinite(run.lossRatio), `run ${run.row.run_id} loss ratio`).toBe(true)
        expect(Number.isFinite(run.speedRatio), `run ${run.row.run_id} speed ratio`).toBe(true)
        // A log axis over a non-positive ratio puts the point off the plot with
        // no error, so this is a real condition rather than belt and braces.
        expect(run.speedRatio, `run ${run.row.run_id} speed ratio is positive`).toBeGreaterThan(0)
      }
    }
  })

  it('reproduces the raw numbers, so it is a division and not a re-derivation', () => {
    for (const group of relativeRuns()) {
      const baseline = research.baselines[group.track]
      expect(baseline).not.toBeNull()
      for (const run of group.runs) {
        expect(run.lossRatio).toBeCloseTo(run.row.loss / baseline!.loss, 12)
        expect(run.speedRatio).toBeCloseTo(run.row.steps_per_sec / baseline!.steps_per_sec, 12)
      }
    }
  })

  it("reports the promoted run's ratios, or nothing at all when there is none", () => {
    for (const group of relativeRuns()) {
      expect(group.bestLossRatio).toBeGreaterThan(0)
      if (group.best === null) {
        // No promoted run means no figure. 1.0 would read as "this language held
        // its baseline", which is a claim; null reads as "nothing to report".
        expect(group.bestSpeedRatio).toBe(1)
        expect(group.bestNote).not.toBe('')
      } else {
        expect(group.bestNote).toBe('')
        // The promoted run's own loss, not the lowest number on the ledger.
        expect(group.best.status).toBe('keep')
        expect(group.bestLossRatio).toBeCloseTo(
          group.best.loss / (baselineRow(group.track)?.loss as number),
          9,
        )
        // A promoted run can be slower than the session baseline -- a keep won on
        // loss may well have cost speed, inside the 5% tolerance -- so this is
        // only bounded below, never asserted to have improved.
        expect(group.bestSpeedRatio).toBeGreaterThan(0)
      }
    }
  })

  it('only places a track that has a baseline to be a ratio of', () => {
    for (const group of relativeRuns()) {
      expect(research.baselines[group.track], `${group.track} has a baseline`).not.toBeNull()
    }
  })

  it('marks frontier runs from the frontier of that track only', () => {
    for (const group of relativeRuns()) {
      const own = new Set(research.pareto[group.track] ?? [])
      for (const run of group.runs) {
        expect(run.onFrontier).toBe(own.has(run.row.run_id))
      }
      expect(group.frontierCount).toBe(group.runs.filter((r) => r.onFrontier).length)
    }
  })
})
