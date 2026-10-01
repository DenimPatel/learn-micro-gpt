/**
 * `#/research` — what the autoresearch loop has tried on the Rust track.
 *
 * The page exists because the loop's real output is not a number, it is a
 * *record*: what was proposed, what it measured, and which of those things were
 * kept. A page that showed only the best loss curve would misrepresent the whole
 * exercise, which is a search. So the discards and the crashes are on the page
 * next to the keeps, with the same prominence, and the empty state says plainly
 * when nothing has been tried.
 *
 * Three things are deliberately absent:
 *
 *  - No runtime fetch. Every number and every patch is a build-time `?raw` or
 *    JSON import, for the reasons in `data/sources.ts`.
 *  - No chart library. The site has two `<svg>` elements and no dependency on a
 *    charting package, and the two charts here are about forty lines of geometry
 *    each. Adding `recharts` to draw a scatter plot would be a worse trade than
 *    it looks: it would add a runtime dependency to a repository whose whole
 *    argument is that you can read all of its code.
 *  - No second opinion on the numbers. `data/research.ts` selects and scales, and
 *    `autoresearch/results.json` is the ledger. The page draws what it is given.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import {
  baselineRow,
  bestRow,
  diffCounts,
  experiments,
  formatGain,
  formatLoss,
  formatRatio,
  formatSpeed,
  frontierIds,
  linearScale,
  logScale,
  lossTicks,
  measuredRows,
  ratioErrorPercent,
  relativeRuns,
  shortDigest,
  statusCount,
  trackRows,
  tracks,
} from '../data/research'
// The three loaders live in `sources` beside the rest of the build-time imports,
// because they are data loading rather than arithmetic. `research.ts` selects and
// scales; it does not fetch, and keeping that line crisp is what stops the page
// growing a second, private data path.
import {
  bestVsBaselinePatch,
  loadCandidate,
  loadPatch,
  loadedCandidate,
  research,
} from '../data/sources'
import type { ResearchRow, ResearchTrack } from '../data/types'

const STATUS_LABEL: Record<ResearchRow['status'], string> = {
  baseline: 'baseline',
  keep: 'kept',
  discard: 'discarded',
  crash: 'crashed',
}

function StatusBadge({ status }: { status: ResearchRow['status'] }) {
  return (
    <span className={`research__badge research__badge--${status}`}>{STATUS_LABEL[status]}</span>
  )
}

/* --- plot geometry ---------------------------------------------------------
   The two charts below are measured, not assumed.

   Both used to be a `0 0 100 100` viewBox stretched into a box several times
   wider than it was tall with `preserveAspectRatio="none"`, which is why neither
   had an axis: a `<text>` in that coordinate system comes out stretched along
   one axis only, so a label was either unreadable or left out. The fix is not a
   nicer label, it is a coordinate system that matches the box. The viewBox is
   now the container's own width in CSS pixels with a height derived from it, so
   one user unit is one CSS pixel, `preserveAspectRatio` does its default thing
   (`xMidYMid meet`, an exact fit at this aspect), and a tick label is 11px of
   type whatever the screen. `vector-effect: non-scaling-stroke` is kept on the
   strokes anyway: it is free at a 1:1 transform and it keeps every stroke a
   device-pixel hairline if the box is ever a fraction of a pixel off. */

const clamp = (value: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, value))

/**
 * Below this rendered width the charts drop ticks rather than shrink them.
 *
 * A tick label is ~40px of monospace. Four of them on a 300px plot is a tick
 * every 60px, which reads; six is a picket fence. So the count is a function of
 * the space, not only of the number of points.
 */
const NARROW = 520

/**
 * The rendered width of a chart's container, in CSS pixels.
 *
 * This is what the viewBox is built from, and it is measured rather than
 * inferred from a media query because the chart is not laid out against the
 * viewport: it sits in a widget in a column whose width is the viewport minus a
 * 270px rail, and that column goes single-column at 980px. A media query would
 * have to be a second copy of the layout's breakpoints, and would still be wrong
 * about the widget's own padding.
 */
function usePlotWidth(fallback: number): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(fallback)
  useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    const measure = () => {
      const next = Math.round(node.clientWidth)
      if (next > 0) setWidth(next)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}

/**
 * Ticks for a logarithmic axis, on the decade grid.
 *
 * Decades, and not a uniform step in value, because a uniform step in value is
 * not what a log axis is measuring. Over the speed range this page shows -- 62
 * to 1362 steps per second, a little over one decade -- `speedTicks` returns
 * either 500, 1000, or 200, 400, 600, 800, 1000, 1200, and the second set is
 * evenly spaced *in value* and visibly bunched into the top quarter of the
 * plot. The spacing would be asserting the opposite of the thing the axis is
 * for. Speed here is compared as a ratio ("3x the reference"), so its grid is
 * 1, 2 and 5 inside each decade with the decades themselves marked, and equal
 * ratios get equal distance.
 *
 * Coarsening is by removing mantissas, not by widening a step: a narrow chart
 * gets the decades alone, which is the grid a reader can still do arithmetic
 * on. It lives here rather than beside `speedTicks` in `data/research.ts`
 * because that file is not part of this change; it belongs there, `speedTicks`
 * should become its linear-axis sibling or go away, and that is a separate edit.
 */
function decadeTicks(lo: number, hi: number, count: number): number[] {
  if (!(hi > lo) || lo <= 0) return []
  const grid = (mantissas: number[]): number[] => {
    const ticks: number[] = []
    const from = Math.floor(Math.log10(lo)) - 1
    const to = Math.ceil(Math.log10(hi)) + 1
    for (let exponent = from; exponent <= to; exponent += 1) {
      for (const mantissa of mantissas) {
        const value = Number((mantissa * 10 ** exponent).toFixed(6))
        if (value >= lo && value <= hi) ticks.push(value)
      }
    }
    return ticks
  }
  const full = grid([1, 2, 5])
  if (full.length <= count) return full
  const decades = grid([1])
  if (decades.length <= count) return decades
  // More than `count` decades is a range wider than any axis here, but it has
  // to degrade rather than overflow, so every other decade is kept.
  return decades.filter((_, index) => index % Math.ceil(decades.length / count) === 0)
}

/**
 * One colour per track, as a CSS custom property name.
 *
 * The four are aliases of ink tokens that already exist in both themes rather
 * than new hex values, so the dark theme is the light theme's tokens rather than
 * a second palette somebody has to remember to tune. The hues are also close to
 * each language's own -- amber for Rust, green for Go, the site's accent for
 * TypeScript because that is the language the playground on the home page runs,
 * and rose for C.
 *
 * The channel is *only* track. Status is carried by size and opacity, and frontier
 * by size and a separating stroke, because a third channel on a chart that already
 * has four hues and two sizes is a chart nobody can read at a glance.
 */
const TRACK_INK: Record<ResearchTrack, string> = {
  rust: 'var(--track-rust)',
  go: 'var(--track-go)',
  typescript: 'var(--track-typescript)',
  c: 'var(--track-c)',
}

/**
 * Every track's runs on one pair of axes, as multiples of their own comparators.
 *
 * The chart is `relativeRuns()` drawn: x is `loss / its own baseline loss`, y is
 * `steps_per_sec / its own baseline speed`. See `relativeRuns` for why the
 * ratios are the only version of this comparison that means anything, and
 * `ParetoScatter` for why the axes are linear and log respectively.
 *
 * What it adds over four separate scatters is the one thing four separate
 * scatters cannot show: that the four tracks occupy the same plane at all. The
 * comparators coincide at (1, 1) by construction, so the reader's eye has a
 * single origin to measure every language's progress from, and a language whose
 * cloud is tight against it is one the loop could not move. Judged track by
 * track that question is unanswerable, because each track's own spread is all the
 * reader ever sees.
 */
function CrossTrackScatter() {
  const [ref, width] = usePlotWidth(880)
  const [selected, setSelected] = useState<string | null>(null)
  const placed = useMemo(() => relativeRuns(), [])
  const runs = useMemo(() => placed.flatMap((group) => group.runs), [placed])
  const narrow = width < NARROW
  const xScale = useMemo(() => linearScale(runs.map((run) => run.lossRatio)), [runs])
  const yScale = useMemo(() => logScale(runs.map((run) => run.speedRatio)), [runs])
  const xTicks = useMemo(() => lossTicks(xScale, narrow ? 3 : 6), [xScale, narrow])
  const yTicks = useMemo(
    () => decadeTicks(yScale.min, yScale.max, narrow ? 3 : 6),
    [yScale, narrow],
  )

  if (runs.length === 0) {
    return (
      <p className="research__empty">
        No track has both a seeded baseline and a measured run, so there is nothing to compare. Run{' '}
        <code>make autoresearch-rust</code> against at least one track first.
      </p>
    )
  }

  const height = Math.round(clamp(320, width * 0.46, 500))
  const pad = { top: 34, right: 16, bottom: 52, left: narrow ? 52 : 60 }
  const plot = {
    w: Math.max(120, width - pad.left - pad.right),
    h: Math.max(160, height - pad.top - pad.bottom),
  }
  const axisY = pad.top + plot.h
  const xOf = (value: number) => pad.left + (xScale.at(value) / 100) * plot.w
  const yOf = (value: number) => pad.top + (1 - yScale.at(value) / 100) * plot.h

  const chosen = runs.find((run) => run.row.run_id === selected) ?? null
  const excluded = tracks().filter((track) => !placed.some((group) => group.track === track))

  const summary = placed
    .map(
      (group) =>
        `${group.language}: ${group.runs.length} runs, ${group.keepCount} kept, ${group.frontierCount} on its frontier`,
    )
    .join('. ')
  const all = [
    `Cross-track scatter of ${runs.length} measured runs from ${placed.length} languages, each plotted as a multiple of its own track's frozen comparator.`,
    `Horizontal axis: mean loss divided by that track's baseline loss, ${formatRatio(xScale.min)} to ${formatRatio(xScale.max)}, lower is better.`,
    `Vertical axis: steps per second divided by that track's baseline speed, on a log scale, ${formatRatio(yScale.min)} to ${formatRatio(yScale.max)}, higher is better.`,
    'Every comparator sits at 1.0 on both axes, so that point is the origin for all four.',
    summary,
  ].join(' ')

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Every language the loop has worked on, on one pair of axes. Each run is drawn as a{' '}
        <em>multiple of its own track&rsquo;s frozen comparator</em>, not as its own loss and steps
        per second, because those are not comparable: the C port runs at 27,000 steps/s and the
        TypeScript port at 245, and that 110&times; gap is a fact about two implementations rather
        than about anything the loop did. Divided by their own starting points, all four comparators
        land on exactly 1.0 &mdash; the hollow ring &mdash; and every experiment becomes a
        measurement of how far the loop moved that language from where it started. Down and to the
        left is better, up and to the right is better, and down-and-right is the trade the rule
        exists to price.
      </figcaption>

      <p className="chart__legend research__key research__key--tracks">
        {placed.map((group) => (
          <span key={group.track}>
            <i className="research__key-dot" style={{ background: TRACK_INK[group.track] }} />
            {group.language} ({group.runs.length})
          </span>
        ))}
        <span>
          <i className="research__key-dot research__key-dot--hollow" />
          all {placed.length} comparators, at 1.0 &times; 1.0
        </span>
        <span>
          <i className="research__key-dot research__key-dot--ring" />
          that track&rsquo;s frontier
        </span>
      </p>

      <div className="research__plot" ref={ref}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          className="chart research__scatter"
          role="img"
          aria-label={all}
        >
          {xTicks.map((tick) => (
            <line
              key={`g${tick}`}
              className="research__grid"
              x1={xOf(tick)}
              x2={xOf(tick)}
              y1={pad.top}
              y2={axisY}
            />
          ))}
          {yTicks.map((tick) => (
            <line
              key={`h${tick}`}
              className="research__grid"
              x1={pad.left}
              x2={pad.left + plot.w}
              y1={yOf(tick)}
              y2={yOf(tick)}
            />
          ))}

          {/*
            The 1.0 crosshair, drawn heavier than the grid and in `--ink` rather
            than `--faint`, because it is the only two lines on this chart that
            mean something: everything left of the vertical is a lower loss than
            the frozen port reached, everything above the horizontal is faster
            than it. A reader who finds that pair has found the reference.
          */}
          <line
            className="research__axis research__axis--reference"
            x1={xOf(1)}
            x2={xOf(1)}
            y1={pad.top}
            y2={axisY}
          />
          <line
            className="research__axis research__axis--reference"
            x1={pad.left}
            x2={pad.left + plot.w}
            y1={yOf(1)}
            y2={yOf(1)}
          />

          <line
            className="research__axis"
            x1={pad.left}
            x2={pad.left + plot.w}
            y1={axisY}
            y2={axisY}
          />
          <line className="research__axis" x1={pad.left} x2={pad.left} y1={pad.top} y2={axisY} />
          {xTicks.map((tick) => (
            <line
              key={`tx${tick}`}
              className="research__axis research__axis--tick"
              x1={xOf(tick)}
              x2={xOf(tick)}
              y1={axisY}
              y2={axisY + 4}
            />
          ))}
          {yTicks.map((tick) => (
            <line
              key={`ty${tick}`}
              className="research__axis research__axis--tick"
              x1={pad.left - 4}
              x2={pad.left}
              y1={yOf(tick)}
              y2={yOf(tick)}
            />
          ))}

          {/*
            One frontier staircase per track, in that track's ink. Per track and
            not one frontier across all four, deliberately: "nothing dominates
            this" is a statement about a candidate against its own frozen
            comparator, and a C run is never dominated by a TypeScript run in
            either direction. Drawing a single cross-track frontier would imply
            the languages compete, which is the claim the ratio axes exist to
            refuse.
          */}
          {placed.map((group) => {
            const ordered = group.runs
              .filter((run) => run.onFrontier)
              .sort((a, b) => a.lossRatio - b.lossRatio)
            return ordered.map((run, index) => {
              const previous = index > 0 ? ordered[index - 1] : undefined
              if (!previous) return null
              return (
                <path
                  key={`f${group.track}${run.row.run_id}`}
                  className="research__frontier"
                  style={{ stroke: TRACK_INK[group.track] }}
                  fill="none"
                  d={`M${xOf(previous.lossRatio)},${yOf(previous.speedRatio)} L${xOf(run.lossRatio)},${yOf(previous.speedRatio)} L${xOf(run.lossRatio)},${yOf(run.speedRatio)}`}
                />
              )
            })
          })}

          {/* The crowd, then what carries the argument, on top of it. */}
          {runs
            .filter((run) => !run.onFrontier && run.row.status !== 'baseline')
            .map((run) => (
              <circle
                key={run.row.run_id}
                className="research__point research__point--relative"
                style={{ fill: TRACK_INK[run.track] }}
                cx={xOf(run.lossRatio)}
                cy={yOf(run.speedRatio)}
                r={run.row.status === 'keep' ? 3.6 : 2.9}
              >
                <title>
                  {`${run.language} run ${run.row.run_id} (${STATUS_LABEL[run.row.status]}): ${formatRatio(run.lossRatio)}x its baseline loss, ${formatRatio(run.speedRatio)}x its baseline speed — ${run.row.description}`}
                </title>
              </circle>
            ))}
          {runs
            .filter((run) => run.onFrontier)
            .map((run) => (
              <circle
                key={`f${run.row.run_id}`}
                className="research__point research__point--relative is-frontier"
                style={{ fill: TRACK_INK[run.track] }}
                cx={xOf(run.lossRatio)}
                cy={yOf(run.speedRatio)}
                r="5"
              >
                <title>
                  {`${run.language} run ${run.row.run_id} (${STATUS_LABEL[run.row.status]}, on its track's frontier): ${formatRatio(run.lossRatio)}x baseline loss, ${formatRatio(run.speedRatio)}x baseline speed — ${run.row.description}`}
                </title>
              </circle>
            ))}

          {/*
            The comparators, as one mark rather than four. They all sit at exactly
            (1, 1) -- that is what dividing by them achieves -- so drawing four
            rings would draw one ring and three invisible ones. The legend says
            how many are there.
          */}
          <circle
            className="research__point research__point--baseline"
            cx={xOf(1)}
            cy={yOf(1)}
            r="6"
          >
            <title>
              {`The frozen comparator of each of the ${placed.length} languages, drawn once because they are all at exactly 1.0 x 1.0 by construction: ${placed
                .map(
                  (group) =>
                    `${group.language} run ${group.runs.find((run) => run.row.status === 'baseline')?.row.run_id ?? '—'}`,
                )
                .join(', ')}`}
            </title>
          </circle>
          {chosen ? (
            <circle
              className="research__point research__point--selected"
              cx={xOf(chosen.lossRatio)}
              cy={yOf(chosen.speedRatio)}
              r="9"
            >
              <title>
                {`${chosen.language} run ${chosen.row.run_id}: ${formatRatio(chosen.lossRatio)}x baseline loss, ${formatRatio(chosen.speedRatio)}x baseline speed`}
              </title>
            </circle>
          ) : null}

          {xTicks.map((tick) => (
            <text
              key={`lx${tick}`}
              className="research__tick"
              x={xOf(tick)}
              y={axisY + 17}
              textAnchor="middle"
            >
              {formatRatio(tick)}
            </text>
          ))}
          {yTicks.map((tick) => (
            <text
              key={`ly${tick}`}
              className="research__tick"
              x={pad.left - 8}
              y={yOf(tick)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {formatRatio(tick)}
            </text>
          ))}
          <text
            className="research__axis-title"
            x={pad.left + plot.w / 2}
            y={height - 10}
            textAnchor="middle"
          >
            {narrow
              ? 'loss x baseline'
              : 'mean loss as a multiple of that track\u2019s own baseline, and 1.0 is the frozen port'}
          </text>
          <text className="research__axis-title" x={pad.left} y={pad.top - 10} textAnchor="start">
            {narrow
              ? 'speed x baseline'
              : 'steps per second as a multiple of that track\u2019s own baseline, log scale'}
          </text>
        </svg>
      </div>

      <div className="widget__controls research__lookup">
        <label>
          find a run
          <select
            value={selected ?? ''}
            onChange={(event) => setSelected(event.target.value || null)}
          >
            <option value="">
              {runs.length} runs across {placed.length} languages &mdash; choose one&hellip;
            </option>
            {runs.map((run) => (
              <option key={run.row.run_id} value={run.row.run_id}>
                {run.language} {run.row.run_id} — {STATUS_LABEL[run.row.status]} —{' '}
                {run.row.description}
              </option>
            ))}
          </select>
        </label>
        {chosen ? (
          <span className="research__lookup-readout">
            {chosen.language} run {chosen.row.run_id}: {formatRatio(chosen.lossRatio)}x baseline
            loss, {formatRatio(chosen.speedRatio)}x baseline speed
            {chosen.onFrontier ? ' — on its track\u2019s frontier' : ''}
          </span>
        ) : null}
      </div>

      {/*
        The per-language numbers, because a scatter says which language moved
        furthest and not by how much on each axis, and "the loop got C to 0.91x
        its baseline loss" is a sentence this table lets a reader write.
      */}
      <table className="research__matrix">
        <caption>
          What the loop achieved per language, against that language&rsquo;s own baseline.
          &ldquo;Best&rdquo; is each language&rsquo;s <em>promoted</em> run &mdash; the lowest-loss
          keep whose code the repository still holds, not the lowest number ever recorded. A track
          can have a hundred rows on the ledger and no promoted run, because keeps get rolled back;
          where one language&rsquo;s best is a dash, its loss and speed read as nothing rather than
          as a number from a run the tree does not contain. &ldquo;Held out&rdquo; is that
          run&rsquo;s loss on the 250 documents every 128th one puts outside training: the number a
          candidate cannot move by choosing which documents it trains on.
        </caption>
        <thead>
          <tr>
            <th scope="col">language</th>
            <th scope="col">runs</th>
            <th scope="col">kept</th>
            <th scope="col">frontier</th>
            <th scope="col">best loss</th>
            <th scope="col">best speed</th>
          </tr>
        </thead>
        <tbody>
          {placed.map((group) => (
            <tr key={group.track}>
              <th scope="row">
                <i
                  className="research__key-dot research__key-dot--inline"
                  style={{ background: TRACK_INK[group.track] }}
                />
                {group.language}
              </th>
              <td>{group.runs.length}</td>
              <td>{group.keepCount}</td>
              <td>{group.frontierCount}</td>
              <td>{formatRatio(group.bestLossRatio)}</td>
              <td>{formatRatio(group.bestSpeedRatio)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {excluded.length > 0 ? (
        <p className="widget__note">
          Not plotted:{' '}
          {excluded.map((track) => research.tracks[track]?.language ?? track).join(', ')}{' '}
          {excluded.length === 1 ? 'has' : 'have'} no seeded baseline, and a run with no baseline is
          not a ratio of anything.
        </p>
      ) : null}
    </figure>
  )
}

/**
 * The Pareto scatter: loss on x, speed on y, one point per measured run.
 *
 * This is the objective, drawn. The rule is "win one axis, do not lose the other
 * by more than the tolerance", so a scatter is the only honest picture of it: a
 * scalar would have to throw one of the two away, and the frontier it produces
 * is a set of points in a plane rather than a ranking. The alternative -- a
 * table, a beeswarm, or a pair of small multiples -- trades the plane for rows
 * a reader then has to sort in their head, which is the thing this chart is
 * for.
 *
 * Density is the objection to sixty-odd marks, and it is real, but it is not
 * the objection it looks like: no two runs land on the same pair of numbers.
 * What crowds is a column of runs at one loss value (a dozen at exactly the
 * baseline's) and a band of runs inside a hundredth of a nat, and at 3px those
 * overlap. Three things answer it:
 *
 *  - The crowd is drawn at partial opacity, so overlapping runs read as a dark
 *    patch. That is not a loss of precision, it is the answer: it says *where*
 *    the loop spent its runs, at a resolution no individual run is legible at
 *    and does not need to be.
 *  - The marks that carry the argument are drawn over the top of the crowd, at
 *    full opacity and a size up: the frontier, the best run, and the baseline.
 *    The baseline is a ring rather than a dot as well as a different colour,
 *    because a comparator that differs only in colour is invisible to a reader
 *    who cannot see colour, and the baseline is the one mark that must be
 *    found. It is also drawn last and filled with the widget's own background,
 *    so it punches a hole in whatever it is sitting on.
 *  - The run selector under the plot names any single run and rings it, so
 *    "which one is the point I cannot see" has an exact answer rather than a
 *    shrug. A control rather than a tooltip, because this page has to work
 *    without a mouse, and the runs are already listed in the ledger above under
 *    the same ids. That is also why every mark keeps its `<title>`.
 */
function ParetoScatter({ track }: { track: ResearchTrack }) {
  const rows = measuredRows(track)
  const [selected, setSelected] = useState<string | null>(null)
  const [ref, width] = usePlotWidth(760)
  const loss = useMemo(() => linearScale(rows.map((row) => row.loss)), [rows])
  const speed = useMemo(() => logScale(rows.map((row) => row.steps_per_sec)), [rows])
  const narrow = width < NARROW
  const ticks = useMemo(() => lossTicks(loss, narrow ? 3 : 6), [loss, narrow])
  const speedMarks = useMemo(
    () => decadeTicks(speed.min, speed.max, narrow ? 3 : 6),
    [speed, narrow],
  )
  const frontier = new Set(frontierIds(track))
  const best = bestRow(track)

  if (rows.length === 0) {
    return (
      <p className="research__empty">
        No run has produced a measurement yet, so there is nothing to plot.
      </p>
    )
  }

  // One user unit is one CSS pixel, so these are pixel paddings: a tick label is
  // ~40px of monospace and needs its own gutter on the left, 34px of headroom
  // carries the y axis's title, and 46px of foot carries the x tick labels and
  // the x title.
  const height = Math.round(clamp(300, width * 0.46, 460))
  const pad = { top: 34, right: 16, bottom: 46, left: narrow ? 52 : 56 }
  const plot = {
    w: Math.max(120, width - pad.left - pad.right),
    h: Math.max(150, height - pad.top - pad.bottom),
  }
  const axisY = pad.top + plot.h
  const xOf = (value: number) => pad.left + (loss.at(value) / 100) * plot.w
  const yOf = (value: number) => pad.top + (1 - speed.at(value) / 100) * plot.h

  const summary = rows
    .map(
      (row) =>
        `run ${row.run_id}: loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps per second, ${STATUS_LABEL[row.status]}`,
    )
    .join('. ')
  const chosen = rows.find((row) => row.run_id === selected) ?? null

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Every measured run, on the two axes the loop actually optimises. Left is the mean loss over
        the last {research.protocol.trend_window} of {research.protocol.steps} steps &mdash; lower
        is better. Up is steps per second &mdash; higher is better. A point down and to the left
        dominates a point up and to the right, and the joined line is the frontier of points nothing
        dominates. Speed is on a log scale because it is compared as a ratio, and a linear axis
        makes a 3&times; difference look like a nudge; its ticks are therefore at decades rather
        than at equal intervals of steps per second.
      </figcaption>

      {/*
        A legend, because the plot has four kinds of mark and no way to ask
        which is which. The counts are in it because a reader who sees a hundred
        small marks wants to know how they divide before they start looking at
        any one of them.
      */}
      <p className="chart__legend research__key">
        <span>
          <i className="research__key-dot research__key-dot--keep" />
          kept ({statusCount('keep', track)})
        </span>
        <span>
          <i className="research__key-dot research__key-dot--discard" />
          discarded ({statusCount('discard', track)})
        </span>
        <span>
          <i className="research__key-dot research__key-dot--baseline" />
          session baseline
        </span>
        <span>
          <i className="research__key-line" />
          the frontier &mdash; {frontier.size} runs nothing dominates
        </span>
      </p>

      <div className="research__plot" ref={ref}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          className="chart research__scatter"
          role="img"
          aria-label={`Pareto scatter of ${rows.length} measured runs on two axes. Horizontal axis: mean loss over the last ${research.protocol.trend_window} of ${research.protocol.steps} steps, ${formatLoss(loss.min)} to ${formatLoss(loss.max)}, lower is better. Vertical axis: steps per second on a log scale, ${formatSpeed(speed.min)} to ${formatSpeed(speed.max)}, higher is better. ${summary}.`}
        >
          {/* Grid, then axes. The grid is `--edge-soft` and the axes are
              `--edge-strong`: a gridline the eye can mistake for an axis is a
              gridline that cannot be read as "nothing is measured here". */}
          {ticks.map((tick) => (
            <line
              key={`l${tick}`}
              className="research__grid"
              x1={xOf(tick)}
              x2={xOf(tick)}
              y1={pad.top}
              y2={axisY}
            />
          ))}
          {speedMarks.map((tick) => (
            <line
              key={`s${tick}`}
              className="research__grid"
              x1={pad.left}
              x2={pad.left + plot.w}
              y1={yOf(tick)}
              y2={yOf(tick)}
            />
          ))}
          <line
            className="research__axis"
            x1={pad.left}
            x2={pad.left + plot.w}
            y1={axisY}
            y2={axisY}
          />
          <line className="research__axis" x1={pad.left} x2={pad.left} y1={pad.top} y2={axisY} />
          {ticks.map((tick) => (
            <line
              key={`lx${tick}`}
              className="research__axis research__axis--tick"
              x1={xOf(tick)}
              x2={xOf(tick)}
              y1={axisY}
              y2={axisY + 4}
            />
          ))}
          {speedMarks.map((tick) => (
            <line
              key={`sx${tick}`}
              className="research__axis research__axis--tick"
              x1={pad.left - 4}
              x2={pad.left}
              y1={yOf(tick)}
              y2={yOf(tick)}
            />
          ))}

          {/* The frontier, drawn as a staircase. Straight segments would imply
              interpolations between points that were never measured, and the
              region between two frontier points is dominated, not intermediate. */}
          {rows
            .filter((row) => frontier.has(row.run_id))
            .sort((a, b) => a.loss - b.loss)
            .map((row, index, ordered) => {
              const previous = index > 0 ? ordered[index - 1] : undefined
              if (!previous) return null
              return (
                <path
                  key={`f${row.run_id}`}
                  className="research__frontier"
                  fill="none"
                  d={`M${xOf(previous.loss)},${yOf(previous.steps_per_sec)} L${xOf(row.loss)},${yOf(previous.steps_per_sec)} L${xOf(row.loss)},${yOf(row.steps_per_sec)}`}
                />
              )
            })}

          {/*
            The crowd, then the marks that matter, in that order: a frontier run
            that lands under three discards is still visible, which it would not
            be if the order were by status.
          */}
          {rows.map((row) => {
            const isBest = best?.run_id === row.run_id
            const isOnFrontier = frontier.has(row.run_id)
            if (isBest || isOnFrontier || row.status === 'baseline') return null
            return (
              <circle
                key={row.run_id}
                className={`research__point research__point--${row.status}`}
                cx={xOf(row.loss)}
                cy={yOf(row.steps_per_sec)}
                r="3.1"
              >
                <title>
                  {`run ${row.run_id} (${STATUS_LABEL[row.status]}): loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps/s — ${row.description}`}
                </title>
              </circle>
            )
          })}
          {rows
            .filter((row) => frontier.has(row.run_id))
            .map((row) => (
              <circle
                key={`g${row.run_id}`}
                className={`research__point research__point--${row.status} is-frontier`}
                cx={xOf(row.loss)}
                cy={yOf(row.steps_per_sec)}
                r="4.3"
              >
                <title>
                  {`run ${row.run_id} (${STATUS_LABEL[row.status]}, on the frontier): loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps/s — ${row.description}`}
                </title>
              </circle>
            ))}
          {/* The baseline is a ring rather than a dot: it is the one mark a
              reader has to find among a hundred, and half a dozen candidates
              share its loss, so it is drawn on top, opaque, and hollow. */}
          {rows
            .filter((row) => row.status === 'baseline')
            .map((row) => (
              <circle
                key={`b${row.run_id}`}
                className="research__point research__point--baseline"
                cx={xOf(row.loss)}
                cy={yOf(row.steps_per_sec)}
                r="4.6"
              >
                <title>
                  {`run ${row.run_id} (session baseline): loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps/s — ${row.description}`}
                </title>
              </circle>
            ))}
          {rows
            .filter((row) => best?.run_id === row.run_id && !frontier.has(row.run_id))
            .map((row) => (
              <circle
                key={`n${row.run_id}`}
                className="research__point research__point--keep is-best"
                cx={xOf(row.loss)}
                cy={yOf(row.steps_per_sec)}
                r="5.2"
              >
                <title>{`run ${row.run_id} (best kept): loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps/s — ${row.description}`}</title>
              </circle>
            ))}
          {chosen ? (
            <circle
              className="research__point research__point--selected"
              cx={xOf(chosen.loss)}
              cy={yOf(chosen.steps_per_sec)}
              r="8"
            />
          ) : null}

          {ticks.map((tick) => (
            <text
              key={`tx${tick}`}
              className="research__tick"
              x={xOf(tick)}
              y={axisY + 17}
              textAnchor="middle"
            >
              {formatLoss(tick)}
            </text>
          ))}
          {speedMarks.map((tick) => (
            <text
              key={`ty${tick}`}
              className="research__tick"
              x={pad.left - 8}
              y={yOf(tick)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {formatSpeed(tick)}
            </text>
          ))}
          <text
            className="research__axis-title"
            x={pad.left + plot.w / 2}
            y={height - 8}
            textAnchor="middle"
          >
            {narrow
              ? 'mean loss'
              : `mean loss, last ${research.protocol.trend_window} of ${research.protocol.steps} steps`}
          </text>
          {/*
            The y axis's title, horizontal, in the headroom above the axis and
            ending flush with it. Rotated down the left edge is the other
            convention and it does not survive contact with a 40px tick label:
            the title occupies the same 12px column the labels hang into, and
            whichever tick happens to be at the middle of the axis collides with
            it. This cannot collide, and it costs the plot 12px of height.
          */}
          <text className="research__axis-title" x={pad.left} y={pad.top - 10} textAnchor="start">
            {narrow ? 'steps/s' : 'steps per second — log scale'}
          </text>
        </svg>
      </div>

      {/*
        The way to read a point you cannot see. Not a tooltip: a tooltip is a
        mouse affordance, this page has to work on a phone, and the runs are
        already listed in the ledger above with the same ids, so the selector is
        a lookup rather than a control.
      */}
      <div className="widget__controls research__lookup">
        <label>
          find a run
          <select
            value={selected ?? ''}
            onChange={(event) => setSelected(event.target.value || null)}
          >
            <option value="">{rows.length} measured runs &mdash; choose one&hellip;</option>
            {rows.map((row) => (
              <option key={row.run_id} value={row.run_id}>
                {row.run_id} — {STATUS_LABEL[row.status]} — {row.description}
              </option>
            ))}
          </select>
        </label>
        {chosen ? (
          <span className="research__lookup-readout">
            run {chosen.run_id}: {formatLoss(chosen.loss)}, {formatSpeed(chosen.steps_per_sec)}{' '}
            steps/s ({formatGain(chosen.loss_gain)} loss, {formatGain(chosen.speed_gain)} speed)
            {frontier.has(chosen.run_id) ? ' — on the frontier' : ''}
          </span>
        ) : null}
      </div>

      <dl className="stats">
        <div>
          <dt>loss range</dt>
          <dd>
            {formatLoss(Math.min(...rows.map((r) => r.loss)))} &ndash;{' '}
            {formatLoss(Math.max(...rows.map((r) => r.loss)))}
          </dd>
        </div>
        <div>
          <dt>speed range</dt>
          <dd>
            {formatSpeed(Math.min(...rows.map((r) => r.steps_per_sec)))} &ndash;{' '}
            {formatSpeed(Math.max(...rows.map((r) => r.steps_per_sec)))} steps/s
          </dd>
        </div>
        <div>
          <dt>on the frontier</dt>
          <dd>{frontier.size}</dd>
        </div>
        <div>
          <dt>measured</dt>
          <dd>{rows.length}</dd>
        </div>
      </dl>
    </figure>
  )
}

/**
 * What each experiment did to each axis, against the session baseline.
 *
 * The two bands are the tolerances and they are drawn, because the whole rule
 * turns on them: a point inside the lossy band and below the material line is a
 * discard, and a reader who cannot see where the band is cannot tell a
 * near-miss from a non-event.
 *
 * The y scale is symmetric-log, and it has to be. Both series are ratios against
 * one baseline, and they are not the same size: the loss series lives inside
 * ±10% because that is all the loss axis moves, while the speed series runs to
 * +2200% because fusing two matmuls into one is worth twenty-two times the
 * wall clock. On the linear axis this chart used to draw, a ±2440% domain puts
 * the whole tolerance band in the middle two tenths of one percent of the plot
 * and the loss series is a straight line lying on the zero line -- which reads
 * as "no experiment changed the loss", the one claim this chart must never make
 * by accident. So the scale is linear out to ±10%, which is twice the loss
 * tolerance and the entire range the loss axis occupies, and logarithmic above
 * it, where the speed axis actually is. The same objection that puts speed on a
 * log scale in the scatter applies to a change that is itself a ratio.
 */
const GAIN_LINEAR = 0.1
/** How much of each half of the plot the linear region gets. */
const GAIN_LINEAR_SHARE = 0.42

function GainsChart({ track }: { track: ResearchTrack }) {
  const rows = experiments(track).filter((row) => row.status !== 'crash')
  const [ref, width] = usePlotWidth(760)
  const narrow = width < NARROW

  if (rows.length === 0) {
    return (
      <p className="research__empty">
        No experiment has been measured yet. Run <code>make autoresearch-rust</code>.
      </p>
    )
  }
  const all = rows.flatMap((row) => [row.loss_gain, row.speed_gain])
  const bound = Math.max(0.05, ...all.map((value) => Math.abs(value))) * 1.1

  const height = Math.round(clamp(220, width * 0.3, 320))
  // 66px of left gutter, because `formatGain` runs to eight characters wide at
  // "+1000.0%" and a clipped tick label is worse than a narrow plot.
  const pad = { top: 34, right: 16, bottom: 54, left: narrow ? 66 : 70 }
  const plot = {
    w: Math.max(120, width - pad.left - pad.right),
    h: Math.max(130, height - pad.top - pad.bottom),
  }
  const axisY = pad.top + plot.h
  const zeroY = pad.top + plot.h / 2

  // Symmetric log: linear to `GAIN_LINEAR`, then log out to `bound`.
  const y = (value: number) => {
    const magnitude = Math.abs(value)
    const decades = Math.log10(bound / GAIN_LINEAR)
    const share =
      magnitude <= GAIN_LINEAR
        ? (magnitude / GAIN_LINEAR) * GAIN_LINEAR_SHARE
        : GAIN_LINEAR_SHARE +
          (Math.log10(magnitude / GAIN_LINEAR) / decades) * (1 - GAIN_LINEAR_SHARE)
    return zeroY - Math.sign(value) * share * (plot.h / 2)
  }
  const x = (index: number) =>
    rows.length === 1 ? pad.left + plot.w / 2 : pad.left + (index / (rows.length - 1)) * plot.w

  const tolerance = research.thresholds.loss_tolerance
  const linearMarks = narrow
    ? [-GAIN_LINEAR, 0, GAIN_LINEAR]
    : [-GAIN_LINEAR, -tolerance, 0, tolerance, GAIN_LINEAR]
  // The decade grid starts one decade above the linear region, so +10% is the
  // tolerance and not also a decade label.
  const logMarks = decadeTicks(GAIN_LINEAR * 10, bound, 3)
  // A negative decade is only labelled if a run actually reached it: this
  // ledger's worst regression is under 5%, and an axis labelled to -1000% on
  // the strength of a rule rather than a measurement is decoration.
  const slowest = Math.min(...rows.map((row) => Math.min(row.loss_gain, row.speed_gain)))
  const gainMarks = [
    ...linearMarks,
    ...logMarks.flatMap((mark) => (slowest <= -mark ? [-mark, mark] : [mark])),
  ]
  // Ordinal positions, not run ids: the ids skip, because a crash is a row that
  // never reached this chart, and a tick every n-th *experiment* is the only
  // spacing that does not claim runs exist between two labels.
  const xMarkCount = Math.min(narrow ? 3 : 6, rows.length)
  const xMarks = Array.from({ length: xMarkCount }, (_, index) =>
    xMarkCount === 1 ? 0 : Math.round((index * (rows.length - 1)) / (xMarkCount - 1)),
  )
  const step = rows.length > 1 ? plot.w / (rows.length - 1) : plot.w
  const cell = Math.max(1.5, Math.min(step * 0.66, 10))

  const series = (key: 'loss_gain' | 'speed_gain') =>
    rows
      .map(
        (row, index) =>
          `${index === 0 ? 'M' : 'L'}${x(index).toFixed(2)},${y(row[key]).toFixed(2)}`,
      )
      .join(' ')

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Each experiment&rsquo;s change against the session baseline: loss on one line, speed on the
        other. Zero is the baseline. A candidate is kept when one line clears its material threshold
        while the other stays inside its tolerance, so a point outside the shaded band on one axis
        is a discard no matter how far the other went. The axis is linear to &plusmn;10% &mdash;
        twice the loss tolerance, and outside the whole range the loss axis moves in &mdash; and
        logarithmic above it, because a change against a baseline is a ratio. It has to be: the loss
        series reaches {formatGain(Math.max(...rows.map((row) => row.loss_gain)))} and the speed
        series {formatGain(Math.max(...rows.map((row) => row.speed_gain)))} on one pair of axes.
      </figcaption>

      <p className="chart__legend research__key">
        <span>
          <i className="research__key-line research__key-line--loss" />
          change in loss
        </span>
        <span>
          <i className="research__key-line research__key-line--speed" />
          change in speed
        </span>
        <span>
          <i className="research__key-band" />
          the &plusmn;{(tolerance * 100).toFixed(0)}% tolerance
        </span>
        <span>
          <i className="research__key-cell research__key-cell--keep" />
          kept ({statusCount('keep', track)})
        </span>
        <span>
          <i className="research__key-cell" />
          discarded ({statusCount('discard', track)})
        </span>
      </p>

      <div className="research__plot" ref={ref}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          className="chart research__gains"
          role="img"
          aria-label={`Change against the session baseline for ${rows.length} experiments, in the order they ran. Vertical axis: percentage change, zero at the session baseline, linear to ${formatGain(GAIN_LINEAR)} and logarithmic above, up to ${formatGain(bound / 1.1)}. ${rows
            .map(
              (row) =>
                `run ${row.run_id}: loss ${formatGain(row.loss_gain)}, speed ${formatGain(row.speed_gain)}`,
            )
            .join('. ')}.`}
        >
          {gainMarks.map((mark) => (
            <line
              key={`g${mark}`}
              className="research__grid"
              x1={pad.left}
              x2={pad.left + plot.w}
              y1={y(mark)}
              y2={y(mark)}
            />
          ))}
          {xMarks.map((mark) => (
            <line
              key={`x${mark}`}
              className="research__grid"
              x1={x(mark)}
              x2={x(mark)}
              y1={pad.top}
              y2={axisY}
            />
          ))}

          {/* The regression tolerance, drawn rather than described: a point
              outside this band on either axis is a discard however far the
              other axis went. The top edge is the *smaller* of the two, not
              `y(-tolerance)`: this used to anchor the rect at -5% and grow it
              downwards, which put the whole band below the zero line. On the
              linear scale it drew before, 10% of a domain in the thousands of
              percent was invisible either way, so nothing caught it. */}
          <rect
            className="research__band"
            x={pad.left}
            y={Math.min(y(-tolerance), y(tolerance))}
            width={plot.w}
            height={Math.abs(y(tolerance) - y(-tolerance))}
          />
          <line
            className="research__zero"
            x1={pad.left}
            x2={pad.left + plot.w}
            y1={zeroY}
            y2={zeroY}
          />
          <line
            className="research__axis"
            x1={pad.left}
            x2={pad.left + plot.w}
            y1={axisY}
            y2={axisY}
          />
          <line className="research__axis" x1={pad.left} x2={pad.left} y1={pad.top} y2={axisY} />
          {gainMarks.map((mark) => (
            <line
              key={`gy${mark}`}
              className="research__axis research__axis--tick"
              x1={pad.left - 4}
              x2={pad.left}
              y1={y(mark)}
              y2={y(mark)}
            />
          ))}
          {xMarks.map((mark) => (
            <line
              key={`xx${mark}`}
              className="research__axis research__axis--tick"
              x1={x(mark)}
              x2={x(mark)}
              y1={axisY}
              y2={axisY + 4}
            />
          ))}

          <path
            className="research__series research__series--loss"
            d={series('loss_gain')}
            fill="none"
          />
          <path
            className="research__series research__series--speed"
            d={series('speed_gain')}
            fill="none"
          />

          {/*
            Which runs were kept, as a status strip under the axis rather than a
            tick per run. Sixty-five tick marks on a 300px axis fuse into one
            green bar and then say "all of them" for two thirds of the ledger;
            the strip is a barcode, which is what it is, and each cell keeps its
            own title so the exact run is still one hover away.
          */}
          {rows.map((row, index) => (
            <rect
              key={row.run_id}
              className={`research__strip research__strip--${row.status}`}
              x={x(index) - cell / 2}
              y={axisY + 8}
              width={cell}
              height="4"
              rx="1"
            >
              <title>
                {`run ${row.run_id} (${STATUS_LABEL[row.status]}): loss ${formatGain(row.loss_gain)}, speed ${formatGain(row.speed_gain)}`}
              </title>
            </rect>
          ))}

          {gainMarks.map((mark) => (
            <text
              key={`ty${mark}`}
              className="research__tick"
              x={pad.left - 8}
              y={y(mark)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {formatGain(mark)}
            </text>
          ))}
          {xMarks.map((mark) => (
            <text
              key={`tx${mark}`}
              className="research__tick"
              x={x(mark)}
              y={axisY + 28}
              // The first and last labels are anchored inwards: a run id
              // centred on the first experiment hangs 20px off the left of the
              // viewBox and is half a character wide.
              textAnchor={mark === 0 ? 'start' : mark === rows.length - 1 ? 'end' : 'middle'}
            >
              {rows[mark]?.run_id}
            </text>
          ))}
          <text
            className="research__axis-title"
            x={pad.left + plot.w / 2}
            y={height - 8}
            textAnchor="middle"
          >
            {narrow ? 'experiment' : 'experiment, in the order the loop ran them'}
          </text>
          {/* Horizontal, in the headroom, ending flush with the axis — see the
              scatter for why this is not rotated down the left edge. */}
          <text className="research__axis-title" x={pad.left} y={pad.top - 10} textAnchor="start">
            {narrow ? 'change' : 'change against the baseline'}
          </text>
        </svg>
      </div>
      <dl className="stats">
        <div>
          <dt>loss tolerance</dt>
          <dd>±{(research.thresholds.loss_tolerance * 100).toFixed(0)}%</dd>
        </div>
        <div>
          <dt>speed tolerance</dt>
          <dd>±{(research.thresholds.speed_tolerance * 100).toFixed(0)}%</dd>
        </div>
        <div>
          <dt>kept</dt>
          <dd>
            {statusCount('keep', track)} of {experiments(track).length}
          </dd>
        </div>
        <div>
          <dt>crashed</dt>
          <dd>{statusCount('crash', track)}</dd>
        </div>
      </dl>
    </figure>
  )
}

function Ledger({ track }: { track: ResearchTrack }) {
  const rows = trackRows(track)
  return (
    <div className="table-scroll">
      <table className="research__ledger">
        <caption>
          The whole ledger, in order, with the failures in it. A search that only shows its hits is
          not a search &mdash; the discarded rows are what make the kept ones mean something.
          Sourced from <code>autoresearch/results.tsv</code> by{' '}
          <code>tools/autoresearch.py --render</code>.
        </caption>
        <thead>
          <tr>
            <th scope="col">run</th>
            <th scope="col">verdict</th>
            <th scope="col">loss</th>
            <th scope="col">&Delta;loss</th>
            <th scope="col">steps/s</th>
            <th scope="col">&Delta;speed</th>
            <th scope="col">grad ratio</th>
            <th scope="col">what it tried</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.run_id} className={`research__row research__row--${row.status}`}>
              <th scope="row">
                <code>{row.run_id}</code>
              </th>
              <td>
                <StatusBadge status={row.status} />
              </td>
              {row.status === 'crash' ? (
                <td colSpan={5} className="research__failure">
                  {row.reason || 'no reason recorded'}
                </td>
              ) : (
                <>
                  <td>{formatLoss(row.loss)}</td>
                  <td>{formatGain(row.loss_gain)}</td>
                  <td>{formatSpeed(row.steps_per_sec)}</td>
                  <td>{formatGain(row.speed_gain)}</td>
                  <td>{formatRatio(row.grad_ratio)}</td>
                </>
              )}
              <td>{row.description || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * The gradient ratio column, called out.
 *
 * It is the most interesting number on the page and the least likely to be read
 * as a score, so it gets a section of its own. A correct gradient measures 1.0;
 * the frozen track measures 1.063, and the research loop's first experiment took
 * it to 0.9999 &mdash; which was then discarded, because a correct gradient that
 * costs 5% throughput is not obviously a win. Both facts are in
 * `docs/KNOWN-ISSUES.md` (issues 1 and 5).
 */
function GradientSection({ track }: { track: ResearchTrack }) {
  const rows = measuredRows(track)
  const baseline = baselineRow(track)
  const candidates = rows.filter((row) => row.status !== 'baseline')
  return (
    <>
      <h3>The gradient ratio</h3>
      <p>
        Every candidate is finite-difference checked before it is allowed to compete on loss. A loss
        number cannot tell a correct gradient from a wrong one: the C port&rsquo;s hand-written
        backward pass was wrong by a factor of −8, its loss curve still tracked the reference to
        within 7%, and it still trained (<code>docs/KNOWN-ISSUES.md</code> issue 1, fixed since). So
        the ratio is the directional derivative of the loss over all parameters against a central
        difference, where a correct gradient is 1.0.
      </p>
      {baseline ? (
        <p>
          The frozen Rust track measures <strong>{formatRatio(baseline.grad_ratio)}</strong>, which
          is {ratioErrorPercent(baseline.grad_ratio).toFixed(1)}% off a correct gradient. That is
          not a measurement error: <code>rmsnorm</code> reads values rather than handles, so the
          whole normalisation sits outside the autograd tape (issue 5). It is frozen on purpose,
          because it is the thing every candidate is measured against.
        </p>
      ) : null}
      {candidates.length > 0 ? (
        <ul className="research__ratios">
          {candidates.map((row) => (
            <li key={row.run_id}>
              run {row.run_id} measured {formatRatio(row.grad_ratio)} &mdash;{' '}
              {row.grad_ratio > 0
                ? `${Math.abs(ratioErrorPercent(row.grad_ratio)).toFixed(1)}% ${row.grad_ratio > 1 ? 'high' : 'low'}`
                : 'not measured'}{' '}
              ({STATUS_LABEL[row.status]})
            </li>
          ))}
        </ul>
      ) : (
        <p className="research__empty">No candidate has been measured yet.</p>
      )}
    </>
  )
}

/** The frozen track and the current candidate, and the patch between them. */
function Divergence({ patch, track }: { patch: string; track: ResearchTrack }) {
  const provenance = research.provenance_of[track]
  const [open, setOpen] = useState(false)

  /*
   * `useEffect`, not `useMemo`, and the cache read in the *state initialiser*.
   * The candidate is a dynamic `?raw` import in its own chunk (see
   * `data/sources.ts`), so loading it is a side effect with an asynchronous
   * completion: doing it in `useMemo` runs a promise during render, and reading
   * the cache inside the effect body would call `setState` on every mount. This is
   * the same shape `CodePanel` uses, for the same reasons, and eslint's
   * `set-state-in-effect` and `react-hooks/immutability` rules are right about
   * both of the other arrangements.
   */
  const [candidate, setCandidate] = useState<string | null>(() => loadedCandidate(track) ?? null)
  useEffect(() => {
    if (candidate !== null) return
    let cancelled = false
    void loadCandidate(track).then((text) => {
      if (!cancelled) setCandidate(text)
    })
    return () => {
      cancelled = true
    }
    // `candidate` is a guard, not a dependency: including it would restart the
    // import on every load and discard the result it was about to deliver.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const counts = diffCounts(patch)

  return (
    <>
      <h3>How far it has moved</h3>
      <p>
        The candidate is a byte-for-byte copy of the frozen track plus one method, so the patch
        below starts as a single <code>set_data</code> &mdash; the one thing a finite-difference
        probe cannot do without, because the arena is private. Everything after that is the
        loop&rsquo;s.
      </p>
      <p>
        <strong>
          {counts.added} line{counts.added === 1 ? '' : 's'} added, {counts.removed} removed
        </strong>{' '}
        against <code>{provenance.baseline_source}</code>, whose sha256 is{' '}
        <code>{shortDigest(provenance.baseline_sha256)}</code>.
      </p>
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        {open ? 'hide the patch' : 'show the patch'}
      </button>
      {open ? (
        <>
          <pre className="research__diff">
            <code>{patch || 'the candidate is byte-identical to the frozen track'}</code>
          </pre>
          <p className="widget__note">
            Current candidate: {candidate ? `${candidate.split('\n').length} lines` : 'loading…'}.
            Full text is on <code>#/compare</code> for the frozen track, and in{' '}
            <code>{provenance.candidate_source}</code> for this one.
          </p>
        </>
      ) : null}
    </>
  )
}

function RunPatches({ track }: { track: ResearchTrack }) {
  const rows = experiments(track).filter((row) => row.status !== 'crash')
  const [selected, setSelected] = useState<string | null>(null)
  const [patch, setPatch] = useState<string | null>(null)
  const [missing, setMissing] = useState(false)

  if (rows.length === 0) return null

  return (
    <>
      <h3>What each experiment changed</h3>
      <p>
        The patch the model proposed, exactly as <code>git apply</code> received it. One chunk per
        experiment, loaded when you ask for it &mdash; a hundred experiments is a hundred patches
        and nobody should download them to look at one.
      </p>
      <div className="widget__controls">
        <label>
          experiment
          <select
            value={selected ?? ''}
            onChange={(event) => {
              const next = event.target.value
              setSelected(next || null)
              setPatch(null)
              setMissing(false)
              if (!next) return
              void loadPatch(next).then((text) => {
                if (text === null) setMissing(true)
                else setPatch(text)
              })
            }}
          >
            <option value="">choose one…</option>
            {rows.map((row) => (
              <option key={row.run_id} value={row.run_id}>
                {row.run_id} — {STATUS_LABEL[row.status]} — {row.description}
              </option>
            ))}
          </select>
        </label>
      </div>
      {missing ? <p className="research__empty">No patch was recorded for that run.</p> : null}
      {patch !== null ? <pre className="research__diff">{patch}</pre> : null}
    </>
  )
}

export function ResearchPage() {
  const [track, setTrack] = useState<ResearchTrack>('rust')
  const all = tracks()
  // A track the loop has never been run on still gets a row here, with zeros, so
  // that "not measured yet" is a thing the page can say rather than a thing it
  // hides by omission. The per-track counts are derived from the rows rather than
  // read from `research.counts`, which is the ledger-wide total and would put a
  // Rust experiment count next to a Go chart.
  const rows = trackRows(track)
  const counts = {
    experiments: rows.filter((row) => row.status !== 'baseline').length,
    keep: rows.filter((row) => row.status === 'keep').length,
    discard: rows.filter((row) => row.status === 'discard').length,
    crash: rows.filter((row) => row.status === 'crash').length,
  }
  const noExperiments = counts.experiments === 0
  const info = research.tracks[track]

  return (
    <article className="research">
      <h1>What the loop has tried</h1>
      <p className="lede">
        A model rewrites one track, the harness measures it against that track&rsquo;s own frozen
        build, and a rule decides whether the rewrite is kept. The loop runs on{' '}
        {/* Written from the data, not from a list. This sentence named three
            languages in prose and went stale the moment a fourth was added, and
            it would have gone stale again silently — there is no test that can
            tell a hardcoded language list is out of date, because the sentence
            is still perfectly valid English. */}
        {all.map((name, index) => (
          <span key={name}>
            {index > 0 ? (index === all.length - 1 ? ' and ' : ', ') : ''}
            {research.tracks[name]?.language ?? name}
          </span>
        ))}
        ; pick one below. Both the kept and the discarded attempts are here, because a search that
        only shows its hits is not a search. Every number was measured on one machine in one session
        and is committed to the repository; CI re-measures the loss axis on every push and fails if
        the code on this page no longer produces the number quoted beside it.
      </p>

      <fieldset className="research__tracks">
        <legend>Which implementation</legend>
        {all.map((name) => (
          <button
            key={name}
            type="button"
            className="research__track"
            aria-pressed={name === track}
            onClick={() => setTrack(name)}
          >
            <span className="research__track-label">{research.tracks[name].language}</span>
            <span className="research__track-note">
              {trackRows(name).filter((row) => row.status === 'keep').length} kept
            </span>
          </button>
        ))}
      </fieldset>
      <p className="widget__note">
        Each track is measured against its own frozen comparator in its own candidate directory
        &mdash; {info?.candidate_dir}, a <code>{info?.source}</code> built with {info?.build}. A
        steps-per-second on one track is not a steps-per-second on another, so nothing here is
        compared across tracks.
      </p>

      {noExperiments ? (
        <p className="research__empty">
          No experiment has been run yet. The baseline below was measured when the workspace was
          created; the loop itself is started with <code>make autoresearch-rust</code>, and{' '}
          <code>docs/AUTORESEARCH.md</code> explains what it costs and what it will not let a
          candidate do.
        </p>
      ) : null}

      <h2>The ledger</h2>
      <dl className="stats">
        <div>
          <dt>experiments</dt>
          <dd>{counts.experiments}</dd>
        </div>
        <div>
          <dt>kept</dt>
          <dd>{counts.keep}</dd>
        </div>
        <div>
          <dt>discarded</dt>
          <dd>{counts.discard}</dd>
        </div>
        <div>
          <dt>crashed</dt>
          <dd>{counts.crash}</dd>
        </div>
      </dl>
      <Ledger track={track} />

      <h2>The objective</h2>
      <p>
        Two axes, not one. A candidate is <strong>kept</strong> if the loss falls by at least{' '}
        {(research.thresholds.loss_material * 100).toFixed(0)}% while speed stays within{' '}
        {(research.thresholds.speed_tolerance * 100).toFixed(0)}% of the baseline, <em>or</em> the
        speed rises by at least {(research.thresholds.speed_material * 100).toFixed(0)}% while the
        loss stays within {(research.thresholds.loss_tolerance * 100).toFixed(0)}%. A tie is a
        discard, and so is anything that improved its windowed loss by less than{' '}
        {(research.thresholds.min_relative_improvement * 100).toFixed(0)}% overall, however fast it
        is.
      </p>
      <p>
        The four numbers are not the same number four times. The two &ldquo;at least&rdquo; values
        are noise floors &mdash; the measured spread of a {research.protocol.trend_window}-step
        window of this loss curve &mdash; and the two &ldquo;within&rdquo; values are regression
        limits. They happen to come in pairs because one axis is far noisier than the other: a fixed
        seed makes the loss axis <em>deterministic</em>, while wall clock moves with whatever else
        the machine is doing.
      </p>
      <p>
        The model proposes and the harness disposes. <code>verdict()</code> is a pure function of
        measured medians, and the model never states an outcome &mdash; an LLM told &ldquo;keep it
        if the number went down&rdquo; drifts toward keeping its own bad ideas within about ten
        experiments, and then the log stops being evidence of anything.
      </p>

      <h2>Every language, on one pair of axes</h2>
      <p>
        The next chart is the only one on this page that puts two languages next to each other, and
        it is placed here for that reason rather than under the switcher: everything below it is
        about whichever track you have selected, and the question it answers &mdash; can the loop
        improve a model <em>in this language</em>, or only in the one it has been run against
        &mdash; is a question about all of them at once.
      </p>
      <CrossTrackScatter />

      <h2>Every run, on both axes</h2>
      <ParetoScatter track={track} />
      <GainsChart track={track} />

      <h2>The gate that is not a score</h2>
      <GradientSection track={track} />

      <h2>The code</h2>
      <Divergence patch={bestVsBaselinePatch(track)} track={track} />
      <RunPatches track={track} />

      <h2>How to read these numbers</h2>
      <ul className="caveats">
        {research.caveats.map((caveat) => (
          <li key={caveat.slice(0, 40)}>{caveat}</li>
        ))}
      </ul>

      <h2>Where this came from</h2>
      <p>
        The loop descends from{' '}
        <a href="https://github.com/karpathy/autoresearch">
          <code>karpathy/autoresearch</code>
        </a>{' '}
        (MIT, © Andrej Karpathy): a fixed protocol, one editable file, a ledger, and a keep/discard
        decision. Everything that differs here &mdash; a fixed step count instead of a wall-clock
        budget, a two-axis objective, a baseline re-measured every session, and failures committed
        rather than discarded &mdash; is in <code>docs/AUTORESEARCH.md</code> with the reason.
      </p>

      {research.model?.model ? (
        <p className="widget__note">
          Proposed by <code>{research.model.model}</code> at{' '}
          <code>
            {research.model.reasoning_parameter}={research.model.reasoning_effort}
          </code>
          , pinned in <code>autoresearch/model.json</code> and verified against the provider&rsquo;s
          catalogue on {research.model.verified}. {research.model.verification}
        </p>
      ) : null}

      {research.runner?.rustc ? (
        <p className="widget__note">
          Measured on {research.runner.os} ({research.runner.machine}) with{' '}
          <code>{research.runner.rustc}</code> on {research.runner.measured}.
        </p>
      ) : null}
    </article>
  )
}
