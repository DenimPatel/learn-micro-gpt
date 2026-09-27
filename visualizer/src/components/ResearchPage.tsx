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

import { useEffect, useMemo, useState } from 'react'
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
  isFrontier,
  linearScale,
  logScale,
  lossTicks,
  measuredRows,
  ratioErrorPercent,
  shortDigest,
  speedTicks,
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

/**
 * The Pareto scatter: loss on x, speed on y, one point per measured run.
 *
 * This is the objective, drawn. The rule is "win one axis, do not lose the other
 * by more than the tolerance", so a scatter is the only honest picture of it: a
 * scalar would have to throw one of the two away.
 */
function ParetoScatter({ track }: { track: ResearchTrack }) {
  const rows = measuredRows(track)
  const loss = useMemo(() => linearScale(rows.map((row) => row.loss)), [rows])
  const speed = useMemo(() => logScale(rows.map((row) => row.steps_per_sec)), [rows])
  const ticks = useMemo(() => lossTicks(loss), [loss])
  const speedMarks = useMemo(() => speedTicks(speed), [speed])
  const frontier = new Set(frontierIds(track))
  const best = bestRow(track)

  if (rows.length === 0) {
    return (
      <p className="research__empty">
        No run has produced a measurement yet, so there is nothing to plot.
      </p>
    )
  }

  const summary = rows
    .map(
      (row) =>
        `run ${row.run_id}: loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps per second, ${STATUS_LABEL[row.status]}`,
    )
    .join('. ')

  return (
    <figure className="widget">
      <figcaption className="widget__caption">
        Every measured run, on the two axes the loop actually optimises. Left is the mean loss over
        the last {research.protocol.trend_window} of {research.protocol.steps} steps &mdash; lower
        is better. Up is steps per second &mdash; higher is better. A point down and to the left
        dominates a point up and to the right, and the joined line is the frontier of points nothing
        dominates. Speed is on a log scale because it is compared as a ratio, and a linear axis
        makes a 3&times; difference look like a nudge.
      </figcaption>

      <svg
        viewBox="0 0 100 100"
        className="chart research__scatter"
        role="img"
        aria-label={`Pareto scatter of ${rows.length} measured runs. ${summary}.`}
        preserveAspectRatio="none"
      >
        {speedMarks.map((tick) => (
          <g key={`s${tick}`}>
            <line
              className="research__grid"
              x1="0"
              x2="100"
              y1={100 - speed.at(tick)}
              y2={100 - speed.at(tick)}
            />
          </g>
        ))}
        {ticks.map((tick) => (
          <line
            key={`l${tick}`}
            className="research__grid"
            x1={loss.at(tick)}
            x2={loss.at(tick)}
            y1="0"
            y2="100"
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
                d={`M${loss.at(previous.loss)},${100 - speed.at(previous.steps_per_sec)} L${loss.at(row.loss)},${100 - speed.at(previous.steps_per_sec)} L${loss.at(row.loss)},${100 - speed.at(row.steps_per_sec)}`}
              />
            )
          })}

        {rows.map((row) => {
          const isBest = best?.run_id === row.run_id
          return (
            <circle
              key={row.run_id}
              className={`research__point research__point--${row.status}${isFrontier(row.run_id, track) ? ' is-frontier' : ''}`}
              cx={loss.at(row.loss)}
              cy={100 - speed.at(row.steps_per_sec)}
              r={isBest ? 1.8 : 1.1}
            >
              <title>
                {`run ${row.run_id} (${STATUS_LABEL[row.status]}): loss ${formatLoss(row.loss)}, ${formatSpeed(row.steps_per_sec)} steps/s — ${row.description}`}
              </title>
            </circle>
          )
        })}
      </svg>

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
 */
function GainsChart({ track }: { track: ResearchTrack }) {
  const rows = experiments(track).filter((row) => row.status !== 'crash')
  if (rows.length === 0) {
    return (
      <p className="research__empty">
        No experiment has been measured yet. Run <code>make autoresearch-rust</code>.
      </p>
    )
  }
  const all = rows.flatMap((row) => [row.loss_gain, row.speed_gain])
  const bound = Math.max(0.05, ...all.map((value) => Math.abs(value))) * 1.1
  const y = (value: number) => 50 - (value / bound) * 50
  const x = (index: number) => (rows.length === 1 ? 50 : (index / (rows.length - 1)) * 92 + 4)

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
        is a discard no matter how far the other went.
      </figcaption>
      <svg
        viewBox="0 0 100 100"
        className="chart research__gains"
        role="img"
        aria-label={`Change against the session baseline for ${rows.length} experiments. ${rows
          .map(
            (row) =>
              `run ${row.run_id}: loss ${formatGain(row.loss_gain)}, speed ${formatGain(row.speed_gain)}`,
          )
          .join('. ')}.`}
        preserveAspectRatio="none"
      >
        <rect
          className="research__band"
          x="0"
          y={y(-research.thresholds.loss_tolerance)}
          width="100"
          height={Math.abs(
            y(research.thresholds.loss_tolerance) - y(-research.thresholds.loss_tolerance),
          )}
        />
        <line className="research__zero" x1="0" x2="100" y1="50" y2="50" />
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
        {rows.map((row, index) =>
          row.status === 'keep' ? (
            <line
              key={row.run_id}
              className="research__keep-tick"
              x1={x(index)}
              x2={x(index)}
              y1="96"
              y2="100"
            />
          ) : null,
        )}
      </svg>
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
        backward pass is wrong by a factor of −8 and its loss curve still tracks the reference to
        within 7% (<code>docs/KNOWN-ISSUES.md</code> issue 1). So the ratio is the directional
        derivative of the loss over all parameters against a central difference, where a correct
        gradient is 1.0.
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
        build, and a rule decides whether the rewrite is kept. The loop runs on Rust, Go and
        TypeScript; pick one below. Both the kept and the discarded attempts are here, because a
        search that only shows its hits is not a search. Every number was measured on one machine in
        one session and is committed to the repository; CI re-measures the loss axis on every push
        and fails if the code on this page no longer produces the number quoted beside it.
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
            {research.tracks[name].language}
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
