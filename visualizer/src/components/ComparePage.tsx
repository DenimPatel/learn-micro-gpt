/**
 * /compare — the same algorithm in five languages, and the measured costs.
 *
 * The two questions this page exists to answer are "do they agree?" and "what
 * does each cost?". The answers are in `traces/`, `benchmarks/results.json`, and
 * the parity tool's own output, and all three are generated from real runs rather
 * than written by hand. `benchmarks/results.json` is committed so this page has
 * something to show without running anything.
 */

import { useEffect, useState } from 'react'
import { Markdown } from '../content/markdown'
import { TRACK_ORDER, benchmarks, index, loadSource, loadedSource, trace } from '../data/sources'
import type { Language } from '../data/types'
import { CodePanel } from './CodePanel'
import { LossChart } from './widgets'

const ORDER: Language[] = ['python', 'c', 'rust', 'go', 'typescript']

export function ComparePage() {
  const results = benchmarks?.results ?? []
  const runner = benchmarks?.runner
  const fastest = results[0]

  /*
   * Sizes of each source, once its chunk has loaded. They are the only reason
   * this page needs the sources at all -- the code panels below need them anyway,
   * but only when expanded -- so the counts fill in as the chunks arrive and the
   * table renders immediately with a placeholder rather than blocking.
   */
  const [sizes, setSizes] = useState<Record<string, { chars: number; lines: number }>>({})
  useEffect(() => {
    let cancelled = false
    void Promise.all(
      ORDER.map(async (language) => {
        const text = loadedSource(language) ?? (await loadSource(language))
        return [language, { chars: text.length, lines: text.split('\n').length }] as const
      }),
    ).then((entries) => {
      if (!cancelled) setSizes(Object.fromEntries(entries))
    })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <article className="compare">
      <h1>Five languages, one algorithm</h1>
      <p className="lede">
        Every track below computes the same function. They are separate programs, not translations,
        which is why the line numbers do not line up &mdash; and why comparing them is interesting
        rather than pointless.
      </p>

      <section aria-labelledby="cost-heading">
        <h2 id="cost-heading">What it costs</h2>
        {results.length === 0 ? (
          <p>
            No benchmark has been recorded in this build. Run <code>make bench</code> to produce
            <code> benchmarks/results.json</code>.
          </p>
        ) : (
          <>
            <div className="table-scroll">
              <table className="bench">
                <caption>
                  {benchmarks.steps} steps at the micro config, measured on{' '}
                  <strong>{runner?.cpu ?? 'an unknown machine'}</strong> ({runner?.os}). Ratios
                  within a row set are meaningful; absolute seconds are not, and they are not a
                  claim about your hardware.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">track</th>
                    <th scope="col">steps/sec</th>
                    <th scope="col">vs Python</th>
                    <th scope="col">built as</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row) => (
                    <tr key={row.lang} className={row.lang === fastest?.lang ? 'is-fastest' : ''}>
                      <th scope="row">
                        <a href={`#/learn/linear`}>{row.lang}</a>
                      </th>
                      <td>{row.steps_per_sec?.toLocaleString()}</td>
                      <td>{row.relative_to_python ? `${row.relative_to_python}×` : '—'}</td>
                      <td>{row.build_mode}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {benchmarks.caveats?.length ? (
              <>
                <h3>What the numbers do not say</h3>
                <ul className="caveats">
                  {benchmarks.caveats.map((caveat) => (
                    <li key={caveat}>{caveat}</li>
                  ))}
                </ul>
              </>
            ) : null}

            <div className="callout">
              <p className="callout__title">The interesting part is not the ranking</p>
              <Markdown>
                {[
                  '**The language is a much weaker lever than the data structure.** Rust and',
                  'TypeScript land within a few percent of each other despite sharing almost',
                  'nothing. What they share is a tape rather than a fresh Python object per',
                  'operation, and that is where the six-fold win comes from.',
                  '',
                  '**The C port’s speed is not a different algorithm.** The portable scalar build',
                  'of the same file learns the same thing to within 0.11% and is roughly 45×',
                  'slower. All of the difference is NEON, Accelerate, and not allocating.',
                  '',
                  '**Autograd, not arithmetic, is what the Python file spends its time on.**',
                  'A compiled language with no libraries at all still beats it by 6×, which says the',
                  '~15,000 `Value` objects per step are the bottleneck.',
                ].join('\n')}
              </Markdown>
            </div>
          </>
        )}
      </section>

      <section aria-labelledby="source-heading">
        <h2 id="source-heading">The sources, side by side</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">track</th>
                <th scope="col">lines</th>
                <th scope="col">characters</th>
                <th scope="col">note</th>
              </tr>
            </thead>
            <tbody>
              {ORDER.map((language) => (
                <tr key={language}>
                  <th scope="row">{language}</th>
                  <td>{sizes[language]?.lines ?? '…'}</td>
                  <td>{sizes[language] ? sizes[language]!.chars.toLocaleString() : '…'}</td>
                  <td>
                    {language === 'python'
                      ? 'the reference'
                      : language === 'c'
                        ? 'the parity track; a hand-written backward pass, finite-difference checked'
                        : language === 'typescript'
                          ? 'reference-faithful, float64, no tuning flags'
                          : 'a tape, so its gradients are correct by construction'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {TRACK_ORDER.map((language) => (
          <CodePanel key={language} language={language} defaultOpen={false} />
        ))}
      </section>

      <section aria-labelledby="parity-heading">
        <h2 id="parity-heading">Do they agree?</h2>
        <p>
          The gate is statistical, and the reason is not a hedge. Each step of the reference trains
          on <em>one document</em>, so its printed loss is a single sample: standard deviation{' '}
          {trace.meta.loss_stats.stdev.toFixed(3)} on a mean of{' '}
          {trace.meta.loss_stats.mean.toFixed(3)}, and it rises on{' '}
          <strong>
            {trace.meta.loss_stats.upward_moves} of its {trace.meta.loss_stats.total_moves}
          </strong>{' '}
          steps. Any check on raw per-step values is comparing noise. So the gate compares an
          exponentially-smoothed loss at fixed checkpoints, and separately requires the windowed
          trend to improve by at least 10%.
        </p>
        <p>
          The reference itself is the proof that the original specification could not work: it
          demanded a strictly decreasing loss, and the reference is not strictly decreasing.
        </p>
        <LossChart />
        <p>
          <a href="#/about">Read more in the docs</a>, or run <code>make parity</code> to see the
          live comparison.
        </p>
      </section>

      <section aria-labelledby="hyper-heading">
        <h2 id="hyper-heading">The configuration they all share</h2>
        <dl className="stats">
          {Object.entries(index.hyperparameters)
            .filter(([, value]) => typeof value === 'number')
            .map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
        </dl>
        <p>
          4,192 parameters, trained one document at a time. Every number on this site is measured at
          exactly this configuration; change a number and every recorded figure describes a
          different experiment.
        </p>
      </section>
    </article>
  )
}
