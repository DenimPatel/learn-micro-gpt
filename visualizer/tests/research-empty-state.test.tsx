import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { ResearchPage } from '../src/components/ResearchPage'
import type {
  ResearchResults,
  ResearchRow,
  ResearchTrack,
  ResearchVerdict,
} from '../src/data/types'

/**
 * The research page's empty states, rendered.
 *
 * `tests/research.test.ts` deliberately asserts on derived values rather than on
 * markup, and its header says why: the unit tests are `environment: 'node'`, and
 * a test that needs a DOM is a Playwright test. That is true, and it is also how
 * this coverage disappeared. Both empty states on the research page are reachable
 * only with a ledger that has not been produced -- no track in
 * `autoresearch/results.json` has zero experiments, and none has a chart with
 * nothing to plot -- so the only place they could ever be exercised was the e2e
 * suite, where the data is the real `results.json` and cannot be substituted. The
 * e2e test written to cover them ("says so when a track has no experiments",
 * `e2e/research.spec.ts`) asserted on `.research`'s inner text, which contains
 * the track switcher, so it was satisfied by *another* track's keep count rather
 * than by the track under test. It passed for as long as TypeScript had zero
 * keeps, recorded nothing about why, and went red the day TypeScript recorded its
 * first one. A test that reads the whole article and matches a regex can be
 * satisfied by a control it did not mean to read; the way to prevent that is to
 * assert on the branch's own text, which is what this file does.
 *
 * So, two things:
 *
 *  1. Cover the branches. Every `research__empty` paragraph that means "nothing
 *     has been measured here" is asserted by its own sentence, against a fixture
 *     ledger, so no assertion here can be satisfied by a sibling element.
 *  2. Keep the "bare zero" honest. A populated track must render the ledger
 *     counts and neither empty state -- and because `make autoresearch-rust`
 *     appears in this page *only* inside those two empty states, the populated
 *     case asserts the command is absent from the whole document. That makes the
 *     negative half of the property symmetric with the positive half instead of
 *     leaving it to a reader to notice.
 *
 * Why `renderToStaticMarkup` and not jsdom. The charts are measured with
 * `ResizeObserver` inside a `useLayoutEffect` (`usePlotWidth`), and neither runs
 * under a static render, so the charts fall back to the width the hook was handed
 * and the markup is complete. `react-dom` is already a dependency; adding a DOM
 * implementation to a repository whose stated argument is that you can read all
 * of its code is a poor trade, and `vitest.config.ts` already says anything
 * needing a DOM is a Playwright test. This is not that. It needs no DOM; it needs
 * the branch.
 *
 * Why `vi.mock` rather than a prop. `ResearchPage` takes no props and its
 * default track is `rust`, so the fixture is built with `rust` as the interesting
 * track. That is a constraint, not a problem. The mock path `../src/data/sources`
 * resolves to the same file `ResearchPage.tsx` imports as `'../data/sources'` and
 * `data/research.ts` as `'./sources'`; `vi.mock` keys on the resolved module, so
 * the arithmetic in `data/research.ts` reads the fixture too -- which is the
 * point, since the counts under test are `trackRows(track)` filtered by status.
 * The object identity of `research` is stable across cases and only its contents
 * are replaced, because the component and the selectors both hold a reference to
 * it for the whole module's life.
 */

const TRACKS: readonly ResearchTrack[] = ['rust', 'go', 'typescript', 'c']

/**
 * `vi.hoisted` because the mock factory is hoisted above the imports, and the
 * factory has to hand back a `research` whose identity outlives the case that
 * filled it. `buildFixture` is a plain function below, called at test time, so it
 * does not need hoisting and is not affected.
 */
const state = vi.hoisted(() => ({ ledger: {} as ResearchResults }))

vi.mock('../src/data/sources', () => ({
  research: state.ledger,
  // The `?raw` loaders never resolve under a static render: `runPatches` reaches
  // `loadPatch` from an effect and `Divergence` reaches `loadCandidate` from
  // another, and neither effect runs. They have to exist because the module is
  // imported by name.
  loadedCandidate: () => undefined,
  loadCandidate: () => Promise.resolve(''),
  loadPatch: () => Promise.resolve(null),
  bestVsBaselinePatch: () => '',
}))

/**
 * Render the page and pull the assertions out of the markup.
 *
 * The probes are deliberately specific rather than generic:
 *
 *  - `emptyStates` matches one tag level deep, which is safe because every
 *    `research__empty` is a `<p>` containing no other `<p>`.
 *  - `stat` matches a `<dt>`/`<dd>` *pair* by label, so it reads the ledger
 *    heading and not a same-worded `<dd>` in one of the charts' own stat lists.
 *  - The command is matched against the whole document on purpose; see the
 *    header.
 */
function render(): { html: string; emptyStates: string[]; stat: (label: string) => string | null } {
  const html = renderToStaticMarkup(<ResearchPage />)
  return {
    html,
    emptyStates: [...html.matchAll(/<p class="research__empty">([\s\S]*?)<\/p>/g)].map((match) =>
      text(match[1] ?? ''),
    ),
    stat: (label: string) => {
      const found = new RegExp(`<dt>${label}</dt>(?:<!--.*?-->)*<dd>([^<]*)</dd>`).exec(html)
      return found ? (found[1] ?? null) : null
    },
  }
}

/** The visible text of a markup fragment: no tags, no React text separators. */
function text(fragment: string): string {
  return fragment
    .replace(/<!--.*?-->/g, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Assert that one of the page's empty-state paragraphs says what it must say. */
function expectEmptyState(states: string[], pattern: RegExp): string {
  const match = states.find((state) => pattern.test(state))
  expect(
    match,
    `no research__empty paragraph matched ${pattern}\nfound:\n${states.join('\n') || '(none)'}`,
  ).toBeDefined()
  return match as string
}

function baseline(track: ResearchTrack, overrides: Partial<ResearchRow> = {}): ResearchRow {
  return {
    run_id: '0000',
    parent: 'f942bd4ed5e6abb04a3dd47ef12b0ccfc88f7f52',
    track,
    loss: 2.433882,
    steps_per_sec: 95.382084,
    loss_gain: 0,
    speed_gain: 0,
    grad_ratio: 1.06316,
    status: 'baseline',
    reason: 'frozen track, re-measured this session',
    description: "the frozen track, unmodified, as this session's comparator",
    has_record: true,
    ...overrides,
  }
}

function attempt(
  runId: string,
  status: ResearchVerdict,
  overrides: Partial<ResearchRow> = {},
): ResearchRow {
  const measured = status !== 'crash'
  return {
    run_id: runId,
    parent: 'a92c731',
    track: 'rust',
    loss: measured ? 2.4 : 0,
    steps_per_sec: measured ? 100 : 0,
    loss_gain: measured ? 0.03 : 0,
    speed_gain: measured ? 0.01 : 0,
    grad_ratio: measured ? 1.05 : 0,
    status,
    reason: measured ? '' : 'the candidate did not compile',
    description: measured
      ? 'a rewrite the harness measured'
      : 'a rewrite the harness could not run',
    has_record: true,
    ...overrides,
  }
}

/**
 * A `ResearchResults` built from the real shapes in `data/types.ts`.
 *
 * Every required field is present, including the ones this page never reads
 * (`objective`), because a fixture that satisfies the type is the cheapest way
 * to notice that a field was added and this test was not updated.
 * `tracks` carries every language, because `tracks()` is
 * `Object.keys(research.tracks)` and the switcher is one button per entry -- a
 * single-track fixture would render a page that could never fail the multi-track
 * e2e tests if it ever reached a browser.
 *
 * `TRACKS` is the single place the list is written, and `tracks` and `pareto` are
 * the two objects that have to be spelled out because they are keyed by it. That
 * is three places to update for a new language, and the cost of missing one is a
 * typecheck error rather than a wrong number, which is the good kind of cost.
 */
function buildFixture(runs: ResearchRow[]): ResearchResults {
  const perTrack = (pick: (row: ResearchRow) => boolean) =>
    Object.fromEntries(
      TRACKS.map((track) => [track, runs.find((row) => row.track === track && pick(row)) ?? null]),
    ) as ResearchResults['baselines']
  return {
    $comment: 'a fixture, not a measurement: see tests/research-empty-state.test.tsx',
    generator: 'tests/research-empty-state.test.tsx',
    track: 'multi',
    tracks: {
      rust: {
        language: 'Rust',
        candidate_dir: 'candidate',
        source: 'src/lib.rs',
        probe: 'tests/gradient_check.rs',
        build: '`cargo build --release` and `cargo clippy --release -- -D warnings`',
      },
      go: {
        language: 'Go',
        candidate_dir: 'candidate-go',
        source: 'main.go',
        probe: 'probe_test.go',
        build: '`go build` and `go vet ./...`',
      },
      typescript: {
        language: 'TypeScript',
        candidate_dir: 'candidate-ts',
        source: 'src/index.ts',
        probe: 'src/probe.ts',
        build: '`npx tsc --noEmit`',
      },
      c: {
        language: 'C',
        candidate_dir: 'candidate-c',
        source: 'microgpt.c',
        probe: 'probe.c',
        build: '`cc -O3 -o microgpt microgpt.c -lm`',
      },
    },
    provenance_of: Object.fromEntries(
      TRACKS.map((track) => [
        track,
        {
          baseline_source: `implementations/${track}/src/lib.rs`,
          baseline_sha256: '4bced112f2db2bf8662082051fa550f82ab793bed251be9398c6f672947a2f15',
          candidate_source: `autoresearch/candidate-${track}/src/lib.rs`,
          candidate_sha256: '99326ce38891f27483c0db2e0cb953f65c516f525ad81a2e167f4876a8c7c785',
          gradient_probe: `autoresearch/candidate-${track}/tests/gradient_check.rs`,
          gradient_probe_sha256: 'b5beae0d11579987c954867846127d5119d4cb07ed2ea47078cf1c2e29807c6b',
        },
      ]),
    ) as ResearchResults['provenance_of'],
    protocol: {
      steps: 1000,
      seed: 42,
      repeats: 3,
      trend_window: 50,
      run_timeout_seconds: 600,
      loss_axis: 'mean loss over the last 50 of 1000 steps, at seed 42',
      speed_axis: 'steps per second, median of 3 runs of the whole process',
    },
    thresholds: {
      loss_material: 0.02,
      loss_tolerance: 0.05,
      speed_material: 0.1,
      speed_tolerance: 0.05,
      min_relative_improvement: 0.005,
      rule: 'keep if one axis clears its material threshold and the other stays inside tolerance',
      note: 'a tie is a discard',
    },
    objective: { kind: 'two-axis', axes: ['loss', 'steps_per_sec'], decided_by: 'verdict()' },
    model: { model: 'fixture/model', reasoning_parameter: 'effort', reasoning_effort: 'high' },
    runner: {
      os: 'linux',
      machine: 'fixture',
      rustc: 'rustc 1.0.0',
      measured: '2026-01-01T00:00:00Z',
    },
    baselines: perTrack((row) => row.status === 'baseline'),
    bests: perTrack((row) => row.status === 'keep'),
    counts: {
      experiments: runs.filter((row) => row.status !== 'baseline').length,
      keep: runs.filter((row) => row.status === 'keep').length,
      discard: runs.filter((row) => row.status === 'discard').length,
      crash: runs.filter((row) => row.status === 'crash').length,
    },
    pareto: { rust: [], go: [], typescript: [], c: [] },
    runs,
    caveats: ['every number here is a fixture, and none of it was measured'],
  }
}

/** Install a ledger, in place: the component holds one reference for the file's life. */
function useLedger(runs: ResearchRow[]): void {
  Object.assign(state.ledger, buildFixture(runs))
}

describe('a track with no experiments at all', () => {
  it('says so and names the command, rather than showing a bare zero', () => {
    useLedger([baseline('rust')])
    const { html, emptyStates, stat } = render()

    // The page-level branch, `ResearchPage.tsx`'s `noExperiments ? ... : null`.
    // Asserted by its own sentence and not by a count: `experiments 0` is on the
    // page in this state too, and it is the sentence beside it that is the point.
    const page = expectEmptyState(emptyStates, /No experiment has been run yet\./)
    expect(page).toContain('make autoresearch-rust')
    expect(page).toContain('docs/AUTORESEARCH.md')

    // The per-track branch in `GainsChart`, reached because the only row is the
    // baseline and `experiments()` excludes it.
    expectEmptyState(emptyStates, /No experiment has been measured yet\./)

    // The zero is still printed, because the ledger is real; what must not happen
    // is the zero standing alone. `make autoresearch-rust` appears in this
    // document *only* inside the two empty states above, so finding it here is
    // proof that a reader is told what to do rather than shown a number.
    expect(stat('experiments')).toBe('0')
    expect(html).toContain('make autoresearch-rust')
  })

  it('draws no gains chart, because one cell per experiment would read as a result', () => {
    useLedger([baseline('rust')])
    const { html, emptyStates } = render()
    // `GainsChart` plots one cell per experiment and bails to a paragraph when
    // there are none. The *scatter* still draws, and correctly so: the baseline
    // is a measured row, and its empty branch is about a track with no
    // measurement at all rather than one with nothing to compare.
    expect(html).not.toContain('research__gains')
    expect(html).toContain('research__scatter')
    expect(emptyStates.some((state) => /nothing to plot/.test(state))).toBe(false)
  })

  it('draws nothing at all for a track that has never been seeded', () => {
    // The same property one step further out, and the case `baselines[track]:
    // ResearchRow | null` in `data/types.ts` exists to describe. With no baseline
    // there is nothing to compare against, so the scatter bails too -- which is
    // the third empty branch on the page, and the only one whose text does not
    // mention the command.
    useLedger([])
    const { html, emptyStates } = render()
    expectEmptyState(emptyStates, /No experiment has been run yet\./)
    expectEmptyState(emptyStates, /No experiment has been measured yet\./)
    expectEmptyState(emptyStates, /No run has produced a measurement yet/)
    expect(html).not.toContain('research__gains')
    expect(html).not.toContain('research__scatter')
  })
})

describe('a track whose every row is a crash', () => {
  it('shows the per-track empty state but not the page-level one', () => {
    // Distinct from "no experiments": the ledger is not empty, the counts are not
    // all zero, and the *chart* still has nothing to plot -- a crash is a row
    // with no measurement, and `GainsChart` filters crashes out before checking
    // length. Conflating the two branches would let a page that drew a chart over
    // crash rows pass as honest.
    useLedger([baseline('rust'), attempt('0001', 'crash'), attempt('0002', 'crash')])
    const { html, emptyStates, stat } = render()

    expectEmptyState(emptyStates, /No experiment has been measured yet\./)
    expect(emptyStates.some((state) => /No experiment has been run yet\./.test(state))).toBe(false)
    expect(stat('experiments')).toBe('2')
    expect(stat('crashed')).toBe('2')
    // The scatter still draws the baseline, which is a real measurement; the
    // gains chart does not, because a crash row has no gain to plot.
    expect(html).toContain('research__scatter')
    expect(html).not.toContain('research__gains')
  })
})

describe('a track with measurements', () => {
  it('renders the ledger statistics and neither empty state', () => {
    useLedger([baseline('rust'), attempt('0001', 'keep'), attempt('0002', 'discard')])
    const { html, emptyStates, stat } = render()

    // The counts come from `trackRows(track)`, so these are this track's and not
    // the ledger-wide `research.counts`, which would put a Rust number on another
    // track's page.
    expect(stat('experiments')).toBe('2')
    expect(stat('kept')).toBe('1')
    expect(stat('discarded')).toBe('1')
    expect(stat('crashed')).toBe('0')

    expect(emptyStates.filter((state) => /No experiment has been/.test(state))).toEqual([])

    // The sharp form of "does not show a bare zero": the command is what both
    // empty states exist to print, and it appears nowhere else on the page. A
    // populated track that rendered an empty state anyway would trip this.
    expect(html).not.toContain('make autoresearch-rust')
    expect(html).toContain('research__scatter')
    expect(html).toContain('research__gains')
  })
})
