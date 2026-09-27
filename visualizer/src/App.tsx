/**
 * The app shell: chrome, routing, and the four static pages.
 *
 * Accessibility decisions that are not obvious, recorded so they do not get
 * "cleaned up":
 *
 *  - A skip link, because a site with 18 concepts and a sidebar is unusable
 *    without one.
 *  - `<nav aria-label>` on every landmark. Two navs with no labels are two
 *    navs a screen reader cannot tell apart.
 *  - The theme toggle is a real `<button>`, not a styled div.
 *  - Focus is moved to the main region on navigation, so a keyboard user who
 *    follows a "next concept" link lands in the new content rather than at the
 *    top of the document with no idea anything happened.
 *  - `aria-current` is set by `NavLink`, not at the call sites, because the
 *    route-to-section mapping is not an exact match: every concept page belongs
 *    to the "concepts" section.
 *
 * Two things here are about feel rather than structure. The header reports
 * reading position, through one passive scroll listener that writes one custom
 * property and one class -- a state update per scroll frame would re-render the
 * code panel and both charts. And the concept rail becomes a drawer below
 * 980px, because the alternative was eighteen links stacked above the content of
 * every page on a phone.
 */

import { useEffect, useRef, useState } from 'react'
import { useRoute, hrefFor, type Route, type RouteName } from './app/router'
import { useProgress, useTheme } from './app/state'
import { TRACK_ORDER, conceptsById, index } from './data/sources'
import { AtlasGraph } from './components/AtlasGraph'
import { ComparePage } from './components/ComparePage'
import { ConceptNotFound, ConceptPage } from './components/ConceptPage'
import { Icon } from './components/Icon'
import { RunPlayground } from './components/RunPlayground'
import { ResearchPage } from './components/ResearchPage'
import { LossChart, SamplesList } from './components/widgets'

function sameRoute(a: Route, b: Route): boolean {
  return a.name === b.name && a.conceptId === b.conceptId
}

const TITLES: Record<Exclude<RouteName, 'concept'>, string> = {
  home: 'a concept atlas for 199 lines — learn microgpt',
  learn: 'Concepts — learn microgpt',
  explore: 'Explore the graph — learn microgpt',
  compare: 'Five languages, one algorithm — learn microgpt',
  research: 'The research loop — learn microgpt',
  run: 'Run it in your browser — learn microgpt',
  about: 'About this atlas — learn microgpt',
}

export function App() {
  const [route, navigate] = useRoute()
  const [theme, toggleTheme] = useTheme()
  const [seen, markSeen, resetSeen] = useProgress()
  // Which route the drawer was opened *from*, rather than a boolean. Navigation
  // then closes it by derivation -- a new route is not the route the drawer was
  // opened on -- instead of by an effect that resets state after a render.
  const [railRoute, setRailRoute] = useState<Route | null>(null)
  const [stuck, setStuck] = useState(false)
  const mainRef = useRef<HTMLElement | null>(null)
  const firstRender = useRef(true)

  useEffect(() => {
    document.title =
      route.name === 'concept'
        ? `${conceptsById[route.conceptId ?? '']?.title ?? 'Concept'} — learn microgpt`
        : TITLES[route.name]
  }, [route.name, route.conceptId])

  // Reading position, reported without re-rendering: one custom property for the
  // progress hairline, one boolean for the header's condensed state.
  useEffect(() => {
    let frame = 0
    const report = () => {
      frame = 0
      const scrollable = document.documentElement.scrollHeight - window.innerHeight
      const ratio = scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0
      document.documentElement.style.setProperty('--progress', ratio.toFixed(4))
    }
    const onScroll = () => {
      setStuck(window.scrollY > 6)
      if (frame) return
      frame = requestAnimationFrame(report)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', report)
    return () => {
      if (frame) cancelAnimationFrame(frame)
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', report)
    }
  }, [])

  // Move focus to the main region on navigation, but not on first paint -- that
  // would steal focus from the document on load, which is its own accessibility
  // bug.
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    mainRef.current?.focus()
  }, [route])

  const railOpen = railRoute !== null && sameRoute(railRoute, route)
  const toggleRail = () =>
    setRailRoute((current) => (current !== null && sameRoute(current, route) ? null : route))

  useEffect(() => {
    if (!railOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setRailRoute(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [railOpen])

  const total = index.chapters.reduce((sum, chapter) => sum + chapter.concepts.length, 0)
  const readPct = total === 0 ? 0 : Math.round((seen.size / total) * 100)

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className={stuck ? 'app__header is-stuck' : 'app__header'}>
        <div className="app__bar">
          <button
            type="button"
            className="app__nav-toggle"
            aria-expanded={railOpen}
            aria-controls="concept-rail"
            onClick={toggleRail}
          >
            <Icon name={railOpen ? 'close' : 'menu'} />
            concepts
          </button>

          <a className="app__brand" href={hrefFor({ name: 'home' })}>
            <span className="app__brand-mark" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            learn<span className="app__brand-dim">microgpt</span>
          </a>

          <nav className="app__nav" aria-label="Main">
            <NavLink route={{ name: 'learn' }} current={route.name}>
              concepts
            </NavLink>
            <NavLink route={{ name: 'explore' }} current={route.name}>
              explore
            </NavLink>
            <NavLink route={{ name: 'compare' }} current={route.name}>
              compare
            </NavLink>
            <NavLink route={{ name: 'research' }} current={route.name}>
              research
            </NavLink>
            <NavLink route={{ name: 'run' }} current={route.name}>
              run
            </NavLink>
            <NavLink route={{ name: 'about' }} current={route.name}>
              about
            </NavLink>
          </nav>

          <div className="app__tools">
            <button
              type="button"
              className="app__icon-button"
              onClick={toggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            >
              <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={17} />
            </button>
          </div>
        </div>

        <div className="app__progress" aria-hidden="true">
          <span />
        </div>
      </header>

      {railOpen ? (
        <button
          type="button"
          className="app__scrim"
          aria-label="Close the concept list"
          onClick={() => setRailRoute(null)}
        />
      ) : null}

      <div className="app__body">
        <aside
          id="concept-rail"
          className={railOpen ? 'app__sidebar is-open' : 'app__sidebar'}
          aria-label="Concepts"
        >
          <div className="app__sidebar-head">
            <p className="app__sidebar-kicker">
              <span>curriculum</span>
              <span className="app__sidebar-count">
                {seen.size} of {total}
              </span>
            </p>
            <div
              className="app__progress-track"
              role="progressbar"
              aria-label="Reading progress"
              aria-valuenow={readPct}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span className="app__progress-fill" style={{ width: `${readPct}%` }} />
            </div>
            {seen.size > 0 ? (
              <button type="button" className="app__reset" onClick={resetSeen}>
                <Icon name="reset" size={12} /> reset progress
              </button>
            ) : null}
          </div>

          {index.chapters.map((chapter) => (
            <section key={chapter.id} className="app__sidebar-group">
              <h2>{chapter.title}</h2>
              <ol>
                {chapter.concepts.map((entry) => {
                  const current = route.name === 'concept' && route.conceptId === entry.id
                  return (
                    <li key={entry.id}>
                      <a
                        href={hrefFor({ name: 'concept', conceptId: entry.id })}
                        className={seen.has(entry.id) ? 'is-opened' : undefined}
                        aria-current={current ? 'page' : undefined}
                      >
                        {entry.title}
                      </a>
                    </li>
                  )
                })}
              </ol>
            </section>
          ))}
        </aside>

        <main id="main" ref={mainRef} tabIndex={-1} className="app__main">
          <Router route={route} navigate={navigate} markSeen={markSeen} seen={seen} total={total} />
        </main>
      </div>

      <footer className="app__footer">
        <div className="app__footer-inner">
          <p>
            The reference is Andrej Karpathy&rsquo;s, reproduced byte-for-byte and pinned by sha256.{' '}
            <a href="#/about">Attribution, licenses, and what is known to be wrong</a>.
          </p>
          <p className="app__footer-count">
            {TRACK_ORDER.length} tracks &middot; {total} concepts &middot; every number measured
            from a real run
          </p>
        </div>
      </footer>
    </div>
  )
}

/**
 * A nav item, marked current when the reader is anywhere inside its section.
 *
 * `aria-current="page"` is only correct on an exact match, and it is an
 * accessibility signal: it is what a screen reader announces. But the mapping
 * from a route to its section is not exact -- every concept page belongs to the
 * "concepts" section -- so the section test lives here rather than at each call
 * site. Without it, no main-nav item is marked current while reading a concept,
 * which is the whole time a reader is here.
 */
const SECTION_OF: Partial<Record<RouteName, RouteName>> = { concept: 'learn' }

function NavLink({
  route,
  current,
  children,
}: {
  route: Route
  current: RouteName
  children: React.ReactNode
}) {
  const section = SECTION_OF[current] ?? current
  return (
    <a href={hrefFor(route)} aria-current={section === route.name ? 'page' : undefined}>
      {children}
    </a>
  )
}

function Router({
  route,
  navigate,
  markSeen,
  seen,
  total,
}: {
  route: Route
  navigate: (route: Route) => void
  markSeen: (id: string) => void
  seen: ReadonlySet<string>
  total: number
}) {
  switch (route.name) {
    case 'home':
      return <HomePage seen={seen} total={total} />
    case 'learn':
      return <LearnPage seen={seen} />
    case 'concept':
      return <ConceptRoute id={route.conceptId ?? ''} navigate={navigate} markSeen={markSeen} />
    case 'explore':
      return <ExplorePage />
    case 'compare':
      return <ComparePage />
    case 'research':
      return <ResearchPage />
    case 'run':
      return <RunPlayground />
    case 'about':
      return <AboutPage />
    default:
      return <HomePage seen={seen} total={total} />
  }
}

function ConceptRoute({
  id,
  navigate,
  markSeen,
}: {
  id: string
  navigate: (route: Route) => void
  markSeen: (id: string) => void
}) {
  const concept = conceptsById[id]
  if (!concept) return <ConceptNotFound id={id} />
  return (
    <ConceptPage
      concept={concept}
      onNavigate={(cid) => navigate({ name: 'concept', conceptId: cid })}
      markSeen={markSeen}
    />
  )
}

/**
 * The home page.
 *
 * A hero, the two ways in, the three reasons to believe it, and the recorded run.
 * The four numbers under the title are the ones a reader can check against
 * `content/index.json` and `traces/`; they are not decoration and they are not
 * rounded.
 */
function HomePage({ seen, total }: { seen: ReadonlySet<string>; total: number }) {
  const first = index.chapters[0]?.concepts[0]
  const next = index.chapters.flatMap((c) => c.concepts).find((c) => !seen.has(c.id))
  const parameters = Number(index.hyperparameters.num_params ?? 0)

  return (
    <article className="home">
      <header className="home__hero">
        <p className="home__kicker">
          <Icon name="file" size={13} />
          microgpt.py &mdash; {index.reference.lines} lines &mdash; {index.reference.author}
        </p>
        <h1>A guided reading of {index.reference.lines} lines</h1>
        <p className="lede">
          Andrej Karpathy&rsquo;s <code>microgpt.py</code> is the smallest complete GPT that still
          trains, generates text, and can be read in one sitting. This site goes through it concept
          by concept &mdash; with the code, the tensor shapes, the recorded numbers, and the same
          program in four other languages.
        </p>

        <div className="home__cta">
          {next ? (
            <a
              className="button button--primary"
              href={hrefFor({ name: 'concept', conceptId: next.id })}
            >
              {seen.size === 0 ? 'start reading' : `continue: ${next.title}`}
              <Icon name="arrow" size={14} />
            </a>
          ) : (
            <a
              className="button button--primary"
              href={hrefFor({ name: 'concept', conceptId: first?.id ?? '' })}
            >
              start again from the beginning
              <Icon name="arrow" size={14} />
            </a>
          )}
          <a className="button" href={hrefFor({ name: 'run' })}>
            run it in your browser
          </a>
        </div>

        <dl className="home__stats">
          <div>
            <dt>reference</dt>
            <dd>
              {index.reference.lines}
              <small>lines</small>
            </dd>
          </div>
          <div>
            <dt>concepts</dt>
            <dd>{total}</dd>
          </div>
          <div>
            <dt>tracks</dt>
            <dd>{TRACK_ORDER.length}</dd>
          </div>
          <div>
            <dt>parameters</dt>
            <dd>{parameters.toLocaleString()}</dd>
          </div>
        </dl>
      </header>

      <h2>What makes it worth reading</h2>
      <ul className="home__points">
        <li>
          <strong>Every number is measured.</strong> The loss curve, the attention weights, the
          speed comparison &mdash; all from real runs, committed to the repository and regenerated
          by CI. Nothing on this site is a plausible-looking invention.
        </li>
        <li>
          <strong>Nothing rots silently.</strong> Concepts declare AST selectors, not line numbers,
          so editing the reference moves the highlights with it. A selector that resolves to nothing
          is a build failure.
        </li>
        <li>
          <strong>What is wrong is written down.</strong> The C port&rsquo;s hand-written backward
          pass gets a gradient wrong by a factor of &minus;0.12, and its loss curve still looks
          fine. <a href="#/about">That is documented, measured, and tested</a> &mdash; and it is the
          most instructive thing on the site.
        </li>
      </ul>

      <h2>The recorded run</h2>
      <p className="home__body-text">
        1,000 steps of the reference on 32,033 names. The samples at the end are a mix of names in
        the dataset and names that are not.
      </p>
      <LossChart />
      <SamplesList />
    </article>
  )
}

function LearnPage({ seen }: { seen: ReadonlySet<string> }) {
  return (
    <article className="learn">
      <h1>Eighteen concepts, in reading order</h1>
      <p className="lede">
        The chapters are the order. The links between them are the prerequisites, drawn from each
        concept&rsquo;s own <code>prereqs</code>, so the graph and the order cannot disagree.
      </p>
      {index.chapters.map((chapter) => (
        <section key={chapter.id}>
          <h2>
            <span className="learn__chapter-order">{chapter.order}</span>
            {chapter.title}
          </h2>
          <p>{chapter.blurb}</p>
          <ul className="learn__grid">
            {chapter.concepts.map((entry) => {
              const opened = seen.has(entry.id)
              return (
                <li key={entry.id}>
                  <a
                    className={opened ? 'learn__card is-opened' : 'learn__card'}
                    href={hrefFor({ name: 'concept', conceptId: entry.id })}
                  >
                    <span className="learn__card__title">
                      {entry.title}
                      {opened ? (
                        <span className="learn__tick" title="opened in this browser">
                          <Icon name="check" size={12} /> read
                        </span>
                      ) : null}
                    </span>
                    <span className="learn__card__summary">{entry.summary}</span>
                  </a>
                </li>
              )
            })}
          </ul>
        </section>
      ))}
    </article>
  )
}

function ExplorePage() {
  return (
    <article className="explore">
      <h1>Explore</h1>
      <p className="lede">
        The whole concept graph, and the recorded data behind the claims. The graph is a diagram;
        the list below it is the same information, and is what a screen reader gets.
      </p>
      <h2>Reading order</h2>
      <AtlasGraph />
    </article>
  )
}

function AboutPage() {
  return (
    <article className="about">
      <h1>About this site</h1>

      <h2>What the thing is</h2>
      <p>
        A reading aid for <code>reference/microgpt.py</code>, 199 lines by Andrej Karpathy,
        originally published with his <em>Intro to Large Language Models</em> talk in November 2023.
        The reference is reproduced here byte-for-byte and its sha256 is checked on every CI run.
      </p>
      <p>
        The C ports descend from <code>vixhal-baraiya/microgpt-c</code> (MIT, © 2026 Vishal), and
        <code> data/input.txt</code> is <code>karpathy/makemore@988aa59</code>&rsquo;s{' '}
        <code>names.txt</code>, byte-identical. Full detail, with digests, is in{' '}
        <code>docs/PROVENANCE.md</code>.
      </p>

      <h2>What is known to be wrong</h2>
      <p>
        The C port&rsquo;s hand-written backward pass does not correctly propagate the key and value
        gradients of earlier positions back to their embeddings. Only the output head&rsquo;s
        gradient is right. Measured as a directional derivative over all 4,192 parameters, its
        gradient comes out at <strong>&minus;0.12&times;</strong> the true value, where a correct
        gradient is 1.0&times;.
      </p>
      <p>
        The instructive part:{' '}
        <strong>
          its loss curve still tracks the reference to within 7% and the model still trains.
        </strong>{' '}
        Adam divides by an estimate of the gradient&rsquo;s own magnitude, so a gradient that is
        wrong by a factor barely moves the step. A loss curve is evidence that something learned,
        not evidence that the thing that learned was the gradient. The full measurements are in{' '}
        <code>docs/KNOWN-ISSUES.md</code>.
      </p>

      <h2>How the numbers are produced</h2>
      <ul>
        <li>
          <code>tools/trace.py</code> runs the real reference file, unmodified, in a scratch
          directory seeded from <code>data/input.txt</code>, and records the loss curve, the
          attention weights and the samples.
        </li>
        <li>
          <code>tools/parity.py</code> compares every track against the reference on an
          exponentially-smoothed loss. It cannot compare raw per-step values, because the reference
          itself is not monotonically decreasing and its per-step standard deviation is 16% of its
          mean.
        </li>
        <li>
          <code>tools/bench.py</code> measures the tracks and records the machine. Ratios within a
          row set are meaningful; absolute seconds are not, and are not a claim about your hardware.
        </li>
      </ul>

      <h2>License</h2>
      <p>
        This site and its tooling are MIT. The reference is Karpathy&rsquo;s; the C ports carry the
        upstream MIT notice. Nothing here is vendored without attribution &mdash; see{' '}
        <code>CREDITS.md</code> and <code>NOTICE</code>.
      </p>
    </article>
  )
}
