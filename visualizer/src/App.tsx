/**
 * The app shell: chrome, routing, and the home page.
 *
 * Accessibility decisions that are not obvious, recorded so they do not get
 * "cleaned up":
 *
 *  - A skip link, because a site with 18 concepts and a sidebar is unusable
 *    without one.
 *  - `<nav aria-label>` on every landmark. Two navs with no labels are two
 *    navs a screen reader cannot tell apart.
 *  - The theme toggle is a real `<button>` with `aria-pressed`, not a styled div.
 *  - Focus is moved to the main heading on navigation, so a keyboard user who
 *    follows a "next concept" link lands in the new content rather than at the
 *    top of the document with no idea anything happened.
 */

import { useEffect, useRef } from 'react'
import { useRoute, hrefFor, type Route, type RouteName } from './app/router'
import { useProgress, useTheme } from './app/state'
import { conceptsById, index } from './data/sources'
import { AtlasGraph } from './components/AtlasGraph'
import { ComparePage } from './components/ComparePage'
import { ConceptNotFound, ConceptPage } from './components/ConceptPage'
import { RunPlayground } from './components/RunPlayground'
import { LossChart, SamplesList } from './components/widgets'

export function App() {
  const [route, navigate] = useRoute()
  const [theme, toggleTheme] = useTheme()
  const [seen, markSeen, resetSeen] = useProgress()
  const mainRef = useRef<HTMLElement | null>(null)
  const firstRender = useRef(true)

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

  const total = index.chapters.reduce((sum, chapter) => sum + chapter.concepts.length, 0)

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className="app__header">
        <a className="app__brand" href="#/">
          learn<span>microgpt</span>
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
          <NavLink route={{ name: 'run' }} current={route.name}>
            run
          </NavLink>
          <NavLink route={{ name: 'about' }} current={route.name}>
            about
          </NavLink>
        </nav>
        <button
          type="button"
          className="app__theme"
          onClick={toggleTheme}
          aria-pressed={theme === 'dark'}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? '☾' : '☀'}
        </button>
      </header>

      <div className="app__body">
        <aside className="app__sidebar" aria-label="Concepts">
          <p className="app__progress-label">
            {seen.size} of {total} opened
          </p>
          <progress value={seen.size} max={total} aria-label="Reading progress" />
          {index.chapters.map((chapter) => (
            <section key={chapter.id}>
              <h2>{chapter.title}</h2>
              <ol>
                {chapter.concepts.map((concept) => (
                  <li key={concept.id}>
                    <a
                      href={hrefFor({ name: 'concept', conceptId: concept.id })}
                      className={seen.has(concept.id) ? 'is-seen' : undefined}
                      aria-current={
                        route.name === 'concept' && route.conceptId === concept.id
                          ? 'page'
                          : undefined
                      }
                    >
                      {concept.title}
                    </a>
                  </li>
                ))}
              </ol>
            </section>
          ))}
          {seen.size > 0 ? (
            <button type="button" className="app__reset" onClick={resetSeen}>
              reset progress
            </button>
          ) : null}
        </aside>

        <main id="main" ref={mainRef} tabIndex={-1} className="app__main">
          <Router route={route} navigate={navigate} markSeen={markSeen} seen={seen} total={total} />
        </main>
      </div>

      <footer className="app__footer">
        <p>
          The reference is Andrej Karpathy&rsquo;s, reproduced byte-for-byte and pinned by sha256.{' '}
          <a href="#/about">Attribution, licenses, and what is known to be wrong</a>.
        </p>
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

function HomePage({ seen, total }: { seen: ReadonlySet<string>; total: number }) {
  const first = index.chapters[0]?.concepts[0]
  const next = index.chapters.flatMap((c) => c.concepts).find((c) => !seen.has(c.id))
  return (
    <article className="home">
      <h1>A guided reading of 199 lines</h1>
      <p className="lede">
        Andrej Karpathy&rsquo;s <code>microgpt.py</code> is the smallest complete GPT that still
        trains, generates text, and can be read in one sitting. This site goes through it concept by
        concept &mdash; with the code, the tensor shapes, the recorded numbers, and the same program
        in four other languages.
      </p>

      <div className="home__cta">
        {next ? (
          <a className="button button--primary" href={`#/learn/${next.id}`}>
            {seen.size === 0 ? 'start reading' : `continue: ${next.title}`}
          </a>
        ) : (
          <p>
            You have opened all {total} concepts.{' '}
            <a href={`#/learn/${first?.id}`}>Start again from the beginning</a>.
          </p>
        )}
        <a className="button" href="#/run">
          run it in your browser
        </a>
      </div>

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
          pass gets a gradient wrong by a factor of −0.12, and its loss curve still looks fine.{' '}
          <a href="#/about">That is documented, measured, and tested</a> &mdash; and it is the most
          instructive thing on the site.
        </li>
      </ul>

      <h2>The recorded run</h2>
      <p>
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
            {chapter.order}. {chapter.title}
          </h2>
          <p>{chapter.blurb}</p>
          <ul>
            {chapter.concepts.map((concept) => (
              <li key={concept.id}>
                <a href={`#/learn/${concept.id}`}>{concept.title}</a>
                {seen.has(concept.id) ? <span className="learn__tick"> ✓ read</span> : null}
                <p>{concept.summary}</p>
              </li>
            ))}
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
        gradient comes out at <strong>−0.12×</strong> the true value, where a correct gradient is
        1.0×.
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
