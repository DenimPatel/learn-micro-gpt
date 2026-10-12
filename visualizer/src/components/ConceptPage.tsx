/**
 * A concept page: prose, code, shapes, maths, and inline trace widgets.
 *
 * The body arrives from `tools/render_content.py` already split into typed
 * blocks, so this file is a switch over block kinds and nothing else. Rendering
 * lives in the Python tool and validation lives in `tools/validate_concepts.py`,
 * which means a malformed directive fails the build rather than rendering as
 * nothing.
 *
 * ## The contents rail
 *
 * The reading column is 68 characters wide because that is about as long as a
 * line can be before the eye loses the return sweep to the next one. On a wide
 * screen that leaves a few hundred pixels beside it, and a contents rail is what
 * belongs there: the page's own headings, with the current one tracked as the
 * reader scrolls.
 *
 * It scrolls to a heading rather than linking to one. The site routes on the URL
 * *hash* (`#/learn/softmax`, see `app/router.ts`), so an in-page anchor in that
 * same hash is indistinguishable from a route -- and the one that loses is the
 * deep link, which has to keep working on GitHub Pages.
 */

import { useEffect, useMemo, useState } from 'react'
import { Math, Markdown, headings } from '../content/markdown'
import { withinAnchor } from '../content/cite'
import { anchorFor, anchors, conceptsById, languagesFor } from '../data/sources'
import { chapterOf, neighbours } from '../data/sequence'
import type { Block, Concept, Language } from '../data/types'
import { CodePanel } from './CodePanel'
import { Icon } from './Icon'
import { AttentionHeatmap, LossChart, SamplesList, ShapeTable, SoftmaxBars } from './widgets'
import { GlossaryLink } from './Glossary'

export function ConceptPage({
  concept,
  onNavigate,
  markSeen,
}: {
  concept: Concept
  onNavigate: (id: string) => void
  markSeen: (id: string) => void
}) {
  useEffect(() => {
    markSeen(concept.id)
  }, [concept.id, markSeen])

  const languages = languagesFor(concept.id)
  const [primary, ...rest] = languages
  const chapter = chapterOf(concept.id)
  const difficulty = concept.difficulty ?? 1

  /*
   * Which line range the primary panel is focused on, when a citation has moved
   * it. `null` means "the concept's own anchor", which is what the panel shows by
   * default -- so the state is an *override*, not the value, which is why it can
   * start empty and why a citation that resolves to the concept's own range is
   * still a meaningful click rather than a no-op.
   */
  const [cited, setCited] = useState<{ start: number; end: number } | null>(null)

  /*
   * A citation resolves against the concept's *declared anchors*, never against a
   * line number baked into the markup (acceptance criterion 2). The prose sits
   * beside one panel -- the primary language -- so a range has to fall inside
   * *that* language's resolved window to be clickable: `data-loading` anchors
   * Python 13-20 and C 397-484, and a citation drawn from the C window would
   * highlight nothing in the Python panel the reader is actually looking at.
   * `withinAnchor` is the same containment check, against the same primary
   * language, that `tools/validate_concepts.py` runs at build time; the two agree
   * because they ask the identical question, not because they happened to.
   *
   * The resolution is split in two on purpose. `canCite` is a pure predicate the
   * renderer asks "would this be clickable?" while building elements -- it must
   * not touch state, or the first citation would focus the panel during render,
   * before the reader clicked anything. `followCite` is the click, and it is the
   * only place `setCited` is called.
   */
  const canCite = (range: { start: number; end: number }): boolean => {
    if (!primary) return false
    return withinAnchor(range, anchorFor(concept.id, primary))
  }

  const followCite = (range: { start: number; end: number }): void => {
    if (!canCite(range)) return
    setCited(range)
  }

  return (
    <div className="concept-page">
      <article className="concept" aria-labelledby="concept-title">
        <nav className="concept__breadcrumb" aria-label="Breadcrumb">
          <a href="#/learn">All concepts</a>
          {chapter ? (
            <>
              <Icon name="arrow" size={12} />
              <span>{chapter.title}</span>
            </>
          ) : null}
        </nav>

        <header className="concept__header">
          <h1 id="concept-title">{concept.title}</h1>
          <div className="concept__meta">
            {chapter ? <span className="chapter-badge">{chapter.title}</span> : null}
            {chapter ? (
              <span className="meta-pair">
                <span>in chapter</span>
                <b>
                  {concept.order} of {chapter.concepts.length}
                </b>
              </span>
            ) : null}
            <span className="meta-pair">
              <span>difficulty</span>
              <span className="difficulty" role="img" aria-label={`difficulty ${difficulty} of 3`}>
                {[1, 2, 3].map((step) => (
                  <i key={step} className={step <= difficulty ? 'is-on' : undefined} />
                ))}
              </span>
            </span>
          </div>
          <p className="concept__summary">{concept.summary}</p>
        </header>

        {concept.math ? (
          <div className="concept__math">
            <Math value={concept.math} display label="The formula for this concept" />
          </div>
        ) : null}

        {/*
          The reference range, first and always. Every other track is a comparison
          to this one rather than a peer of it, so it gets the open panel and the
          others get closed ones.
        */}
        {primary ? (
          <CodePanel
            language={primary as Language}
            conceptId={concept.id}
            focus={cited ?? undefined}
          />
        ) : null}

        {rest.length > 0 ? (
          <section className="concept__ports" aria-label="The same code in other languages">
            <h2>The same algorithm, in {rest.length + 1} languages</h2>
            <p>
              The range below is the same concept in the other tracks. These are separate programs,
              not translations of one another, so the line numbers do not line up &mdash; and that
              is the interesting part.
            </p>
            {rest.map((language) => (
              <CodePanel
                key={language}
                language={language}
                conceptId={concept.id}
                defaultOpen={false}
              />
            ))}
          </section>
        ) : null}

        {concept.shapes?.length ? <ShapeTable shapes={concept.shapes} /> : null}

        <div className="prose">
          {concept.blocks.map((block, blockIndex) => (
            <BlockView
              key={blockIndex}
              block={block}
              onCite={primary ? canCite : undefined}
              onActivate={primary ? followCite : undefined}
            />
          ))}
        </div>

        <nav className="concept__nav" aria-label="Concept navigation">
          <PrevNext concept={concept} onNavigate={onNavigate} />
        </nav>
      </article>

      <Contents concept={concept} />
    </div>
  )
}

function BlockView({
  block,
  onCite,
  onActivate,
}: {
  block: Block
  /** Whether a prose citation resolves against the panel's visible range. */
  onCite?: ((range: { start: number; end: number }) => boolean) | undefined
  /** Follows a prose citation the reader clicked. */
  onActivate?: ((range: { start: number; end: number }) => void) | undefined
}) {
  switch (block.kind) {
    case 'prose':
      return (
        <Markdown onCite={onCite} onActivate={onActivate}>
          {block.markdown}
        </Markdown>
      )

    case 'callout':
    case 'note':
      return (
        <aside className={block.kind === 'note' ? 'callout callout--note' : 'callout'}>
          {block.title ? (
            <p className="callout__title">
              <Icon name={block.kind === 'note' ? 'measure' : 'note'} size={13} />
              {block.title}
            </p>
          ) : null}
          <Markdown onCite={onCite} onActivate={onActivate}>
            {block.markdown ?? ''}
          </Markdown>
        </aside>
      )

    case 'shape':
      return <ShapeTable names={[block.name]} />

    case 'term':
      return <GlossaryLink term={block.term} />

    case 'trace': {
      const args = block.args
      const num = (key: string, fallback: number) => {
        const raw = args[key]
        const parsed = raw === undefined ? Number.NaN : Number(raw)
        return Number.isFinite(parsed) ? parsed : fallback
      }
      switch (args.kind) {
        case 'attention':
          return (
            <AttentionHeatmap step={num('step', 0)} pos={num('pos', 3)} head={num('head', 0)} />
          )
        case 'softmax':
          return <SoftmaxBars step={num('step', 0)} pos={num('pos', 0)} />
        case 'loss-curve':
          return <LossChart />
        case 'tokens':
          return <TraceTokens step={num('step', 0)} />
        case 'samples':
          return args.step === undefined ? (
            <SamplesList />
          ) : (
            <SamplesList step={num('step', 1000)} />
          )
        default:
          return null
      }
    }

    case 'lang':
      return null

    default:
      return null
  }
}

function TraceTokens({ step }: { step: number }) {
  // The committed trace records loss, attention and probabilities, but not the
  // token sequence: `parse_stdout` in tools/trace.py keeps it in the raw output
  // rather than the trace, on the grounds that a reader who wants to see which
  // characters produced a given position can read the name in the stdout file
  // more easily than decode 5 integers out of JSON. So this points there rather
  // than inventing a widget for data that was not recorded.
  return (
    <p className="widget widget--empty">
      The documents and sampled names for step {step} are in <code>traces/python/micro/</code>, and
      the attention and probability data behind them is on the <a href="#/explore">Explore</a> page.
    </p>
  )
}

/**
 * The contents rail, and the scroll spy that keeps it honest.
 *
 * The observer's `rootMargin` puts the "current" band just below the sticky
 * header, so the entry that lights up is the one whose heading the reader has
 * most recently passed. It is the only reason a table of contents earns the
 * width it takes, and it is why the headings need ids.
 */
function Contents({ concept }: { concept: Concept }) {
  const items = useMemo(
    () => concept.blocks.flatMap((block) => headings(block.markdown ?? '')),
    [concept],
  )
  const [active, setActive] = useState<string | undefined>(items[0]?.id)

  useEffect(() => {
    const targets = items
      .map((item) => document.getElementById(item.id))
      .filter((node): node is HTMLElement => node !== null)
    if (targets.length === 0) return

    const shown = new Set<string>()
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) shown.add(entry.target.id)
          else shown.delete(entry.target.id)
        }
        // Document order, not intersection order: two headings can sit inside the
        // band at once on a short section, and the earlier one is the one being
        // read.
        setActive((items.find((item) => shown.has(item.id)) ?? items[0])?.id)
      },
      { rootMargin: '-96px 0px -68% 0px' },
    )
    for (const target of targets) observer.observe(target)
    return () => observer.disconnect()
  }, [items])

  // Fewer than two headings is not a contents rail, it is a label.
  if (items.length < 2) return null

  return (
    <nav className="toc" aria-label="On this page">
      <span className="toc__label">On this page</span>
      <ol>
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              className={item.id === active ? 'is-active' : undefined}
              aria-current={item.id === active ? 'true' : undefined}
              onClick={(event) => {
                event.preventDefault()
                const target = document.getElementById(item.id)
                if (!target) return
                setActive(item.id)
                target.scrollIntoView({
                  behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
                    ? 'auto'
                    : 'smooth',
                  block: 'start',
                })
              }}
            >
              {item.text}
            </a>
          </li>
        ))}
      </ol>
      <p className="toc__foot">
        <b>{concept.blocks.length}</b> sections,{' '}
        {concept.blocks.filter((b) => b.kind === 'trace').length} of them measured
      </p>
    </nav>
  )
}

function PrevNext({ concept, onNavigate }: { concept: Concept; onNavigate: (id: string) => void }) {
  const { previous, next } = neighbours(concept.id)

  return (
    <div className="prevnext">
      {previous ? (
        <a
          className="prevnext__link prevnext__link--prev"
          href={`#/learn/${previous.id}`}
          onClick={() => onNavigate(previous.id)}
        >
          <span className="prevnext__label">
            <Icon name="back" size={12} /> previous
          </span>
          {previous.title}
        </a>
      ) : (
        <span />
      )}
      {next ? (
        <a
          className="prevnext__link prevnext__link--next"
          href={`#/learn/${next.id}`}
          onClick={() => onNavigate(next.id)}
        >
          <span className="prevnext__label">
            next <Icon name="arrow" size={12} />
          </span>
          {next.title}
        </a>
      ) : (
        <span />
      )}
    </div>
  )
}

/**
 * Levenshtein distance, two rows instead of a full matrix.
 *
 * The alphabet here is concept ids (`multi-head-attention`), so the strings are
 * short and the table would be wasteful; the rolling rows keep it O(min(a,b))
 * in memory. Ties are not meaningful -- the caller only asks for the nearest
 * few -- so no backtrace is kept.
 */
export function editDistance(a: string, b: string): number {
  if (a === b) return 0
  let prev: number[] = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i += 1) {
    const curr: number[] = [i]
    for (let j = 1; j <= b.length; j += 1) {
      const substitute = (prev[j - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1)
      curr[j] = globalThis.Math.min((prev[j] ?? 0) + 1, (curr[j - 1] ?? 0) + 1, substitute)
    }
    prev = curr
  }
  return prev[b.length] ?? 0
}

/**
 * The concepts closest to a mistyped id, best first.
 *
 * Distance alone would offer `cross-entropy` to someone who typed `softmax`,
 * which is a correct answer to the wrong question, so the ids are compared by
 * both their full string and their final hyphen-separated segment: a reader who
 * half-remembers a term usually remembers the tail (`attention`, `softmax`,
 * `sampling`). The segment match is a bonus rather than a filter, so an id with
 * no close tail still falls back to the whole-string edit distance.
 *
 * Ties break on id so the order is stable across runs and testable.
 */
export function nearestConcepts(id: string, ids: string[], limit = 3): string[] {
  const needle = id.toLowerCase()
  const tail = needle.split('-').pop() ?? needle
  return ids
    .map((candidate) => ({
      id: candidate,
      distance:
        editDistance(needle, candidate.toLowerCase()) +
        editDistance(tail, candidate.toLowerCase().split('-').pop() ?? candidate),
    }))
    .sort((x, y) => x.distance - y.distance || x.id.localeCompare(y.id))
    .slice(0, limit)
    .map((entry) => entry.id)
}

export function ConceptNotFound({ id }: { id: string }) {
  // The hash is deliberately left alone: a shared `#/learn/typo` link stays the
  // link the sender sent, so the recipient sees the same repair the sender did
  // rather than a silently corrected URL that hides the mistake.
  const suggestions = nearestConcepts(id, Object.keys(conceptsById))
  return (
    <article className="concept">
      <h1>No such concept</h1>
      <p className="lede">
        There is no concept called <code>{id}</code>. It may have been renamed.{' '}
        <a href="#/learn">Back to all concepts</a>.
      </p>
      {suggestions.length > 0 && (
        <nav className="concept__suggestions" aria-label="Did you mean">
          <h2>Did you mean</h2>
          <ul>
            {suggestions.map((candidate) => (
              <li key={candidate}>
                <a href={`#/learn/${candidate}`}>{conceptsById[candidate]?.title}</a>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </article>
  )
}

export { anchors, conceptsById }
