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
import { anchors, conceptsById, index, languagesFor } from '../data/sources'
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
  const chapter = index.chapters.find((c) => c.id === concept.chapter)
  const difficulty = concept.difficulty ?? 1

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
        {primary ? <CodePanel language={primary as Language} conceptId={concept.id} /> : null}

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
            <BlockView key={blockIndex} block={block} />
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

function BlockView({ block }: { block: Block }) {
  switch (block.kind) {
    case 'prose':
      return <Markdown>{block.markdown}</Markdown>

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
          <Markdown>{block.markdown ?? ''}</Markdown>
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
          return <SamplesList />
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
  const chapter = index.chapters.find((c) => c.id === concept.chapter)
  if (!chapter) return null
  const at = chapter.concepts.findIndex((c) => c.id === concept.id)
  const previous = at > 0 ? chapter.concepts[at - 1] : undefined
  const next = at < chapter.concepts.length - 1 ? chapter.concepts[at + 1] : undefined

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

export function ConceptNotFound({ id }: { id: string }) {
  return (
    <article className="concept">
      <h1>No such concept</h1>
      <p className="lede">
        There is no concept called <code>{id}</code>. It may have been renamed.{' '}
        <a href="#/learn">Back to all concepts</a>.
      </p>
    </article>
  )
}

export { anchors, conceptsById }
