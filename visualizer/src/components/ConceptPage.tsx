/**
 * A concept page: prose, code, shapes, maths, and inline trace widgets.
 *
 * The body arrives from `tools/render_content.py` already split into typed
 * blocks, so this file is a switch over block kinds and nothing else. Rendering
 * lives in the Python tool and validation lives in `tools/validate_concepts.py`,
 * which means a malformed directive fails the build rather than rendering as
 * nothing.
 */

import { useEffect } from 'react'
import { Math, Markdown } from '../content/markdown'
import { anchors, conceptsById, index, languagesFor } from '../data/sources'
import type { Block, Concept, Language } from '../data/types'
import { CodePanel } from './CodePanel'
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
    document.title = `${concept.title} — learn microgpt`
  }, [concept.id, markSeen])

  const languages = languagesFor(concept.id)
  const [primary, ...rest] = languages
  const chapter = index.chapters.find((c) => c.id === concept.chapter)
  const position = chapter ? `${chapter.title} · ${concept.order} of ${chapter.concepts.length}` : ''

  return (
    <article className="concept" aria-labelledby="concept-title">
      <nav className="concept__breadcrumb" aria-label="Breadcrumb">
        <a href="#/learn">All concepts</a>
        {chapter ? <span aria-hidden="true"> / </span> : null}
        {chapter ? <span>{chapter.title}</span> : null}
      </nav>

      <header className="concept__header">
        <h1 id="concept-title">{concept.title}</h1>
        <p className="concept__meta">
          {position}
          {concept.difficulty ? <> · difficulty {concept.difficulty}/3</> : null}
        </p>
        <p className="concept__summary">{concept.summary}</p>
      </header>

      {concept.math ? (
        <div className="concept__math">
          <Math value={concept.math} display label="The formula for this concept" />
        </div>
      ) : null}

      {primary ? (
        <CodePanel language={primary as Language} conceptId={concept.id} />
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
            <CodePanel key={language} language={language} conceptId={concept.id} defaultOpen={false} />
          ))}
        </section>
      ) : null}

      {concept.shapes?.length ? <ShapeTable names={concept.shapes.map((s) => s.name)} /> : null}

      <div className="prose">
        {concept.blocks.map((block, index) => (
          <BlockView key={index} block={block} />
        ))}
      </div>

      <nav className="concept__nav" aria-label="Concept navigation">
        <PrevNext concept={concept} onNavigate={onNavigate} />
      </nav>
    </article>
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
          {block.title ? <p className="callout__title">{block.title}</p> : null}
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
          return <AttentionHeatmap step={num('step', 0)} pos={num('pos', 3)} head={num('head', 0)} />
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
      The documents and sampled names for step {step} are in{' '}
      <code>traces/python/micro/</code>, and the attention and probability data behind them is on
      the <a href="#/explore">Explore</a> page.
    </p>
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
        <a className="prevnext__link prevnext__link--prev" href={`#/learn/${previous.id}`} onClick={() => onNavigate(previous.id)}>
          <span className="prevnext__label">previous</span>
          {previous.title}
        </a>
      ) : (
        <span />
      )}
      {next ? (
        <a className="prevnext__link prevnext__link--next" href={`#/learn/${next.id}`} onClick={() => onNavigate(next.id)}>
          <span className="prevnext__label">next</span>
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
      <p>
        There is no concept called <code>{id}</code>. It may have been renamed.{' '}
        <a href="#/learn">Back to all concepts</a>.
      </p>
    </article>
  )
}

export { anchors, conceptsById }
