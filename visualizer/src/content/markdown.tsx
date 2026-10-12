/**
 * Markdown and inline math.
 *
 * `marked` renders the GitHub-flavoured markdown; KaTeX renders `$...$` and
 * `$$...$$` inside the result. Two things worth stating plainly:
 *
 * **No raw HTML.** `marked` is configured with `mangle: false` and the renderer
 * passes `html` through as escaped text. This is a documentation site whose
 * content comes from files in the same repository, but "the content is trusted"
 * is exactly the assumption that rots: a fork with a bad edit should render
 * broken, not execute. Every element this module creates is a React element, so
 * markdown and the surrounding prose never go through `dangerouslySetInnerHTML`
 * (the app has three call sites in total; see the note on the one below).
 *
 * **The math delimiters are stripped before markdown sees them**, so a formula
 * containing `*` or `_` cannot be read as emphasis. That is the usual way
 * markdown-plus-math goes wrong: `softmax_{i}` becomes italic soup.
 */

import { marked } from 'marked'
import katex from 'katex'
import type { ReactNode } from 'react'
import { createElement, Fragment } from 'react'
import { parseCite, type CiteRange } from './cite'

// `mangle` was removed in marked v13: email obfuscation is off, and the
// `headerIds` option it sat next to is gone too. Both defaults are what we want.
marked.setOptions({ gfm: true, breaks: false })

export interface Segment {
  kind: 'text' | 'inline' | 'display'
  value: string
}

const INLINE = /\$([^$\n]+?)\$/g

function clamp(value: number, low: number, high: number): number {
  return value < low ? low : value > high ? high : value
}
const DISPLAY = /\$\$([\s\S]+?)\$\$/g

/**
 * Split a block into markdown, inline-math and display-math segments.
 *
 * Display math is matched first so a `$$...$$` block is not mistaken for two
 * empty inline formulas.
 */
export function segment(input: string): Segment[] {
  const out: Segment[] = []
  let cursor = 0
  const pattern = new RegExp(`${DISPLAY.source}|${INLINE.source}`, 'g')

  for (const match of input.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index > cursor) out.push({ kind: 'text', value: input.slice(cursor, index) })
    if (match[1] !== undefined) {
      out.push({ kind: 'display', value: match[1] })
    } else if (match[2] !== undefined) {
      out.push({ kind: 'inline', value: match[2] })
    }
    cursor = index + match[0].length
  }
  if (cursor < input.length) out.push({ kind: 'text', value: input.slice(cursor) })
  return out
}

/**
 * Render one KaTeX expression to a React element.
 *
 * Named `Math` to match the macro a reader expects in a formula, rather than
 * something like `Formula` that a reader would have to learn.
 */
export function Math({
  value,
  display = false,
  label,
}: {
  value: string
  display?: boolean
  label?: string
}): ReactNode {
  const html = katex.renderToString(value, {
    displayMode: display,
    // A bad formula renders its own source in red rather than blanking the page.
    // One broken equation should not make a concept unreadable.
    throwOnError: false,
    strict: false,
  })
  return createElement('span', {
    className: display ? 'math-display' : 'math-inline',
    // Trust boundary: one of the app's three `dangerouslySetInnerHTML` call
    // sites, and the only one fed markdown-adjacent content. It receives KaTeX's
    // own rendered output for `value` -- a formula literal from a frontmatter
    // file, not user input and not raw markdown. (The other two are in
    // `CodePanel.tsx`, which feeds Shiki's syntax-highlight HTML -- `html` for
    // the light theme and `darkHtml` for the dark one -- into the two theme
    // spans.) Nothing a reader types reaches this line; the call-site allow-list
    // is pinned by `tests/trust-boundary.test.ts`, so a fourth site fails the
    // suite rather than appearing silently.
    dangerouslySetInnerHTML: { __html: html },
    // A screen reader should hear the formula's source, not "visual formula".
    ...(label ? { 'aria-label': label } : { 'aria-hidden': 'true' }),
  })
}

/** Inline math only: `$x$` inside a sentence. */
function renderInlineMath(input: string): ReactNode {
  return segment(input).map((part, index) => {
    if (part.kind === 'inline') {
      return <Math key={index} value={part.value} label={part.value} />
    }
    return <Fragment key={index}>{part.value}</Fragment>
  })
}

/**
 * Split a document at its first heading of a given depth.
 *
 * Used by the home page to lift the introduction out of the prose so it can be a
 * hero, and by the concept page to build a table of contents from the headings
 * it already has. The split happens on the *token* stream and each half is
 * re-assembled from the tokens' own `raw` source, so no text is re-serialised and
 * no heading has to be matched by its wording: an author who renames "The
 * recorded run" cannot break the layout.
 */
export function splitAtHeading(markdown: string, depth: number): [string, string] {
  const tokens = marked.lexer(markdown)
  const at = tokens.findIndex(
    (token) =>
      (token as { type?: string }).type === 'heading' &&
      (token as { depth?: number }).depth === depth,
  )
  if (at < 0) return [markdown, '']
  const raw = (from: number, to: number) =>
    tokens
      .slice(from, to)
      .map((token) => (token as { raw?: string }).raw ?? '')
      .join('')
  return [raw(0, at), raw(at, tokens.length)]
}

/**
 * The headings in a document, in order, for a table of contents.
 *
 * Depth 2 and 3 only: a concept page's `##` sections and their `###`
 * subsections, and nothing deeper. A document with no headings yields an empty
 * list, which the caller renders as nothing at all.
 *
 * The ids are derived exactly as `Markdown` derives them -- same slug, same
 * counter, over *every* heading including the ones not listed -- so a toc entry
 * and the heading it scrolls to cannot disagree.
 */
export function headings(markdown: string): { depth: number; text: string; id: string }[] {
  const tokens = marked.lexer(markdown)
  const out: { depth: number; text: string; id: string }[] = []
  headingIds(tokens).forEach((id, index) => {
    const node = tokens[index] as { depth?: number; text?: string }
    const depth = node.depth ?? 0
    if (depth < 2 || depth > 3) return
    const text = (node.text ?? '').replace(/\s+/g, ' ').trim()
    if (text) out.push({ depth, text, id })
  })
  return out
}

function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
}

/** Token index to heading id, for one document. Counters make them unique. */
function headingIds(tokens: unknown[]): Map<number, string> {
  const ids = new Map<number, string>()
  const used = new Map<string, number>()
  tokens.forEach((token, index) => {
    const node = token as { type?: string; text?: string }
    if (node.type !== 'heading') return
    const base = slug((node.text ?? '').replace(/\s+/g, ' ').trim()) || 'section'
    const count = used.get(base) ?? 0
    used.set(base, count + 1)
    ids.set(index, count === 0 ? base : `${base}-${count + 1}`)
  })
  return ids
}

/**
 * Markdown to React elements.
 *
 * `marked.lexer` is used rather than `marked.parse` so the output is a token tree
 * this function walks itself. That is more code than parsing an HTML string, and
 * it is what lets math be rendered as React elements and links be given real
 * `href`s -- no `dangerouslySetInnerHTML` on the markdown path, and every element
 * is inspectable.
 *
 * ## Citations
 *
 * A link whose href is a `cite:` range is a *citation*: instead of an `<a>` that
 * navigates, it becomes a button that scrolls the code panel to the line range.
 * Whether it should be one is decided here, from the href alone, and the decision
 * is handed to `onCite`. That predicate returns whether the range can resolve; a
 * `false` -- no panel, no such line, a concept that declares no anchor in any
 * language -- falls back to rendering the citation as plain text, because a
 * button that does nothing when clicked is worse than a number that was never
 * made clickable. It runs during render and must be pure; the click itself goes
 * through `onActivate`, so the state change happens on the event, not on the
 * pass that builds the elements.
 *
 * The distinction matters because the *same* markdown is rendered in places that
 * have no code panel at all (the glossary, the home page cards). There is no
 * "current concept" there, so there is nothing to cite against, and the plain
 * text is the honest rendering rather than a broken control.
 */
export function Markdown({
  children,
  onCite,
  onActivate,
}: {
  children: string
  /**
   * Whether a citation's range can be resolved right now; `true` if it can. A
   * *pure* predicate: the renderer calls it while building elements to decide
   * button-versus-text, so it must not mutate state (a `setState` here is a
   * state update during render, which React will apply on the same pass and
   * which would leave the panel focused on the first citation before any click).
   */
  onCite?: ((range: CiteRange) => boolean) | undefined
  /** Called when the reader actually clicks a citation. */
  onActivate?: ((range: CiteRange) => void) | undefined
}) {
  const tokens = marked.lexer(children)
  const ids = headingIds(tokens)
  return (
    <>
      {tokens.map((token, index) => renderToken(token, index, ids.get(index), onCite, onActivate))}
    </>
  )
}

function renderToken(
  token: unknown,
  key: number,
  id?: string,
  onCite?: (range: CiteRange) => boolean,
  onActivate?: (range: CiteRange) => void,
): ReactNode {
  const node = token as {
    type?: string
    text?: string
    raw?: string
    depth?: number
    tokens?: unknown[]
    href?: string
    lang?: string
    ordered?: boolean
    items?: unknown[]
    header?: { text: string; depth: number }
    start?: number | null
  }

  switch (node.type) {
    case 'paragraph':
      return <p key={key}>{renderInlineChildren(node.tokens ?? [], onCite, onActivate)}</p>

    case 'heading': {
      const depth = node.depth ?? 2
      // Not `Math.min`: the exported `Math` component above shadows the global.
      const Tag = `h${clamp(depth, 2, 6)}` as 'h2'
      return (
        <Tag key={key} id={id}>
          {renderInlineChildren(node.tokens ?? [], onCite, onActivate)}
        </Tag>
      )
    }

    case 'code':
      return (
        <pre key={key} className="code-block" data-lang={node.lang ?? 'text'}>
          <code>{node.text ?? ''}</code>
        </pre>
      )

    case 'blockquote':
      return (
        <blockquote key={key}>
          {renderInlineChildren(node.tokens ?? [], onCite, onActivate)}
        </blockquote>
      )

    case 'list': {
      const Tag = node.ordered ? 'ol' : 'ul'
      return (
        <Tag key={key}>
          {(node.items ?? []).map((item, index) => (
            <li key={index}>
              {renderInlineChildren(
                (item as { tokens?: unknown[] }).tokens ?? [],
                onCite,
                onActivate,
              )}
            </li>
          ))}
        </Tag>
      )
    }

    case 'table':
      return (
        <div className="table-scroll" key={key}>
          <table>
            <thead>
              <tr>
                {(node.header?.text ? [{ text: node.header.text }] : []).map((cell, index) => (
                  <th key={index} scope="col">
                    {cell.text}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(node.items ?? []).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {((row as { text?: string }).text ?? '').split('|').map((cell, cellIndex) => (
                    <td key={cellIndex}>{renderInlineMath(cell.trim())}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )

    case 'hr':
      return <hr key={key} />

    case 'space':
      return null

    default:
      // Unknown token type: fall back to the raw text rather than dropping it.
      // A silently missing paragraph is the worst outcome for a reading site.
      return node.text ? <p key={key}>{renderInlineMath(node.text)}</p> : null
  }
}

/**
 * The plain text of an inline token list, for use in an accessible name.
 *
 * `aria-label` takes a string, not elements, so the citation's rendered children
 * cannot be reused there. Nesting is walked (`**bold**` is `strong` wrapping
 * `text`) because a truncated label is worse than no label on the one control
 * whose whole purpose is to name a location precisely.
 */
function labelText(tokens: unknown[]): string {
  return tokens
    .map((token) => {
      const node = token as { text?: string; tokens?: unknown[] }
      if (node.tokens?.length) return labelText(node.tokens)
      return node.text ?? ''
    })
    .join('')
}

function renderInlineChildren(
  tokens: unknown[],
  onCite?: (range: CiteRange) => boolean,
  onActivate?: (range: CiteRange) => void,
): ReactNode {
  return tokens.map((token, index) => {
    const node = token as {
      type?: string
      text?: string
      tokens?: unknown[]
      href?: string
      title?: string
    }
    switch (node.type) {
      case 'text':
        /*
         * A `text` token that carries children is a *block* of inline content
         * handed over whole -- which is what `marked` does for a list item -- and
         * its children still have to be walked. Rendering `node.text` directly
         * showed the reader the literal `**bold**` and `` `code` `` markers
         * instead of the emphasis and the code span.
         */
        if (node.tokens?.length)
          return (
            <Fragment key={index}>{renderInlineChildren(node.tokens, onCite, onActivate)}</Fragment>
          )
        return <Fragment key={index}>{renderInlineMath(node.text ?? '')}</Fragment>
      case 'strong':
        return (
          <strong key={index}>{renderInlineChildren(node.tokens ?? [], onCite, onActivate)}</strong>
        )
      case 'em':
        return <em key={index}>{renderInlineChildren(node.tokens ?? [], onCite, onActivate)}</em>
      case 'codespan':
        return <code key={index}>{node.text}</code>
      case 'br':
        return <br key={index} />
      case 'paragraph':
        return (
          <Fragment key={index}>
            {renderInlineChildren(node.tokens ?? [], onCite, onActivate)}
          </Fragment>
        )
      case 'link': {
        const label = renderInlineChildren(node.tokens ?? [], onCite, onActivate)
        const cite = parseCite(node.href)
        if (cite) {
          const landed = onCite?.(cite) ?? false
          // No handler, or a range that resolved to nothing: keep the author's
          // words as text. See the `Markdown` doc comment.
          if (!landed) return <Fragment key={index}>{label}</Fragment>
          return (
            <button
              key={index}
              type="button"
              className="cite"
              // The visible label is the author's own phrase, which is right for
              // reading; the label names the target so the same phrase in two
              // places is still distinguishable to a screen reader, and so the
              // e2e suite can assert the citation points where it claims.
              aria-label={`show ${labelText(node.tokens ?? [])} in the source`}
              data-cite={`${cite.start}-${cite.end}`}
              onClick={() => onActivate?.(cite)}
            >
              {label}
            </button>
          )
        }
        return (
          <a
            key={index}
            href={node.href}
            {...(node.href?.startsWith('http')
              ? { target: '_blank', rel: 'noreferrer noopener' }
              : {})}
          >
            {label}
          </a>
        )
      }
      case 'del':
        return <del key={index}>{renderInlineChildren(node.tokens ?? [], onCite, onActivate)}</del>
      default:
        return node.text ? <Fragment key={index}>{renderInlineMath(node.text)}</Fragment> : null
    }
  })
}
