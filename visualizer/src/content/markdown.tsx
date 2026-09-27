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
 * broken, not execute. Every element below is created by React, so there is no
 * `dangerouslySetInnerHTML` anywhere in the app.
 *
 * **The math delimiters are stripped before markdown sees them**, so a formula
 * containing `*` or `_` cannot be read as emphasis. That is the usual way
 * markdown-plus-math goes wrong: `softmax_{i}` becomes italic soup.
 */

import { marked } from 'marked'
import katex from 'katex'
import type { ReactNode } from 'react'
import { createElement, Fragment } from 'react'

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
    // The one `dangerouslySetInnerHTML` in the app, and it is KaTeX's own output
    // from a string literal in a frontmatter file -- not from user input, and not
    // from markdown.
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
 * `href`s -- no `dangerouslySetInnerHTML`, and every element is inspectable.
 */
export function Markdown({ children }: { children: string }) {
  const tokens = marked.lexer(children)
  const ids = headingIds(tokens)
  return <>{tokens.map((token, index) => renderToken(token, index, ids.get(index)))}</>
}

function renderToken(token: unknown, key: number, id?: string): ReactNode {
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
      return <p key={key}>{renderInlineChildren(node.tokens ?? [])}</p>

    case 'heading': {
      const depth = node.depth ?? 2
      // Not `Math.min`: the exported `Math` component above shadows the global.
      const Tag = `h${clamp(depth, 2, 6)}` as 'h2'
      return (
        <Tag key={key} id={id}>
          {renderInlineChildren(node.tokens ?? [])}
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
      return <blockquote key={key}>{renderInlineChildren(node.tokens ?? [])}</blockquote>

    case 'list': {
      const Tag = node.ordered ? 'ol' : 'ul'
      return (
        <Tag key={key}>
          {(node.items ?? []).map((item, index) => (
            <li key={index}>
              {renderInlineChildren((item as { tokens?: unknown[] }).tokens ?? [])}
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

function renderInlineChildren(tokens: unknown[]): ReactNode {
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
          return <Fragment key={index}>{renderInlineChildren(node.tokens)}</Fragment>
        return <Fragment key={index}>{renderInlineMath(node.text ?? '')}</Fragment>
      case 'strong':
        return <strong key={index}>{renderInlineChildren(node.tokens ?? [])}</strong>
      case 'em':
        return <em key={index}>{renderInlineChildren(node.tokens ?? [])}</em>
      case 'codespan':
        return <code key={index}>{node.text}</code>
      case 'br':
        return <br key={index} />
      case 'paragraph':
        return <Fragment key={index}>{renderInlineChildren(node.tokens ?? [])}</Fragment>
      case 'link':
        return (
          <a
            key={index}
            href={node.href}
            {...(node.href?.startsWith('http')
              ? { target: '_blank', rel: 'noreferrer noopener' }
              : {})}
          >
            {renderInlineChildren(node.tokens ?? [])}
          </a>
        )
      case 'del':
        return <del key={index}>{renderInlineChildren(node.tokens ?? [])}</del>
      default:
        return node.text ? <Fragment key={index}>{renderInlineMath(node.text)}</Fragment> : null
    }
  })
}
