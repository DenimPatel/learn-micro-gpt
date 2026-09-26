/**
 * The code panel: a source file, a line range, and the ability to see the rest.
 *
 * ## Why the whole file is loaded and then cropped
 *
 * Every language file is bundled at build time (`src/data/sources.ts`), so there
 * is no fetch and no URL to get wrong. The panel then shows the concept's
 * resolved range, with a couple of lines of context on each side, and offers the
 * whole file behind a disclosure.
 *
 * The alternative -- highlighting only the range -- is what makes line numbers
 * and copy-paste awkward, and it re-introduces the "extract a slice of highlighted
 * HTML" problem that broke the previous implementation. Cropping the *text* and
 * highlighting the result keeps one code path.
 *
 * ## Accessibility
 *
 * The highlighted `<pre>` is the accessible content, and it carries an
 * `aria-label` naming the file and the lines. A screen reader will otherwise read
 * a hundred `span`s with no indication of what file they came from. The line
 * gutter is `aria-hidden` because the numbers are presentational: the same
 * information is in the label and in the plain-text `<pre>` fallback that every
 * code panel has before Shiki loads.
 */

import { useEffect, useId, useMemo, useState } from 'react'
import { useHighlight } from '../content/highlight'
import { HIGHLIGHT_LANG, LANGUAGES, anchorFor, loadSource, loadedSource } from '../data/sources'
import type { Language } from '../data/types'
import { Icon } from './Icon'

const CONTEXT_LINES = 3

/**
 * Copy the visible slice, and say so for a moment.
 *
 * The failure is the interesting part: `navigator.clipboard` is undefined on an
 * insecure origin, which includes a bundle opened from `file://` -- one of the
 * four ways this site is meant to work (see `app/router.ts`). So a rejected or
 * missing clipboard is a no-op with no state change, rather than an unhandled
 * rejection and a button that claims success.
 */
async function copy(text: string, setCopied: (value: boolean) => void): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    return
  }
  setCopied(true)
  setTimeout(() => setCopied(false), 1600)
}

export function CodePanel({
  language,
  conceptId,
  focus,
  defaultOpen = true,
}: {
  language: Language
  /** When set, the panel opens focused on this concept's resolved range. */
  conceptId?: string
  /** An explicit 1-based inclusive line range, which wins over `conceptId`. */
  focus?: { start: number; end: number }
  defaultOpen?: boolean
}) {
  const [expanded, setExpanded] = useState(defaultOpen)
  const panelId = useId()

  /*
   * The source arrives asynchronously, because each language is a dynamic
   * `?raw` import in its own chunk (see `data/sources.ts` for why). Until it
   * lands the panel shows a one-line placeholder rather than an empty box, so
   * the page never reflows into looking broken.
   *
   * The cache is read in the state *initialiser*, not in an effect. Every code
   * panel on a concept page asks for the same source, so after the first one
   * loads the rest would each setState on mount -- a synchronous state update
   * inside an effect, which is a cascading render, and eslint's
   * `set-state-in-effect` rule is right about that.
   */
  const [source, setSource] = useState<string | null>(() => loadedSource(language) ?? null)
  useEffect(() => {
    if (source !== null) return
    let cancelled = false
    void loadSource(language).then((text) => {
      if (!cancelled) setSource(text)
    })
    return () => {
      cancelled = true
    }
    // `source` is a guard, not a dependency: including it would re-run the
    // effect on every load and restart the import that just resolved.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language])

  const range = useMemo(() => {
    if (focus) return focus
    if (!conceptId) return null
    return anchorFor(conceptId, language) ?? null
  }, [focus, conceptId, language])

  const text = source ?? ''
  const lines = useMemo(() => text.split('\n'), [text])
  const from = range ? Math.max(0, range.start - 1 - CONTEXT_LINES) : 0
  const to = range ? Math.min(lines.length, range.end + CONTEXT_LINES) : lines.length
  const slice = lines.slice(from, to).join('\n')
  const { html, ready } = useHighlight(slice, HIGHLIGHT_LANG[language], 'light')
  const { html: darkHtml, ready: darkReady } = useHighlight(slice, HIGHLIGHT_LANG[language], 'dark')
  const [copied, setCopied] = useState(false)

  if (source === null) {
    const name = LANGUAGES[language].label
    return (
      <div className="code-panel" role="group" aria-label={`loading the ${language} source`}>
        <div className="code-panel__bar">
          <span className="code-panel__name">{name}</span>
        </div>
        <p className="code-panel__missing">loading the {name} source&hellip;</p>
      </div>
    )
  }

  const label = range
    ? `${language} source, lines ${range.start} to ${range.end} of ${lines.length}`
    : `${language} source, ${lines.length} lines`

  const gutter = Array.from({ length: to - from }, (_, index) => {
    const number = from + index + 1
    const focused = range ? number >= range.start && number <= range.end : false
    return (
      <span key={number} className={focused ? 'code-gutter__line is-focused' : 'code-gutter__line'}>
        {number}
      </span>
    )
  })

  return (
    <div className="code-panel" role="group" aria-label={label}>
      <div className="code-panel__bar">
        <span className="code-panel__name">{LANGUAGES[language].label}</span>
        {range ? (
          <span className="code-panel__range">
            lines {range.start}&ndash;{range.end}
          </span>
        ) : null}
        <span className="code-panel__spacer" />
        <button
          type="button"
          className="code-panel__copy"
          onClick={() => void copy(slice, setCopied)}
        >
          <Icon name={copied ? 'check' : 'copy'} size={13} />
          {copied ? 'copied' : 'copy'}
        </button>
        <button
          type="button"
          className="code-panel__toggle"
          aria-expanded={expanded}
          aria-controls={panelId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? 'show less' : 'show whole file'}
        </button>
      </div>

      <div id={panelId} hidden={!expanded} className="code-panel__body">
        <div className="code-gutter" aria-hidden="true">
          {gutter}
        </div>
        {/*
          A `div`, not a `pre`.
          
          Shiki's `codeToHtml` returns a complete `<pre class="shiki">`. Wrapping
          that in another `<pre>` nests one preformatted element inside another,
          which is invalid HTML and which the browser re-parses -- silently
          changing the structure the stylesheet and the e2e selectors were written
          against. So the highlighted output goes in as-is, and the fallback gets
          its own `<pre>` below.
        */}
        <div className="code-panel__code" tabIndex={0} role="group" aria-label={label}>
          {ready && darkReady ? (
            <>
              <span className="light-only" dangerouslySetInnerHTML={{ __html: html ?? '' }} />
              <span className="dark-only" dangerouslySetInnerHTML={{ __html: darkHtml ?? '' }} />
            </>
          ) : (
            <pre className="code-block">
              <code>
                {slice}
                {/* A polite notice, not a spinner: the text is already readable. */}
                <span className="code-panel__loading"> (highlighting&hellip;)</span>
              </code>
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}
