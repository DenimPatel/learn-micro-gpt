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
import { ok, type Language } from '../data/types'
import { Icon } from './Icon'
import { LoadableResult, type Loadable } from './LoadableResult'

const CONTEXT_LINES = 3

/**
 * Copy the visible slice, and say so for a moment.
 *
 * The failure is the interesting part: `navigator.clipboard` is undefined on an
 * insecure origin, which includes a bundle served over plain HTTP (and, in
 * principle, one opened from `file://` — though the app does not boot there; see
 * `app/router.ts`). So a rejected or missing clipboard is a no-op with no state
 * change, rather than an unhandled rejection and a button that claims success.
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
   *
   * The state is `Loadable<string>` -- the loader's own `Result` while it is in
   * flight, `undefined` before. A failed chunk is therefore a *rendered* error
   * state (`LoadableResult` below) rather than the empty box this used to fall
   * back to: it is now a value the promise resolves, so there is somewhere for
   * it to go.
   */
  const [source, setSource] = useState<Loadable<string>>(() => {
    const cached = loadedSource(language)
    return cached === undefined ? undefined : ok(cached)
  })
  useEffect(() => {
    if (source !== undefined) return
    let cancelled = false
    void loadSource(language).then((result) => {
      if (!cancelled) setSource(result)
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

  if (source === undefined || !source.ok) {
    const name = LANGUAGES[language].label
    return (
      <div className="code-panel" role="group" aria-label={`loading the ${language} source`}>
        <div className="code-panel__bar">
          <span className="code-panel__name">{name}</span>
        </div>
        <LoadableResult
          result={source}
          loading={`loading the ${name} source\u2026`}
          error={(message) => (
            <>
              the {name} source could not be loaded ({message}).
            </>
          )}
        >
          {() => null}
        </LoadableResult>
      </div>
    )
  }

  return (
    <CodeBody language={language} source={source.value} range={range} defaultOpen={defaultOpen} />
  )
}

/**
 * The loaded panel: a `Result` narrowed to its success variant.
 *
 * Split out so the hooks below it (`useHighlight` twice over the slice) run
 * unconditionally, and so the body deals in a plain `string` rather than
 * unwrapping a `Result` at every read. The `Result` is the prop, not the text,
 * because that is what "this panel has a source" is *typed* as.
 */
function CodeBody({
  language,
  source,
  range,
  defaultOpen,
}: {
  language: Language
  source: string
  range: { start: number; end: number } | null
  defaultOpen: boolean
}) {
  const [expanded, setExpanded] = useState(defaultOpen)
  const panelId = useId()
  const lines = useMemo(() => source.split('\n'), [source])
  const from = range ? Math.max(0, range.start - 1 - CONTEXT_LINES) : 0
  const to = range ? Math.min(lines.length, range.end + CONTEXT_LINES) : lines.length
  const slice = lines.slice(from, to).join('\n')
  const { html, ready } = useHighlight(slice, HIGHLIGHT_LANG[language], 'light')
  const { html: darkHtml, ready: darkReady } = useHighlight(slice, HIGHLIGHT_LANG[language], 'dark')
  const [copied, setCopied] = useState(false)

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
          {/*
            Trust boundary: these two `dangerouslySetInnerHTML` sites -- the
            app's second and third, the first is in `content/markdown.tsx` --
            receive Shiki's syntax-highlight HTML (`html` for the light theme,
            `darkHtml` for the dark one) for the pinned source this panel was
            handed. The input is the reference file, read at build time; raw
            markdown and other user text do *not* reach here, and neither does
            anything the reader types. If that ever changes -- a panel fed
            concept prose, say -- this becomes an injection path, which is why
            the guard lives in `tests/trust-boundary.test.ts`.
          */}
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
