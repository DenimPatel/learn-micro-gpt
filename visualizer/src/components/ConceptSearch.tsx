/**
 * The rail's filter field -- the concept search, in the rail itself.
 *
 * ## Why a filter and not a second page
 *
 * A reader's goal when they search is to *read a concept*, and a results page
 * would make that two navigations. Filtering the rail keeps the answer and the
 * map in one place: the list they were scanning narrows to what they typed.
 *
 * ## The states, because they are the feature
 *
 * Four of them, and each is a deliberate decision rather than a default:
 *
 *  - **Empty query** -- the ordinary rail, chapter by chapter. Nothing is
 *    hidden and no "0 results" is shown for a query nobody typed.
 *  - **Matches** -- the same markup, chapter by chapter, so the reader's mental
 *    map survives the filter. A concept that matched on its *summary* carries
 *    that summary underneath it, in the rail's own voice, because "Walking the
 *    graph backwards" is not recognisable as the answer to "chain rule" without
 *    it.
 *  - **No matches** -- an explicit sentence naming the query, plus how many
 *    concepts were searched, so the screen is not a blank void the reader reads
 *    as broken. It also offers the count of what is actually there.
 *  - **Too little of a query** -- one character matches over half the rail
 *    (every "the", every "a"), which is noise dressed as a result. Below
 *    `MIN_QUERY` characters the rail stays unfiltered and a hint explains that
 *    more is needed, rather than pretending to have searched.
 *
 * ## Keyboard
 *
 * The field is where a keyboard reader starts, and it must be possible to leave
 * the search *without* a pointer:
 *
 *  - Escape clears the query when there is one, and clears focus when there is
 *    not, so the second Escape puts the reader back on the document. It stops
 *    propagation: the shell closes the phone drawer on Escape, and a reader
 *    clearing a half-typed term has not asked to close the drawer.
 *  - ArrowDown moves into the results (a real `<a>` per hit), so a reader never
 *    has to Tab past the query and the hint to reach the first concept.
 *  - Tab reaches the clear button, then the results.
 */

import { useEffect, useRef, useState } from 'react'
import { hrefFor } from '../app/router'
import { Icon } from './Icon'
import { searchConcepts } from '../data/search'
import { chapters } from '../data/sequence'

/** Below this many characters the rail does not filter; see the header. */
export const MIN_QUERY = 2

export function ConceptSearch({
  seen,
  currentId,
  onNavigate,
}: {
  seen: ReadonlySet<string>
  /** The concept being read, so a filtered result keeps its `aria-current`. */
  currentId: string | undefined
  /** Close the phone drawer and return focus to the reading column. */
  onNavigate: () => void
}) {
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLOListElement | null>(null)

  const trimmed = query.trim()
  const active = trimmed.length >= MIN_QUERY
  const hits = active ? searchConcepts(chapters, trimmed) : []
  const total = chapters.reduce((sum, chapter) => sum + chapter.concepts.length, 0)

  // Escape needs to read the *current* query without re-binding a document
  // listener on every keystroke, so the value is mirrored in a ref.
  const queryRef = useRef(query)
  useEffect(() => {
    queryRef.current = query
  }, [query])

  const clear = () => {
    setQuery('')
    inputRef.current?.focus()
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    // The shell listens for Escape to close the drawer. A reader clearing a
    // half-typed term has not asked to close anything, so this stays local.
    if (event.key === 'Escape') {
      event.stopPropagation()
      if (queryRef.current !== '') setQuery('')
      else inputRef.current?.blur()
      return
    }
    if (event.key === 'ArrowDown' && active) {
      const first = listRef.current?.querySelector('a')
      if (first instanceof HTMLElement) {
        event.preventDefault()
        first.focus()
      }
    }
  }

  return (
    <div className="app__search">
      <label className="app__search-field">
        <Icon name="search" size={14} />
        <span className="visually-hidden">Search concepts</span>
        <input
          ref={inputRef}
          type="search"
          className="app__search-input"
          value={query}
          placeholder="find a concept…"
          autoComplete="off"
          spellCheck={false}
          aria-label="Search concepts by title or summary"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
        />
        {query !== '' ? (
          <button
            type="button"
            className="app__search-clear"
            onClick={clear}
            aria-label="Clear the search"
            title="Clear the search"
          >
            <Icon name="close" size={13} />
          </button>
        ) : null}
      </label>

      {/* Two live regions, one short, so a reader using a screen reader hears
          the shape of the result rather than the whole list being announced on
          every keystroke. */}
      <p className="app__search-status" role="status" aria-live="polite">
        {!active
          ? trimmed === ''
            ? ''
            : `keep typing — ${MIN_QUERY} letters or more searches all ${total} concepts`
          : hits.length === 0
            ? `no concept matches “${trimmed}” — ${total} concepts searched`
            : `${hits.length} of ${total} concepts match “${trimmed}”`}
      </p>

      {!active ? (
        <div className="app__sidebar-rail" data-filtered="false">
          {chapters.map((chapter) => (
            <ChapterGroup
              key={chapter.id}
              title={chapter.title}
              entries={chapter.concepts}
              seen={seen}
              currentId={currentId}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      ) : hits.length === 0 ? (
        <div className="app__search-empty">
          <p>
            No concept mentions <strong>“{trimmed}”</strong> in its title or summary.
          </p>
          <p className="app__search-hint">
            Try a shorter word, or{' '}
            <button type="button" className="app__search-link" onClick={clear}>
              clear the search
            </button>{' '}
            to see all {total} in reading order.
          </p>
        </div>
      ) : (
        <ol className="app__search-results" ref={listRef}>
          {hits.map((hit) => {
            const current = hit.id === currentId
            return (
              <li key={hit.id}>
                <a
                  href={hrefFor({ name: 'concept', conceptId: hit.id })}
                  className={seen.has(hit.id) ? 'is-opened' : undefined}
                  aria-current={current ? 'page' : undefined}
                  onClick={onNavigate}
                >
                  <span className="app__search-title">{hit.title}</span>
                  {/* Only for a summary-only match: on a title match the summary
                      is noise, and the reader already recognises the answer. */}
                  {hit.titleMatch ? null : (
                    <span className="app__search-snippet">{snippet(hit.summary, trimmed)}</span>
                  )}
                </a>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}

/**
 * The part of the summary around the match, so the reason is visible.
 *
 * A snippet rather than the whole summary: the summaries run to two or three
 * sentences, and the rail is 270px wide -- an unsnipped hit would push every
 * other result off the screen, which is the scanning problem this feature exists
 * to solve. Falls back to the head of the summary if the match is somehow not
 * found (the needle is matched against a folded copy, so the offsets can differ
 * from the original).
 */
function snippet(summary: string, needle: string): string {
  const at = summary.toLowerCase().indexOf(needle)
  if (at < 0) return summary
  const start = Math.max(0, at - 32)
  const raw = summary.slice(start, at + needle.length + 60).trim()
  return `${start > 0 ? '…' : ''}${raw}…`
}

/** One chapter's head and its concepts, in reading order. */
function ChapterGroup({
  title,
  entries,
  seen,
  currentId,
  onNavigate,
}: {
  title: string
  entries: readonly { id: string; title: string }[]
  seen: ReadonlySet<string>
  currentId: string | undefined
  onNavigate: () => void
}) {
  return (
    <section className="app__sidebar-group">
      <h2>{title}</h2>
      <ol>
        {entries.map((entry) => {
          const current = entry.id === currentId
          return (
            <li key={entry.id}>
              <a
                href={hrefFor({ name: 'concept', conceptId: entry.id })}
                className={seen.has(entry.id) ? 'is-opened' : undefined}
                aria-current={current ? 'page' : undefined}
                onClick={onNavigate}
              >
                {entry.title}
              </a>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
