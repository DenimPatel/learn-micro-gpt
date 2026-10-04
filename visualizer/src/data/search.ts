/**
 * Concept search: the words a reader has, resolved to the concept they mean.
 *
 * ## Why this is a module and not three lines in `App.tsx`
 *
 * With eighteen concepts and slugs like `attention-block`, a reader who
 * half-remembers a term has to scan the whole rail. Filtering it is a feature
 * whose *whole* behaviour is the matching rule, and the matching rule is the
 * part that is easy to get subtly wrong -- case, whitespace, which fields it
 * looks at, whether a summary-only hit tells the reader why it matched. Kept
 * here it is a pure function over plain data: no DOM, no state, no React, so the
 * unit tests in `tests/search.test.ts` can pin every one of those rules, and the
 * Playwright case only has to prove the input is wired to it.
 *
 * ## What it is not
 *
 * It is not a data pipeline. The corpus is the `concepts` array `sources.ts`
 * already builds from the generated `content.json` -- matching runs in memory
 * over objects the bundle inlines, so nothing here can introduce a request, a
 * cache or a second source of truth. `App.tsx` passes `chapters` in from
 * `sequence.ts`, which is the one owner of reading order, and the results come
 * back in the order it was handed: a search reorders nothing, so the rail and
 * the results list agree about what "the next concept" is.
 */

import type { Chapter, ChapterEntry } from './types'

/** One concept that matched, with the field that matched. */
export interface SearchHit {
  id: string
  title: string
  summary: string
  /**
   * `true` when the title matches and `false` when only the summary does.
   *
   * This exists so the rail can render the summary for a summary-only hit --
   * a reader who typed "chain rule" and got "Walking the graph backwards" has to
   * be told *why* that is the answer. It is a boolean rather than a `match` shape
   * because the rail only needs to decide whether to show the snippet.
   */
  titleMatch: boolean
}

/** Fold a query or a field down to what the comparison is really about. */
function fold(value: string): string {
  return value.trim().toLowerCase()
}

/**
 * Every concept matching `query`, in the order the chapters were given.
 *
 * - Case-insensitive: the query and both fields are folded before comparing.
 * - Whitespace-insensitive at the edges: `"  softmax  "` finds softmax. Interior
 *   spaces are not collapsed -- a reader pasting a phrase with a typo in its
 *   spacing is asking for something that is not there, and silently pretending
 *   otherwise would make the no-match state unreachable from a real mistake.
 * - Title *and* summary, which is the acceptance criterion: `"chain rule"` is in
 *   `autograd-backward`'s summary and nowhere in its title.
 * - An empty (or all-whitespace) query matches nothing rather than everything.
 *   The caller renders the unfiltered rail for that case, so "no query" and
 *   "a query that matched nothing" stay different screens, which is the
 *   distinction a reader depends on when their term is not in the corpus.
 * - Chapter titles are searched too: a reader who remembers the section
 *   ("training") gets that section's concepts, which is the other half of
 *   finding a page you cannot name.
 */
export function searchConcepts(chapters: readonly Chapter[], query: string): SearchHit[] {
  const needle = fold(query)
  if (needle === '') return []

  const hits: SearchHit[] = []
  for (const chapter of chapters) {
    const chapterMatches = fold(chapter.title).includes(needle)
    for (const entry of chapter.concepts) {
      const hit = match(entry, needle)
      if (hit) hits.push(hit)
      else if (chapterMatches) hits.push({ ...entry, titleMatch: false })
    }
  }
  return hits
}

/** One concept against an already-folded needle, or `undefined` if it misses. */
function match(entry: ChapterEntry, needle: string): SearchHit | undefined {
  const title = fold(entry.title)
  const summary = fold(entry.summary)
  const titleMatch = title.includes(needle)
  if (titleMatch || summary.includes(needle)) {
    return { id: entry.id, title: entry.title, summary: entry.summary, titleMatch }
  }
  return undefined
}
