/**
 * The one reading sequence, derived from `index.json`.
 *
 * ## Why this module exists
 *
 * The sidebar, the previous/next links, the home page's "continue" button and
 * the graph's chapter grouping all have to agree on one order: the order the
 * curriculum is written in. That order lives in exactly one place -- the
 * `chapters` array of `content/index.json`, which `tools/render_content.py`
 * copies verbatim into `generated/index.json` -- and every surface must read it
 * from there rather than restate it.
 *
 * The failure this prevents is quiet: a literal id list in a component compiles,
 * renders, and passes every test until someone edits the curriculum, at which
 * point the app confidently shows the old order and nobody notices. Deriving the
 * sequence here, once, means a concept added to the data appears in the rail,
 * gains a previous/next link and joins its chapter with no code change at all.
 *
 * ## What is *not* here
 *
 * Nothing in this file is presentation. There is no chapter accent colour, no
 * display label, no grouping head -- those are not in `index.json` and inventing
 * them from it would be worse than keeping them. They belong beside the markup
 * that uses them, keyed by the derived `Chapter['id']` so the typechecker is the
 * thing that notices a new chapter has appeared.
 */

import { index } from './sources'
import type { Chapter, ChapterEntry } from './types'

/**
 * The chapters in reading order.
 *
 * This is the `chapters` array as written. `order` is carried on each chapter as
 * a display number, but the array *is* the sequence -- `index.json` says so in
 * its own comment ("this file is authoritative for *sequence*"), and
 * `tools/validate_concepts.py` rejects a chapter whose number disagrees with its
 * position. Sorting here would be a second opinion about a question the data
 * already answers, so the array order is used directly and the test pins the two
 * to each other.
 */
export const chapters: readonly Chapter[] = index.chapters

/** Every concept, chapter by chapter: the reading order, flattened. */
export const conceptSequence: readonly ChapterEntry[] = chapters.flatMap(
  (chapter) => chapter.concepts,
)

/** The total the rail counts the reader against; `index.json` decides, not a number. */
export const conceptCount = conceptSequence.length

/** The chapter ids, in sequence. Derived, so a new chapter is valid the moment it is data. */
export const chapterIds: readonly string[] = chapters.map((chapter) => chapter.id)

/** The chapter a concept belongs to, or `undefined` if the data has no such chapter. */
export function chapterOf(conceptId: string): Chapter | undefined {
  return chapters.find((chapter) => chapter.concepts.some((entry) => entry.id === conceptId))
}

/**
 * The concepts either side of this one, within its chapter.
 *
 * Reading order is chapter order and a chapter is a run: the previous and next
 * link move inside the chapter, and a chapter boundary is the end of the road
 * (the concept pages say so by rendering no link on that side). Keyed off the
 * same flattened sequence the rail uses, so the two cannot disagree about what
 * "next" means.
 */
export function neighbours(conceptId: string): {
  previous: ChapterEntry | undefined
  next: ChapterEntry | undefined
} {
  const chapter = chapterOf(conceptId)
  if (!chapter) return { previous: undefined, next: undefined }
  const at = chapter.concepts.findIndex((entry) => entry.id === conceptId)
  if (at < 0) return { previous: undefined, next: undefined }
  return {
    previous: at > 0 ? chapter.concepts[at - 1] : undefined,
    next: at < chapter.concepts.length - 1 ? chapter.concepts[at + 1] : undefined,
  }
}

/** Where to begin: the first concept in the first chapter, if there is one. */
export const firstConcept: ChapterEntry | undefined = conceptSequence[0]

/** The first concept a reader has not opened yet, in reading order. */
export function firstUnread(seen: ReadonlySet<string>): ChapterEntry | undefined {
  return conceptSequence.find((entry) => !seen.has(entry.id))
}
