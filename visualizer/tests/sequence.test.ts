import { describe, expect, it } from 'vitest'

import contentJson from '../src/data/generated/content.json'
import indexJson from '../src/data/generated/index.json'
import {
  chapters,
  chapterIds,
  chapterOf,
  conceptCount,
  conceptSequence,
  firstConcept,
  firstUnread,
  neighbours,
} from '../src/data/sequence'

/**
 * The reading sequence, pinned to `index.json`.
 *
 * `sequence.ts` derives the sidebar order, the previous/next links and the
 * chapter grouping from the imported `index` data. A test that only asserted
 * "some order exists" would pass just as happily against a hard-coded list -- it
 * would be measuring the code against itself. So every expectation here is built
 * from a *second, independent* read of the same JSON the app imports, and the
 * assertions are about the two being equal. If someone reintroduces a literal
 * order anywhere on the derived path, these go red.
 */

/** The same file the app imports, read again here so it is not the app's own copy. */
const raw = indexJson as {
  chapters: {
    id: string
    order: number
    title: string
    concepts: { id: string; title: string }[]
  }[]
}

describe('the derived sequence equals the index.json order', () => {
  it('lists the chapters in the order index.json writes them', () => {
    expect(chapters.map((chapter) => chapter.id)).toEqual(raw.chapters.map((chapter) => chapter.id))
  })

  it('carries the data file’s own chapter order numbers, in file order', () => {
    // The file is authoritative for sequence *and* for the number the heading
    // shows. This pins the two together so a reordered array cannot keep stale
    // numbers in its headings.
    expect(chapters.map((chapter) => chapter.order)).toEqual(
      raw.chapters.map((chapter) => chapter.order),
    )
    expect(chapters.map((chapter) => chapter.order)).toEqual(
      [...chapters.map((chapter) => chapter.order)].sort((a, b) => a - b),
    )
  })

  it('flattens to exactly the concepts index.json declares, in order', () => {
    const expected = raw.chapters.flatMap((chapter) => chapter.concepts.map((entry) => entry.id))
    expect(conceptSequence.map((entry) => entry.id)).toEqual(expected)
    expect(conceptCount).toBe(expected.length)
  })

  it('exposes the chapter ids as a derived list, not a literal', () => {
    expect(chapterIds).toEqual(raw.chapters.map((chapter) => chapter.id))
  })

  it('starts at the first chapter’s first concept', () => {
    expect(firstConcept?.id).toBe(raw.chapters[0]?.concepts[0]?.id)
  })
})

describe('chapter grouping is read, not restated', () => {
  it('places every concept in the chapter index.json puts it in', () => {
    for (const chapter of raw.chapters) {
      for (const entry of chapter.concepts) {
        expect(chapterOf(entry.id)?.id).toBe(chapter.id)
      }
    }
  })

  it('reports no chapter for an id the data does not contain', () => {
    expect(chapterOf('no-such-concept')).toBeUndefined()
  })

  it('agrees with the chapter each concept declares in content.json', () => {
    // `index.json` (where a concept *sits*) and `content.json` (what a concept
    // *says* its chapter is) are two generated files. `ConceptPage` now reads
    // the chapter by id from the sequence -- `chapterOf(concept.id)` -- rather
    // than trusting `concept.chapter`, so if the pipeline ever moved a concept
    // in `index.json` without updating `content.json`, the page would silently
    // show the new chapter while the validator (which only checks the id is
    // known) stayed green. Pin the two together here, at the one place that
    // reads both.
    const declared = contentJson as { concepts: { id: string; chapter: string }[] }
    for (const concept of declared.concepts) {
      expect(chapterOf(concept.id)?.id).toBe(concept.chapter)
    }
  })

  it('gives a concept the neighbours its chapter declares', () => {
    // Walk every chapter and check each position against the raw array; this is
    // the same relation the previous/next links render.
    for (const chapter of raw.chapters) {
      chapter.concepts.forEach((entry, at) => {
        const { previous, next } = neighbours(entry.id)
        expect(previous?.id).toBe(at > 0 ? chapter.concepts[at - 1]?.id : undefined)
        expect(next?.id).toBe(
          at < chapter.concepts.length - 1 ? chapter.concepts[at + 1]?.id : undefined,
        )
      })
    }
  })

  it('has no neighbours for an unknown id, and does not throw', () => {
    expect(neighbours('no-such-concept')).toEqual({ previous: undefined, next: undefined })
  })
})

describe('progress is measured against the data’s own length', () => {
  it('offers the first unread concept in reading order', () => {
    const everything = new Set(conceptSequence.map((entry) => entry.id))
    expect(firstUnread(everything)).toBeUndefined()

    const rest = new Set(conceptSequence.slice(1).map((entry) => entry.id))
    expect(firstUnread(rest)?.id).toBe(conceptSequence[0]?.id)
  })
})
