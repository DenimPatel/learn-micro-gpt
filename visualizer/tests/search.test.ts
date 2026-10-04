import { describe, expect, it } from 'vitest'

import contentJson from '../src/data/generated/content.json'
import indexJson from '../src/data/generated/index.json'
import { searchConcepts } from '../src/data/search'
import { chapters } from '../src/data/sequence'
import type { Chapter } from '../src/data/types'

/**
 * The concept matcher.
 *
 * `searchConcepts` is the whole feature: everything in `ConceptSearch.tsx` is
 * markup around an answer this function produced. So the rules a reader will
 * notice -- case, which field matched, order, and what an empty query does --
 * are pinned here, against the *real* generated content, and the e2e case only
 * has to prove the input is wired to it.
 *
 * The expectations are built from a second read of the same JSON the app
 * imports, following `sequence.test.ts`: a test that restated a hit from
 * memory would keep passing after the curriculum changed the summary it names.
 */

/** The generated index, typed, read again here so it is not the app's copy. */
const raw = indexJson as {
  chapters: { id: string; title: string; concepts: { id: string; title: string }[] }[]
}

/** Chapters in reading order, as `searchConcepts` is called. */
const readingOrder = chapters.map((chapter) => chapter.id)

describe('matching is by title and by summary', () => {
  it('finds a concept by its title, whatever the case', () => {
    for (const query of ['softmax', 'SOFTMAX', 'SoftMax', '  softmax  ']) {
      const ids = searchConcepts(chapters, query).map((hit) => hit.id)
      // The reader's term is a title; other concepts may mention it in a
      // summary, so the assertion is "softmax is in the results and is a title
      // match", not "it is the only one" -- the sole-result case belongs to a
      // word the content uses once (`softmax`'s own summary is quoted in
      // other concepts' prose).
      expect(ids, `“${query}” should find softmax`).toContain('softmax')
      const softmax = searchConcepts(chapters, query).find((hit) => hit.id === 'softmax')
      expect(softmax?.titleMatch).toBe(true)
    }
    // ...and the case does not change the answer.
    expect(searchConcepts(chapters, 'SOFTMAX')).toEqual(searchConcepts(chapters, 'softmax'))
  })

  it('finds a concept by a word that is only in its summary', () => {
    // The acceptance criterion the rail alone cannot satisfy: "chain rule" is a
    // phrase the reader has, and it is nowhere in any concept *title*.
    const hits = searchConcepts(chapters, 'chain rule')
    expect(hits.length).toBeGreaterThan(0)
    for (const hit of hits) {
      const concept = contentJson.concepts.find((c) => c.id === hit.id)
      expect(concept, `${hit.id} is a real concept`).toBeDefined()
      expect(concept?.title.toLowerCase()).not.toContain('chain rule')
      expect(concept?.summary.toLowerCase()).toContain('chain rule')
    }
    // ...and the title-only reader would not have found it: `softmax` is a
    // title search that does not contain the summary-only hit.
    expect(searchConcepts(chapters, 'softmax').map((hit) => hit.id)).not.toContain(hits[0]?.id)
  })

  it('flags which field matched, so the rail knows when to show the snippet', () => {
    const byTitle = searchConcepts(chapters, 'softmax')[0]
    expect(byTitle?.titleMatch).toBe(true)
    const bySummary = searchConcepts(chapters, 'chain rule')[0]
    expect(bySummary?.titleMatch).toBe(false)
  })

  it('searches chapter titles too, so a remembered section is reachable', () => {
    const chapter = (indexJson as { chapters: { id: string; title: string }[] }).chapters[0]!
    const hits = searchConcepts(chapters, chapter.title)
    const expected = raw.chapters[0]?.concepts.map((entry) => entry.id) ?? []
    expect(hits.map((hit) => hit.id)).toEqual(expected)
  })

  it('is case-insensitive on the summary as well as the title', () => {
    const lower = searchConcepts(chapters, 'chain rule').map((hit) => hit.id)
    const upper = searchConcepts(chapters, 'CHAIN RULE').map((hit) => hit.id)
    expect(upper).toEqual(lower)
    expect(upper.length).toBeGreaterThan(0)
  })

  it('returns the concept once even when both fields match', () => {
    // "attention" is in several titles and several summaries; a naive
    // concatenation would list such a concept twice.
    const hits = searchConcepts(chapters, 'attention')
    expect(new Set(hits.map((hit) => hit.id)).size).toBe(hits.length)
  })

  it('finds nothing for a term that is not in the content', () => {
    expect(searchConcepts(chapters, 'quantum entanglement')).toEqual([])
  })
})

describe('order is the reading order, not a ranking', () => {
  it('returns hits in index.json order', () => {
    const hits = searchConcepts(chapters, 'the').map((hit) => {
      const chapter = chapters.find((c) => c.concepts.some((e) => e.id === hit.id))
      return [chapters.indexOf(chapter as Chapter), hit.id] as const
    })
    expect(hits).toEqual([...hits].sort((a, b) => a[0] - b[0] || 0))
  })

  it('lists the chapter title matches in that chapter’s reading order', () => {
    const last = (indexJson as { chapters: { id: string; title: string }[] }).chapters.at(-1)!
    const expected = raw.chapters.at(-1)!.concepts.map((entry) => entry.id)
    expect(searchConcepts(chapters, last.title).map((hit) => hit.id)).toEqual(expected)
    expect(readingOrder.at(-1)).toBe(last.id)
  })
})

describe('the states around a real query', () => {
  it('treats an empty query as "not searching", not as "matches everything"', () => {
    // The rail renders the unfiltered list for this case; if the matcher
    // returned all 18 the component could not tell the two apart.
    for (const query of ['', ' ', '\n', '   ']) {
      expect(searchConcepts(chapters, query), `“${query}”`).toEqual([])
    }
  })

  it('carries the summary through, so a summary-only hit can say why', () => {
    const hit = searchConcepts(chapters, 'chain rule')[0]
    const concept = contentJson.concepts.find((c) => c.id === hit?.id)
    expect(hit?.summary).toBe(concept?.summary)
    expect(hit?.summary).toContain('chain rule')
  })

  it('searches every concept the rail lists', () => {
    // A query per listed concept id *and* its title: the slug is not the title
    // (`autograd-value` is "The Value class"), so this pins both halves --
    // every concept is reachable by the title the rail shows, and the summary
    // is searched too.
    const listed = raw.chapters.flatMap((chapter) => chapter.concepts)
    expect(listed).toHaveLength(18)
    const unreachable = listed.filter(
      (entry) => !searchConcepts(chapters, entry.title).some((hit) => hit.id === entry.id),
    )
    expect(unreachable.map((entry) => entry.id)).toEqual([])
    expect(searchConcepts(chapters, 'chain rule').length).toBeGreaterThan(0)
  })
})
