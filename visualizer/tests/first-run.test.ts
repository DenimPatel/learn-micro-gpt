import { describe, expect, it } from 'vitest'

import indexJson from '../src/data/generated/index.json'
import { conceptSequence, conceptCount } from '../src/data/sequence'
import { startPrompt } from '../src/app/firstRun'

/**
 * The home page's "start here" decision.
 *
 * ## What is being pinned
 *
 * Three things, and each of them is something the inline-JSX version could get
 * wrong without any test noticing:
 *
 * 1. the empty set means "begin", and it points at the *first* concept of the
 *    curriculum -- read from a second, independent parse of `index.json` so the
 *    test is not agreeing with the code's own copy of the order;
 * 2. *any* seen concept (not "the first concept is seen", and not "the first
 *    chapter") means the reader has started, so the state is `resume`;
 * 3. `resume` continues at the first *unread* concept, in reading order.
 *
 * The fourth case -- every concept seen -- is the one that would otherwise
 * render a link with an empty id, so it is asserted to have no target at all.
 */

/** The order as `index.json` writes it, read again here rather than imported. */
const rawOrder: string[] = (
  indexJson as { chapters: { concepts: { id: string }[] }[] }
).chapters.flatMap((chapter) => chapter.concepts.map((entry) => entry.id))

describe('startPrompt: deciding what the home page offers', () => {
  it('a fresh reader is invited to begin, at the data chapter’s first concept', () => {
    const prompt = startPrompt(new Set())
    expect(prompt.state).toBe('begin')
    // The curriculum's first concept, per the data, is "data-loading" -- the
    // `data` chapter, order 1. Asserted against the second parse above rather
    // than the literal, so a reordered `index.json` moves this expectation too.
    expect(prompt.target?.id).toBe(rawOrder[0])
    expect(prompt.target?.id).toBe('data-loading')
  })

  it('one seen concept is enough to have started -- the state is resume, not begin', () => {
    // Deliberately a concept from the *middle* of the curriculum: "I have read
    // the first concept" must not be the test for having started.
    const prompt = startPrompt(new Set(['attention']))
    expect(prompt.state).toBe('resume')
  })

  it('resume continues at the first unread concept, in reading order', () => {
    const first = rawOrder[0] as string
    const prompt = startPrompt(new Set([first]))
    expect(prompt.state).toBe('resume')
    // The first concept is read, so the next one is the target -- not the first.
    expect(prompt.target?.id).toBe(rawOrder[1])
    expect(prompt.target?.id).not.toBe(first)
  })

  it('resume skips a gap and lands on the earliest unread concept, not the latest read', () => {
    const prompt = startPrompt(new Set([rawOrder[3] as string, rawOrder[5] as string]))
    expect(prompt.state).toBe('resume')
    expect(prompt.target?.id).toBe(rawOrder[0])
  })

  it('a fully-read curriculum has nothing to continue to', () => {
    const prompt = startPrompt(new Set(rawOrder))
    expect(rawOrder).toHaveLength(conceptCount)
    expect(prompt.state).toBe('done')
    expect(prompt.target).toBeUndefined()
  })

  it('an id that is not in the curriculum still counts as having started', () => {
    // Progress is a set of ids, and a stale export -- a concept that has since
    // been renamed -- must not resurrect the first-run invitation.
    const prompt = startPrompt(new Set(['not-a-concept']))
    expect(prompt.state).toBe('resume')
    // ...and it continues at the real first concept, because every real concept
    // is still unread.
    expect(prompt.target?.id).toBe(conceptSequence[0]?.id)
  })
})
