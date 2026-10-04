/**
 * The home page's "start here" decision, as a pure function.
 *
 * ## Why this is a function and not a conditional in the JSX
 *
 * "Is this a first-time reader?" is the one question the home page asks that a
 * learner can feel when it is answered wrongly: a returning reader who is
 * shouted at to "start reading" is being told the site forgot them, and a
 * first-timer who is shown a "continue" button has no idea where they are
 * continuing *from*. The question is also the one the acceptance criteria talk
 * about ("de-emphasises once at least one concept is seen"), so it is worth
 * being able to answer it in a test -- the version inline in the JSX could only
 * be exercised by launching a browser, which is why it would eventually drift.
 *
 * Nothing here is presentation: the state names what the page should say, and
 * the markup and the stylesheet decide what that looks like.
 */

import type { ChapterEntry } from '../data/types'
import { conceptSequence, firstUnread } from '../data/sequence'

/**
 * What a reader's progress says the home page should offer.
 *
 * - `begin` -- nothing has been opened yet; this is the first time.
 * - `resume` -- some concepts are open; the reader has started, so the page
 *   must not invite them to "start" again.
 * - `done` -- every concept in the curriculum is marked seen, so there is no
 *   unread concept left to continue to.
 */
export type StartState = 'begin' | 'resume' | 'done'

/** The "start here" offering the home page renders. */
export type StartPrompt = {
  /** Which of the three states produced this prompt. */
  state: StartState
  /**
   * The concept the link points at.
   *
   * `begin` points at the first concept in the curriculum; `resume` at the
   * first unread one, which is `sequence.ts`'s `firstUnread` and therefore
   * `index.json`'s order rather than a literal. `done` has nothing to point at,
   * so this is `undefined` -- and the type says so, which is what stops the
   * markup from rendering a "start again" link out of an empty id.
   */
  target: ChapterEntry | undefined
}

/**
 * Decide what the home page should offer, given the reader's progress.
 *
 * The rule, in one line: *any* seen concept means the reader has started.
 * Deliberately not "the first concept is seen" -- a course where someone has
 * read `softmax` is not a course they have never opened, and asking them to
 * start from the top would be wrong. The seen-set is the only input, so this is
 * a total function of it.
 */
export function startPrompt(seen: ReadonlySet<string>): StartPrompt {
  const target = seen.size > 0 ? firstUnread(seen) : conceptSequence[0]
  if (!target) return { state: 'done', target: undefined }
  return { state: seen.size > 0 ? 'resume' : 'begin', target }
}
