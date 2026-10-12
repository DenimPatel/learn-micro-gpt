import { describe, expect, it } from 'vitest'
import { parseCite, withinAnchor } from '../src/content/cite'

/**
 * The citation grammar.
 *
 * `cite:` is understood by two runtimes that cannot share code -- this one, and
 * the Python validator in `tools/validate_concepts.py` -- so the grammar has to
 * be small enough to hold in the head. These tests are the statement of what it
 * is; the Python tests state the same thing against the same examples, which is
 * what keeps the two copies from drifting.
 *
 * The interesting cases are the rejections. A form the parser accepts but the
 * validator rejects (or the reverse) is a citation that either fails the build
 * for no reason or renders as a control that scrolls nowhere, and both are worse
 * than not having the feature.
 */
describe('parseCite', () => {
  it('parses a single line', () => {
    expect(parseCite('cite:11')).toEqual({ start: 11, end: 11 })
  })

  it('parses an inclusive range', () => {
    // The real corpus example: the seed line and the shuffle it enables.
    expect(parseCite('cite:11-12')).toEqual({ start: 11, end: 12 })
  })

  it('parses a one-line range spelled as one', () => {
    expect(parseCite('cite:7-7')).toEqual({ start: 7, end: 7 })
  })

  it('accepts line one, the first line of a file', () => {
    // Off-by-one bait: a naive `start > 1` guard would reject the first line.
    expect(parseCite('cite:1')).toEqual({ start: 1, end: 1 })
  })

  it('rejects an inverted range rather than normalising it', () => {
    // Silently swapping the ends would highlight something the author did not
    // write; the prose number and the highlighted number must be the same number.
    expect(parseCite('cite:20-11')).toBeUndefined()
  })

  it('rejects line zero and other non-positive lines', () => {
    // 1-based, so 0 does not exist. `-1` would parse as an inverted range if the
    // digits-only guard were loosened.
    expect(parseCite('cite:0')).toBeUndefined()
    expect(parseCite('cite:0-3')).toBeUndefined()
  })

  it('rejects a bare scheme', () => {
    expect(parseCite('cite:')).toBeUndefined()
  })

  it('rejects whitespace around the number', () => {
    // `cite: 11` is a typo, and a lenient parser is one that stops catching
    // pasted-from-a-doc formats.
    expect(parseCite('cite: 11')).toBeUndefined()
    expect(parseCite('cite:11 ')).toBeUndefined()
  })

  it('rejects a non-numeric span', () => {
    expect(parseCite('cite:eleven')).toBeUndefined()
    expect(parseCite('cite:11-')).toBeUndefined()
    expect(parseCite('cite:-11')).toBeUndefined()
    expect(parseCite('cite:1.5')).toBeUndefined()
  })

  it('rejects a three-part range', () => {
    expect(parseCite('cite:1-2-3')).toBeUndefined()
  })

  it('leaves ordinary links alone', () => {
    // The whole feature hangs on this: every other href on the site must keep
    // the behaviour it has today.
    expect(parseCite('https://example.com/')).toBeUndefined()
    expect(parseCite('#/learn/softmax')).toBeUndefined()
    expect(parseCite('cite')).toBeUndefined()
    expect(parseCite('../concepts/softmax.md')).toBeUndefined()
  })

  it('leaves a missing href alone', () => {
    // `marked` can hand through a link token with no href at all.
    expect(parseCite(undefined)).toBeUndefined()
    expect(parseCite(null)).toBeUndefined()
  })
})

/**
 * The containment check the validator mirrors.
 *
 * Anchors are whole-concept ranges, so this can only say "inside the window the
 * panel crops to" -- which is exactly what the citation needs, and no more. It
 * is deliberately not a claim that the line is the *right* line: no build can
 * know that. See the module comment.
 */
describe('withinAnchor', () => {
  const anchor = { start: 13, end: 20 }

  it('accepts a range wholly inside', () => {
    expect(withinAnchor({ start: 14, end: 16 }, anchor)).toBe(true)
  })

  it('accepts the window itself, at both edges', () => {
    expect(withinAnchor({ start: 13, end: 20 }, anchor)).toBe(true)
    expect(withinAnchor({ start: 13, end: 13 }, anchor)).toBe(true)
    expect(withinAnchor({ start: 20, end: 20 }, anchor)).toBe(true)
  })

  it('rejects a range that starts before the window', () => {
    expect(withinAnchor({ start: 12, end: 14 }, anchor)).toBe(false)
  })

  it('rejects a range that ends after the window', () => {
    expect(withinAnchor({ start: 19, end: 21 }, anchor)).toBe(false)
  })

  it('rejects a range that misses the window entirely', () => {
    expect(withinAnchor({ start: 100, end: 101 }, anchor)).toBe(false)
  })

  it('rejects everything when there is no anchor at all', () => {
    // A concept with no resolved range in this language has no window to be
    // inside, so no citation can resolve against it.
    expect(withinAnchor({ start: 1, end: 1 }, undefined)).toBe(false)
  })
})
