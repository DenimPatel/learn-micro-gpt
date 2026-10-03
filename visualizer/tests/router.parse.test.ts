import { describe, expect, it } from 'vitest'
import { hrefFor, parseRoute, sameRoute, type Route } from '../src/app/router'

/**
 * The router's pure halves: parsing a hash into a `Route`, and route identity.
 *
 * The navigation guard itself is *not* tested here. It is exercised through the
 * real `useRoute().navigate` against a mocked `window` in `router.test.tsx`,
 * because a test that restates the guard's rule (`!sameRoute(target,
 * parseRoute(hash))`) cannot fail when the real rule changes -- which is exactly
 * how IMP-051 slipped through the first time. What is left here is the parsing
 * contract the guard is built on.
 */

describe('parseRoute', () => {
  it('reads a bare concept link', () => {
    expect(parseRoute('#/learn/softmax')).toEqual({ name: 'concept', conceptId: 'softmax' })
  })

  it('ignores a second in-page anchor and still reads the route', () => {
    // A hash carrying `#sections` is the concept page with a scroll position,
    // not a different page: a browser given `#/learn/softmax#sections` is
    // showing softmax. `parseRoute` drops everything after the second `#` so
    // every reader of the hash agrees on which route is showing, which is the
    // question IMP-051 was about.
    expect(parseRoute('#/learn/softmax#sections')).toEqual({
      name: 'concept',
      conceptId: 'softmax',
    })
    expect(parseRoute('#/run#output').name).toBe('run')
  })

  it('reads the fixed routes', () => {
    expect(parseRoute('#/learn').name).toBe('learn')
    expect(parseRoute('#/explore').name).toBe('explore')
    expect(parseRoute('#/compare').name).toBe('compare')
    expect(parseRoute('#/research').name).toBe('research')
    expect(parseRoute('#/run').name).toBe('run')
    expect(parseRoute('#/about').name).toBe('about')
  })

  it('treats an empty hash as home', () => {
    expect(parseRoute('')).toEqual({ name: 'home' })
    expect(parseRoute('#/')).toEqual({ name: 'home' })
  })

  it('falls back to home for an unknown route instead of throwing', () => {
    // A mistyped deep link should land a reader on the front page, not a blank
    // screen with a stack trace in the console.
    expect(parseRoute('#/nope')).toEqual({ name: 'home' })
  })
})

describe('hrefFor', () => {
  it('round-trips every route through parseRoute', () => {
    // The property that matters: whatever `navigate` writes, `parseRoute` reads
    // back as the same route. If these ever disagree, a click writes a hash the
    // router then interprets as a different page.
    const routes: Route[] = [
      { name: 'home' },
      { name: 'learn' },
      { name: 'concept', conceptId: 'softmax' },
      { name: 'explore' },
      { name: 'compare' },
      { name: 'research' },
      { name: 'run' },
      { name: 'about' },
    ]
    for (const route of routes) {
      expect(parseRoute(hrefFor(route))).toEqual(route)
    }
  })
})

describe('sameRoute', () => {
  it('separates two concepts', () => {
    expect(
      sameRoute(
        { name: 'concept', conceptId: 'softmax' },
        { name: 'concept', conceptId: 'rmsnorm' },
      ),
    ).toBe(false)
  })

  it('agrees on one concept', () => {
    expect(
      sameRoute(
        { name: 'concept', conceptId: 'softmax' },
        { name: 'concept', conceptId: 'softmax' },
      ),
    ).toBe(true)
  })

  it('separates a concept from its chapter', () => {
    expect(sameRoute({ name: 'learn' }, { name: 'concept', conceptId: 'softmax' })).toBe(false)
  })
})
