import { describe, expect, it } from 'vitest'
import { ROUTES, hrefFor, parseRoute, type Route, type RouteName } from './router'

/**
 * The router's round trip, from *beside* the router.
 *
 * ## Why this file is colocated rather than in `tests/`
 *
 * It is the IMP-074 proof that the colocated family runs at all. `vitest.config.ts`
 * included only `tests/**`, so a `src/**\/*.test.ts` file was silently never
 * executed -- the suite stayed green while having run nothing in it. A test in
 * `tests/` could not catch that, because a test in `tests/` was already running.
 * This file exists partly for its own sake and partly to fail the build's
 * `test:unit` step if the include glob ever narrows back: delete the `src/**`
 * patterns and this file vanishes from the run without an error, which is the
 * bug, so the fix has to be a file that only runs when the glob is right.
 *
 * ## Why it iterates `ROUTES` and does not restate the list
 *
 * The pre-existing round-trip test in `tests/router.parse.test.ts` lists the
 * eight routes by hand. That list is a copy of the router's own table, so adding
 * a route to the router does not fail that test -- it just stops covering the
 * new route. This test reads `ROUTES` itself, so a ninth route is covered the
 * moment it is declared.
 */

/** A concrete example of each route name, for the names that take no id. */
const SAMPLE_HASH: Record<RouteName, string> = {
  home: '#/',
  learn: '#/learn',
  concept: '#/learn/softmax',
  explore: '#/explore',
  compare: '#/compare',
  research: '#/research',
  run: '#/run',
  about: '#/about',
}

describe('the route table', () => {
  it('is the whole set the samples cover, with nothing left over', () => {
    // Guards the test itself: if `RouteName` grows a member, this object literal
    // stops type-checking (it is exhaustive by type), and if `ROUTES` gains an
    // entry the comparison below fails. Either way the gap is loud.
    const declared = new Set(ROUTES.map((route) => route.name))
    const sampled = new Set(Object.keys(SAMPLE_HASH) as RouteName[])
    expect(declared).toEqual(sampled)
  })

  it('matches a route pattern for every sample, so no sample is a dead one', () => {
    // A sample hash that no pattern matches would make the round trip below
    // vacuous for that route: `parseRoute` would return `home` and any assertion
    // about it would be about the wrong thing.
    for (const route of ROUTES) {
      const sample = parseRoute(SAMPLE_HASH[route.name])
      expect(sample.name, `no pattern matched the sample for "${route.name}"`).toBe(route.name)
    }
  })
})

describe('hrefFor/parseRoute round trip', () => {
  it('reads back every route in ROUTES as the same route', () => {
    // The property that matters: whatever `hrefFor` writes, `parseRoute` reads
    // back. If the two disagree, a click writes a hash the router then interprets
    // as a different page -- a dead link that only exists inside the app.
    //
    // `concept` is the one route that carries data, so it is round-tripped with a
    // real id; the others take no argument. Asking `hrefFor` for a concept with
    // no id writes the literal `#/learn/undefined`, which is why that case is not
    // in the loop -- see the dedicated concept test below for the id handling.
    for (const route of ROUTES) {
      const written =
        route.name === 'concept'
          ? hrefFor({ name: 'concept', conceptId: 'softmax' })
          : hrefFor({ name: route.name })
      const expected: Route =
        route.name === 'concept' ? { name: 'concept', conceptId: 'softmax' } : { name: route.name }
      expect(
        parseRoute(written),
        `hrefFor(${route.name}) => "${written}" did not round-trip`,
      ).toEqual(expected)
    }
  })

  it('reads back a concept, id and all, for every concept id', () => {
    // `concept` is the one route carrying data, so it is the one where a round
    // trip can lose information rather than merely name the wrong page.
    const ids = ['softmax', 'multi-head-attention', 'rmsnorm', 'a', 'a-1', 'x9-y8']
    for (const conceptId of ids) {
      const route: Route = { name: 'concept', conceptId }
      expect(parseRoute(hrefFor(route))).toEqual(route)
    }
  })

  it('writes a hash the route table actually matches, not just one that parses', () => {
    // Stronger than the round trip: `parseRoute` falls back to `home` for an
    // unknown hash, so a `home` round trip would pass even if `hrefFor` wrote
    // garbage. Requiring a non-fallback match for every non-home route closes
    // that hole.
    for (const route of ROUTES) {
      if (route.name === 'home') continue
      const written =
        route.name === 'concept'
          ? hrefFor({ name: 'concept', conceptId: 'softmax' })
          : hrefFor({ name: route.name })
      expect(route.pattern.test(written), `"${written}" is not matched by its own pattern`).toBe(
        true,
      )
    }
  })
})
