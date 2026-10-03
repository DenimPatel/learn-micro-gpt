/**
 * The router.
 *
 * ## Why hash routing, and not the history API
 *
 * The site is a single static bundle on GitHub Pages, and the whole point of the
 * `?raw` imports in `src/data/sources.ts` is that nothing is fetched at
 * runtime. The one URL the app does own is the route, and it has to work:
 *
 *   - at the root of a domain,
 *   - under a subpath (`/learn-micro-gpt/`) on Pages,
 *   - and on a fork, at whatever subpath the fork's owner is.
 *
 * History routing needs the server to rewrite `/learn` to `/index.html`. GitHub
 * Pages does not do that for arbitrary paths, so every deep link 404s — and a
 * 404 in a teaching site is a broken lesson, not a cosmetic problem. Hash
 * routing needs no server configuration and behaves identically in all three
 * cases.
 *
 * The scheme *would* also be correct from a `file://` URL, but the site does not
 * run there: `file://` is an opaque origin, so the browser blocks the
 * `type="module"` script (and the CSS) with a CORS error before the router ever
 * runs. That is browser security, not a routing decision, and no hash scheme can
 * work around it. `index.html` detects the case and shows a "serve this over
 * HTTP instead" message rather than a blank page; the `file://` claim used to
 * live in this comment and was simply wrong.
 *
 * The cost is uglier URLs (`#/learn/softmax` rather than `/learn/softmax`).
 * Given the choice between prettier URLs that 404 on GitHub Pages and working
 * ones, this takes the working ones. A `dist/404.html` copy is also written at
 * build time, so switching to history routing later is a two-line change if the
 * repo ever moves to a host with rewrite support.
 *
 * 40 lines instead of a routing dependency, for a site with five routes.
 */

import { useCallback, useEffect, useState } from 'react'

export type RouteName =
  'home' | 'learn' | 'concept' | 'explore' | 'compare' | 'research' | 'run' | 'about'

export interface Route {
  name: RouteName
  conceptId?: string
}

/**
 * Every route the app recognises, in match order.
 *
 * Exported so a test can iterate the real table instead of restating it. A
 * hand-written copy in a test is the failure mode this export exists to stop: if
 * a route is added here and the copy is not updated, the round-trip test keeps
 * passing while covering less and less -- and a route no test ever round-trips
 * is exactly the one that ships with a `hrefFor`/`parseRoute` disagreement.
 * The `pattern` is a matcher for a hash, not the hash `hrefFor` writes; that
 * pair is what the round-trip test proves agree.
 */
export const ROUTES: {
  pattern: RegExp
  name: RouteName
  concept?: (m: RegExpMatchArray) => string | undefined
}[] = [
  { pattern: /^#\/learn\/([a-z0-9-]+)$/, name: 'concept', concept: (m) => m[1] },
  { pattern: /^#\/learn\/?$/, name: 'learn' },
  { pattern: /^#\/explore\/?$/, name: 'explore' },
  { pattern: /^#\/compare\/?$/, name: 'compare' },
  { pattern: /^#\/research\/?$/, name: 'research' },
  { pattern: /^#\/run\/?$/, name: 'run' },
  { pattern: /^#\/about\/?$/, name: 'about' },
  { pattern: /^#?\/?$/, name: 'home' },
]

/**
 * The route named by a hash.
 *
 * A `#` in the raw hash is not always the route's own: a browser that has
 * followed an in-page anchor carries `#/learn/softmax#sections`, and a fragment
 * on a fixed route (`#/run#output`) does the same. The route is the part before
 * that second `#`; anything after it is a scroll position, not a page. Dropping
 * it here -- rather than in the guard -- means every reader of the hash agrees
 * on which route is showing, which is the question IMP-051 was really about.
 *
 * An empty or route-less hash (`#sections` alone, or `''`) still falls through
 * to whichever route pattern matches what is left, which is `home`.
 */
export function parseRoute(hash: string): Route {
  const anchor = hash.indexOf('#', 1)
  const path = anchor === -1 ? hash : hash.slice(0, anchor)
  for (const route of ROUTES) {
    const match = route.pattern.exec(path)
    if (match) {
      const conceptId = route.concept?.(match)
      return conceptId ? { name: route.name, conceptId } : { name: route.name }
    }
  }
  return { name: 'home' }
}

/** Route identity: the name and, for a concept, which concept. */
export function sameRoute(a: Route, b: Route): boolean {
  return a.name === b.name && a.conceptId === b.conceptId
}

export function hrefFor(route: Route): string {
  switch (route.name) {
    case 'home':
      return '#/'
    case 'learn':
      return '#/learn'
    case 'concept':
      return `#/learn/${route.conceptId}`
    case 'explore':
      return '#/explore'
    case 'compare':
      return '#/compare'
    case 'research':
      return '#/research'
    case 'run':
      return '#/run'
    case 'about':
      return '#/about'
  }
}

/**
 * Does `navigate(next)` need to write the hash, given where the browser is now?
 *
 * The guard is the *parsed route*, not the hash string, and that is the whole of
 * this function:
 *
 *   - The app writes the hash itself in some paths, so the two disagree more
 *     often than they look like they should. A concept is reached by a plain
 *     `<a href="#/learn/softmax">`, which changes the hash without going through
 *     `navigate` at all, and a browser that has been given an in-page anchor
 *     carries `#/learn/softmax#sections` -- a second `#` this router does not
 *     parse as a route.
 *   - So `window.location.hash === hrefFor(next)` is a different question from
 *     "is the target route already showing". A navigate whose target *differs*
 *     must move even when the raw hash already equals the target's href, and a
 *     navigate to the route already showing must be a no-op even when the hashes
 *     differ by an in-page anchor.
 *
 * Pulled out of the hook so it can be tested directly. It used to be a closure
 * inside `useRoute`, which meant the only way to test the decision was to
 * re-state it in the test file -- and a restated decision cannot fail when the
 * real one regresses.
 */
export function shouldNavigate(next: Route, currentHash: string): boolean {
  return !sameRoute(next, parseRoute(currentHash))
}

/**
 * The current route, and a way to change it.
 *
 * `hashchange` rather than a `popstate` shim: it is the event the platform
 * actually fires for hash changes, it fires for the back button, and it does not
 * need a history entry pushed per navigation.
 */
export function useRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(() =>
    parseRoute(typeof window === 'undefined' ? '' : window.location.hash),
  )

  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  const navigate = useCallback((next: Route) => {
    if (!shouldNavigate(next, window.location.hash)) return
    window.location.hash = hrefFor(next)
  }, [])

  return [route, navigate]
}
