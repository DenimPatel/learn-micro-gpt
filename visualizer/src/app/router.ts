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
 *   - from a `file://` URL, which is how someone will open a downloaded build,
 *   - and on a fork, at whatever subpath the fork's owner is.
 *
 * History routing needs the server to rewrite `/learn` to `/index.html`. GitHub
 * Pages does not do that for arbitrary paths, so every deep link 404s — and a
 * 404 in a teaching site is a broken lesson, not a cosmetic problem. Hash
 * routing needs no server configuration and behaves identically in all four
 * cases.
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

export type RouteName = 'home' | 'learn' | 'concept' | 'explore' | 'compare' | 'run' | 'about'

export interface Route {
  name: RouteName
  conceptId?: string
}

const ROUTES: { pattern: RegExp; name: RouteName; concept?: (m: RegExpMatchArray) => string | undefined }[] =
  [
    { pattern: /^#\/learn\/([a-z0-9-]+)$/, name: 'concept', concept: (m) => m[1] },
    { pattern: /^#\/learn\/?$/, name: 'learn' },
    { pattern: /^#\/explore\/?$/, name: 'explore' },
    { pattern: /^#\/compare\/?$/, name: 'compare' },
    { pattern: /^#\/run\/?$/, name: 'run' },
    { pattern: /^#\/about\/?$/, name: 'about' },
    { pattern: /^#?\/?$/, name: 'home' },
  ]

export function parseRoute(hash: string): Route {
  for (const route of ROUTES) {
    const match = route.pattern.exec(hash)
    if (match) {
      const conceptId = route.concept?.(match)
      return conceptId ? { name: route.name, conceptId } : { name: route.name }
    }
  }
  return { name: 'home' }
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
    case 'run':
      return '#/run'
    case 'about':
      return '#/about'
  }
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
    const href = hrefFor(next)
    if (window.location.hash === href) return
    window.location.hash = href
  }, [])

  return [route, navigate]
}
