import { act, createElement, useEffect, type ComponentType } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createMockWindow } from './support/mock-window'
import { shouldNavigate, useRoute, type Route } from '../src/app/router'

/**
 * `navigate()`'s guard, tested through the real hook.
 *
 * ## The bug (IMP-051), and why this file exists
 *
 * `navigate()` used to bail on `if (window.location.hash === href) return`. The
 * failure is not two routes with the same hash -- it is that the *string* is a
 * different question from "is this route already showing":
 *
 *   - the app writes the hash without going through `navigate` (a concept is
 *     reached by a plain `<a href="#/learn/softmax">`), so the hash can name a
 *     route the user is not looking at; and
 *   - a browser that has been given an in-page anchor carries
 *     `#/learn/softmax#sections`, which is not a string `hrefFor` ever writes.
 *
 * An earlier version of `tests/router.test.ts` tested a *copy* of the decision
 * (`wouldWriteHash(target, hash) = !sameRoute(target, parseRoute(hash))`)
 * instead of the decision itself. It passed with the guard reverted, because a
 * restated rule cannot fail when the real rule changes. These cases call the
 * real `useRoute().navigate` against a mocked `window`, so they can.
 *
 * ## Why a mocked `window` rather than jsdom
 *
 * `vitest.config.ts` is `environment: 'node'` on purpose, and adding jsdom for
 * one guard is a large dependency for a small claim. See
 * `tests/support/mock-window.ts`: it provides a *live* `location.hash` whose
 * setter fires the `hashchange` listeners, which is what makes "the hash changed,
 * so the app re-rendered" mean something here.
 *
 * ## Why `useRoute` and not a bare `navigate`
 *
 * The mocked `window` has a *live* `location.hash` whose setter fires the
 * `hashchange` listeners, so a write becomes a real re-render. The last block
 * drives the hook both ways: a bare `<a href>` assignment (the browser's own
 * path, which never calls `navigate`) and `navigate` itself, and asserts the
 * hook follows both. Rendering the whole `App` was tried and dropped: it needs
 * a full DOM emulation for the concept widgets, and none of that is IMP-051.
 */

const WINDOW_KEY = '__visualizerMockWindow__'

// `react-dom/client` only treats `act()` as the test environment's when this
// flag is set; without it every `act` below logs a warning. Vitest's node
// environment does not set it for us.
;(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true

afterEach(() => {
  delete (globalThis as Record<string, unknown>)[WINDOW_KEY]
  vi.unstubAllGlobals()
})

function installMockWindow(hash = ''): ReturnType<typeof createMockWindow> {
  const mock = createMockWindow(hash)
  vi.stubGlobal('window', mock.window)
  // The container's `ownerDocument` is the stub, so React and the element it is
  // handed agree about which document they are in.
  vi.stubGlobal('document', (mock.container as unknown as { ownerDocument: unknown }).ownerDocument)
  vi.stubGlobal('localStorage', mock.localStorage)
  vi.stubGlobal(
    'matchMedia',
    () => ({ matches: false, addEventListener() {}, removeEventListener() {} }) as unknown,
  )
  ;(globalThis as Record<string, unknown>)[WINDOW_KEY] = mock
  return mock
}

/** The `useRoute` pair, isolated from the app: the guard's own decision. */
function Routes({ onNavigate }: { onNavigate?: (navigate: (route: Route) => void) => void }) {
  const [, navigate] = useRoute()
  useEffect(() => {
    onNavigate?.(navigate)
  }, [navigate, onNavigate])
  return null
}

describe('navigate guard (the real hook, a mocked window)', () => {
  let root: Root | undefined

  afterEach(() => {
    act(() => root?.unmount())
    root = undefined
  })

  /**
   * Mount `Routes` on the given hash and return the `navigate` the real hook
   * handed out, plus the mock's hash reader.
   */
  function mountRoutes(hash: string) {
    const mock = installMockWindow(hash)
    let navigate: ((route: Route) => void) | undefined
    act(() => {
      root = createRoot(mock.container)
      root.render(
        createElement(Routes as ComponentType<Record<string, unknown>>, {
          onNavigate: (fn: (route: Route) => void) => {
            navigate = fn
          },
        }),
      )
    })
    if (!navigate) throw new Error('useRoute() never handed out a navigate')
    return { navigate, mock }
  }

  it('writes the hash when the route differs but the target href is not what is on screen', () => {
    // The plain-anchor case: a concept was reached by `<a href="#/learn/softmax">`,
    // so the hash names softmax and the caller now asks for rmsnorm. Both rules
    // agree here; it is the baseline the next two cases refine.
    const { navigate, mock } = mountRoutes('#/learn/softmax')
    act(() => navigate({ name: 'concept', conceptId: 'rmsnorm' }))
    expect(mock.readHash()).toBe('#/learn/rmsnorm')
  })

  it('compares routes, not hash strings', () => {
    // `hrefFor` and `parseRoute` are inverses on well-formed hashes, so on a
    // well-formed hash the two rules cannot disagree -- which is exactly why the
    // original test could never fail. They disagree on a hash the router did not
    // write: `#/learn/softmax#foo` is a second `#`, and `shouldNavigate` must
    // still say "not yet there" for a *different* concept whose href is not that
    // string. Asserting the predicate here is what separates the rules; the two
    // `it` blocks around it fail on the old guard through `navigate` itself.
    expect(shouldNavigate({ name: 'concept', conceptId: 'rmsnorm' }, '#/learn/softmax#foo')).toBe(
      true,
    )
    expect(shouldNavigate({ name: 'concept', conceptId: 'softmax' }, '#/learn/softmax#foo')).toBe(
      false,
    )
  })

  it('moves when the target href differs from the two-part hash on screen', () => {
    // The concept was reached by `<a href="#/learn/softmax">` with an in-page
    // anchor, so the raw hash is `#/learn/softmax#foo`; the caller now asks for
    // rmsnorm. The old string rule happened to agree here, so this is the
    // baseline the anchor no-op case refines -- but it is a real navigation and
    // must not be skipped.
    const { navigate, mock } = mountRoutes('#/learn/softmax#foo')
    act(() => navigate({ name: 'concept', conceptId: 'rmsnorm' }))
    expect(mock.readHash()).toBe('#/learn/rmsnorm')
  })

  it('no-ops when only an in-page anchor differs', () => {
    // `#/learn/softmax#foo` is the concept page with a fragment on the end.
    // The old guard compared `'#/learn/softmax#foo'` against
    // `hrefFor(softmax)` = `'#/learn/softmax'`, which differs -- so it wrote a
    // redundant hash, and its `hashchange` is what scrolls the reader to the top.
    // The routes are the same, so the honest answer is "do nothing".
    const { navigate, mock } = mountRoutes('#/learn/softmax#foo')
    const before = mock.hashChanges()
    expect(shouldNavigate({ name: 'concept', conceptId: 'softmax' }, '#/learn/softmax#foo')).toBe(
      false,
    )
    act(() => navigate({ name: 'concept', conceptId: 'softmax' }))
    expect(mock.readHash()).toBe('#/learn/softmax#foo')
    expect(mock.hashChanges()).toBe(before)
  })

  it('writes nothing when the hash already is the target route', () => {
    const { navigate, mock } = mountRoutes('#/learn')
    const before = mock.hashChanges()
    act(() => navigate({ name: 'learn' }))
    expect(mock.readHash()).toBe('#/learn')
    expect(mock.hashChanges()).toBe(before)
  })
})

/**
 * The hook's other half: after `navigate` writes the hash, the mock fires
 * `hashchange`, and the hook must read the new route back. This is what makes
 * the guard's write a *navigation* rather than a string assignment, and it is
 * also the path the app uses when a bare `<a href>` changes the hash.
 */
describe('useRoute reads the hash back', () => {
  let root: Root | undefined

  afterEach(() => {
    act(() => root?.unmount())
    root = undefined
  })

  function RouteProbe({ onRoute }: { onRoute: (route: ReturnType<typeof useRoute>[0]) => void }) {
    const [route, navigate] = useRoute()
    useEffect(() => {
      onRoute(route)
    }, [route, onRoute])
    useEffect(() => {
      ;(globalThis as Record<string, unknown>).__navigate = navigate
    }, [navigate])
    return null
  }

  it('tracks a hash written by an anchor and a hash written by navigate', () => {
    const mock = installMockWindow('')
    const seen: string[] = []
    const onRoute = (route: ReturnType<typeof useRoute>[0]) => {
      seen.push(route.name + (route.conceptId ? `/${route.conceptId}` : ''))
    }
    act(() => {
      root = createRoot(mock.container)
      root.render(createElement(RouteProbe as ComponentType<Record<string, unknown>>, { onRoute }))
    })

    // A plain `<a href="#/learn/softmax">`: the browser writes the hash and the
    // hook, which never saw a `navigate`, must still follow it.
    act(() => {
      mock.window.location.hash = '#/learn/softmax'
    })
    expect(seen.at(-1)).toBe('concept/softmax')

    // And now via `navigate`, the app's own path.
    const navigate = (globalThis as Record<string, unknown>).__navigate as (route: {
      name: 'concept'
      conceptId: string
    }) => void
    act(() => navigate({ name: 'concept', conceptId: 'rmsnorm' }))
    expect(mock.readHash()).toBe('#/learn/rmsnorm')
    expect(seen.at(-1)).toBe('concept/rmsnorm')
  })
})
