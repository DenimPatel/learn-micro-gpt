import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { PageErrorBoundary } from '../src/components/ErrorBoundary'
import type { Route } from '../src/app/router'

/**
 * The page error boundary.
 *
 * ## Why this drives the boundary's own hooks instead of throwing under a renderer
 *
 * React error boundaries are a *client* feature. Neither `renderToStaticMarkup`
 * nor the streaming server renderers (`prerenderToNodeStream`,
 * `renderToPipeableStream`) consult a boundary: a throw during server render
 * fails the render, which was tried here and reproduced (`Error: widget index 7
 * is missing` propagated straight out of `renderToStaticMarkup`). React says as
 * much in its docs -- boundaries do not run on the server.
 *
 * `vitest.config.ts` is `environment: 'node'` on purpose -- "anything needing a
 * DOM is a Playwright test" -- and this repository has no jsdom/happy-dom and no
 * `react-test-renderer`, so the two honest options are:
 *
 *   1. an e2e case with a deliberate `throw` behind a query flag, which would
 *      ship a crash switch in the production bundle to cover a branch the unit
 *      suite can reach directly, and would land in a suite with two specs known
 *      to be flaky on a clean tree (`recon-runbook.md` §3); or
 *   2. call the boundary's error-render path directly.
 *
 * The item names option 2 as acceptable, and it is the smaller change: the
 * contract that matters is "given an error in state, `render` returns the
 * page-shaped fallback", plus "a caught error is logged locally and goes
 * nowhere else". Both are reachable without a renderer that runs boundaries.
 *
 * So: `getDerivedStateFromError` is called exactly as React would call it, its
 * result is moved into `state`, and the resulting node is rendered. That is the
 * real code under test; only React's dispatch to it is stubbed, and React's
 * dispatch is the part that is not in doubt. `getDerivedStateFromError` is a
 * static on the class, so calling it is not a fiction.
 *
 * What this does *not* prove, and how it is covered instead: that App wires the
 * boundary around the routed content only is visible in `App.tsx` (inside
 * `<main>`, wrapping `<Router>`, outside the header/sidebar/footer), and that
 * the two controls work needs a browser (`#/learn` is a hash link the router
 * owns; the reload button calls `window.location.reload`).
 */

function boundaryFor(route: Route): PageErrorBoundary {
  return new PageErrorBoundary({ route, children: null })
}

/** React's own step, reproduced: derive the state, then render against it. */
function renderError(route: Route, error: Error): string {
  const boundary = boundaryFor(route)
  boundary.state = PageErrorBoundary.getDerivedStateFromError(error)
  return renderToStaticMarkup(boundary.render())
}

describe('PageErrorBoundary', () => {
  it('renders the page-shaped error state when a child throws', () => {
    const html = renderError(
      { name: 'concept', conceptId: 'softmax' },
      new Error('widget index 7 is missing'),
    )

    // A page, not a blank screen and not the chrome.
    expect(html).toContain('This page could not be drawn')
    // Name the route, so the reader can say which page failed.
    expect(html).toContain('#/learn/softmax')
    // The error itself, so the message survives to the screen.
    expect(html).toContain('widget index 7 is missing')
    // The way out, and the way to try again.
    expect(html).toContain('href="#/learn"')
    expect(html).toContain('Back to all concepts')
    expect(html).toContain('Reload this page')
    // Announced: a reader on a screen reader is told the page broke.
    expect(html).toContain('role="alert"')
    // Reporting nothing to any network: the fallback carries no URL, so it
    // cannot be a report link, and the boundary has no fetch to make.
    expect(html).not.toContain('http://')
    expect(html).not.toContain('https://')
  })

  it('names the route by its path, so the reader can say what failed', () => {
    expect(renderError({ name: 'concept', conceptId: 'softmax' }, new Error('x'))).toContain(
      '#/learn/softmax',
    )
    expect(renderError({ name: 'research' }, new Error('x'))).toContain('#/research')
  })

  it('reports the error to the developer, locally and only locally', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    try {
      const boundary = boundaryFor({ name: 'learn' })
      const error = new Error('bang')
      boundary.componentDidCatch(error, { componentStack: '\n  at Boom' })

      // The developer still sees the stack: the boundary must not swallow it.
      expect(consoleError).toHaveBeenCalledTimes(1)
      expect(consoleError.mock.calls[0]).toContain(error)
      // And nothing left the machine.
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      consoleError.mockRestore()
      fetchSpy.mockRestore()
    }
  })

  it('renders its children untouched when nothing throws', () => {
    const html = renderToStaticMarkup(
      <PageErrorBoundary route={{ name: 'learn' }}>
        <p className="marker">a page that drew</p>
      </PageErrorBoundary>,
    )
    expect(html).toContain('a page that drew')
    expect(html).not.toContain('This page could not be drawn')
  })
})
