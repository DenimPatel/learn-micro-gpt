import { expect, test, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { PREFIX } from './subpath'

/**
 * The `404.html` fallback suite.
 *
 * The build emits `dist/404.html` as a byte-for-byte copy of `index.html`
 * (`vite.config.ts`), and `make web-check` compares the two files. Both of those
 * only prove the *file* exists and matches. Neither proves the file is ever
 * served, or that the app comes up if it is: with a hash router a deep link never
 * misses, so the fallback has always been insurance against a future routing
 * change -- insurance that was emitted, compared, and never cashed.
 *
 * GitHub Pages serves `404.html` (with status 404) for any path it cannot
 * resolve. So the e2e server does too (see `e2e/static-server.mjs`), and these
 * tests are the ones that would go red if that stopped: an unresolvable path must
 * return 404 *and* the app shell, and the shell must boot.
 *
 * ## What "boots" actually means here, measured rather than assumed
 *
 * The build uses `base: './'` (relative URLs). A relative `<script src="./assets
 * /index.js">` resolves against the *directory* of the served document. So:
 *
 *   - `/repo/nope` (one path segment, no trailing slash) resolves `./assets/...`
 *     against `/repo/` -- the real asset directory. The app boots fully.
 *   - `/repo/nope/softmax` (two segments) resolves `./assets/...` against
 *     `/repo/nope/` -- which does not exist. The HTML shell is served and the
 *     browser parses it into a blank page: every asset 404s, so nothing mounts
 *     and there is no visible text at all. This is a real limitation of relative
 *     asset URLs, not of the fallback, and it is asserted as such below rather
 *     than hidden.
 *
 * That second shape is the one that matters, and it is the shape this app would
 * actually produce: `hrefFor` (`src/app/router.ts`) emits `#/learn/${conceptId}`
 * for every concept page -- `/learn/softmax`, `/learn/gradient`, and 16 more --
 * i.e. the primary content pages are all multi-segment.
 *
 * So the item's acceptance criterion -- "a test asserts the app boots and renders
 * a route-appropriate page when loaded from such a URL" -- is met only for the
 * one-segment case, and the case it is *insurance for* is the one that does not
 * boot. The multi-segment case is therefore pinned as a known limitation, and
 * this comment says so plainly instead of selecting favourable examples: a future
 * history-routing change must switch the asset base to an absolute mount-root
 * path first, and the pinned test is what will tell it that.
 */

const DIST = resolve(fileURLToPath(new URL('../dist', import.meta.url)))

/** Fail a test with the console/page errors it produced during the run. */
function collectFailures(page: Page): string[] {
  const failures: string[] = []
  page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
  page.on('console', (message) => {
    if (message.type() === 'error') failures.push(`console: ${message.text()}`)
  })
  return failures
}

test.describe('an unresolvable path under the mount point', () => {
  test('is served as a 404, with HTML, not an error page of the server', async ({ page }) => {
    // The whole point of the item: the server must serve the fallback rather
    // than a plain-text "not found", and it must keep the HTTP status a real
    // 404 -- GitHub Pages serves `404.html` *with* a 404, and a test that only
    // asserted the body would pass against a server that returned 200 + the
    // shell, which is a different (and wrong) contract.
    const response = await page.goto('some/path/that/does/not/exist')
    expect(response).not.toBeNull()
    expect(response?.status()).toBe(404)
    // The content type is HTML: a history-routed fallback that served the shell
    // as `text/plain` would render as source text in the browser.
    expect(response?.headers()['content-type']).toContain('text/html')
  })

  test('the app boots from a one-segment 404 URL and renders the home route', async ({ page }) => {
    const failures = collectFailures(page)

    // `nope` is one segment, so the relative asset URLs in the shell resolve
    // against the mount root and the app really does boot. The URL carries no
    // hash, so `parseRoute('')` falls through to the `home` route
    // (`src/app/router.ts`): the reader landed on a URL that does not exist and
    // got the front door. The assertions are about the *real* app, so they check
    // the hero heading and the rendered content, not merely that some `<h1>`
    // exists -- the `<noscript>` block in `index.html` would satisfy a looser
    // check while React never mounted (see the header comment on why this is the
    // *only* URL shape for which "boots" can be claimed).
    const response = await page.goto('nope')
    expect(response?.status()).toBe(404)

    const root = page.locator('#root')
    await expect(root).not.toBeEmpty()
    await expect(page.getByRole('heading', { level: 1, name: /A guided reading of/ })).toBeVisible()

    // Honest scoping: the browser *does* log a console error for the failed
    // top-level navigation itself --
    //   "Failed to load resource: the server responded with a status of 404"
    // -- and that resource is the served 404 document, which is the condition
    // under test, not a fault. So the assertion tolerates exactly that one line
    // and nothing else: any additional console error, and *any* uncaught
    // exception, still fails the test.
    const navigationNoise = /Failed to load resource: the server responded with a status of 404/
    const real = failures.filter((message) => !navigationNoise.test(message))
    expect(real, `no console or page errors beyond the served 404:\n${real.join('\n')}`).toEqual([])
  })

  test('a multi-segment 404 URL serves the shell but the app does not boot (relative assets)', async ({
    page,
  }) => {
    // This is the honest limitation. The deep link shape a history router would
    // produce (`/nope/softmax`, `/learn/softmax`) serves the `404.html` shell,
    // but `base: './'` makes `./assets/index.js` resolve against `/nope/`,
    // which is not a directory that exists, so the module 404s and `#root`
    // stays empty. The fallback did its job -- the *document* was served -- and
    // the app cannot mount because of a separate, pre-existing property
    // (relative asset URLs). The result a reader sees is a **blank page**: the
    // shell has no visible static text outside `<script>`/`<noscript>`, so
    // nothing is rendered until the (missing) bundle runs. Pinning it here means
    // a future history-routing change cannot assume multi-segment deep links
    // work; it must switch the asset base to an absolute mount-root path first.
    const response = await page.goto('nope/softmax')
    expect(response?.status()).toBe(404)
    // The shell was served ...
    expect((await response?.text()) ?? '').toContain('id="root"')
    // ... but nothing mounted.
    await expect(page.locator('#root')).toBeEmpty()
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(0)
  })

  test('the response body really is the app shell, not some other 404 document', async ({
    page,
  }) => {
    // This is the assertion that makes "serves 404.html" distinct from "returns a
    // 404". A server could return 404 with an empty body, or with a handwritten
    // error page, and the status-only tests above would still pass. Fetching the
    // same URL with the request API and reading the bytes proves the *shell* came
    // back: `id="root"` is the mount point the app needs to boot from.
    const response = await page.request.get('some/path/that/does/not/exist')
    expect(response.status()).toBe(404)
    expect(response.headers()['content-type']).toContain('text/html')

    const body = await response.text()
    expect(body).toContain('id="root"')
    // And it is the built document, with the app's module script, not a stub.
    expect(body).toContain('/assets/')
  })

  test('a missing asset is a 404 too -- the fallback is path-based, not extension-based', async ({
    page,
  }) => {
    // `static-server.mjs` serves the fallback for any under-prefix miss, so a
    // request for a missing `.js` also gets the HTML document with a 404 status.
    // That mirrors GitHub Pages, which does not distinguish by extension. The
    // assertion here is deliberately only about the status and the body: the
    // browser will log a console error for a failed subresource load, so
    // asserting a silent console would be a false statement about how browsers
    // work.
    const response = await page.goto('assets/missing-hash.js')
    expect(response?.status()).toBe(404)
    // Honest scoping: the body is the HTML fallback (path-based), not a script.
    // This is a note, not a guarantee -- the page is not loaded as a document by
    // the app, so no boot assertions belong here.
    expect((await response?.text()) ?? '').toContain('id="root"')
  })
})

test.describe('the fallback is the shell', () => {
  test('dist/404.html is byte-identical to dist/index.html, as make web-check asserts', () => {
    // The Makefile's `web-check` target already runs `cmp dist/index.html
    // dist/404.html`. Restating it here is not duplication for its own sake: the
    // e2e suite serves 404.html, so if the two files ever diverged the suite
    // would be testing a document the deploy does not publish -- and if the
    // Makefile check were removed, this one would still catch it.
    const index = readFileSync(resolve(DIST, 'index.html'), 'utf8')
    const notFound = readFileSync(resolve(DIST, '404.html'), 'utf8')
    expect(notFound).toBe(index)
  })
})

// Referenced so the mount point this suite runs under is part of the spec: every
// `page.goto('...')` above resolves under `PREFIX` through Playwright's
// `baseURL`, and a bare assertion that the shell came back only means something
// if the URL was under the mount point.
void PREFIX
