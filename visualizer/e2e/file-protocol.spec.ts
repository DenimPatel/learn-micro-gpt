import { expect, test } from '@playwright/test'
import { resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

/**
 * The `file://` suite.
 *
 * Opening a downloaded build straight from disk is the obvious first thing a
 * curious reader tries, and it cannot work: `file://` is an opaque origin, so the
 * browser blocks every `type="module"` script and the CSS with a CORS error. The
 * whole bundle fails to load, the router never runs, and `#root` stays empty --
 * a blank page with no explanation, and no `<noscript>` help because JavaScript
 * *is* enabled. That was the state of the world until this test existed.
 *
 * The app cannot fix the browser policy, so it does the honest thing instead:
 * `index.html` detects `location.protocol === 'file:'` and swaps the empty root
 * for a "serve this over HTTP instead" message. This test asserts that message
 * appears -- i.e. the failure is *explained*, not silently blank.
 *
 * It deliberately does not run through `baseURL`; that is the point. The served
 * suite proves the app works over HTTP, this one proves the one URL that cannot
 * work says so out loud.
 */

const DIST_INDEX = resolve(fileURLToPath(new URL('../dist/index.html', import.meta.url)))

test.describe('opened from the filesystem', () => {
  test('explains the file:// limitation instead of showing a blank page', async ({ page }) => {
    // The module scripts WILL be blocked and WILL log CORS errors. That is the
    // condition under test, not a failure, so this page's console is expected to
    // be noisy and nothing is asserted about it.
    await page.goto(pathToFileURL(DIST_INDEX).href)

    const root = page.locator('#root')
    await expect(root).toContainText('does not run from a')
    await expect(root).toContainText('file://')
    // Names a way out, rather than only naming the problem.
    await expect(root).toContainText('HTTP')

    // The specific regression: `#root` used to be empty here.
    expect((await root.innerText()).trim().length).toBeGreaterThan(40)
  })
})
