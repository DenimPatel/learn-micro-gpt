import { defineConfig, devices } from '@playwright/test'

import { ORIGIN, PREFIX } from './e2e/subpath'

/**
 * Playwright config.
 *
 * The important line is the web server. The e2e suite runs against a **production
 * build served from a subpath**, not the dev server and not a root-served
 * production build.
 *
 * That is the check the previous visualizer needed and did not have. It used
 * `fetch('/microgpt.py')` with a `../microgpt.py` fallback and `<script
 * src="/main.js">`: absolute URLs that ignore whatever subpath GitHub Pages
 * served from. Every test would pass when the site was served from `/`, and the
 * site would 404 in production. Serving from `/learn-micro-gpt/` and pointing
 * the browser at it is the only way to catch that class of bug, and it is why
 * `base: './'` is in the vite config and `server.fs.allow` is in the dev config.
 *
 * The mount point is `PREFIX` (see `e2e/subpath.ts`), not a constant, so the
 * suite is not tied to `/learn-micro-gpt/`. `PREFIX=/some-fork/ npx playwright
 * test` re-runs every test under a fork-like prefix through the same code path,
 * which is what the router's "works at any subpath" claim actually needs.
 *
 * The build is skipped when `SKIP_BUILD=1`: `base: './'` makes the asset URLs
 * relative, so one `dist/` is correct at every prefix and the second run reuses
 * it rather than paying for an identical rebuild.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  // Set unconditionally rather than conditionally: `exactOptionalPropertyTypes`
  // rejects an explicit `undefined` for an optional number.
  workers: process.env.CI ? 1 : 4,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',

  use: {
    baseURL: `${ORIGIN}${PREFIX}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],

  /*
   * A real static server with a non-empty mount point, serving `dist/`.
   *
   * `vite preview` would not do: it strips the base path rather than serving
   * under it, so it cannot reproduce the failure. A 60-line handler that serves
   * files under a prefix is enough, and it has the useful property of 404ing
   * anything outside the prefix -- which is exactly what Pages does.
   */
  webServer: {
    command: `${process.env.SKIP_BUILD ? '' : 'npm run build && '}node e2e/static-server.mjs`,
    url: `${ORIGIN}${PREFIX}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: 'pipe',
  },
})
