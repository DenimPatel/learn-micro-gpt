import { expect, test } from '@playwright/test'

/**
 * `#/research`, against a production build served from a subpath.
 *
 * The unit tests in `tests/research.test.ts` cover the arithmetic. This file
 * covers the three things only a browser can tell you:
 *
 *  1. The page renders with no console or page errors. A `?raw` import that
 *     resolves at build time but a lazy `import.meta.glob` entry that does not
 *     will produce a blank section and a console error, and neither shows up in
 *     a node-environment unit test.
 *  2. Every `<svg>` has a real `aria-label`. The accessibility sweep in
 *     `atlas.spec.ts` walks the whole document and fails on an unlabelled
 *     image, but it visits the routes it knows about — this one is new, and a
 *     chart with no label is invisible to a screen reader.
 *  3. The research data is in the bundle, not fetched. Same reason as the
 *     language sources: this site's whole argument is that there is no URL to
 *     get wrong, and a new data path is exactly where that would come back.
 */

test.describe('the research page', () => {
  test('loads with no console or page errors', async ({ page }) => {
    const failures: string[] = []
    page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(`console: ${message.text()}`)
    })

    const response = await page.goto('#/research')
    expect(response?.status()).toBe(200)
    await expect(
      page.getByRole('heading', { level: 1, name: 'What the loop has tried' }),
    ).toBeVisible()
    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })

  test('is reachable from the main nav and marked current there', async ({ page }) => {
    await page.goto('#/')
    const nav = page.locator('nav[aria-label="Main"]')
    await expect(nav.locator('a[href="#/research"]')).toHaveText('research')
    await nav.locator('a[href="#/research"]').click()
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('What the loop has tried')
    await expect(nav.locator('a[aria-current="page"]')).toHaveText('research')
  })

  test('every chart carries a quantitative label, not a caption', async ({ page }) => {
    await page.goto('#/research')
    const charts = page.locator('.research svg')
    const count = await charts.count()
    expect(count).toBeGreaterThan(0)
    for (let i = 0; i < count; i += 1) {
      const label = await charts.nth(i).getAttribute('aria-label')
      expect(label, `svg ${i} has no aria-label`).toBeTruthy()
      // A label that is just the title is useless to a screen reader: the point
      // is that the numbers are in it, which is what `LossChart` does too.
      expect(label?.length ?? 0, `svg ${i} has a suspiciously short label`).toBeGreaterThan(40)
      expect(label, `svg ${i} label should mention runs or experiments`).toMatch(/run|experiment/i)
    }
  })

  test('the ledger shows every run, crashes included', async ({ page }) => {
    await page.goto('#/research')
    const rows = page.locator('.research__ledger tbody tr')
    // One row per ledger entry. The count is cross-checked against the summary
    // figure rather than hard-coded, so adding an experiment does not break it.
    const experiments = await page
      .locator('.research .stats')
      .first()
      .locator('div', { hasText: 'experiments' })
      .locator('dd')
      .innerText()
    const expected = Number(experiments) + 1 // plus the baseline row
    expect(await rows.count()).toBe(expected)
    await expect(rows.first()).toContainText('baseline')
  })

  test('the research numbers are in the bundle, not fetched', async ({ page }) => {
    const fetched: string[] = []
    page.on('request', (request) => {
      const path = new URL(request.url()).pathname
      if (/\.(json|patch|rs)$/.test(path) && /autoresearch/.test(path)) {
        fetched.push(request.url())
      }
    })

    await page.goto('#/research')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    await page.waitForTimeout(1200)

    expect(fetched, `research data fetched at runtime:\n${fetched.join('\n')}`).toEqual([])
  })

  test('the caveat list is on the page, not just in the source', async ({ page }) => {
    // A table of numbers with no caveats is a lie, and the caveats live in a JSON
    // field that nothing else forces the page to render.
    await page.goto('#/research')
    const caveats = page.locator('.research .caveats li')
    expect(await caveats.count()).toBeGreaterThan(0)
    await expect(caveats.first()).toContainText(/benchmarks|comparable|session/i)
  })

  test('the frozen baseline and the probe digest are both on the page', async ({ page }) => {
    await page.goto('#/research')
    const text = await page.locator('.research').innerText()
    expect(text).toMatch(/[0-9a-f]{12}/) // a shortened digest
    expect(text).toContain('set_data')
  })

  test('the empty state names the command, rather than showing a bare zero', async ({ page }) => {
    // When nothing has been kept, the page must say so and say what to do. A
    // chart with one point and no caption looks like a broken chart.
    await page.goto('#/research')
    const text = await page.locator('.research').innerText()
    if (/experiments[\s\S]{0,40}0/.test(text)) {
      expect(text).toContain('make autoresearch-rust')
    }
  })
})

test.describe('accessibility on the research page', () => {
  test('no unlabelled graphics', async ({ page }) => {
    await page.goto('#/research')
    const offenders: string[] = []
    const nodes = await page.locator('.research svg, .research img').all()
    for (const node of nodes) {
      const tag = await node.evaluate((element) => element.tagName.toLowerCase())
      const hasLabel = await node.evaluate((element) =>
        Boolean(
          element.getAttribute('aria-label') ??
          element.getAttribute('aria-labelledby') ??
          element.getAttribute('title') ??
          element.closest('[aria-label], [aria-hidden], [role="img"]'),
        ),
      )
      if (!hasLabel) offenders.push(tag)
    }
    expect(offenders, `unlabelled graphics: ${offenders.join(', ')}`).toEqual([])
  })

  test('the page has exactly one h1 and a heading order that does not skip', async ({ page }) => {
    await page.goto('#/research')
    await expect(page.locator('article.research h1')).toHaveCount(1)
    const levels = await page
      .locator('article.research :is(h1, h2, h3, h4)')
      .evaluateAll((nodes) => nodes.map((node) => Number(node.tagName.slice(1))))
    let previous = 0
    for (const level of levels) {
      expect(
        level - previous,
        `heading jumped to h${level} after h${previous}`,
      ).toBeLessThanOrEqual(1)
      previous = level
    }
  })
})
