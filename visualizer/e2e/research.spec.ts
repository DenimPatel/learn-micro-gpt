import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test } from '@playwright/test'

/**
 * The number of research tracks, read from the data the page is built against
 * rather than written down here.
 *
 * A literal `3` in this file was a second copy of the track list, and the second
 * copy is what failed: adding C made the page render four buttons and this
 * assertion still asked for three. It failed loudly, which is the good outcome,
 * but "add a track" should not mean "find the literals".
 *
 * Read from `autoresearch/results.json` rather than from the page, so that the
 * assertion still has teeth: it compares the switcher against the data it claims
 * to render, and a track present in the bundle but missing from the switcher
 * would still fail.
 */
const RESEARCH_TRACKS = (
  JSON.parse(
    readFileSync(
      fileURLToPath(new URL('../../autoresearch/results.json', import.meta.url)),
      'utf-8',
    ),
  ) as { tracks: Record<string, { language: string }> }
).tracks
const RESEARCH_TRACK_COUNT = Object.keys(RESEARCH_TRACKS).length
/** The button labels, which are languages rather than track keys. */
const RESEARCH_LANGUAGES = Object.values(RESEARCH_TRACKS).map((track) => track.language)

/**
 * `#/research`, against a production build served from a subpath.
 *
 * The unit tests in `tests/research.test.ts` cover the arithmetic, and
 * `tests/research-empty-state.test.tsx` covers the empty states -- which cannot
 * be reached from here, because this file runs against the real
 * `autoresearch/results.json` and every track in it has been run on. This file
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
      if (/\.(json|patch|rs|go|ts)$/.test(path) && /autoresearch/.test(path)) {
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
    // The patch, not the method name: which accessor the probe needed is a
    // property of the port, and Go and TypeScript expose their parameters already,
    // so asserting `set_data` here would only ever have described Rust.
    expect(text).toContain('lines added')
  })

  test('offers every track the harness knows, and switching changes the page', async ({ page }) => {
    // The whole point of the multi-track change: a Go number must never be
    // reachable under a Rust heading, or vice versa. So the switcher has to be
    // present, it has to mark which track is showing, and pressing a different
    // one has to actually change what the page says.
    await page.goto('#/research')
    const buttons = page.locator('.research__track')
    await expect(buttons).toHaveCount(RESEARCH_TRACK_COUNT)
    // The exact set of labels, compared against the data and in the data's own
    // order -- which is a stronger claim than a set, and is what the switcher
    // renders since it iterates `Object.keys(research.tracks)`.
    //
    // A count alone would pass on four buttons reading "Rust Rust Go Go", so the
    // labels are compared. They are read from `.research__track-label` rather than
    // with `filter({ hasText })`, because Playwright's `hasText` string form is a
    // case-insensitive *substring* match: `hasText: 'C'` matches "TypeScript" too,
    // which is the same trap that let a page-wide innerText assertion pass against
    // another track's control.
    await expect(buttons.locator('.research__track-label')).toHaveText(RESEARCH_LANGUAGES)
    await expect(buttons.filter({ hasText: 'Rust' })).toHaveAttribute('aria-pressed', 'true')

    // Asserted on the source file and the build command rather than on the
    // candidate directory: "candidate" is a substring of "candidate-go" and
    // "candidate-ts", so a directory assertion would pass for all three tracks and
    // catch nothing.
    await expect(page.locator('.research')).toContainText('src/lib.rs')
    await expect(page.locator('.research')).toContainText('cargo build --release')

    await buttons.filter({ hasText: 'Go' }).click()
    await expect(buttons.filter({ hasText: 'Go' })).toHaveAttribute('aria-pressed', 'true')
    await expect(buttons.filter({ hasText: 'Rust' })).toHaveAttribute('aria-pressed', 'false')
    await expect(page.locator('.research')).toContainText('main.go')
    await expect(page.locator('.research')).not.toContainText('cargo build --release')

    await buttons.filter({ hasText: 'TypeScript' }).click()
    await expect(page.locator('.research')).toContainText('src/index.ts')
    await expect(page.locator('.research')).not.toContainText('main.go')
  })

  test('a track with experiments renders its ledger statistics, not an empty state', async ({
    page,
  }) => {
    // Every track in the ledger has been run on, and the loop below asserts it
    // per track rather than assuming it. (The test this replaces carried the
    // comment "Go and TypeScript have never been run", which was the premise of
    // the whole test and was already false when it was written -- Go had been
    // kept from run 0107. Restating a fact about the ledger in prose is a
    // maintenance burden that fails silently; asserting it costs one line and
    // goes red the moment the ledger moves, which is what should have happened
    // here instead.)
    //
    // Why this is scoped and the test it replaces was not. That test clicked the
    // Go button and then matched a regex against `.research`'s inner text --
    // the text of the whole article, which *includes the switcher*. Every button
    // in the switcher is labelled `{language}` plus `{N} kept`, so
    // `/no experiment|0 kept|experiments\s*0/i` could be satisfied by any track's
    // control and never by the track under test. It passed for as long as
    // TypeScript had zero keeps, recorded nothing about why, and went red on run
    // 0232 when TypeScript recorded its first -- not because the page changed
    // behaviour, but because a *different* track's label stopped matching. A
    // regex over a container that holds the control you are not testing measures
    // the control.
    //
    // So the fix is scope, in both directions:
    //
    //  - `.research > dl.stats` is the ledger's own definition list and a direct
    //    child of the article. Each chart carries a `<dl class="stats">` of its
    //    own, nested inside a `<figure>`, so the child combinator picks the
    //    ledger and only the ledger. It is the element the prose at
    //    `ResearchPage.tsx:1090` describes, so it moves if and when the ledger
    //    moves rather than needing this test to be edited alongside it.
    //  - `p.research__empty` is the empty state's own class and appears nowhere
    //    in the switcher, so "no empty state" cannot be satisfied by a sibling
    //    track's label either.
    await page.goto('#/research')
    for (const label of ['Rust', 'Go', 'TypeScript']) {
      const button = page.locator('.research__track').filter({ hasText: label })
      await button.click()
      await expect(button).toHaveAttribute('aria-pressed', 'true')

      const ledger = page.locator('article.research > dl.stats')
      // All four figures, and their order, rather than just the one under test:
      // a `<dt>` that went missing is a page that lost a number, and checking
      // the list catches that without a separate assertion per figure.
      await expect(ledger.locator('dt')).toHaveText(['experiments', 'kept', 'discarded', 'crashed'])
      const experiments = Number(
        await ledger.locator('div', { hasText: 'experiments' }).locator('dd').innerText(),
      )
      expect(experiments, `${label} has experiments in the ledger`).toBeGreaterThan(0)

      // Cross-checked against this track's table, not hard-coded, and not the
      // ledger-wide total: the figures come from `trackRows(track)`, so a
      // mismatch would mean the heading and the table below it were describing
      // two different tracks.
      const experimentRows = page.locator(
        '.research__ledger tbody tr:not(.research__row--baseline)',
      )
      expect(await experimentRows.count(), `${label} ledger rows`).toBe(experiments)

      // The property, negatively: a track with measurements must not tell the
      // reader there are none.
      await expect(page.locator('article.research p.research__empty')).toHaveCount(0)
    }
  })

  test('the switcher and the ledger count the same track, not the whole ledger', async ({
    page,
  }) => {
    // This replaces a test that read `make autoresearch-rust` out of the page if
    // and only if the article matched /experiments[\s\S]{0,40}0/. That condition
    // had been false since the first keep, so the test had been passing
    // vacuously, and it was reading the same article-wide inner text that made
    // the empty-state test beside it meaningless. The empty state itself is now
    // covered where it can actually be reached -- a fixture ledger, in
    // `tests/research-empty-state.test.tsx` -- because no real ledger has a track
    // with nothing on it.
    //
    // What *is* reachable here, with the real data, is the mistake the switcher's
    // own comment warns about: `research.counts` is the ledger-wide total, and
    // reading it would put a Rust keep count next to a Go chart. Both figures on
    // screen are derived from `trackRows(track)`, so switching must change both
    // of them together -- and the last assertion is what keeps the pair of
    // checks from being a tautology: TypeScript has one keep and Rust has
    // dozens, so a ledger-wide number leaking into either would show up as the
    // two agreeing with each other and disagreeing with the table.
    await page.goto('#/research')
    const kept = async (label: string) => {
      const button = page.locator('.research__track').filter({ hasText: label })
      await button.click()
      await expect(button).toHaveAttribute('aria-pressed', 'true')
      const inLedger = Number(
        await page
          .locator('article.research > dl.stats')
          .locator('div', { hasText: 'kept' })
          .locator('dd')
          .innerText(),
      )
      const inSwitcher = Number(
        (await button.locator('.research__track-note').innerText()).replace(/[^\d]/g, ''),
      )
      expect(inSwitcher, `${label} keep count in the switcher`).toBe(inLedger)
      return inLedger
    }

    const rust = await kept('Rust')
    const go = await kept('Go')
    const typescript = await kept('TypeScript')
    // The tracks are nowhere near each other, which is the point: these three
    // numbers being different is what makes the agreement above mean something.
    expect(typescript).toBeLessThan(go)
    expect(go).toBeLessThan(rust)
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
