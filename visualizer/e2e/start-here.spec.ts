import { expect, test, type Page } from '@playwright/test'

/**
 * The guided "start here" surface on the home page.
 *
 * A first-time reader lands on a hero with a graph and a rail of eighteen
 * concepts and no ordered invitation to begin. The unit test
 * (`tests/first-run.test.ts`) pins the *decision* -- ordering plus seen-set to a
 * state -- against the real `index.json`; what only a browser can prove is the
 * part a reader touches:
 *
 *  - the link the home page actually renders points at the real first concept
 *    of the curriculum, not at something that merely looks like it;
 *  - visiting a concept really flips the surface to its de-emphasised state, on
 *    the *storage* path the app uses rather than a seeded one;
 *  - the same holds at 390px, where the hero stacks.
 *
 * The "first concept" is read here from the app's own rail rather than typed as
 * a slug: the rail is the curriculum in `index.json` order, so the assertion is
 * "the CTA points at whatever this build's first concept is", which is what the
 * acceptance criterion asks for and which survives a reordered curriculum.
 */

/** The CTA as it reads to a screen reader / a reader, by accessible name. */
function startHere(page: Page) {
  return page.getByRole('link', { name: /start here/i })
}

/** The demoted "continue" link a reader who has started gets. */
function continueLink(page: Page) {
  return page.getByRole('link', { name: /continue:/i })
}

/**
 * The curriculum's first concept, taken from the rail: the first group under
 * `#/learn` is the `data` chapter, and its first link is the first concept.
 */
async function firstConceptHref(page: Page): Promise<string> {
  await page.goto('./#/learn')
  const href = await page.locator('.app__sidebar-group a').first().getAttribute('href')
  expect(href, 'the rail lists a first concept').toBeTruthy()
  return href as string
}

/** Fail a test with the console/page errors it produced during the run. */
function collectFailures(page: Page): string[] {
  const failures: string[] = []
  page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
  page.on('console', (message) => {
    if (message.type() === 'error') failures.push(`console: ${message.text()}`)
  })
  return failures
}

test.describe('home: the guided start', () => {
  test('a fresh reader is invited to begin, and the link opens the real first concept', async ({
    page,
  }) => {
    const failures = collectFailures(page)

    const firstHref = await firstConceptHref(page)
    await page.goto('./')
    // Storage really is empty, so the assertions below are about the first-run
    // state and not about a profile left behind by another test.
    expect(await page.evaluate(() => localStorage.getItem('atlas:progress'))).toBeNull()

    const cta = startHere(page)
    await expect(cta).toBeVisible()
    // The acceptance criterion, exactly: the CTA links to the first concept.
    // `data-loading` is the `data` chapter's order-1 concept, reached through the
    // rail above rather than written here.
    await expect(cta).toHaveAttribute('href', firstHref)
    expect(firstHref).toMatch(/#\/learn\/data-loading$/)
    // It is a *primary* call to action while the reader has not started.
    await expect(cta).toHaveClass(/button--primary/)

    // And the link really works -- clicking it lands on the first concept, which
    // is the whole point of an invitation.
    await cta.click()
    await expect(page).toHaveURL(/#\/learn\/data-loading$/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(/Loading the dataset/)

    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })

  test('visiting the first concept de-emphasises the CTA on return to home', async ({ page }) => {
    const failures = collectFailures(page)
    await page.goto('./')

    // Earn the state the way a reader does: click the invitation, read the
    // concept, come back. No seeded storage -- that would prove the markup can
    // be driven by a fixture, not that visiting a concept marks it. The heading
    // is awaited first: the concept page writes progress on mount, and leaving
    // before that lands would test the timing, not the CTA.
    await startHere(page).click()
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(/Loading the dataset/)

    await page.goto('./')
    // The reader has started: the loud invitation is gone...
    await expect(startHere(page)).toHaveCount(0)
    await expect(page.locator('.home__start')).toHaveClass(/is-started/)
    // ...and what remains is a quieter link to the *next* concept, so the
    // curriculum is still reachable from home (see the spec's header).
    const next = continueLink(page)
    await expect(next).toBeVisible()
    await expect(next).not.toHaveClass(/button--primary/)
    await expect(next).toHaveAttribute('href', '#/learn/tokenizer')
    // The playground link takes over the primary slot, so the page still has one
    // obvious action -- it is just no longer "begin".
    await expect(page.getByRole('link', { name: /run it in your browser/i })).toHaveClass(
      /button--primary/,
    )

    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })

  test('the start surface stacks on a phone without overflowing', async ({ page }) => {
    const failures = collectFailures(page)
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('./')

    await expect(startHere(page)).toBeVisible()
    await expect(page.locator('.home__start-note')).toContainText('Loading the dataset')

    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }))
    expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth)

    // The de-emphasised state stacks too: read the first concept and come back.
    // The concept page marks it seen on mount, so the heading is waited for --
    // navigating away before the write lands would test a race, not the phone.
    await startHere(page).click()
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(/Loading the dataset/)
    await page.goto('./')
    await expect(continueLink(page)).toBeVisible()
    await expect(page.locator('.home__start-note')).toContainText('1 of 18 read')

    const overflowAfter = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }))
    expect(overflowAfter.scrollWidth).toBeLessThanOrEqual(overflowAfter.clientWidth)

    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })
})
