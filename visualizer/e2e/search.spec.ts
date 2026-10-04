import { expect, test, type Page } from '@playwright/test'

/**
 * Concept search in the rail.
 *
 * The feature is a filter field at the top of the concept rail that narrows the
 * list by title *and* summary, in reading order, entirely in memory. The
 * matching rules are pinned in `tests/search.test.ts` against the real generated
 * content; what only a browser can prove is the part a reader actually touches:
 *
 *  - the field is wired to the matcher at all,
 *  - the "sole result" case is reachable from a typed term,
 *  - a summary-only match explains itself with the words that matched,
 *  - the states a filter can fall into: half-typed, no match, cleared,
 *  - keyboard-only use, including Escape, which the shell also binds,
 *  - the 390px drawer, where the rail stacks over the content.
 *
 * Everything is asserted against the rendered DOM, and the terms used are real
 * words from the generated content rather than fixtures.
 */

/** The rail's filter field, by its accessible name. */
function searchBox(page: Page) {
  return page.getByLabel('Search concepts by title or summary')
}

/** The rail's filtered result links, in the order they render. */
function results(page: Page) {
  return page.locator('.app__search-results a')
}

/** The count line under the field. */
function status(page: Page) {
  return page.locator('.app__search-status')
}

test.describe('concept search: narrowing the rail', () => {
  test('a typed term narrows the rail to the concept that names it, and it opens', async ({
    page,
  }) => {
    const failures: string[] = []
    page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(`console: ${message.text()}`)
    })

    await page.goto('#/learn')
    // Unfiltered, the rail is the whole curriculum -- otherwise a one-result
    // assertion below would be vacuous.
    await expect(page.locator('.app__sidebar-group a')).toHaveCount(18)

    // No other title or summary in the curriculum contains "rmsnorm", so this
    // query demonstrates the acceptance criterion in the small: one term, one
    // result, and that result opens the concept it names.
    await searchBox(page).fill('rmsnorm')
    await expect(results(page)).toHaveCount(1)
    await expect(results(page)).toHaveText(/RMSNorm/)
    await expect(status(page)).toHaveText('1 of 18 concepts match “rmsnorm”')

    await results(page).click()
    await expect(page).toHaveURL(/#\/learn\/rmsnorm/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(/RMSNorm/)

    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })

  test('a query that matches only a summary says why, with the matching words', async ({
    page,
  }) => {
    // "chain rule" is the reader's phrase; it is in a summary and in no title.
    await page.goto('#/learn')
    await searchBox(page).fill('chain rule')

    await expect(results(page)).toHaveCount(1)
    await expect(results(page).first()).toContainText('Walking the graph backwards')
    // The reason it matched is on screen, not just implied. (The summary spells
    // it "chain-rule", and the matcher lowercases but does not normalise
    // punctuation, so the word "rule" is what appears.)
    await expect(results(page).first().locator('.app__search-snippet')).toContainText('rule')
  })

  test('a word that is nowhere in the content says so instead of showing a blank rail', async ({
    page,
  }) => {
    await page.goto('#/learn')
    await searchBox(page).fill('quantum entanglement')

    await expect(results(page)).toHaveCount(0)
    await expect(page.locator('.app__search-empty')).toBeVisible()
    await expect(page.locator('.app__search-empty')).toContainText('No concept mentions')
    await expect(page.locator('.app__search-empty')).toContainText('quantum entanglement')
    await expect(status(page)).toContainText('no concept matches')
    // The way out is on screen: the full curriculum is still reachable.
    await page.locator('.app__search-clear').click()
    await expect(page.locator('.app__sidebar-group a')).toHaveCount(18)
  })

  test('one letter does not filter, because a single letter matches half the rail', async ({
    page,
  }) => {
    await page.goto('#/learn')
    await searchBox(page).fill('e')
    // The rail is untouched and the reader is told why.
    await expect(page.locator('.app__sidebar-group a')).toHaveCount(18)
    await expect(status(page)).toContainText('keep typing')
    await expect(results(page)).toHaveCount(0)

    // A second character, that matches nothing, reaches the empty state.
    await searchBox(page).fill('eo')
    await expect(results(page)).toHaveCount(0)
    await expect(status(page)).toContainText('no concept matches')
  })

  test('an empty query shows the ordinary rail, chapter by chapter, in reading order', async ({
    page,
  }) => {
    await page.goto('#/learn')
    const groups = await page
      .locator('.app__sidebar-group h2')
      .evaluateAll((nodes) => nodes.map((node) => node.textContent))
    expect(groups.length).toBeGreaterThan(1)

    await searchBox(page).fill('attention')
    await expect(page.locator('.app__sidebar-group')).toHaveCount(0)

    // Clearing restores the chapter heads exactly as they were.
    await searchBox(page).fill('')
    const after = await page
      .locator('.app__sidebar-group h2')
      .evaluateAll((nodes) => nodes.map((node) => node.textContent))
    expect(after).toEqual(groups)
    await expect(status(page)).toHaveText('')
  })

  test('filtered results keep reading order, not an arbitrary ranking', async ({ page }) => {
    // The full, unfiltered order, read from the rail itself.
    await page.goto('#/learn')
    const full: string[] = await page
      .locator('.app__sidebar-group a')
      .evaluateAll((nodes) => nodes.map((node) => node.textContent?.trim() ?? ''))

    await searchBox(page).fill('the')
    const filtered: string[] = await results(page).evaluateAll((nodes) =>
      nodes.map((node) => node.querySelector('.app__search-title')?.textContent?.trim() ?? ''),
    )
    expect(filtered.length).toBeGreaterThan(1)
    // The filtered list is exactly the unfiltered list with entries removed --
    // i.e. the same relative order. A sort by relevance would break this.
    const expected = full.filter((title) => filtered.includes(title))
    expect(filtered).toEqual(expected)
  })
})

test.describe('concept search: keyboard', () => {
  test('ArrowDown moves from the field into the results', async ({ page }) => {
    await page.goto('#/learn')
    await searchBox(page).click()
    await searchBox(page).fill('rmsnorm')
    await page.keyboard.press('ArrowDown')
    await expect(results(page).first()).toBeFocused()
  })

  test('Escape clears a typed query, and a second Escape leaves the field', async ({ page }) => {
    await page.goto('#/learn')
    await searchBox(page).fill('rmsnorm')
    await expect(results(page).first()).toBeVisible()

    await page.keyboard.press('Escape')
    // The rail is back, the query is gone, and focus stayed in the field so the
    // reader can retype without reaching for the pointer.
    await expect(page.locator('.app__sidebar-group a')).toHaveCount(18)
    await expect(searchBox(page)).toHaveValue('')
    await expect(searchBox(page)).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(searchBox(page)).not.toBeFocused()
  })

  test('a keyboard reader can arrow into the results and open one', async ({ page }) => {
    await page.goto('#/learn')
    await searchBox(page).click()
    await searchBox(page).fill('rmsnorm')
    await page.keyboard.press('ArrowDown')
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/#\/learn\/rmsnorm/)
  })
})

test.describe('concept search: the 390px drawer', () => {
  test('the field stacks with the drawer, filters it, and closes it on navigate', async ({
    page,
  }) => {
    const failures: string[] = []
    page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(`console: ${message.text()}`)
    })

    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('#/learn')

    // The rail is a drawer at this width: it starts off-screen...
    await page.getByRole('button', { name: 'concepts' }).click()
    await expect(page.locator('.app__sidebar')).toHaveClass(/is-open/)

    // ...and the field is inside it, usable, and does not push the page wider
    // than the viewport.
    await searchBox(page).fill('rmsnorm')
    await expect(results(page).first()).toBeVisible()
    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }))
    expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth)

    // Following a result closes the drawer, so the reader sees the concept.
    await results(page).first().click()
    await expect(page).toHaveURL(/#\/learn\/rmsnorm/)
    await expect(page.locator('.app__sidebar')).not.toHaveClass(/is-open/)
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(/RMSNorm/)

    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })
})
