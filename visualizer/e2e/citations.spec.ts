import { expect, test } from '@playwright/test'

/**
 * Citations: a phrase in the prose that scrolls the code panel to a line range.
 *
 * The claim being tested is end-to-end because the feature is: the citation is
 * authored in `content/concepts/data-loading.md` as a markdown link, resolved by
 * the build against the concept's declared anchor, rendered by `Markdown` into a
 * button, and handled by `ConceptPage` by moving the code panel's focus. Any one
 * of those four could be broken while the page still *looks* right -- the phrase
 * would render, just as inert text -- so a DOM assertion is the only check that
 * covers the whole path.
 *
 * The page's own default focus is the concept's whole anchor (lines 13-20 of the
 * reference). The interesting assertion is therefore not "something is
 * highlighted" -- that is already true before the click -- but that the
 * highlighted *set* changes to exactly what the citation names. Both tests below
 * pin that down.
 */

test.describe('prose citations', () => {
  test('a citation renders as a control that scrolls the panel to its range', async ({ page }) => {
    await page.goto('#/learn/data-loading')
    await expect(page.getByRole('heading', { name: 'Loading the dataset' })).toBeVisible()

    const panel = page.locator('.code-panel').first()

    // Before the click the panel shows the concept's whole anchor: eight focused
    // lines, 13 through 20. Asserting this first is what makes the click's effect
    // meaningful -- a panel that was already focused on line 19 would pass the
    // assertion below without the citation doing anything.
    const focused = panel.locator('.code-gutter__line.is-focused')
    await expect(focused).toHaveCount(8)
    await expect(focused.first()).toHaveText('13')

    // The citation is authored as `[random.shuffle(docs)](cite:19)`. It is a
    // button, not a link: it acts on the page, it does not navigate. Finding it
    // by role also asserts that distinction -- the whole point of the design is
    // that it is *not* an anchor.
    const citation = page.getByRole('button', {
      name: /show random\.shuffle\(docs\) in the source/i,
    })
    await expect(citation).toBeVisible()
    await expect(citation).toHaveAttribute('data-cite', '19-19')

    await citation.click()

    // Exactly one line, and it is line 19 -- the `random.shuffle(docs)` line. A
    // citation that highlighted the whole anchor, or the wrong single line, is
    // the failure mode this guards against.
    await expect(focused).toHaveCount(1)
    await expect(focused.first()).toHaveText('19')
    await expect(focused.first()).toHaveAttribute('data-line', '19')
  })

  test('the citation resolves from the anchor, not a frozen line number', async ({ page }) => {
    /*
     * Acceptance criterion 2: the link must resolve from the concept's declared
     * anchors, not a line number in the markup. The observable consequence is
     * that the citation's target is the *same number* the anchor itself resolves
     * to -- so this reads both off the page and compares them, rather than
     * hard-coding 19 a second time.
     *
     * The panel's own range label ("lines 13–20") is rendered from the resolved
     * anchor. If the reference file ever moved the section, the anchor would move
     * and the build would fail (`tools/validate_concepts.py`), and the citation
     * would have to be rewritten to match. Nothing on this page is a literal.
     */
    await page.goto('#/learn/data-loading')
    const panel = page.locator('.code-panel').first()
    await expect(panel.getByText('lines 13–20')).toBeVisible()

    const target = (await page
      .getByRole('button', { name: /show random\.shuffle\(docs\) in the source/i })
      .getAttribute('data-cite')) as string
    const [start, end] = target.split('-').map(Number)

    // Inside the anchor window the page reports. That is the contract the Python
    // gate enforces, restated against the rendered page rather than the file.
    expect(start).toBeGreaterThanOrEqual(13)
    expect(end).toBeLessThanOrEqual(20)

    // And the panel really lands there, so the two halves agree at runtime.
    const focused = panel.locator('.code-gutter__line.is-focused')
    await page.getByRole('button', { name: /show random\.shuffle\(docs\) in the source/i }).click()
    await expect(focused.first()).toHaveAttribute('data-line', String(start))
    await expect(focused.last()).toHaveAttribute('data-line', String(end))
  })

  test('a citation is a button, and an ordinary link is still a link', async ({ page }) => {
    /*
     * The regression this exists for: the renderer decides "citation" from the
     * href scheme alone, and every other link on every concept page goes through
     * the same branch. Turning the *whole* link path into buttons would break the
     * cross-references (`[Cross-entropy loss](../concepts/cross-entropy-loss.md)`)
     * that the corpus is full of, and it would do it quietly.
     */
    await page.goto('#/learn/data-loading')

    // A real link, from the same concept's prose, still navigates. Scoped to the
    // article: the sidebar's contents rail links to the same concept by title.
    const link = page.locator('#main').getByRole('link', { name: 'Cross-entropy loss' })
    await expect(link).toBeVisible()
    await expect(link).toHaveAttribute('href', '../concepts/cross-entropy-loss.md')
  })
})
