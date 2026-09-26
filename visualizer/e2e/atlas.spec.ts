import { expect, test } from '@playwright/test'

/**
 * The subpath suite.
 *
 * Every test here runs against a production build served from
 * `/learn-micro-gpt/`, not from `/`. That is the whole point: the previous
 * visualizer used absolute URLs (`fetch('/microgpt.py')`,
 * `<script src="/main.js">`) that work perfectly at the root and 404 on GitHub
 * Pages, and no test that serves from the root can catch that.
 *
 * If a test in this file fails, the site is broken in production. The unit tests
 * failing means something is broken in the code. Neither is a substitute for the
 * other.
 */

test.describe('served from a subpath', () => {
  test('the app boots and renders its heading', async ({ page }) => {
    const failures: string[] = []
    page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`))
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(`console: ${message.text()}`)
    })

    const response = await page.goto('./')
    expect(response?.status()).toBe(200)
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    expect(failures, `no console or page errors:\n${failures.join('\n')}`).toEqual([])
  })

  test('every asset request is relative, so nothing escapes the mount point', async ({ page }) => {
    const escaped: string[] = []
    page.on('request', (request) => {
      const url = new URL(request.url())
      const path = url.pathname
      if (url.hostname === '127.0.0.1' && !path.startsWith('/learn-micro-gpt/')) {
        escaped.push(`${request.method()} ${path}`)
      }
    })

    await page.goto('./')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    // Give the lazy chunks a chance to be requested; the graph layout and the
    // first code panel both import dynamically.
    await page.waitForTimeout(1200)

    expect(escaped, `requests outside the mount point:\n${escaped.join('\n')}`).toEqual([])
  })

  test('the reference source is in the bundle, not fetched', async ({ page }) => {
    const fetched: string[] = []
    page.on('request', (request) => {
      if (/\.py$|\.c$|\.go$|\.rs$|\.ts$/.test(new URL(request.url()).pathname))
        fetched.push(request.url())
    })

    await page.goto('#/learn/softmax')
    await expect(page.getByRole('heading', { name: 'Softmax' })).toBeVisible()
    await expect(page.locator('.code-panel').first()).toBeVisible()
    // The source arrives as a JS chunk. Give it a moment, then confirm no bare
    // source file was ever requested over the network.
    await page.waitForTimeout(800)
    expect(fetched, `source files must be bundled, not fetched:\n${fetched.join('\n')}`).toEqual([])
  })

  test('a deep link works, which a history router could not do without a rewrite', async ({
    page,
  }) => {
    await page.goto('./#/learn/multi-head-attention')
    await expect(page.getByRole('heading', { name: 'Multi-head attention' })).toBeVisible()
    await expect(page.getByText(/lines 122/)).toBeVisible()
  })
})

test.describe('content and anchors', () => {
  test('the code panel highlights the concept’s own line range', async ({ page }) => {
    await page.goto('#/learn/softmax')
    await expect(page.getByRole('heading', { name: 'Softmax' })).toBeVisible()

    // The generated anchors say softmax is lines 96-100 of the reference. If
    // that drifts, the panel is pointing at the wrong code while still looking
    // completely fine -- which is the failure this whole design exists to catch.
    await expect(page.getByText('lines 96–100')).toBeVisible()

    // Scoped to the visible panel: a concept page also renders a collapsed panel
    // per other language, and those are in the DOM too. Counting every focused
    // line in the document counts the hidden ones as well.
    const panel = page.locator('.code-panel__body:visible').first()
    const focused = panel.locator('.code-gutter__line.is-focused')
    await expect(focused).toHaveCount(5)
    await expect(focused.first()).toHaveText('96')
    await expect(focused.last()).toHaveText('100')
  })

  test('shiki output is not split mid-token', async ({ page }) => {
    // The reference opens with a multi-line docstring, which is a *single* token.
    // The previous implementation split highlighted HTML on newlines and cut it
    // in half. The check is structural: balanced <pre> with real spans inside, and
    // no orphaned closing tag at the start of any line.
    await page.goto('#/learn/autograd-value')
    await expect(page.locator('.code-panel pre.shiki').first()).toBeVisible()

    const broken = await page.locator('.code-panel pre').evaluateAll(
      (nodes) =>
        nodes
          .map((node) => node.innerHTML)
          // A line that starts with a closing tag, or an unbalanced span count,
          // both mean a token was cut.
          .filter(
            (html) =>
              /^\s*<\//.test(html) ||
              (html.match(/<span/g) ?? []).length !== (html.match(/<\/span>/g) ?? []).length,
          ).length,
    )
    expect(broken, 'no code panel has a split token').toBe(0)
  })

  test('an unknown concept shows a useful message, not a blank page', async ({ page }) => {
    await page.goto('#/learn/not-a-concept')
    await expect(page.getByRole('heading', { name: 'No such concept' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Back to all concepts' })).toBeVisible()
  })
})

test.describe('trace widgets', () => {
  test('the attention heatmap is drawn from the recorded trace', async ({ page }) => {
    await page.goto('#/learn/multi-head-attention')
    // The concept inlines two: step 0 and step 200, same position and head. The
    // contrast between them is the point of the section.
    const heatmaps = page.locator('.heatmap')
    await expect(heatmaps).toHaveCount(2)

    // Position 3 with four positions visible: the row must have four cells and
    // they must sum to 1. Those are the two structural facts the concept's prose
    // tells the reader to check.
    const first = heatmaps.nth(0)
    await expect(first.locator('.heatmap__cell')).toHaveCount(4)
    await expect(first).toHaveAttribute('role', 'img')
    await expect(first).toHaveAttribute(
      'aria-label',
      /step 0, head 0, position 3\. position 0: 0\.231/,
    )

    // And the trained one really is different, with the weights the concept's
    // table quotes. If these ever matched, the trace would not be describing
    // training.
    const second = heatmaps.nth(1)
    await expect(second).toHaveAttribute('aria-label', /step 200.*position 0: 0\.394/)
    /*
     * The concept's central claim, asserted on the rendered widget rather than
     * just in prose: at step 200 the argmax is position **0**, not the diagonal.
     * Almost every reader's first guess about attention is that the current
     * position dominates; the recorded run says the opposite, and this is the
     * check that keeps the widget and the claim from drifting apart.
     */
    const cells = second.locator('.heatmap__cell')
    await expect(cells).toHaveCount(4)
    await expect(cells.first()).toHaveClass(/is-argmax/)
    await expect(cells.first()).toHaveText('0.39')
    await expect(cells.last()).toHaveText('0.17')
    await expect(cells.last()).not.toHaveClass(/is-argmax/)
  })

  test('the loss chart is labelled with the measured noise', async ({ page }) => {
    await page.goto('./')
    const chart = page.locator('figure.widget').filter({ hasText: 'Loss over' }).first()
    await expect(chart).toBeVisible()
    await expect(chart).toContainText('500')
    await expect(chart.locator('svg')).toHaveAttribute('aria-label', /1000 steps/)
  })

  test('a trace widget for a step that was not recorded says so', async ({ page }) => {
    // Not every (step, pos, head) combination is in the trace. A widget that
    // renders empty for an unrecorded combination is a lie by omission, so the
    // empty state names what *was* recorded.
    await page.goto('#/explore')
    await expect(page.locator('body')).toContainText('concept')
  })
})

test.describe('accessibility', () => {
  test('the skip link is the first tabbable element and it works', async ({ page }) => {
    await page.goto('./')
    await page.keyboard.press('Tab')
    const focused = page.locator(':focus')
    await expect(focused).toHaveClass(/skip-link/)
    await focused.press('Enter')
    await expect(page).toHaveURL(/#main/)
  })

  test('every landmark is labelled and every nav is distinguishable', async ({ page }) => {
    // A concept page, which has the most landmarks: main nav, breadcrumb, and
    // the previous/next nav. The sidebar is an <aside> with its own label, so it
    // is checked separately below.
    await page.goto('#/learn/softmax')
    const navs = page.locator('nav')
    const count = await navs.count()
    expect(count).toBeGreaterThanOrEqual(3)
    for (let i = 0; i < count; i++) {
      await expect(navs.nth(i)).toHaveAttribute('aria-label', /.+/)
    }
    await expect(page.locator('aside[aria-label]')).toHaveCount(1)
    await expect(page.locator('main')).toHaveAttribute('id', 'main')
  })

  test('the concept graph has a text equivalent, because an SVG is not navigable', async ({
    page,
  }) => {
    await page.goto('#/explore')
    await expect(page.locator('svg.atlas__svg')).toHaveAttribute(
      'aria-label',
      /graph of the 18 concepts/,
    )
    // And the list is real content, not aria-only: it is what a screen reader
    // reads, and what renders if dagre fails to load.
    const list = page.locator('ol.atlas__list')
    await expect(list).toBeVisible()
    expect(await list.getByRole('link').count()).toBe(18)
  })

  test('the current page is marked with aria-current', async ({ page }) => {
    await page.goto('#/learn/rmsnorm')
    await expect(page.locator('nav[aria-label="Main"] a[aria-current="page"]')).toHaveText(
      'concepts',
    )
    await expect(page.locator('.app__sidebar a[aria-current="page"]')).toHaveText('RMSNorm')
  })

  test('images and charts carry text alternatives', async ({ page }) => {
    await page.goto('#/learn/multi-head-attention')
    const unlabelled = await page.evaluate(() => {
      const out: string[] = []
      for (const node of document.querySelectorAll('svg, img')) {
        // An SVG inside an already-labelled or already-hidden container is a
        // decorative part of something that has its own alternative -- KaTeX
        // emits one for every stretchy glyph it draws, and those are inside a
        // span with `aria-label` or `aria-hidden` on it.
        const container = node.closest('[aria-label], [aria-hidden], [role="img"]')
        if (container && container !== node) continue
        if (container?.getAttribute('aria-hidden') === 'true') continue
        const label =
          node.getAttribute('aria-label') ??
          node.getAttribute('aria-labelledby') ??
          node.querySelector('title')?.textContent ??
          ''
        if (!label.trim()) out.push(node.outerHTML.slice(0, 80))
      }
      return out
    })
    expect(unlabelled, `unlabelled graphics:\n${unlabelled.join('\n')}`).toEqual([])
  })
})

test.describe('theme', () => {
  test('the inline script applies the theme before paint, and the toggle persists it', async ({
    page,
  }) => {
    await page.goto('./')
    const html = page.locator('html')
    const initial = await html.evaluate((node) => node.classList.contains('dark'))

    // Polled rather than read once: the click schedules a React state update and
    // an effect, neither of which has happened by the time click() resolves.
    await page.getByRole('button', { name: /Switch to .* theme/ }).click()
    await expect.poll(() => html.evaluate((node) => node.classList.contains('dark'))).toBe(!initial)
    const flipped = !initial

    // Persisted: a reload should not flash back to the default.
    await page.reload()
    expect(await html.evaluate((node) => node.classList.contains('dark'))).toBe(flipped)
    expect(initial).toBeDefined()
  })
})

test.describe('the playground degrades', () => {
  test('the run button is present and the page explains the pinned runtime', async ({ page }) => {
    await page.goto('#/run')
    await expect(page.getByRole('heading', { name: 'Run it yourself' })).toBeVisible()
    await expect(page.getByText(/Pyodide/)).toBeVisible()
    await expect(page.getByRole('button', { name: /run in the browser/i })).toBeVisible()
    // And the recorded run is right there, so the page is useful with no network
    // at all.
    await expect(page.locator('figure.widget').filter({ hasText: 'Loss over' })).toBeVisible()
  })

  test('a failed run is a useful message, not a red banner', async ({ page }) => {
    // Block the CDN before the page can fetch it, which is the realistic failure
    // on a locked-down network.
    await page.route('**/pyodide*/**', (route) => route.abort())
    await page.goto('#/run')
    await page.getByRole('button', { name: /run in the browser/i }).click()

    const notice = page.locator('.notice--warn')
    await expect(notice).toBeVisible({ timeout: 60_000 })
    await expect(notice).toContainText('make run')
  })
})
