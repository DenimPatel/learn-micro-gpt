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

  test('the reading pages talk to no host but their own', async ({ page }) => {
    // Stronger than "every asset request is relative" above: that one only
    // inspects requests to the site's own host. This one fails if *any* request
    // leaves the origin, so an analytics tag, a font CDN or a tag manager added
    // to the shell or a component is caught here. The site's stated value is that
    // its numbers are reproducible, and an unnoted third-party request at boot
    // breaks that promise in a way nothing else in the suite would notice.
    //
    // The Pyodide CDN is the one deliberately allowed third-party dependency,
    // but it is lazy -- it loads only when a reader runs the playground, and
    // neither of the two routes below does that. So on these routes the allowed
    // third-party set is empty.
    const foreign: string[] = []
    page.on('request', (request) => {
      const url = new URL(request.url())
      // Data/blob URLs are not a host at all; the SVG favicon and inline
      // workers arrive this way and never leave the machine.
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return
      if (url.hostname !== '127.0.0.1') foreign.push(`${request.method()} ${url.href}`)
    })

    for (const route of ['#/', '#/learn/softmax']) {
      await page.goto(route)
      await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
    }
    // Let the lazy chunks (graph layout, first code panel) request before we
    // read the list.
    await page.waitForTimeout(1200)

    expect(
      foreign,
      `the reading pages must not contact a third-party host:\n${foreign.join('\n')}`,
    ).toEqual([])
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

  test('the code is actually highlighted, not silently falling back', async ({ page }) => {
    /*
     * `CodePanel` falls back to an unhighlighted `<pre><code>` when highlighting
     * fails, so a broken highlighter renders a page that *looks* fine. That is
     * exactly what happened: the Shiki themes were registered under
     * `undefined-transparent` because a dynamic import of a theme module gives
     * the namespace object rather than the theme, and every `codeToHtml` threw.
     * A screenshot of the working-looking result was read as the fix working.
     *
     * So this asserts the *absence* of the fallback, not the presence of a <pre>.
     */
    const failures: string[] = []
    page.on('console', (message) => {
      if (message.type() === 'error') failures.push(message.text())
    })
    page.on('pageerror', (error) => failures.push(error.message))

    await page.goto('#/learn/softmax')
    const pre = page.locator('.code-panel pre').first()
    await expect(pre).toBeVisible()
    // Shiki marks its output, and wraps every token in a coloured span.
    await expect(pre).toHaveClass(/shiki/)
    expect(await pre.locator('span[style*="color"]').count()).toBeGreaterThan(10)
    // The fallback's marker text must not be present.
    await expect(pre).not.toContainText('highlighting…')
    expect(failures.join('\n')).not.toMatch(/ShikiError|not found/)
  })

  test('the code panel background comes from the design tokens', async ({ page }) => {
    /*
     * Shiki writes an inline `style="background-color:#24292e"` onto the <pre>
     * it returns, and an inline style beats any stylesheet -- so left alone a code
     * panel is an off-grey box in a tokenised page. The themes are registered with
     * `bg: 'transparent'` so the panel's own background applies.
     */
    for (const dark of [false, true]) {
      await page.goto(dark ? '#/learn/softmax' : '#/learn/softmax')
      await page.evaluate((on) => {
        document.documentElement.classList.toggle('dark', on)
        localStorage.setItem('atlas:theme', on ? 'dark' : 'light')
      }, dark)
      await page.reload()
      /*
       * `pre.shiki:visible`, not `.code-panel pre`.
       *
       * CodePanel renders Shiki's output twice, once in a `.light-only` span and
       * once in a `.dark-only` span, and hides the one that does not match the
       * theme. So a plain `.code-panel pre` .first() resolves to whichever copy
       * comes first in the DOM -- the *light* one -- which is `display: none` in
       * dark mode, and the test failed there intermittently depending on whether
       * highlighting had arrived inside the timeout.
       *
       * `:visible` picks the copy actually on screen, and `.shiki` keeps the
       * assertion pointed at the element that has Shiki's inline background,
       * which is the whole point: the plain-text fallback before highlighting
       * arrives carries no inline background and would pass vacuously.
       */
      const pre = page.locator('.code-panel pre.shiki:visible').first()
      await expect(pre, `Shiki never rendered a visible code panel (dark=${dark})`).toBeVisible()
      const background = await pre.evaluate((node) => getComputedStyle(node).backgroundColor)
      const page_ = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
      expect(background, `dark=${dark}: pre ${background} vs body ${page_}`).not.toBe(
        'rgb(36, 41, 46)',
      )
      expect(background, `dark=${dark}: pre ${background} vs body ${page_}`).not.toBe(
        'rgb(255, 255, 255)',
      )
    }
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

  test('the softmax trace panel renders, outside the callout above it', async ({ page }) => {
    // `softmax.md`'s callout used to be missing its closer, so the `:::trace`
    // below it was swallowed into the callout body and rendered as a line of
    // literal colons instead of a widget. A rendered `.bars` figure is the
    // observable difference between the two.
    await page.goto('#/learn/softmax')
    const bars = page.locator('figure.widget .bars')
    await expect(bars).toHaveCount(1)
    await expect(bars.locator('.bars__row')).toHaveCount(8)
  })

  test('the loss chart is labelled with the measured noise', async ({ page }) => {
    await page.goto('./')
    const chart = page.locator('figure.widget').filter({ hasText: 'Loss over' }).first()
    await expect(chart).toBeVisible()
    await expect(chart).toContainText('500')
    await expect(chart.locator('svg')).toHaveAttribute('aria-label', /1000 steps/)
  })

  test('every trace widget on a concept page has the rows it claims to draw', async ({ page }) => {
    /*
     * This case used to be a tautology. It was named "a trace widget for a step
     * that was not recorded says so" and asserted `expect(page.locator('body'))
     * .toContainText('concept')` -- a substring every page of the site contains,
     * so it passed whatever the widgets did. `SoftmaxBars` could `return null`
     * and `SamplesList` could render an empty `<ul>` and this still went green.
     *
     * The empty state itself is now pinned where it can be produced
     * deterministically: `tests/widgets.test.tsx` renders each widget with a
     * deliberately unrecorded (step, pos, head) and asserts the empty-state
     * class and the named args. Every committed `::trace` names a *recorded*
     * combination, so no page in the built site can reach the empty branch --
     * which is why an e2e case here could only ever be a weaker duplicate.
     *
     * What a browser *can* check is the property the tautology was standing in
     * for: that each trace widget on a concept page drew real rows, and that the
     * `widget--empty` figure appears only where the data really is absent (the
     * step-1000 samples widget, which has its own page). A widget that silently
     * rendered nothing would leave its figure without its drawn content, and the
     * assertion below would fail.
     */
    await page.goto('#/learn/multi-head-attention')
    const heatmaps = page.locator('.heatmap')
    await expect(heatmaps).toHaveCount(2)
    await expect(page.locator('figure.widget--empty')).toHaveCount(0)

    await page.goto('#/learn/softmax')
    await expect(page.locator('figure.widget .bars .bars__row')).toHaveCount(8)
    await expect(page.locator('figure.widget--empty')).toHaveCount(0)

    await page.goto('#/learn/inference-sampling')
    // The samples widget is the one that names a step with rows: it must draw
    // its `<ul class="samples">`, not a labelled empty state.
    await expect(page.locator('figure.widget ul.samples li').first()).toBeVisible()
    await expect(page.locator('figure.widget--empty')).toHaveCount(0)
  })
})

test.describe('accessibility', () => {
  test('the skip link is the first tabbable element and it works', async ({ page }) => {
    // From a *concept* page, not `./`. The skip link used to put `#main` in the
    // URL, and `parseRoute('#main')` falls through to `home` -- so activating it
    // here navigated the reader away from the concept they were reading. From
    // home that loss was invisible, which is why the old assertion started at
    // `./` and passed for the wrong reason (IMP-049).
    await page.goto('#/learn/softmax')
    await expect(page.getByRole('heading', { name: 'Softmax' })).toBeVisible()

    await page.keyboard.press('Tab')
    const focused = page.locator(':focus')
    await expect(focused).toHaveClass(/skip-link/)

    await focused.press('Enter')

    // (a) The route did not change: the fragment must not replace the hash.
    await expect(page).toHaveURL(/#\/learn\/softmax/)
    expect(await page.evaluate(() => window.location.hash)).toMatch(/^#\/learn\/softmax/)
    // ...and the concept is still the page being read.
    await expect(page.getByRole('heading', { name: 'Softmax' })).toBeVisible()

    // (b) Focus moved into the main region.
    await expect(page.locator('#main')).toBeFocused()
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

  test('a run in progress draws the loss falling, step by step', async ({ page }) => {
    // The real worker needs a 12 MB CDN download and a minute of CPU, so the
    // suite stubs `Worker` itself rather than the network: the test is about the
    // main thread's rendering, and the protocol between the two is the only thing
    // that matters here. This is the one place the test reaches past the page's
    // own API, and it is deliberate -- without it the live chart is untested,
    // because no test environment can run the real thing quickly or offline.
    await page.addInitScript(() => {
      class StubWorker {
        onmessage: ((event: MessageEvent) => void) | null = null
        postMessage(request: { type: string }) {
          if (request.type !== 'run') return
          const emit = (data: unknown) => this.onmessage?.({ data } as MessageEvent)
          emit({ type: 'status', stage: 'running', detail: 'stub run' })
          // Paced, not synchronous: the page only draws the live chart while the
          // run is in flight, so a stub that finished instantly would leave an
          // empty chart and assert nothing. 100 steps at 10 ms is one second of a
          // run that never ends -- `done` is deliberately never sent, so the chart
          // stays on screen for the assertions below.
          let step = 0
          const timer = setInterval(() => {
            step += 1
            emit({ type: 'step', step, total: 200, loss: 3.2 - step * 0.01 })
            if (step >= 100) clearInterval(timer)
          }, 10)
        }
        terminate() {}
      }
      Object.defineProperty(window, 'Worker', { value: StubWorker, configurable: true })
    })
    await page.goto('#/run')
    await page.getByRole('button', { name: /run in the browser/i }).click()

    // Scoped to the live figure: the recorded chart below carries an aria-label
    // starting "Loss over" too, and would otherwise satisfy this.
    const chart = page.locator('.widget--live svg[aria-label*="Loss over"]')
    await expect(chart).toBeVisible()
    // The run is held open mid-flight, so the label says "so far" and counts the
    // steps that have arrived -- 100, where the stub stops.
    await expect(chart).toHaveAttribute('aria-label', /Loss over 100 steps so far/)
    // Two curves: the raw per-step loss and the smoothed one.
    await expect(chart.locator('path')).toHaveCount(2)
    // The whole point of the item: the caption reads as a live update, not a
    // final summary.
    await expect(page.getByText(/100 steps so far, now at loss 2\.2000/)).toBeVisible()
  })
})

test.describe('scroll on navigation', () => {
  test('navigating to the next concept lands at the top, and an in-page anchor does not', async ({
    page,
  }) => {
    await page.goto('#/learn/softmax')
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()

    // Read to the bottom of the page, the way a reader who finishes a concept
    // does, then take the "next" link.
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
    expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0)

    await page.getByRole('link', { name: /next/i }).click()
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible()

    // Polled: click() resolving only guarantees the event dispatched, not that
    // React's route effect has run and applied the scroll reset.
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await expect
      .poll(() =>
        page
          .getByRole('heading', { level: 1 })
          .evaluate(
            (el) =>
              el.getBoundingClientRect().top >= 0 &&
              el.getBoundingClientRect().top < window.innerHeight,
          ),
      )
      .toBe(true)

    // A same-page table-of-contents link still scrolls to its section instead of
    // being reset to the top.
    const toc = page.locator('.toc a').first()
    await toc.click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(0)
  })

  test('clicking next twice moves two concepts forward, not one', async ({ page }) => {
    // The regression this exists for: `navigate()` once bailed on
    // `window.location.hash === href`, so a click whose target href already
    // matched the hash did nothing and the second "next" looked dead. Asserting
    // the *title* after each click rather than the hash is deliberate -- the hash
    // is what the bug compared and what a stale test would agree with.
    await page.goto('#/learn/softmax')
    await expect(page.getByRole('heading', { level: 1, name: 'Softmax' })).toBeVisible()

    await page.getByRole('link', { name: /next/i }).click()
    await expect(page.getByRole('heading', { level: 1, name: 'RMSNorm' })).toBeVisible()

    await page.getByRole('link', { name: /next/i }).click()
    await expect(page.getByRole('heading', { level: 1, name: 'Embedding' })).toBeVisible()
    expect(new URL(page.url()).hash).toBe('#/learn/embedding')
  })

  test('clicking previous twice moves two concepts back', async ({ page }) => {
    await page.goto('#/learn/multi-head-attention')
    await expect(
      page.getByRole('heading', { level: 1, name: 'Multi-head attention' }),
    ).toBeVisible()

    await page.getByRole('link', { name: /previous/i }).click()
    await expect(page.getByRole('heading', { level: 1, name: 'Embedding' })).toBeVisible()

    await page.getByRole('link', { name: /previous/i }).click()
    await expect(page.getByRole('heading', { level: 1, name: 'RMSNorm' })).toBeVisible()
    expect(new URL(page.url()).hash).toBe('#/learn/rmsnorm')
  })
})

test.describe('no horizontal overflow on a phone', () => {
  /**
   * A 390px viewport is an iPhone 12/13/14 in CSS pixels -- the narrowest a
   * reader is likely to hold. The document must fit it: if any element's
   * intrinsic width exceeds the viewport, `scrollWidth` grows past `clientWidth`
   * and the whole page pans sideways, so the reader loses the gutter and the
   * layout reads as broken.
   *
   * This is the assertion #/research was missing. Its experiment `<select>`
   * listed a long `<option>` per run, and a native `<select>` sizes itself to
   * its widest option; unconstrained inside a flex `.widget__controls` it made
   * the document 1212px wide at a 390px viewport. Nothing checked, so nothing
   * failed -- the page just looked wrong on a phone.
   *
   * It checks `documentElement.scrollWidth`, not a particular element, because
   * the failure mode is always "the whole page is wider than the window" and any
   * of a dozen causes can produce it. The route list is the primary navigation,
   * so a new route that overflows is a new route that has to fix itself.
   */
  const PHONE = { width: 390, height: 844 }
  const ROUTES = [
    ['home', './'],
    ['concepts', '#/learn'],
    ['explore', '#/explore'],
    ['compare', '#/compare'],
    ['research', '#/research'],
    ['run', '#/run'],
    ['about', '#/about'],
    ['a concept', '#/learn/softmax'],
  ] as const

  for (const [name, route] of ROUTES) {
    test(`${name} does not scroll sideways at 390px`, async ({ page }) => {
      await page.setViewportSize(PHONE)
      await page.goto(route)
      // Let the route render and any charts settle before measuring.
      await expect(page.locator('main#main')).toBeVisible()
      const overflow = await page.evaluate(() => {
        const root = document.documentElement
        return { scrollWidth: root.scrollWidth, clientWidth: root.clientWidth }
      })
      expect(
        overflow.scrollWidth,
        `${name} (${route}) is ${overflow.scrollWidth}px wide at a ${overflow.clientWidth}px viewport`,
      ).toBeLessThanOrEqual(overflow.clientWidth)
    })
  }
})
