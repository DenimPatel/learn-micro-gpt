import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { trace } from '../src/data/sources'
import { AttentionHeatmap, SamplesList, SoftmaxBars } from '../src/components/widgets'

/**
 * The trace widgets' empty state.
 *
 * ## Why this exists, and why the e2e case could not cover it
 *
 * The e2e suite had a case named "a trace widget for a step that was not
 * recorded says so" whose assertion was `expect(page.locator('body'))
 * .toContainText('concept')` -- a substring present on *every* page of the site,
 * so it passed no matter what the widgets did. The real empty state was asserted
 * nowhere, which is how `SoftmaxBars` came to `if (!row) return null` (rendering
 * nothing at all) and `SamplesList` to render an empty `<ul>` for an unrecorded
 * step.
 *
 * A unit test is the honest fix: it renders the real component with a
 * deliberately unrecorded combination and inspects the markup. It cannot be a
 * false-positive substring match, because the assertions are on the empty-state
 * class and on the argument values -- both of which are *absent* from the drawn
 * widget. It is deterministic: no build, no server, no navigation.
 *
 * `renderToStaticMarkup`, not a DOM: `vitest.config.ts` is `environment: 'node'`
 * on purpose, matching `tests/markdown.test.ts` and `tests/error-boundary.test.tsx`.
 *
 * ## Derived from the trace, never hardcoded
 *
 * The recorded combinations are read out of `trace`, and the test then asks for
 * one that is provably *not* in that set. If the trace were re-recorded and
 * gained the combination, the "recorded set is not empty" / "this one is
 * unrecorded" assertions would catch it rather than the test silently exercising
 * the drawn branch.
 */

const attnCombos = new Set(trace.attn.map((r) => `${r.step}/${r.pos}/${r.head}`))
const probComboKeys = new Set(trace.probs.map((r) => `${r.step}/${r.pos}`))
const sampleSteps = [...new Set(trace.samples.map((s) => s.step))].sort((a, b) => a - b)

/** A (step, pos, head) the trace provably does not contain. */
function unrecordedAttnCombo(): { step: number; pos: number; head: number } {
  for (const step of [42_424, 42_425, 42_426]) {
    for (const pos of [99, 98, 97]) {
      for (const head of [7, 8, 9]) {
        if (!attnCombos.has(`${step}/${pos}/${head}`)) return { step, pos, head }
      }
    }
  }
  throw new Error('every candidate combination is somehow recorded')
}

/**
 * Render `SamplesList` against a stand-in `trace.samples` and restore it.
 *
 * `SamplesList` takes no props -- its only state is the selected step, seeded
 * from `trace.samples` -- so a branch that depends on the *contents* of that
 * list is only reachable by changing the list. The committed trace is restored
 * in `finally`, and every test that uses this either asserts the real list first
 * or restores before the next test runs.
 */
function renderWithSamples(rows: typeof trace.samples): string {
  const original = trace.samples
  try {
    trace.samples = rows
    return renderToStaticMarkup(createElement(SamplesList))
  } finally {
    trace.samples = original
  }
}

describe('AttentionHeatmap empty state', () => {
  it('names the args it was given and the combinations that exist', () => {
    expect(attnCombos.size, 'the committed trace has attention rows').toBeGreaterThan(0)
    const { step, pos, head } = unrecordedAttnCombo()
    expect(attnCombos.has(`${step}/${pos}/${head}`)).toBe(false)

    const html = renderToStaticMarkup(createElement(AttentionHeatmap, { step, pos, head }))

    // The empty-state class, which the drawn heatmap never carries.
    expect(html).toContain('widget--empty')
    expect(html).not.toContain('class="heatmap"')
    // The args as given, so the reader knows what was asked for.
    expect(html).toContain(`step ${step}`)
    expect(html).toContain(`position ${pos}`)
    expect(html).toContain(`head ${head}`)
    // And a combination that *does* exist, derived from the trace's own rows.
    expect(html).toContain([...attnCombos][0]!)
  })
})

describe('SoftmaxBars empty state', () => {
  it('renders a labelled empty state instead of nothing for an unrecorded (step, pos)', () => {
    const step = 42_424
    expect(probComboKeys.size, 'the committed trace has probability rows').toBeGreaterThan(0)
    expect(trace.probs.some((r) => r.step === step)).toBe(false)

    const html = renderToStaticMarkup(createElement(SoftmaxBars, { step, pos: 0 }))

    // The regression this pins: `if (!row) return null` rendered the empty
    // string, so the page silently lost the widget's whole section.
    expect(html).not.toBe('')
    expect(html).toContain('widget--empty')
    expect(html).not.toContain('class="bars"')
    // The args it was given.
    expect(html).toContain(`step ${step}`)
    expect(html).toContain('position 0')
    // And a (step, pos) pair that does exist, derived from the trace.
    expect(html).toContain([...probComboKeys][0]!)
  })

  it('still draws the bars when the row is recorded', () => {
    // The non-hostile half: the guard must not swallow a real row. Without this,
    // a widget that always rendered the empty state would pass above.
    const row = trace.probs[0]!
    const html = renderToStaticMarkup(createElement(SoftmaxBars, { step: row.step, pos: row.pos }))
    expect(html).toContain('class="bars"')
    expect(html).not.toContain('widget--empty')
  })
})

describe('SamplesList empty state', () => {
  it('draws the recorded samples for the committed trace', () => {
    expect(sampleSteps.length, 'the committed trace has samples').toBeGreaterThan(0)
    const html = renderToStaticMarkup(createElement(SamplesList))
    expect(html).toContain('class="samples"')
    expect(html).not.toContain('widget--empty')
    // Every recorded step is offered by the select, read from the trace's rows.
    for (const step of sampleSteps) expect(html).toContain(`value="${step}"`)
  })

  it('names the step and the steps that do have samples when the selected step has none', () => {
    /*
     * The branch that shipped an empty `<ul>`: rows exist in the trace, but not
     * at the step the widget is pointing at. That is exactly what a
     * `::trace kind=samples step=…` naming a step without samples produces, so
     * the widget takes the step as an optional prop and this covers the missed
     * step against the *committed* trace -- no mutation, no stand-in data.
     */
    expect(sampleSteps.length, 'the committed trace has samples').toBeGreaterThan(0)
    const missing = Math.max(...sampleSteps) + 1000
    expect(sampleSteps).not.toContain(missing)

    const html = renderToStaticMarkup(createElement(SamplesList, { step: missing }))

    expect(html).toContain('widget--empty')
    expect(html).not.toContain('class="samples"')
    // The step it was given, and the steps that do have samples.
    expect(html).toContain(`No samples were recorded for step ${missing}`)
    for (const step of sampleSteps) expect(html).toContain(String(step))
  })

  it('says so when the trace records no samples at all', () => {
    /*
     * The default is `trace.samples[0]?.step ?? 1000`; with no samples at all
     * the select is empty and `rows` is empty. Before the guard that rendered an
     * empty `<ul>` inside a `<figure>` whose caption promises twenty samples --
     * a lie by omission. The empty state is the only thing standing between a
     * reader and a caption with nothing under it.
     */
    const html = renderWithSamples([])
    expect(html).toContain('widget--empty')
    expect(html).toContain('No samples were recorded in this trace at all.')
    expect(html).not.toContain('class="samples"')
    // The figure's caption is still there, so the page keeps its shape.
    expect(html).toContain('class="widget__caption"')
  })
})
