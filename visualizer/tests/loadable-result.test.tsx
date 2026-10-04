import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { LoadableResult } from '../src/components/LoadableResult'
import { err, ok } from '../src/data/types'

/**
 * `LoadableResult` is the one place the three states of an async load are drawn.
 *
 * The point of it is that a failed chunk and a chunk still in flight are
 * *different* renders: before this, both were an empty surface (or an ellipsis),
 * which is how a failed chunk could sit there looking like a slow one forever.
 * These cases pin the three states apart and, in the error case, assert the
 * `Result` actually narrows to its failure variant inside the component -- the
 * `else` of an `isOk` guard has to expose `.error`, or this does not compile.
 *
 * `vitest.config.ts` is `environment: 'node'`, so this renders to static markup
 * rather than mounting (the same idiom as `error-boundary.test.tsx`).
 */

const noop = (value: string) => <span>{value}</span>

describe('LoadableResult', () => {
  it('renders the loading state while the result is undefined', () => {
    const html = renderToStaticMarkup(
      <LoadableResult result={undefined} loading="loading the source…" error={(e) => e}>
        {noop}
      </LoadableResult>,
    )
    expect(html).toContain('role="group"')
    expect(html).toContain('aria-label="loading"')
    expect(html).toContain('loading the source…')
  })

  it('renders the success value once the result is ok', () => {
    const html = renderToStaticMarkup(
      <LoadableResult result={ok('the source')} loading="loading…" error={(e) => e}>
        {noop}
      </LoadableResult>,
    )
    expect(html).toContain('the source')
    expect(html).not.toContain('role="alert"')
  })

  it('renders the failure variant, narrowing to the error branch', () => {
    const html = renderToStaticMarkup(
      <LoadableResult result={err('the chunk vanished')} loading="loading…" error={(e) => e}>
        {noop}
      </LoadableResult>,
    )
    expect(html).toContain('role="alert"')
    // `.error` is only readable because the component's `isOk` guard narrowed
    // the `Result` to its failure variant in the `else`.
    expect(html).toContain('the chunk vanished')
  })
})
