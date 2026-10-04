import { describe, expect, it } from 'vitest'
import { err, isErr, isOk, ok, type Result } from '../src/data/types'
import { intoResult, loadCandidate, loadPatch, loadSource } from '../src/data/sources'

/**
 * The `Result` boundary the content loaders are.
 *
 * `loadSource`, `loadCandidate` and `loadPatch` each await a dynamic `?raw`
 * import, and a dynamic import is the one thing in this app that can fail at
 * run time (see `data/sources.ts` for the rest of it). They used to hand that
 * failure straight back as a rejected promise, which meant every caller had to
 * remember a `.catch` that nothing made it write -- and none of them did, so a
 * failed chunk was an unhandled rejection and a surface that stayed empty
 * forever. The loaders now resolve a `Result` instead, so failure is a value
 * the typechecker makes every caller look at.
 *
 * ## Why the failing-import test goes through `intoResult`
 *
 * The ideal test would fail the import itself, but a Vite `?raw` import cannot
 * be intercepted from a test: the plugin resolves the file to a string *before*
 * Vitest's mock registry is consulted, so neither `vi.mock` nor `vi.doMock` on
 * `'../../../reference/microgpt.py?raw'` has any effect -- the real text comes
 * back either way (verified while writing this file). That is a property of the
 * plugin, not of this code.
 *
 * So the seam is deliberate and small: every loader routes its import through
 * `intoResult`, which is the only `catch` in the file and the exact function a
 * failed chunk runs through. Injecting a loader that throws is therefore not a
 * simulation of the boundary -- it *is* the boundary, one call frame in from
 * the `import`. What that test cannot cover is the plugin's own delivery of the
 * rejection, which is noted here rather than papered over: it is the verifier's
 * integration gap.
 */
describe('the Result boundary', () => {
  it('turns a throwing loader into the failure variant, not a rejection', async () => {
    // The literal property under test: a loader that throws (what a failed
    // dynamic import is) resolves as `{ok:false}` and never rejects. If the
    // promise rejected, this test would fail with an unhandled rejection
    // instead of reaching the assertion.
    const result = await intoResult(() => Promise.reject(new Error('chunk did not arrive')))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toBe('chunk did not arrive')
  })

  it('turns a synchronous throw from the loader into the failure variant', async () => {
    const result = await intoResult(() => {
      throw new Error('thrown before the promise was built')
    })
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toBe('thrown before the promise was built')
  })

  it('turns a non-Error throw into a string message, without throwing itself', async () => {
    // A rejection carries whatever was thrown, and that is not guaranteed to be
    // an `Error`. The message helper has to cope, and must never throw.
    const result = await intoResult(() => Promise.reject('a bare string'))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toBe('a bare string')
  })

  it('passes a resolved value through as the success variant', async () => {
    const result = await intoResult(() => Promise.resolve('the source'))
    expect(result).toEqual({ ok: true, value: 'the source' })
  })
})

describe('the Result helpers', () => {
  it('constructs the success variant and narrows with isOk', () => {
    const result = ok(7)
    expect(isOk(result)).toBe(true)
    expect(isErr(result)).toBe(false)
    if (isOk(result)) expect(result.value).toBe(7)
  })

  it('constructs the failure variant and narrows with isErr', () => {
    const result = err('chunk vanished')
    expect(isErr(result)).toBe(true)
    expect(isOk(result)).toBe(false)
    // The narrowing is the point: `.error` is only reachable once `isErr` has
    // been consulted, so a caller cannot read a value that is not there.
    if (isErr(result)) expect(result.error).toBe('chunk vanished')
  })

  it('narrows to the failure variant in the else branch of an isOk guard', () => {
    // The consumers (and `LoadableResult`) branch on `isOk` and read `.error`
    // in the `else`, so the narrowing has to hold there too, not just via
    // `isErr`. If `isOk` did not carry a type predicate this would not compile:
    // `result.value` is what an un-guarded `Result` offers, and the `else` arm
    // would have no `.error` to read.
    const result: Result<number> = err('chunk vanished')
    const rendered = isOk(result) ? `ok: ${result.value}` : `error: ${result.error}`
    expect(rendered).toBe('error: chunk vanished')
  })

  it('keeps the error type defaulting to a plain string message', () => {
    const result: Result<number> = err('boom')
    expect(result.ok).toBe(false)
    if (!result.ok) expect(typeof result.error).toBe('string')
  })
})

describe('the loaders resolve a Result and never reject', () => {
  it('loadSource reports success for the reference track', async () => {
    // The `ok` branch through a real import: this is the success half of the
    // same boundary the tests above cover on the failure half.
    const result = await loadSource('python')
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.value).toContain('class Value')
  })

  it('loadCandidate treats an unknown track as an empty success, not an error', async () => {
    // There is no candidate for a track the loop has never run on. That is not a
    // failure, and keeping it a success is what keeps it distinguishable from a
    // chunk that did not arrive.
    const result = await loadCandidate('nonesuch')
    expect(result).toEqual({ ok: true, value: '' })
  })

  it('loadPatch treats an unknown run as a null success, not an error', async () => {
    // "No patch was recorded" is an honest answer the page shows. It is a
    // different state from "the chunk failed", and the two must not collapse.
    const result = await loadPatch('no-such-run')
    expect(result).toEqual({ ok: true, value: null })
  })
})
