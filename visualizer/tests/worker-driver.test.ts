import { describe, expect, it } from 'vitest'
import { buildDriver, DATASET_PATH } from '../src/worker/driver'

/**
 * The Python driver the playground hands to Pyodide.
 *
 * Two bugs shipped in this string, and no test noticed, because the only e2e
 * that touched the playground aborted the Pyodide CDN request before the driver
 * ever ran:
 *
 *  1. it `open('/microgpt.py')`ed a file the worker never wrote, so every real
 *     run raised `FileNotFoundError: [Errno 44] /microgpt.py`; and
 *  2. even once fixed, the reference resolves `input.txt` from the *current
 *     working directory* (docs/KNOWN-ISSUES.md #3), while the worker wrote
 *     `/input.txt` -- so the check was false and the reference tried to download
 *     the dataset, failing with `unknown url type: https` in a worker.
 *
 * These are the properties that were silently false. They hold against the pure
 * `(source, steps, dataPath) -> string` helper, with no worker, no DOM and no
 * 12 MB download.
 */
describe('buildDriver', () => {
  const source =
    'num_steps = 1000 # number of training steps\nfor step in range(num_steps):\n    pass\n'

  it('embeds the exact source it was given', () => {
    // The contract with `RunPlayground` is that the run is the source the reader
    // sees in the editor -- not a re-loaded reference, not a file on disk.
    const driver = buildDriver(source, 200, DATASET_PATH)
    expect(driver).toContain(JSON.stringify(source))
  })

  it('never reads the reference source from a file', () => {
    // The whole class of bug: a read path that can disagree with the write path.
    // There is no write path for the *source* any more, so there must be no read.
    const driver = buildDriver(source, 200, DATASET_PATH)
    expect(driver).not.toContain("open('/microgpt.py'")
    expect(driver).not.toContain('open("/microgpt.py"')
  })

  it('chdirs so the reference’s CWD-relative input.txt resolves to the write path', () => {
    // The dataset bug: the worker writes `DATASET_PATH`; the reference reads
    // `input.txt` from the CWD. The driver must reconcile the two, and it must do
    // so from the same path it was handed.
    const driver = buildDriver(source, 200, '/custom/input.txt')
    expect(driver).toContain('os.chdir(os.path.dirname(_dataset))')
    expect(driver).toContain(JSON.stringify('/custom/input.txt'))
    // And a missing dataset is loud, not a silent download of a mutable URL.
    expect(driver).toContain('raise FileNotFoundError')
  })

  it('patches num_steps to the requested value, with no unfilled placeholder', () => {
    // The latent bug behind the first: the replacement target was the literal
    // `"num_steps = %d"` and the `%d` was never interpolated -- so even once the
    // file existed, the patched source would have been a `ValueError`. A `%d`
    // anywhere in the driver is that bug.
    const driver = buildDriver(source, 200, DATASET_PATH)
    expect(driver).toContain('"num_steps = 200"')
    expect(driver).not.toContain('%d')

    expect(buildDriver(source, 37, DATASET_PATH)).toContain('"num_steps = 37"')
  })

  it('compiles under a stable filename for readable tracebacks', () => {
    expect(buildDriver(source, 200, DATASET_PATH)).toContain(
      'compile(patched, "<microgpt.py>", "exec")',
    )
  })

  it('streams per-step progress as JSON the main thread can parse', () => {
    // The `step` message protocol (`protocol.ts`) depends on this exact prefix.
    expect(buildDriver(source, 200, DATASET_PATH)).toContain('"__ATLAS_STEP__"')
  })
})
