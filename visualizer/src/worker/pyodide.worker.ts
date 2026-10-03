/// <reference lib="webworker" />
/**
 * The worker half of the browser playground.
 *
 * Kept deliberately small: load Pyodide, write the dataset into its virtual
 * filesystem, run the reference, stream progress back. Everything it sends is a
 * discriminated `WorkerResponse`, so the main thread can render an exhaustive
 * switch and the typechecker will complain if a case is added here but not there.
 *
 * `loadPyodide` is imported dynamically so the ~12 MB runtime is only fetched if
 * someone actually presses Run. It is pinned; see `protocol.ts` for why that is
 * not negotiable.
 */

import type { WorkerRequest, WorkerResponse } from './protocol'
import { buildDriver, DATASET_PATH } from './driver'

const PYODIDE_VERSION = '0.26.4'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

interface Pyodide {
  runPythonAsync: (code: string) => Promise<unknown>
  FS: { writeFile: (path: string, data: string) => void; readFile: (path: string) => string }
  loadPackage: (name: string) => Promise<void>
}

let pyodide: Pyodide | null = null
let cancelled = false

function post(message: WorkerResponse) {
  ;(self as unknown as Worker).postMessage(message)
}

async function getPyodide(): Promise<Pyodide> {
  if (pyodide) return pyodide
  post({
    type: 'status',
    stage: 'loading',
    detail: `fetching Pyodide ${PYODIDE_VERSION} (~12 MB, once)`,
  })
  // Pinned version, pinned URL. An unpinned runtime means the same URL can serve
  // different code tomorrow.
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs` as string)
  pyodide = (await loadPyodide({ indexURL: PYODIDE_URL })) as Pyodide
  return pyodide
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelled = true
    return
  }
  if (request.type !== 'run') return

  cancelled = false
  const started = performance.now()
  try {
    const py = await getPyodide()

    if (cancelled) return

    /*
     * The dataset, written into Pyodide's virtual filesystem before the
     * reference runs.
     *
     * This is the whole reason the playground is hermetic. The reference does
     * `if not os.path.exists('input.txt')` and, if it is missing, downloads it
     * from a mutable URL. Writing the file first short-circuits that check, so
     * nothing is fetched and the run is exactly the committed one.
     */
    post({
      type: 'status',
      stage: 'writing',
      detail: 'writing data/input.txt into the virtual filesystem',
    })
    py.FS.writeFile(DATASET_PATH, request.dataset)

    if (cancelled) return

    post({ type: 'status', stage: 'compiling', detail: `compiling reference/microgpt.py` })

    /*
     * Run with a reduced step count. The reference is not edited; the single
     * substitution happens inside `buildDriver`, on the source text we hand to
     * Python (which the reader can see in the editor).
     */
    const steps = Math.max(1, Math.min(request.steps, 2000))
    const driver = buildDriver(request.source, steps, DATASET_PATH)
    post({ type: 'status', stage: 'running', detail: `training for ${steps} steps` })
    const raw = await py.runPythonAsync(driver)

    if (cancelled) {
      post({ type: 'error', message: 'Cancelled', recoverable: true })
      return
    }

    /*
     * Samples come back through stdout: the driver's last expression is the text
     * it captured. If that is not a string, the driver and this reader have
     * drifted apart -- which is the exact class of bug that hid here for so long
     * (a `TypeError: ... matchAll of undefined` reads as a mysterious crash).
     * Say so plainly instead of throwing.
     */
    if (typeof raw !== 'string') {
      post({
        type: 'error',
        message: `the run produced no captured output (got ${typeof raw}); this is a bug in the playground driver`,
        recoverable: false,
      })
      return
    }
    const samples = [...raw.matchAll(/^sample\s+\d+:\s*(.*)$/gm)].map((m) => m[1] ?? '')

    post({
      type: 'done',
      wallSeconds: (performance.now() - started) / 1000,
      samples,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    post({
      type: 'error',
      // A network failure is recoverable in the sense that it is worth telling the
      // reader to try again; a bug in the reference is not.
      recoverable: /fetch|network|Failed to fetch|NetworkError/i.test(message),
      message,
    })
  }
}
