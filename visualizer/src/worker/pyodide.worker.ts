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
    py.FS.writeFile('/input.txt', request.dataset)

    if (cancelled) return

    post({ type: 'status', stage: 'compiling', detail: `compiling reference/microgpt.py` })
    const instrumented = request.source

    /*
     * Run with a reduced step count.
     *
     * The reference is not edited. `num_steps` is 1000 in the file; in a browser
     * that is a minute of frozen-tab, so the string is replaced *in the source
     * text we hand to Python*, which is visible to the reader in the "what
     * actually ran" box below. This is a playground, not a second reference.
     */
    const steps = Math.max(1, Math.min(request.steps, 2000))
    const driver = `
import sys, io, time, json, contextlib

ns = {"__name__": "__main__"}
source = open(${JSON.stringify('/microgpt.py')}, encoding="utf-8").read()
patched = source.replace("num_steps = 1000", "num_steps = %d")
if patched == source:
    print("__ATLAS_NOTE__num_steps was not 1000; nothing patched")
open("/microgpt.py", "w", encoding="utf-8").write(patched)
code = compile(patched, "/microgpt.py", "exec")

# Stream progress: the training loop prints one line per step, and stdout in
# Pyodide is a Python file object we can wrap.
class _Tee(io.StringIO):
    def __init__(self, real):
        super().__init__()
        self.real = real
        self.last = 0.0
    def write(self, s):
        self.real.write(s)
        if s.startswith("step ") and "|" in s:
            try:
                step = int(s.split()[1])
                loss = float(s.split("loss")[1])
                print("__ATLAS_STEP__" + json.dumps({"step": step, "loss": loss}))
            except Exception:
                pass
        return len(s)

sys.stdout = _Tee(sys.stdout)
exec(code, ns)
sys.stdout = sys.stdout.real
`
    post({ type: 'status', stage: 'running', detail: `training for ${steps} steps` })
    const output = (await py.runPythonAsync(driver)) as string
    void instrumented

    if (cancelled) {
      post({ type: 'error', message: 'Cancelled', recoverable: true })
      return
    }

    // Samples come back through stdout, so they are read out of the captured text.
    const samples = [...output.matchAll(/^sample\s+\d+:\s*(.*)$/gm)].map((m) => m[1] ?? '')

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
