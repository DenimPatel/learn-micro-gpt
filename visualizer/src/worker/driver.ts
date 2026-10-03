/**
 * The Python driver the playground hands to Pyodide.
 *
 * Deliberately a separate module from `pyodide.worker.ts`: this file has **no
 * side effects** (the worker's file, by contrast, assigns `self.onmessage` at
 * import time), so it can be imported by a plain unit test with no worker, no
 * DOM and no 12 MB Pyodide download. It is a pure `(source, steps) -> string`.
 *
 * ## Why the source is compiled from a string, not read from a file
 *
 * The driver used to `open('/microgpt.py')` and compile that, but the worker
 * only ever wrote `/input.txt` into Pyodide's virtual filesystem, so the `open`
 * raised `FileNotFoundError: [Errno 44] /microgpt.py` on every *real* run.
 * No test noticed, because the only e2e touching the playground aborted the
 * Pyodide CDN request before the driver ever ran.
 *
 * The source arrives in the `run` message (see `protocol.ts`); there was never
 * a reason to round-trip it through a file. Compiling the string directly means
 * there is exactly one source of truth and **no read path that can disagree
 * with a write path** -- the bug is structurally impossible now, not merely
 * fixed. `tests/worker-driver.test.ts` asserts both properties.
 *
 * ## The one edit
 *
 * `num_steps` is 1000 in the reference; a browser tab frozen for a minute is a
 * bad way to learn, so it is replaced with the requested value *in the source
 * text we hand to Python*. The reader can see the result in the editor. This is
 * a playground, not a second reference.
 *
 * ## The dataset path
 *
 * The worker writes the dataset to `DATASET_PATH`; the driver `os.chdir`s to its
 * directory so the reference's CWD-relative `os.path.exists('input.txt')` finds
 * it. Write path and read path come from the same constant, so they cannot drift
 * apart -- which is the original `open('/microgpt.py')` bug, one layer down.
 */
export const DATASET_PATH = '/input.txt'

/**
 * The prefix the driver prints before each step's loss, as a JSON object.
 *
 * The worker reads stdout **while the run happens** (Pyodide's
 * `setStdout({ batched })`) and turns each such line into a `step` message, which
 * is what the live loss curve is drawn from. It is exported so the parser below
 * and the driver string cannot disagree about the marker.
 */
export const STEP_MARKER = '__ATLAS_STEP__'

/**
 * Parse one stdout line into a step event, or `null` if it is an ordinary line.
 *
 * Pure and exported so the worker's streaming path is unit-testable without a
 * worker, a DOM or a Pyodide download. The `step`/`loss` type checks matter: the
 * chart does arithmetic on them, so a malformed marker must be *dropped*, not
 * forwarded as `NaN` and silently drawn as a flat line.
 */
export function parseStepLine(line: string): { step: number; loss: number } | null {
  const at = line.indexOf(STEP_MARKER)
  if (at < 0) return null
  const json = line.slice(at + STEP_MARKER.length).trim()
  let payload: unknown
  try {
    payload = JSON.parse(json)
  } catch {
    return null
  }
  if (payload === null || typeof payload !== 'object') return null
  const { step, loss } = payload as { step?: unknown; loss?: unknown }
  if (typeof step !== 'number' || typeof loss !== 'number') return null
  // `NaN`/`Infinity` survive `typeof === 'number'` but are not drawable.
  if (!Number.isFinite(step) || !Number.isFinite(loss)) return null
  return { step, loss }
}

export function buildDriver(source: string, steps: number, dataPath: string): string {
  return `
import os, sys, io, json

# The reference resolves \`input.txt\` from the **current working directory**
# (see docs/KNOWN-ISSUES.md #3), not from a fixed path. The worker writes the
# dataset at an absolute path before this runs; chdir to its directory so the
# reference's \`os.path.exists('input.txt')\` finds it. Without this the check is
# false, the reference downloads the dataset from a mutable URL -- which fails
# in a worker anyway ("unknown url type: https").
_dataset = ${JSON.stringify(dataPath)}
os.chdir(os.path.dirname(_dataset))
if not os.path.exists(os.path.basename(_dataset)):
    # Loud, not silent: a mismatch between the worker's write path and the
    # reference's read path is the exact bug this file exists to make impossible.
    raise FileNotFoundError(
        "the browser run staged no dataset at " + _dataset + " (cwd=" + os.getcwd() + ")"
    )

ns = {"__name__": "__main__"}
source = ${JSON.stringify(source)}
patched = source.replace("num_steps = 1000", ${JSON.stringify(`num_steps = ${steps}`)})
if patched == source:
    print("__ATLAS_NOTE__num_steps was not 1000; nothing patched")
code = compile(patched, "<microgpt.py>", "exec")

# Stream progress: the training loop prints one line per step, and stdout in
# Pyodide is a Python file object we can wrap.
class _Tee(io.StringIO):
    def __init__(self, real):
        super().__init__()
        self.real = real
        self.last = 0.0
    def write(self, s):
        # Accumulate into this StringIO as well as forwarding to the real stdout:
        # the driver returns getvalue() so the worker can parse the samples out.
        # The original tee forwarded only, so getvalue() was always "" and the
        # samples path silently produced nothing.
        super().write(s)
        self.real.write(s)
        if s.startswith("step ") and "|" in s:
            try:
                step = int(s.split()[1])
                loss = float(s.split("loss")[1])
                print(${JSON.stringify(STEP_MARKER)} + json.dumps({"step": step, "loss": loss}))
            except Exception:
                pass
        return len(s)

sys.stdout = _Tee(sys.stdout)
exec(code, ns)
_output = sys.stdout.getvalue()
sys.stdout = sys.stdout.real
# Return the captured stdout explicitly. The worker normally reads stdout
# *during* the run (via setStdout) and parses the samples and step events from
# that stream; this return value is the fallback for a runtime without
# setStdout. It must be an expression, not an assignment -- runPythonAsync
# resolves to the value of the last expression, and an assignment is None.
_output
`
}
