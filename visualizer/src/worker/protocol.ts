/**
 * The Pyodide worker: runs `reference/microgpt.py` in the reader's browser.
 *
 * ## Why a worker, and why a pinned CDN
 *
 * Pyodide is a ~12 MB WebAssembly CPython plus a standard library. It is fetched
 * from a CDN at a **pinned version**, never "latest": an unpinned runtime means
 * the same URL can serve different code tomorrow, and a teaching site that
 * silently changes what it runs is not a teaching site.
 *
 * It runs in a worker because the whole thing is CPU-bound and would otherwise
 * freeze the page for a minute. The worker reports progress, and the main thread
 * only ever renders.
 *
 * ## The dataset has to be pre-written
 *
 * The reference resolves `input.txt` from the current working directory and, if
 * it is missing, downloads it from a mutable URL. Neither works here: there is no
 * real filesystem, and a 228 KB fetch on every page load is the opposite of the
 * point. So the worker writes `input.txt` into Pyodide's virtual filesystem
 * *before* executing the reference. The `os.path.exists` check then short-circuits
 * and nothing is downloaded. See `docs/KNOWN-ISSUES.md` issue 3.
 *
 * ## Degradation is the normal case
 *
 * Pyodide can fail: no network, a locked-down CSP, a reader on a metered
 * connection who would rather not. None of that is an error worth a red banner.
 * The page says what is missing, offers the same run as a command to paste into a
 * terminal, and links to the trace — which is the same run, already recorded.
 */

import type { Language } from '../data/types'

export type WorkerRequest =
  | { type: 'run'; steps: number; temperature: number; source: string; dataset: string }
  | { type: 'cancel' }

export type WorkerResponse =
  | { type: 'status'; stage: 'loading' | 'writing' | 'compiling' | 'running' | 'sampling'; detail?: string }
  | { type: 'step'; step: number; total: number; loss: number }
  | { type: 'sample'; index: number; text: string }
  | { type: 'done'; wallSeconds: number; samples: string[] }
  | { type: 'error'; message: string; recoverable: boolean }

const PYODIDE_VERSION = '0.26.4'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

/*
 * The driver that runs the reference lives in `pyodide.worker.ts`, next to the
 * Pyodide loading it depends on. It used to live here, which meant this module
 * carried a template string nobody could type-check and nobody could test.
 */

export function createWorker(): Worker {
  return new Worker(new URL('../worker/pyodide.worker.ts', import.meta.url), {
    type: 'module',
  })
}

export const PYODIDE_CDN = PYODIDE_URL
export const PYODIDE_PINNED_VERSION = PYODIDE_VERSION

/**
 * Only the reference is runnable in a browser today.
 *
 * The other tracks need a toolchain, not just a runtime, so they are listed here
 * as unavailable rather than silently absent -- a "run this" link that always
 * fails is worse than no link.
 */
export const RUNNABLE_LANGUAGES: readonly Language[] = ['python']
