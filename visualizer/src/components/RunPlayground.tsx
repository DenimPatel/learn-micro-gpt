/**
 * The playground page: edit the reference, run it in the browser.
 *
 * ## Degradation is the normal case, not the error path
 *
 * Pyodide is 12 MB of WebAssembly fetched from a CDN. It can fail: no network, a
 * restrictive CSP, a reader on a metered connection. None of that deserves a red
 * banner, so the failure states here are all *useful* states:
 *
 *  - a clear statement of what could not run and why,
 *  - the exact command to run the same thing locally, which works offline and is
 *    what most readers will actually prefer,
 *  - and a link to the recorded trace, which is the same run, already done.
 *
 * The run is also deliberately small (200 steps by default, capped at 2000)
 * because a browser tab frozen for a minute is a bad way to learn something.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { loadSource, loadedSource } from '../data/sources'
import { CodePanel } from './CodePanel'
import { LossChart } from './widgets'
import { PYODIDE_PINNED_VERSION, createWorker, type WorkerResponse } from '../worker/protocol'

/**
 * Playground state.
 *
 * Every optional field is typed `T | undefined` rather than `T?`, because
 * `exactOptionalPropertyTypes` is on and several transitions genuinely clear a
 * field (`detail: undefined` when a new stage starts). With `T?` the compiler
 * rejects the assignment, and the fix -- silently leaving a stale message on
 * screen -- is worse than the error.
 */
interface RunState {
  status: 'idle' | 'starting' | 'running' | 'done' | 'error'
  stage: string | undefined
  detail: string | undefined
  step: number | undefined
  total: number | undefined
  losses: { step: number; loss: number }[]
  samples: string[]
  error: string | undefined
  recoverable: boolean | undefined
  wallSeconds: number | undefined
}

const IDLE: RunState = {
  status: 'idle',
  stage: undefined,
  detail: undefined,
  step: undefined,
  total: undefined,
  losses: [],
  samples: [],
  error: undefined,
  recoverable: undefined,
  wallSeconds: undefined,
}

export function RunPlayground() {
  const [run, setRun] = useState<RunState>(IDLE)
  const [editing, setEditing] = useState(false)
  const [edited, setEdited] = useState<string | null>(null)
  // The reference source, loaded from its own chunk. The editor needs it in
  // hand, so it is fetched eagerly here rather than lazily by a CodePanel.
  const [reference, setReference] = useState<string | null>(() => loadedSource('python') ?? null)
  const workerRef = useRef<Worker | null>(null)
  const textRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    let cancelled = false
    void loadSource('python').then((text) => {
      if (!cancelled) setReference(text)
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    return () => {
      workerRef.current?.terminate()
      workerRef.current = null
    }
  }, [])

  const onMessage = useCallback((event: MessageEvent<WorkerResponse>) => {
    const message = event.data
    setRun((current) => {
      switch (message.type) {
        case 'status':
          return { ...current, status: 'running', stage: message.stage, detail: message.detail }
        case 'step':
          return {
            ...current,
            status: 'running',
            step: message.step,
            total: message.total,
            losses: [...current.losses, { step: message.step, loss: message.loss }],
          }
        case 'sample':
          return { ...current, samples: [...current.samples, message.text] }
        case 'done':
          return {
            ...current,
            status: 'done',
            wallSeconds: message.wallSeconds,
            samples: message.samples.length ? message.samples : current.samples,
          }
        case 'error':
          return { ...current, status: 'error', error: message.message, recoverable: message.recoverable }
        default:
          return current
      }
    })
  }, [])

  const start = useCallback(() => {
    if (typeof Worker === 'undefined') {
      setRun({
        ...IDLE,
        status: 'error',
        error: 'This browser has no Web Worker support, so the in-page runner is unavailable.',
        recoverable: false,
      })
      return
    }
    workerRef.current?.terminate()
    const worker = createWorker()
    worker.onmessage = onMessage
    workerRef.current = worker

    setRun({ ...IDLE, status: 'starting' })
    worker.postMessage({
      type: 'run',
      steps: 200,
      temperature: 0.5,
      source: edited ?? reference ?? '',
      dataset: DATASET_PLACEHOLDER,
    })
  }, [edited, onMessage, reference])

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ type: 'cancel' })
    workerRef.current?.terminate()
    workerRef.current = null
    setRun((current) => ({ ...current, status: 'idle', stage: 'cancelled' }))
  }, [])

  const busy = run.status === 'starting' || run.status === 'running'

  return (
    <article className="playground">
      <h1>Run it yourself</h1>
      <p className="lede">
        The 199 lines, unmodified, in your browser. Change{' '}
        <code>n_head</code> or <code>n_layer</code>, press run, and watch the samples change.
      </p>

      <section className="playground__how" aria-labelledby="how-heading">
        <h2 id="how-heading">How this actually works</h2>
        <p>
          A <strong>pinned</strong> copy of Pyodide (CPython compiled to WebAssembly, version{' '}
          {PYODIDE_PINNED_VERSION}) is fetched from a CDN the first time you press run, and cached
          by the browser after that. <code>data/input.txt</code> is written into its virtual
          filesystem first, so the reference's <code>os.path.exists</code> check succeeds and
          nothing is downloaded.
        </p>
        <p className="callout">
          The step count is reduced from 1000 to 200 for the browser run, by substituting the
          string <code>num_steps = 1000</code> in the source text before compiling. That is the
          only difference, it happens in the source you can see below, and it is why this is a
          playground rather than a second reference. For the real thing, with 1000 steps and the
          real dataset, run <code>make run</code> in a terminal.
        </p>
      </section>

      <div className="playground__actions">
        <button type="button" onClick={start} disabled={busy} className="button button--primary">
          {busy ? 'running…' : 'run in the browser'}
        </button>
        {busy ? (
          <button type="button" onClick={cancel} className="button">
            cancel
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => setEditing((value) => !value)}
          aria-expanded={editing}
          className="button"
        >
          {editing ? 'hide the editor' : 'edit the source first'}
        </button>
      </div>

      {run.status === 'error' ? (
        <div className="notice notice--warn" role="status">
          <p>
            <strong>The in-browser run could not finish.</strong> {run.error}
          </p>
          {run.recoverable ? (
            <p>That is usually a network problem. The two options below both work offline.</p>
          ) : null}
          <p>
            Either run it locally, which is the better way to do this anyway:
          </p>
          <pre className="code-block">
            <code>make run</code>
          </pre>
          <p>
            or read the recorded run of all 1,000 steps below — the same code, the same dataset,
            already done.
          </p>
        </div>
      ) : null}

      {run.status === 'starting' || run.status === 'running' ? (
        <div className="notice" role="status" aria-live="polite">
          <p>
            {run.stage === 'loading'
              ? run.detail ?? 'loading Pyodide'
              : run.detail ?? 'working'}
            {run.step !== undefined && run.total ? ` — step ${run.step} of ${run.total}` : ''}
          </p>
          {run.losses.length ? (
            <p className="notice__last">
              latest loss {run.losses[run.losses.length - 1]!.loss.toFixed(4)}
            </p>
          ) : null}
        </div>
      ) : null}

      {run.status === 'done' ? (
        <div className="notice notice--ok" role="status">
          <p>
            Finished {run.wallSeconds?.toFixed(1)}s in the browser. Your samples:
          </p>
          <ul className="samples">
            {run.samples.map((text, i) => (
              <li key={i}>
                <code>{text || <em>(empty)</em>}</code>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {editing ? (
        <section aria-labelledby="editor-heading">
          <h2 id="editor-heading">The source that will run</h2>
          <p>
            Try <code>n_head = 1</code>, or <code>block_size = 8</code>, or{' '}
            <code>learning_rate, ... = 0.05, ...</code>. Reset restores the pinned reference.
          </p>
          <label className="visually-hidden" htmlFor="source-editor">
            The reference source, editable
          </label>
          <textarea
            id="source-editor"
            ref={textRef}
            className="editor"
            spellCheck={false}
            value={edited ?? reference ?? ''}
            onChange={(e) => setEdited(e.target.value)}
            rows={24}
          />
          <button
            type="button"
            className="button"
            onClick={() => {
              setEdited(null)
              textRef.current?.focus()
            }}
          >
            reset to the pinned reference
          </button>
        </section>
      ) : (
        <CodePanel language="python" />
      )}

      <section aria-labelledby="recorded-heading">
        <h2 id="recorded-heading">The recorded run</h2>
        <p>
          All 1,000 steps, recorded from a real run of the pinned reference with{' '}
          <code>data/input.txt</code>. No browser required.
        </p>
        <LossChart />
      </section>
    </article>
  )
}

/**
 * The dataset, imported at build time.
 *
 * Same reasoning as the source files: a runtime `fetch('../data/input.txt')`
 * would be an absolute-or-relative URL guessing game, which is exactly what broke
 * the previous visualizer on GitHub Pages. Bundling it costs 228 KB gzipped to
 * about 60 KB and removes the entire failure mode.
 */
import datasetTxt from '../../../data/input.txt?raw'
const DATASET_PLACEHOLDER = datasetTxt
