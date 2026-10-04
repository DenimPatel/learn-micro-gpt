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
import { LossChart, LiveLossChart } from './widgets'
import { PYODIDE_PINNED_VERSION, createWorker, type WorkerResponse } from '../worker/protocol'
import {
  IDLE,
  cancelRunState,
  failRunState,
  isBusy,
  reduce,
  startRunState,
  type BusyRunState,
  type RunState,
} from './run-state'

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
    // `loadSource` resolves a `Result`, so a failed chunk is a value here rather
    // than an unhandled rejection; the editor simply stays empty, as before.
    void loadSource('python').then((result) => {
      if (!cancelled && result.ok) setReference(result.value)
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
    setRun((current) => reduce(current, message))
  }, [])

  /*
   * Every failure path lands here, and they all render the same notice.
   *
   * The worker's own `error` message is one path; `onerror` (the worker script
   * failed to load or threw before it could post) and a throwing `postMessage`
   * (a structured-clone failure) are the other two. Without this, those two leave
   * the "starting" state on screen forever -- a spinner that never resolves, which
   * is the failure readers report and maintainers cannot reproduce. One typed
   * state, three entry points.
   */
  const failRun = useCallback((message: string, recoverable: boolean) => {
    setRun(failRunState(message, recoverable))
  }, [])

  const start = useCallback(() => {
    if (typeof Worker === 'undefined') {
      failRun(
        'This browser has no Web Worker support, so the in-page runner is unavailable.',
        false,
      )
      return
    }
    workerRef.current?.terminate()
    const worker = createWorker()
    worker.onmessage = onMessage
    // A worker script that fails to load (CSP, a stale hash, a stray syntax
    // error) fires `error` and never posts anything. Without this the page looks
    // like it is loading, forever.
    worker.onerror = (event) => {
      failRun(event.message || 'The in-page runner failed to start.', true)
    }
    workerRef.current = worker

    setRun(startRunState())
    try {
      worker.postMessage({
        type: 'run',
        steps: 200,
        temperature: 0.5,
        source: edited ?? reference ?? '',
        dataset: DATASET_PLACEHOLDER,
      })
    } catch (error) {
      // `postMessage` throws synchronously if the message cannot be cloned. The
      // reference source is a plain string, so this should be unreachable, but a
      // throw here would otherwise escape and leave the run stuck on "starting".
      failRun(
        error instanceof Error ? error.message : 'The in-page runner could not be given the run.',
        false,
      )
    }
  }, [edited, failRun, onMessage, reference])

  const cancel = useCallback(() => {
    workerRef.current?.postMessage({ type: 'cancel' })
    workerRef.current?.terminate()
    workerRef.current = null
    setRun(cancelRunState())
  }, [])

  const busy = isBusy(run)

  return (
    <article className="playground">
      <h1>Run it yourself</h1>
      <p className="lede">
        The 199 lines, unmodified, in your browser. Change <code>n_head</code> or{' '}
        <code>n_layer</code>, press run, and watch the samples change.
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
          The step count is reduced from 1000 to 200 for the browser run, by substituting the string{' '}
          <code>num_steps = 1000</code> in the source text before compiling. That is the only
          difference, it happens in the source you can see below, and it is why this is a playground
          rather than a second reference. For the real thing, with 1000 steps and the real dataset,
          run <code>make run</code> in a terminal.
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
          <p>Either run it locally, which is the better way to do this anyway:</p>
          <pre className="code-block">
            <code>make run</code>
          </pre>
          <p>
            or read{' '}
            <a
              href="#recorded-run"
              onClick={(event) => {
                // The same in-page-anchor pattern `ConceptPage`'s table of
                // contents and the skip link use: the href is real (so no-JS and
                // assistive tech get a working link), but the default is
                // prevented because this site routes on the hash -- following
                // `#recorded-run` verbatim would replace the `#/run` route and
                // throw the reader back to the home page.
                event.preventDefault()
                const target = document.getElementById('recorded-run')
                if (!target) return
                target.scrollIntoView({
                  behavior: matchMedia('(prefers-reduced-motion: reduce)').matches
                    ? 'auto'
                    : 'smooth',
                  block: 'start',
                })
                target.focus()
              }}
            >
              the recorded run of all 1,000 steps
            </a>{' '}
            on this page — the same code, the same dataset, already done.
          </p>
        </div>
      ) : null}

      {/*
        The "starting" phase is the same notice with nothing to report yet: the
        worker exists, no stage message has arrived. Rendering it through the
        running branch keeps that one notice, and the placeholder copy
        ("loading Pyodide", "working") is what the reader sees until the first
        real message lands.
      */}
      {isBusy(run) ? <RunNotice run={run} /> : null}

      {run.status === 'done' ? (
        <div className="notice notice--ok" role="status">
          <p>Finished {run.wallSeconds.toFixed(1)}s in the browser. Your samples:</p>
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

      <section aria-labelledby="recorded-heading" id="recorded-run" tabIndex={-1}>
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
 * The progress notice, for the two states where a run is in flight.
 *
 * It takes the union narrowed to `starting | running` rather than the whole
 * `RunState` so the payloads (`stage`, `detail`, `step`, `total`, `losses`) are
 * the variants' own: `starting` carries nothing yet, and the running branch's
 * optional step read is the only place a `step`/`total` pair is meaningful.
 *
 * `starting` renders "working" rather than a stage-labelled message, because
 * that is what the reader saw before this variant existed: the old state had no
 * stage until the worker's first `status` message. Naming the phase here is
 * tempting, and would be a visible copy change.
 */
function RunNotice({ run }: { run: BusyRunState }) {
  return (
    <div className="notice" role="status" aria-live="polite">
      <p>
        {run.status === 'starting'
          ? 'working'
          : run.stage === 'loading'
            ? (run.detail ?? 'loading Pyodide')
            : (run.detail ?? 'working')}
        {run.status === 'running' && run.step !== undefined && run.total
          ? ` — step ${run.step} of ${run.total}`
          : ''}
      </p>
      {run.status === 'running' && run.losses.length ? (
        // The page's promise is a loss curve you watch fall, not a column of
        // numbers. Draw what has already arrived, so the run has visible
        // feedback from the first step rather than only at the end.
        <LiveLossChart points={run.losses} />
      ) : null}
    </div>
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
