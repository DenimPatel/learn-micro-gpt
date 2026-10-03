/**
 * The playground's run state, as a pure reducer.
 *
 * ## Why a discriminated union, and why it lives outside the component
 *
 * The state used to be one wide interface with a field per datum, every one of
 * them optional. That makes impossible states representable -- running with no
 * stage, an error with no message, done with no wall time -- and each one had to
 * be ruled back out by an ad-hoc conditional at the point of use. The worker
 * protocol already names distinct phases (`status`, `step`, `sample`, `done`,
 * `error`), so the state mirrors it: a payload exists only on the variant where
 * it means something, and the compiler refuses to let anything else read it.
 *
 * This module holds no React import on purpose. The transitions are pure
 * functions from one state to the next, so they can be tested in plain node
 * rather than only through a rendered page -- the behaviour that was previously
 * buried in a `setRun` callback and reachable from nowhere but Playwright.
 *
 * ## `T | undefined`, not `T?`
 *
 * Fields that a transition genuinely clears (`detail` when a new stage starts)
 * are declared required-but-`undefined` rather than optional, because
 * `exactOptionalPropertyTypes` is on: with `T?` the compiler rejects assigning
 * `undefined`, and the fix -- silently leaving a stale message on screen -- is
 * worse than the error.
 */

import type { WorkerResponse } from '../worker/protocol'

export interface LossPoint {
  step: number
  loss: number
}

export type RunState =
  | { status: 'idle' }
  /**
   * The brief window between pressing run and the worker's first progress
   * message: the worker exists, Pyodide has not reported a stage yet.
   */
  | { status: 'starting' }
  | {
      status: 'running'
      stage: string
      detail: string | undefined
      step: number | undefined
      total: number | undefined
      losses: LossPoint[]
      samples: string[]
    }
  | { status: 'done'; wallSeconds: number; losses: LossPoint[]; samples: string[] }
  | { status: 'error'; error: string; recoverable: boolean }

export const IDLE: RunState = { status: 'idle' }

/**
 * Whether the run is in flight.
 *
 * Both the run/cancel buttons and the progress notice key off this, and the
 * two phases differ only in which message has arrived so far. It is a type
 * guard rather than a bare boolean so the notice can take the narrowed union
 * and read each variant's own payloads.
 */
export type BusyRunState = Extract<RunState, { status: 'starting' | 'running' }>

export function isBusy(run: RunState): run is BusyRunState {
  return run.status === 'starting' || run.status === 'running'
}

/**
 * Apply one worker message to the current state.
 *
 * Exhaustive over `message.type` by construction: a new worker message is a
 * compile error here until it is handled. The `never` default makes that
 * failure explicit rather than a silent `return current`.
 */
export function reduce(state: RunState, message: WorkerResponse): RunState {
  switch (message.type) {
    case 'status':
      // The worker's first `status` is what ends the "starting" phase.
      return {
        status: 'running',
        stage: message.stage,
        detail: message.detail,
        step: stageCarryover(state).step,
        total: stageCarryover(state).total,
        losses: carriedLosses(state),
        samples: carriedSamples(state),
      }
    case 'step':
      return {
        status: 'running',
        stage: stageCarryover(state).stage,
        detail: stageCarryover(state).detail,
        step: message.step,
        total: message.total,
        losses: [...carriedLosses(state), { step: message.step, loss: message.loss }],
        samples: carriedSamples(state),
      }
    case 'sample':
      // Samples may arrive after `done` (the worker's final sample list wins
      // there); appending keeps the streamed ones until it does.
      return {
        status: 'running',
        ...stageCarryover(state),
        samples: [...carriedSamples(state), message.text],
        losses: carriedLosses(state),
      }
    case 'done':
      return {
        status: 'done',
        wallSeconds: message.wallSeconds,
        losses: carriedLosses(state),
        // The worker's own list is authoritative when present, but it can be
        // empty (a run that produced nothing), and then the streamed samples
        // are all there is to show.
        samples: message.samples.length ? message.samples : carriedSamples(state),
      }
    case 'error':
      return { status: 'error', error: message.message, recoverable: message.recoverable }
    default: {
      const _never: never = message
      return _never
    }
  }
}

/**
 * The stage carried across a transition that does not itself carry one.
 *
 * Only a `running` state has a stage to keep; a `step` message keeps the stage
 * the preceding `status` message established, which is why the union widens
 * back to `undefined` here rather than dropping the field.
 */
function stageCarryover(state: RunState): {
  stage: string
  detail: string | undefined
  step: number | undefined
  total: number | undefined
} {
  return state.status === 'running'
    ? { stage: state.stage, detail: state.detail, step: state.step, total: state.total }
    : { stage: '', detail: undefined, step: undefined, total: undefined }
}

function carriedLosses(state: RunState): LossPoint[] {
  return state.status === 'running' || state.status === 'done' ? state.losses : []
}

function carriedSamples(state: RunState): string[] {
  return state.status === 'running' || state.status === 'done' ? state.samples : []
}

/** Pressing run: a worker has been created, nothing has reported yet. */
export function startRunState(): RunState {
  return { status: 'starting' }
}

/**
 * Every failure path, in one place.
 *
 * The worker's own `error` message is one; `onerror` (the worker script failed
 * to load or threw before it could post) and a throwing `postMessage` are the
 * other two. Without a single typed state for them, those two leave "starting"
 * on screen forever -- a spinner that never resolves, which is the failure
 * readers report and maintainers cannot reproduce.
 */
export function failRunState(message: string, recoverable: boolean): RunState {
  return { status: 'error', error: message, recoverable }
}

/**
 * Cancelling resets to idle.
 *
 * There is no `cancelled` variant: nothing renders differently for it, and a
 * status that only ever means "idle, but by a different route" would be a
 * variant every exhaustive switch had to handle for no reader-visible gain.
 */
export function cancelRunState(): RunState {
  return IDLE
}

/**
 * A short human label for each variant.
 *
 * Not rendered anywhere -- it exists so the exhaustiveness of the union is a
 * *tested* property, not only a compiler one: adding a variant without handling
 * it here fails the `never` assignment, and `describeForStatus` is called once
 * per variant in the unit test.
 */
export function describeForStatus(run: RunState): string {
  switch (run.status) {
    case 'idle':
      return 'idle'
    case 'starting':
      return 'starting'
    case 'running':
      return `running: ${run.stage}`
    case 'done':
      return `done in ${run.wallSeconds.toFixed(1)}s`
    case 'error':
      return `error: ${run.error}`
    default: {
      const _never: never = run
      return _never
    }
  }
}
