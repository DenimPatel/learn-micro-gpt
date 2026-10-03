import { describe, expect, it } from 'vitest'

import type { WorkerResponse } from '../src/worker/protocol'
import {
  IDLE,
  cancelRunState,
  describeForStatus,
  failRunState,
  isBusy,
  reduce,
  startRunState,
  type RunState,
} from '../src/components/run-state'

/**
 * The playground's run state.
 *
 * These are pure transitions, so they are tested directly instead of through a
 * rendered page: the behaviour used to live inside a `setRun` callback with no
 * DOM-free way to reach it, and the only guard on it was the e2e suite. The
 * point of the union is that impossible states cannot be built, so most of what
 * is pinned here is *which payload each transition carries*, plus the two
 * quirks the page depends on verbatim (the worker's own samples winning at
 * `done`, and cancel returning to idle).
 */

/** Walk a run through a sequence of messages, starting from idle. */
function run(...messages: WorkerResponse[]): RunState {
  return messages.reduce<RunState>(reduce, IDLE)
}

describe('the playground run state', () => {
  it('starts idle, and a status message is what ends the starting phase', () => {
    expect(IDLE).toEqual({ status: 'idle' })
    expect(startRunState()).toEqual({ status: 'starting' })
    expect(isBusy(startRunState())).toBe(true)

    const running = reduce(startRunState(), { type: 'status', stage: 'loading' })
    expect(running.status).toBe('running')
    expect(running).toMatchObject({
      status: 'running',
      stage: 'loading',
      detail: undefined,
      step: undefined,
      total: undefined,
      losses: [],
      samples: [],
    })
  })

  it('carries the stage detail when the worker sends one', () => {
    const running = reduce(IDLE, { type: 'status', stage: 'compiling', detail: 'importing' })
    expect(running).toMatchObject({ status: 'running', stage: 'compiling', detail: 'importing' })
  })

  it('appends a step to the losses and updates step and total', () => {
    const afterStatus = reduce(IDLE, { type: 'status', stage: 'running', detail: 'step 1' })
    const first = reduce(afterStatus, { type: 'step', step: 1, total: 200, loss: 3.1 })
    expect(first).toMatchObject({
      status: 'running',
      step: 1,
      total: 200,
      losses: [{ step: 1, loss: 3.1 }],
      // The stage from the preceding status message survives a step.
      stage: 'running',
      detail: 'step 1',
    })

    const second = reduce(first, { type: 'step', step: 2, total: 200, loss: 2.8 })
    expect(second).toMatchObject({
      step: 2,
      total: 200,
      losses: [
        { step: 1, loss: 3.1 },
        { step: 2, loss: 2.8 },
      ],
    })
  })

  it('collects streamed samples', () => {
    const sampled = run(
      { type: 'status', stage: 'sampling' },
      { type: 'sample', index: 0, text: 'the' },
      { type: 'sample', index: 1, text: 'cat' },
    )
    expect(sampled).toMatchObject({ status: 'running', samples: ['the', 'cat'] })
  })

  it("prefers the worker's own samples at done, and keeps the streamed ones when it sends none", () => {
    const streamed = run(
      { type: 'status', stage: 'sampling' },
      { type: 'sample', index: 0, text: 'x' },
    )

    const withSamples = reduce(streamed, { type: 'done', wallSeconds: 12.34, samples: ['a', 'b'] })
    expect(withSamples).toMatchObject({ status: 'done', wallSeconds: 12.34, samples: ['a', 'b'] })

    const withoutSamples = reduce(streamed, { type: 'done', wallSeconds: 1.5, samples: [] })
    expect(withoutSamples).toMatchObject({ status: 'done', wallSeconds: 1.5, samples: ['x'] })
  })

  it("carries the worker's own error message -- not a generic string -- onto the error variant", () => {
    const errored = reduce(startRunState(), {
      type: 'error',
      message: 'boom',
      recoverable: true,
    })
    expect(errored).toEqual({ status: 'error', error: 'boom', recoverable: true })
    // The message is the worker's, character for character.
    expect(errored.status === 'error' && errored.error).toBe('boom')
  })

  it('failRunState is the error variant the worker path lands in, and is not recoverable by default', () => {
    expect(failRunState('no Web Workers here', false)).toEqual({
      status: 'error',
      error: 'no Web Workers here',
      recoverable: false,
    })
  })

  it('cancel resets the union to idle, dropping every payload', () => {
    const midRun = run(
      { type: 'status', stage: 'running' },
      { type: 'step', step: 5, total: 200, loss: 2 },
    )
    expect(cancelRunState()).toEqual({ status: 'idle' })
    expect(midRun.status).toBe('running')
  })

  it('is not busy in any terminal state', () => {
    expect(isBusy(IDLE)).toBe(false)
    expect(isBusy(reduce(IDLE, { type: 'done', wallSeconds: 1, samples: ['a'] }))).toBe(false)
    expect(isBusy(failRunState('boom', true))).toBe(false)
  })

  /*
   * Exhaustiveness, as a test rather than only a compiler property: every
   * variant reaches `describeForStatus`, whose `never` default makes a new
   * variant a type error. If the reducer stops handling one, the `status`
   * assertions above fail too.
   */
  describe('every status is handled', () => {
    const variants: RunState[] = [
      IDLE,
      startRunState(),
      reduce(IDLE, { type: 'status', stage: 'running' }),
      reduce(IDLE, { type: 'done', wallSeconds: 9.94, samples: [] }),
      failRunState('boom', true),
    ]

    it('labels each variant', () => {
      expect(variants.map(describeForStatus)).toEqual([
        'idle',
        'starting',
        'running: running',
        'done in 9.9s',
        'error: boom',
      ])
    })

    it('reduces each variant to itself for a message it does not carry a payload for', () => {
      // A sample arriving before any stage still lands on `running`, so the
      // notice renders rather than the state being dropped on the floor.
      for (const variant of variants) {
        expect(reduce(variant, { type: 'sample', index: 0, text: 'hi' }).status).toBe('running')
      }
    })
  })
})
