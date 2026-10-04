/**
 * `LoadableResult` — the one place a loadable value's states are drawn.
 *
 * ## Why this exists
 *
 * Three components load an asynchronous chunk and each of them invented its
 * own representation of "not loaded yet" on top of a `Result`: `CodePanel` and
 * `RunPlayground` held a `string | null` and treated `null` as loading,
 * `ComparePage` held a map that was empty until every chunk landed. That is one
 * shape per component, and each one has to re-decide what the loading paragraph
 * and the failure paragraph say — or, as they did before IMP-080, forget the
 * failure case entirely and leave an empty box on screen forever.
 *
 * So the shape is explicit, and it is the shape the loaders actually produce:
 * `Result<T, E>` on its way, `undefined` until the promise settles. The three
 * states are the two variants of `Result` plus "the promise has not settled",
 * which is why this type is `Result<T, E> | undefined` rather than a second
 * union with its own `{ status: 'loading' }` arm — the loaders already speak
 * `Result`, and a second union would be a translation layer between the two.
 *
 * ## Exhaustiveness
 *
 * `render` narrows `result` and defaults to `never`, so a fourth state cannot
 * be added without a compile error here — the same property, and the same
 * `never`-defaulted idiom, as `run-state.ts`'s switches. The narrowings use
 * `isOk` from `data/types.ts`, not `isErr`, so each branch is a positive test
 * and the `else` is genuinely the "not the success variant" case rather than
 * its mirror image.
 */

import type { ReactNode } from 'react'
import { isOk, type Result } from '../data/types'

/**
 * A load's state: settled (a `Result`), or not settled yet (`undefined`).
 *
 * `undefined` rather than a `loading: true` flag because it is what a
 * `useState<Result<T>>()` starts as, so a component does not have to
 * manufacture a placeholder variant to render before its effect runs.
 */
export type Loadable<Value, E = string> = Result<Value, E> | undefined

/**
 * Render one loadable resource's three states.
 *
 * The success arm is a function rather than a node so the value is in scope at
 * the call site without a cast, and so the child cannot be computed — or
 * throw — for a value that never arrived.
 */
export function LoadableResult<Value, E = string>({
  result,
  loading,
  error,
  children,
}: {
  result: Loadable<Value, E>
  /** The loading copy. Plain text, not markup: every site says one sentence. */
  loading: string
  /** How a failure is worded. The `Result`'s own `error` is passed alongside. */
  error: (message: E) => ReactNode
  children: (value: Value) => ReactNode
}) {
  if (result === undefined) return <Loading>{loading}</Loading>
  if (isOk(result)) return <>{children(result.value)}</>
  return <Failure>{error(result.error)}</Failure>
}

/**
 * The loading state's markup.
 *
 * `role="group"` with a label, matching the code panel's existing placeholder:
 * the one visible paragraph is a sentence, and a screen reader should hear what
 * it is waiting for rather than only "loading…".
 */
function Loading({ children }: { children: ReactNode }) {
  return (
    <div role="group" aria-label="loading" className="loadable">
      <p className="loadable__message">{children}</p>
    </div>
  )
}

/** The failure state's markup. `role="alert"`: the wait is over and it failed. */
function Failure({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className="loadable loadable--error">
      <p className="loadable__message">{children}</p>
    </div>
  )
}

/**
 * The loaded text, or `''` when the load has not settled or failed.
 *
 * The playground hands its reference straight to the worker and to the editor,
 * which both want a `string` and neither of which can render more than an empty
 * editor while the chunk is coming. This keeps that read in one place instead of
 * a `reference?.ok ? reference.value : ''` at each site.
 */
export function loadableText(result: Loadable<string>): string {
  return result !== undefined && isOk(result) ? result.value : ''
}
