/**
 * The page-level error boundary.
 *
 * One thrown render error -- a widget indexing a trace row that is not there, a
 * term block whose key is missing from the glossary, a layout that returns a
 * malformed node -- unmounts the whole React tree and leaves a blank page with a
 * console trace and nothing on screen. For an 18-page teaching site the correct
 * failure is a broken *page*, not a broken *site*.
 *
 * So this wraps the routed content only. The header, the concept rail and the
 * footer are outside it, which is the point: when a page throws, the reader can
 * still navigate anywhere else from the chrome, and "back to all concepts" is a
 * real escape rather than a promise.
 *
 * A class component, because React has no hook equivalent. That is the one place
 * in this codebase where a class is not a mistake.
 *
 * Two deliberate choices:
 *
 *  - `componentDidCatch` logs to the console and nothing else. There is no
 *    reporting endpoint here and there will not be one: this site ships a static
 *    bundle that fetches nothing at runtime, and a crash reporter would be the
 *    first network call in it. The reader stays private; the developer still
 *    gets the stack.
 *  - The fallback names the route. "Something went wrong" is not a report a
 *    reader can act on or send. Which page failed is.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react'
import { hrefFor, type Route } from '../app/router'

interface Props {
  children: ReactNode
  /** The route the failed content belongs to, used to name it in the fallback. */
  route: Route
}

interface State {
  error: Error | null
}

export class PageErrorBoundary extends Component<Props, State> {
  override state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // Local only. No network, and no `fetch`/`sendBeacon` -- see the header.
    console.error('Page render failed', error, info.componentStack)
  }

  override render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children

    const label = hrefFor(this.props.route)
    return (
      <article className="page-error" role="alert">
        <h1>This page could not be drawn</h1>
        <p className="lede">
          Something threw while rendering <code>{label}</code>. The rest of the site still works.
        </p>
        <p className="page-error__detail">
          <code>{error.message}</code>
        </p>
        <p className="page-error__actions">
          <a className="page-error__action" href="#/learn">
            Back to all concepts
          </a>
          <button
            type="button"
            className="page-error__action"
            onClick={() => window.location.reload()}
          >
            Reload this page
          </button>
        </p>
      </article>
    )
  }
}
