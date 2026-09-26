/**
 * Entry point.
 *
 * `StrictMode` is on deliberately. It double-invokes effects in development, which
 * is the cheapest way to find an effect that is not idempotent -- and the trace
 * work in this repository was built on exactly that kind of care.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './styles/index.css'
import 'katex/dist/katex.min.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('#root is missing from index.html')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
