/**
 * Light and dark, and the reading-progress memory.
 *
 * Both live in `localStorage` and both are applied before first paint by an
 * inline script in `index.html` -- this module is the React side of the same
 * state, not the only copy. The inline script exists so a reader who chose dark
 * never sees a white flash; duplicating the decision here keeps React in sync
 * with what was already applied.
 */

import { useCallback, useEffect, useState } from 'react'

const THEME_KEY = 'atlas:theme'
const PROGRESS_KEY = 'atlas:progress'

export type Theme = 'light' | 'dark'

function readTheme(): Theme {
  if (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) {
    return 'dark'
  }
  return 'light'
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(readTheme)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    try {
      localStorage.setItem(THEME_KEY, theme)
    } catch {
      // Private browsing, or storage disabled. The theme still applies for this
      // session; it just will not be remembered. Not worth a visible error.
    }
  }, [theme])

  const toggle = useCallback(
    () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    [],
  )
  return [theme, toggle]
}

/**
 * Which concepts a reader has opened.
 *
 * Stored rather than derived, because "have I read this yet" is the only piece
 * of state the site has, and the reading order in `index.json` is already the
 * curriculum. The map is kept sorted on read so the export is stable.
 */
export function useProgress(): [ReadonlySet<string>, (id: string) => void, () => void] {
  const [seen, setSeen] = useState<ReadonlySet<string>>(() => {
    try {
      const raw = localStorage.getItem(PROGRESS_KEY)
      return new Set(raw ? (JSON.parse(raw) as string[]) : [])
    } catch {
      return new Set()
    }
  })

  const mark = useCallback((id: string) => {
    setSeen((current) => {
      if (current.has(id)) return current
      const next = new Set(current)
      next.add(id)
      try {
        localStorage.setItem(PROGRESS_KEY, JSON.stringify([...next].sort()))
      } catch {
        // As above: the state still works, it just is not persisted.
      }
      return next
    })
  }, [])

  const reset = useCallback(() => {
    setSeen(new Set())
    try {
      localStorage.removeItem(PROGRESS_KEY)
    } catch {
      /* nothing to do */
    }
  }, [])

  return [seen, mark, reset]
}
