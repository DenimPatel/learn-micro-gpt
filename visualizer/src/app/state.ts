/**
 * Light and dark, and the reading-progress memory.
 *
 * Both live in `localStorage` and both are applied before first paint by an
 * inline script in `index.html` -- this module is the React side of the same
 * state, not the only copy. The inline script exists so a reader who chose dark
 * never sees a white flash; duplicating the decision here keeps React in sync
 * with what was already applied.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const THEME_KEY = 'atlas:theme'
const PROGRESS_KEY = 'atlas:progress'

/** Matches the paper colour of each theme; see the token block in index.css. */
const THEME_COLORS: Record<Theme, string> = { light: '#fcfcfa', dark: '#0b0b0f' }

export type Theme = 'light' | 'dark'

/**
 * `localStorage`, or a stand-in that quietly does nothing.
 *
 * Storage throws rather than returning null in more situations than it looks
 * like it should: private browsing in some browsers, a disabled-site setting,
 * and a full quota all surface as an exception from `getItem`/`setItem`. The
 * page must not break in any of them -- the reader loses persistence, not the
 * feature -- so every access goes through here and every call is wrapped.
 *
 * Returning a no-op object rather than `null` keeps the call sites free of
 * `?.`, which is what let the same guard drift into three slightly different
 * copies in the first place.
 */
function safeStorage(): Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> {
  try {
    return localStorage
  } catch {
    return { getItem: () => null, setItem: () => {}, removeItem: () => {} }
  }
}

/**
 * State that mirrors `localStorage`, tolerating storage that is absent or throws.
 *
 * `initial` is a value or a function returning one, called lazily on the first
 * render only (the `useState` contract), which is what lets a caller read a
 * default out of the already-painted DOM -- the theme does exactly that, so the
 * React state cannot disagree with what the inline script applied.
 *
 * `serialize`/`deserialize` default to JSON and exist because the caller knows
 * its own shape: progress is a `Set`, which JSON cannot represent, so it is
 * stored as a sorted array and rebuilt into a `Set` on read. Keeping the codec
 * in the caller is what makes one hook cover both callers without special cases.
 *
 * Writes happen in the setter, not in an effect. An effect would re-write the
 * value on mount even when nothing changed, and -- more importantly -- a write
 * the reader triggers should not be able to land after an unmount, which a
 * `setState`-driven effect callback will happily do.
 */
export function useLocalStorage<T>(
  key: string,
  initial: T | (() => T),
  {
    serialize,
    deserialize,
  }: { serialize?: (value: T) => string; deserialize?: (raw: string) => T } = {},
): [T, (next: T | ((current: T) => T)) => void, () => void] {
  // The codec: the *read* below is the first render's job, so it closes over
  // this render's `deserialize` directly -- a ref would be a needless detour and
  // React's rules forbid reading one during render. The *write* inside `set`
  // needs a later render's `serialize`, so that one does go through a ref,
  // refreshed in the effect below (writes are triggered from events/effects,
  // which run after the commit that refreshed it).
  //
  // The type annotation names the parameter `serialized`, deliberately: naming
  // it `value` would shadow the `value` state below and make the arrow read the
  // component's current state instead of the argument it is handed, so every
  // write would persist the *previous* value.
  const serializeRef = useRef<(serialized: T) => string>((serialized) => JSON.stringify(serialized))

  const [value, setValue] = useState<T>(() => {
    const resolveInitial = () => (typeof initial === 'function' ? (initial as () => T)() : initial)
    // Both the read *and* the parse can fail: `getItem` throws outright when the
    // browser has storage disabled (`SecurityError`), and a value written by
    // something else may not be parseable. Either way the caller's default is a
    // better answer than a crash inside a render.
    let raw: string | null
    try {
      raw = safeStorage().getItem(key)
    } catch {
      return resolveInitial()
    }
    if (raw === null) return resolveInitial()
    try {
      return (deserialize ?? ((stored: string) => JSON.parse(stored) as T))(raw)
    } catch {
      return resolveInitial()
    }
  })

  // Keep the write-side codec current without touching a ref during render.
  // `serialize` is usually undefined, in which case the default is already in
  // the ref and this effect only runs once.
  useEffect(() => {
    if (serialize) serializeRef.current = serialize
  }, [serialize])

  // `set` reads the codec out of the ref at *call* time, not out of the render
  // it was created in. Both matter:
  //
  //  - the ref, so `set`'s identity only depends on `key` and a consumer can
  //    safely list it as an effect dependency;
  //  - read late, because the write happens inside the state updater, which
  //    React calls during a later render. Closing over this render's
  //    `serialize` would write whatever this render saw, and for a caller that
  //    passes a fresh inline codec each render that is a stale value in storage
  //    on the first write after a re-render.
  const set = useCallback(
    (next: T | ((current: T) => T)) => {
      setValue((current) => {
        const resolved = typeof next === 'function' ? (next as (c: T) => T)(current) : next
        try {
          safeStorage().setItem(key, serializeRef.current(resolved))
        } catch {
          // Serialization failed, or storage refused the write. The in-memory
          // value still updates: the feature keeps working for this session.
        }
        return resolved
      })
    },
    [key],
  )

  const remove = useCallback(() => {
    try {
      safeStorage().removeItem(key)
    } catch {
      // Nothing to do -- the state below is cleared either way.
    }
    setValue(typeof initial === 'function' ? (initial as () => T)() : initial)
  }, [key, initial])

  return [value, set, remove]
}

function readTheme(): Theme {
  if (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) {
    return 'dark'
  }
  return 'light'
}

export function useTheme(): [Theme, () => void] {
  // The *initial* theme is not read from storage here. The inline script in
  // `index.html` has already applied the theme before first paint, and
  // `readTheme` reports what it did; a stored default would be the same answer,
  // but a stale one would not, and trusting the key would let React disagree
  // with the DOM in the window between the script running and this rendering.
  // The stored value is still consulted: `useLocalStorage` reads the key on
  // mount and prefers it, and because both sides agree on the encoding, it
  // agrees with the script.
  //
  // The codec below is not decoration: the theme is stored as a *raw* string,
  // because `index.html`'s first-paint script does `stored === 'dark'` against
  // `getItem('atlas:theme')`. JSON-quoting it here would make that comparison
  // false and give a reader who chose dark a white flash on every load. The two
  // copies of this decision have to agree byte for byte.
  const [theme, setTheme] = useLocalStorage<Theme>(THEME_KEY, readTheme, {
    serialize: (value) => value,
    deserialize: (raw) => (raw === 'dark' ? 'dark' : 'light'),
  })

  // Applying the theme to the document stays here rather than moving into
  // `useLocalStorage`: it is this hook's job, not storage's. An effect is right
  // for these two -- they touch the DOM, which must not happen during render.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    // Keep the browser chrome (the address bar on mobile, the title bar in some
    // desktop windows) in step with the page. The inline script in index.html
    // does the same thing before first paint; without it there, the first paint
    // would be right and every later change wrong.
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLORS[theme])
  }, [theme])

  const toggle = useCallback(
    () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    [setTheme],
  )
  return [theme, toggle]
}

/**
 * The persisted shape of reading progress, and the codec that produces it.
 *
 * The stored value is the *export*: it is the only durable record of what a
 * reader has opened, it is what survives a reload, and it is what a reader would
 * copy out of devtools if they wanted to move their progress between machines.
 * Its shape is therefore part of the app's contract, not an implementation
 * detail, so it is written down here and pinned by a test:
 *
 *   - the key is `atlas:progress`;
 *   - the value is a JSON array of concept ids (`string[]`), e.g.
 *     `["adam","softmax"]`;
 *   - the array is always **sorted ascending**, so two readers with the same set
 *     of read concepts have byte-identical storage and a diff of two exports is
 *     meaningful;
 *   - duplicates never appear, because the source is a `Set`;
 *   - `[]` is the empty export, not `null` or a missing key.
 *
 * Encoding lives in one place (`encodeProgress`/`decodeProgress`) so the test
 * can call the exact functions the hook uses rather than a copy of the rule.
 */
export const PROGRESS_SHAPE = {
  key: PROGRESS_KEY,
  /** A JSON array of concept-id strings, sorted ascending. */
  empty: '[]',
} as const

/** Set → the documented export (a sorted JSON array of ids). */
export function encodeProgress(value: ReadonlySet<string>): string {
  return JSON.stringify([...value].sort())
}

/** The documented export → Set. Throws only on malformed input, which the
 * `useLocalStorage` read path already treats as "use the default". */
export function decodeProgress(raw: string): Set<string> {
  return new Set(JSON.parse(raw) as string[])
}

/**
 * Which concepts a reader has opened.
 *
 * Stored rather than derived, because "have I read this yet" is the only piece
 * of state the site has, and the reading order in `index.json` is already the
 * curriculum. The map is kept sorted on read so the export is stable -- see
 * `PROGRESS_SHAPE` for exactly what the export is.
 */
export function useProgress(): [ReadonlySet<string>, (id: string) => void, () => void] {
  // A `Set` is not JSON-representable, so it is stored as a sorted array: the
  // sort is what makes the stored value stable across writes, and the `Set`
  // rebuild is what makes membership checks O(1) again.
  const [seen, setSeen, clearSeen] = useLocalStorage<ReadonlySet<string>>(
    PROGRESS_KEY,
    () => new Set<string>(),
    { serialize: encodeProgress, deserialize: decodeProgress },
  )

  const mark = useCallback(
    (id: string) => {
      setSeen((current) => {
        if (current.has(id)) return current
        const next = new Set(current)
        next.add(id)
        return next
      })
    },
    [setSeen],
  )

  return [seen, mark, clearSeen]
}
