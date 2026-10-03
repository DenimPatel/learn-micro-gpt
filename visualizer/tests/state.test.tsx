import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import {
  useLocalStorage,
  useProgress,
  useTheme,
  encodeProgress,
  decodeProgress,
  PROGRESS_SHAPE,
} from '../src/app/state'
import { createMockWindow } from './support/mock-window'

/**
 * `useLocalStorage`, driven against storage that throws.
 *
 * ## Why the stub throws rather than being removed
 *
 * "Storage is unavailable" reaches the code as an *exception*, not as a missing
 * object: private browsing in some browsers, a disabled-site setting, and a full
 * quota all raise from `getItem`/`setItem`. A test that simply deletes
 * `localStorage` exercises the `typeof === 'undefined'` path and proves nothing
 * about the path the reader actually hits. The stub below is a real object whose
 * methods throw, which is the shape the guards were written for.
 *
 * ## Why a real render
 *
 * The behaviour under test is a React hook's: that a throw on read yields the
 * default, that a throw on write still updates the in-memory value, and that
 * neither escapes as an uncaught error. Asserting against the functions directly
 * would test a copy of the hook rather than the hook.
 *
 * `environment: 'node'` (see `vitest.config.ts`) plus `tests/support/mock-window.ts`
 * provides the small DOM slice React's client renderer needs, the same way the
 * router tests do. No jsdom. React warns that this environment "is not configured
 * to support act(...)" unless `globalThis.IS_REACT_ACT_ENVIRONMENT` is set, so it
 * is set below, exactly as `router.test.tsx` does. The flag does not change how
 * `act` flushes -- it only tells React it may treat this as a test environment --
 * which is the whole of what the warning is about.
 */

/** A `localStorage` whose every method throws, as a locked-down browser's does. */
function throwingStorage(message = 'SecurityError: storage is disabled') {
  return {
    getItem: () => {
      throw new Error(message)
    },
    setItem: () => {
      throw new Error(message)
    },
    removeItem: () => {
      throw new Error(message)
    },
  }
}

/**
 * Mount a hook and hand back what it returned, plus a way to unmount.
 *
 * The value is read out of the rendered tree rather than captured in a closure,
 * so what is asserted is what React actually rendered -- a hook that returned
 * correctly but re-rendered wrongly would show up here.
 *
 * ## The two traps this helper exists to avoid
 *
 * `createRoot` on a node that has already been given to `createRoot` logs
 * "container that has already been passed to createRoot()" and then renders
 * nothing, and a root cannot be reused after `unmount()` ("Cannot update an
 * unmounted root"). So the root is created once per test, and re-created only
 * after an unmount.
 *
 * The second trap is subtler: re-rendering with a *freshly defined* component
 * function makes React treat it as a different type, and the resulting update is
 * scheduled asynchronously -- synchronous `act` never flushes it, and the root is
 * silently left broken for every later test in the file. So `Host` is defined
 * once at module scope and the hook under test is handed to it through a ref,
 * which means `rerender()` genuinely re-renders the same component.
 */
const shared = (() => {
  const handle = createMockWindow()
  vi.stubGlobal('window', handle.window)
  vi.stubGlobal(
    'document',
    (handle.container as unknown as { ownerDocument: unknown }).ownerDocument,
  )
  return handle
})()

// `react-dom/client` only treats `act()` as the test environment's when this
// flag is set; without it every `act` in this file logs a warning. The node
// environment does not set it for us. Same line as `router.test.tsx`, for the
// same reason.
;(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true

/** What the currently mounted `Host` should call; swapped per `mountHook`. */
let hookUnderTest: (() => unknown) | null = null
/** What that hook returned on the most recent render. */
let hookResult: unknown

/** One stable component type, so re-rendering never looks like a type change. */
function Host() {
  hookResult = hookUnderTest?.()
  return null
}

let liveRoot: Root | null = null
/** Bumped on every mount so React discards the previous tree instead of reusing it. */
let mountGeneration = 0

/** The root for the current test, re-created after a previous one unmounted. */
function currentRoot(): Root {
  if (!liveRoot) {
    while (shared.container.childNodes.length > 0) {
      shared.container.removeChild(shared.container.childNodes[0] as never)
    }
    liveRoot = createRoot(shared.container)
  }
  return liveRoot
}

/** In-memory storage, built per test so keys cannot leak across them. */
function freshStorage(seed: Map<string, string> = new Map<string, string>()) {
  const map = seed
  return {
    map,
    getItem: (key: string) => map.get(key) ?? null,
    setItem: (key: string, value: string) => {
      map.set(key, value)
    },
    removeItem: (key: string) => {
      map.delete(key)
    },
  }
}

async function mountHook<T>(useHook: () => T): Promise<{
  read: () => T
  rerender: () => Promise<void>
  unmount: () => Promise<void>
}> {
  const root = currentRoot()
  hookUnderTest = useHook as () => unknown
  mountGeneration += 1
  const key = String(mountGeneration)

  await act(async () => {
    root.render(createElement(Host, { key }))
  })

  return {
    read: () => hookResult as T,
    // Re-rendering with the *same* key is what makes this a real re-render: the
    // component stays mounted, so the hook's state and the setters' identity
    // survive, which is exactly what the identity assertions need.
    rerender: () => act(async () => root.render(createElement(Host, { key }))),
    unmount: async () => {
      await act(async () => root.unmount())
      if (liveRoot === root) liveRoot = null
      hookUnderTest = null
    },
  }
}

describe('useLocalStorage', () => {
  it('falls back to the initial value when reading throws, without throwing', async () => {
    vi.stubGlobal('localStorage', throwingStorage())
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'fallback'))
    expect(read()).toEqual(['fallback', expect.any(Function), expect.any(Function)])
    await unmount()
  })

  it('calls a function initial lazily, and keeps working when storage throws', async () => {
    // The theme depends on this: its default is read from the already-painted
    // DOM, so the initializer must be a function *and* must not be needed when
    // storage is empty and unusable at the same time.
    vi.stubGlobal('localStorage', throwingStorage())
    const initial = vi.fn(() => 'from-dom')
    const { read, unmount } = await mountHook(() => useLocalStorage('k', initial))
    expect(initial).toHaveBeenCalled()
    expect(read()[0]).toBe('from-dom')
    await unmount()
  })

  it('still updates the in-memory value when the write throws', async () => {
    // The whole point of the guard: the reader loses persistence, not the
    // feature. A throw on write must not stop the state from advancing.
    vi.stubGlobal('localStorage', throwingStorage())
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'a'))
    expect(read()[0]).toBe('a')
    await act(async () => read()[1]('b'))
    expect(read()[0]).toBe('b')
    await unmount()
  })

  it('survives a write and a remove that both throw', async () => {
    vi.stubGlobal('localStorage', throwingStorage())
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'a'))
    await act(async () => read()[1]('b'))
    await act(async () => read()[2]())
    expect(read()[0]).toBe('a')
    await unmount()
  })

  it('reads an existing stored value on mount', async () => {
    const storage = freshStorage(new Map([['k', JSON.stringify('stored')]]))
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'default'))
    expect(read()[0]).toBe('stored')
    await unmount()
  })

  it('writes the serialized value through to storage', async () => {
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'a'))
    await act(async () => read()[1]('b'))
    expect(storage.map.get('k')).toBe('"b"')
    await unmount()
  })

  it('uses the initial value for a malformed stored value instead of throwing', async () => {
    // Something else wrote this key. A parse error here would take down the
    // page on mount, which is not a proportionate response to a corrupt entry.
    const storage = freshStorage(new Map([['k', '{not json']]))
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useLocalStorage('k', 'safe-default'))
    expect(read()[0]).toBe('safe-default')
    await unmount()
  })

  it('keeps the setters stable across renders when the codec is inline', async () => {
    // Both real callers build their options object fresh on every render, so an
    // implementation that closed over `serialize`/`deserialize` directly would
    // hand back a new `set` each time -- and a consumer that lists the setter as
    // an effect dependency would then re-run that effect forever. Identity is
    // the whole assertion here: `setSecond` must be the same function as
    // `setFirst` even though each render passed a *new* inline `serialize`.
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const serialize = (v: string) => String(v)
    const { read, unmount, rerender } = await mountHook(() =>
      useLocalStorage('k', 'a', { serialize: (v) => serialize(v) }),
    )
    const [, setFirst] = read()
    await act(async () => rerender())
    const [, setSecond] = read()
    expect(setSecond).toBe(setFirst)
    // And the identity is not bought by freezing the codec at its first value:
    // a write after the re-render still serializes through the new codec.
    await act(async () => setSecond('b'))
    expect(storage.map.get('k')).toBe('b')
    await unmount()
  })
})

describe('useProgress on top of useLocalStorage', () => {
  it('marks a concept even when storage throws, and keeps the public shape', async () => {
    vi.stubGlobal('localStorage', throwingStorage())
    const { read, unmount } = await mountHook(() => useProgress())
    const [seen, mark] = read()
    expect(seen).toBeInstanceOf(Set)
    expect(seen.size).toBe(0)
    await act(async () => mark('softmax'))
    expect(read()[0].has('softmax')).toBe(true)
    await unmount()
  })

  it('persists as a sorted array and restores a Set, using a custom codec', async () => {
    // JSON cannot represent a `Set`, so the hook must not assume JSON round-trips
    // the value: this is what the `serialize`/`deserialize` options are for.
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useProgress())
    await act(async () => read()[1]('softmax'))
    await act(async () => read()[1]('adam'))
    expect(storage.map.get('atlas:progress')).toBe(JSON.stringify(['adam', 'softmax']))
    await unmount()
  })

  it('is idempotent: re-marking a concept neither changes the set nor rewrites storage', async () => {
    // The item this guards (IMP-073) is about the *reset*, and a reset that can
    // be followed by a stale write is the failure mode. Marking is the only
    // writer, so it has to be stable: a second `mark` of the same id must be a
    // no-op, not a new Set (which would re-serialize and could race a reset).
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useProgress())
    await act(async () => read()[1]('softmax'))
    const first = read()[0]
    await act(async () => read()[1]('softmax'))
    // Same identity: `mark` returned `current` unchanged rather than a new Set.
    expect(read()[0]).toBe(first)
    expect(storage.map.get('atlas:progress')).toBe(JSON.stringify(['softmax']))
    await unmount()
  })

  it('resets to empty: the set is emptied and storage is cleared, not left stale', async () => {
    // Reset is the one irreversible action in the app. After it, both the
    // in-memory `Set` and the persisted export must agree that nothing is read,
    // or a reload would resurrect the progress the reader just discarded.
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useProgress())
    await act(async () => read()[1]('softmax'))
    await act(async () => read()[1]('adam'))
    expect(read()[0].size).toBe(2)
    const reset = read()[2]
    await act(async () => reset())
    expect(read()[0].size).toBe(0)
    expect(read()[0]).toBeInstanceOf(Set)
    expect(storage.map.has('atlas:progress')).toBe(false)
    await unmount()
  })
})

describe('the progress export', () => {
  // The export is the durable record of what a reader has opened -- it is what a
  // reload restores and what a reader would copy out of devtools. Its shape is a
  // contract, so it is pinned byte-for-byte here rather than left to whichever
  // codec the hook happens to pass. See `PROGRESS_SHAPE` in the source.
  it('is a JSON array of concept-id strings, sorted ascending order', () => {
    expect(encodeProgress(new Set(['softmax', 'adam', 'mlp-block']))).toBe(
      '["adam","mlp-block","softmax"]',
    )
    // Sorted, so two readers with the same set produce byte-identical storage.
    expect(encodeProgress(new Set(['z', 'a', 'm']))).toBe(encodeProgress(new Set(['m', 'z', 'a'])))
  })

  it('has the documented empty export, and round-trips through the codec', () => {
    expect(encodeProgress(new Set())).toBe(PROGRESS_SHAPE.empty)
    // The key in the shape is the key the hook writes under; if that drifts, a
    // reader's existing progress silently vanishes.
    expect(PROGRESS_SHAPE.key).toBe('atlas:progress')
    const ids = ['adam', 'softmax', 'rmsnorm']
    expect([...decodeProgress(encodeProgress(new Set(ids)))].sort()).toEqual([
      'adam',
      'rmsnorm',
      'softmax',
    ])
    expect(decodeProgress(PROGRESS_SHAPE.empty).size).toBe(0)
  })

  it('drops duplicates, because the source is a Set', () => {
    expect(encodeProgress(new Set(['adam', 'adam', 'adam']))).toBe('["adam"]')
  })
})

describe('useTheme on top of useLocalStorage', () => {
  it('persists the theme as a raw string, the encoding index.html compares against', async () => {
    // Not `"dark"`. The first-paint script in index.html does
    // `localStorage.getItem('atlas:theme') === 'dark'`, so a JSON-quoted value
    // would read as "not dark" and a reader who chose dark would get a white
    // flash on every load. This assertion is the only thing standing between a
    // future codec change and that regression, so it compares the stored bytes
    // rather than the round-tripped value.
    const storage = freshStorage()
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useTheme())
    const [theme, toggle] = read()
    expect(theme).toBe('light')
    await act(async () => toggle())
    expect(read()[0]).toBe('dark')
    expect(storage.map.get('atlas:theme')).toBe('dark')
    await unmount()
  })

  it('agrees with index.html about what a stored value means', async () => {
    // Read the literal out of the real file instead of trusting a copy of the
    // rule here: if the inline script ever stops using the bare string, this
    // test fails rather than the two implementations drifting apart silently.
    const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8')
    expect(html).toContain("localStorage.getItem('atlas:theme')")
    expect(html).toContain("stored === 'dark'")
  })

  it('restores a stored dark theme on mount', async () => {
    const storage = freshStorage(new Map([['atlas:theme', 'dark']]))
    vi.stubGlobal('localStorage', storage)
    const { read, unmount } = await mountHook(() => useTheme())
    expect(read()[0]).toBe('dark')
    await unmount()
  })

  it('toggles without throwing when storage is unavailable', async () => {
    vi.stubGlobal('localStorage', throwingStorage())
    // `documentElement` is shared across the whole file and `useTheme` applies
    // its theme to it, so a previously rendered `useTheme` can have left a
    // `dark` class behind. Clear it first: this test is about storage, not
    // about which theme happened to be painted before it.
    document.documentElement.classList.remove('dark')
    const { read, unmount } = await mountHook(() => useTheme())
    // Nothing has painted a `dark` class onto `documentElement`, so the
    // DOM-derived default is `light`; the point of the test is that a throwing
    // storage does not stop the toggle from advancing.
    expect(read()[0]).toBe('light')
    await act(async () => read()[1]())
    expect(read()[0]).toBe('dark')
    await act(async () => read()[1]())
    expect(read()[0]).toBe('light')
    await unmount()
  })
})
