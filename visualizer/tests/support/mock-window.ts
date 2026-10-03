/**
 * A mocked `window` and a minimal `document` for the router tests.
 * ## Why this exists rather than jsdom
 *
 * `vitest.config.ts` is `environment: 'node'` deliberately ("anything needing a
 * DOM is a Playwright test"), and the router's guard needs a very small slice of
 * one: `location.hash`, `addEventListener('hashchange')`, `removeEventListener`,
 * and enough of `document` for React's client renderer to mount the hook's host
 * component into.
 *
 * `react-dom/client` only needs an element to hand `appendChild` and to read
 * `nodeType`/`ownerDocument` from. Providing that is a page of code; providing
 * jsdom is a dependency that re-implements the browser. The slice is written to
 * behave like the browser where the tests depend on it:
 *
 *   - `location.hash` is a *live property with a setter*. Assigning to it fires
 *     the `hashchange` listeners, as a browser does. That is what makes a test
 *     that says "the hash changed, so the app re-rendered" mean anything -- a
 *     plain field would let the router write a hash that never reaches it back.
 *   - the raw hash is stored and handed back untouched, so the parser sees the
 *     second `#` of `#/learn/softmax#sections` exactly as a browser would.
 *
 * What it deliberately does *not* do: `history`, `pushState`, focus, or any
 * navigation side effect. The router does not use them, so implementing them
 * would be untested code pretending to be a browser.
 */
export interface MockWindow {
  location: { hash: string; reload: () => void }
  addEventListener: (type: string, listener: () => void) => void
  removeEventListener: (type: string, listener: () => void) => void
  scrollTo: (options: { top: number; left: number }) => void
  scrollY: number
  innerHeight: number
  /** React's focus bookkeeping reads this while rendering; never focused here. */
  HTMLIFrameElement: new () => unknown
}

export interface MockWindowHandle {
  window: MockWindow
  /** A detached element React can mount into, with the bits it reads. */
  container: HTMLElement
  /** In-memory storage, so `useTheme`/`useProgress` see a clean slate. */
  localStorage: {
    getItem: (key: string) => string | null
    setItem: (key: string, value: string) => void
    removeItem: (key: string) => void
  }
  /** The raw hash string, exactly as the app left it. */
  readHash: () => string
  /** How many `hashchange` events have fired. A redundant write shows up here. */
  hashChanges: () => number
  /** Fire `hashchange` by hand, for a case that mutates the hash directly. */
  emitHashChange: () => void
}

/** The smallest node React will mount into. */
function createElementNode(tagName: string): HTMLElement {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const node: any = {
    nodeType: 1,
    nodeName: tagName.toUpperCase(),
    tagName: tagName.toUpperCase(),
    namespaceURI: 'http://www.w3.org/1999/xhtml',
    ownerDocument: undefined as unknown,
    parentNode: null,
    childNodes: [] as unknown[],
    style: {},
    firstChild: null,
    nextSibling: null,
    appendChild(child: any) {
      node.childNodes.push(child)
      child.parentNode = node
      node.firstChild = node.childNodes[0] ?? null
      return child
    },
    insertBefore(child: any, before: any) {
      const at = node.childNodes.indexOf(before)
      if (at === -1) node.childNodes.push(child)
      else node.childNodes.splice(at, 0, child)
      child.parentNode = node
      node.firstChild = node.childNodes[0] ?? null
      return child
    },
    removeChild(child: any) {
      const at = node.childNodes.indexOf(child)
      if (at !== -1) node.childNodes.splice(at, 1)
      child.parentNode = null
      node.firstChild = node.childNodes[0] ?? null
      return child
    },
    setAttribute() {},
    removeAttribute() {},
    addEventListener() {},
    removeEventListener() {},
    classList: {
      _set: new Set<string>(),
      contains(name: string) {
        return node.classList._set.has(name)
      },
      add(name: string) {
        node.classList._set.add(name)
      },
      remove(name: string) {
        node.classList._set.delete(name)
      },
      toggle(name: string, on?: boolean) {
        const next = on ?? !node.classList._set.has(name)
        if (next) node.classList._set.add(name)
        else node.classList._set.delete(name)
        return next
      },
    },
    querySelector() {
      return null
    },
    set textContent(_value: string) {},
    get textContent() {
      return ''
    },
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */
  return node as HTMLElement
}

/** The piece of `document` React's client renderer reaches for. */
function createMockDocument(mockWindow: MockWindow) {
  const container = createElementNode('div')
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const doc: any = {
    nodeType: 9,
    defaultView: mockWindow,
    documentElement: createElementNode('html'),
    body: createElementNode('body'),
    createElement: (tag: string) => {
      const node: any = createElementNode(tag)
      node.ownerDocument = doc
      return node
    },
    createElementNS: (ns: string, tag: string) => {
      const node: any = createElementNode(tag)
      node.ownerDocument = doc
      node.namespaceURI = ns
      return node
    },
    createTextNode: (text: string) => ({ nodeType: 3, nodeValue: text, data: text }),
    activeElement: null,
    title: '',
    getElementById: () => null,
    querySelector: () => null,
    addEventListener() {},
    removeEventListener() {},
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */
  ;(container as unknown as { ownerDocument: unknown }).ownerDocument = doc
  doc.body.appendChild(container)
  return { doc, container }
}

export function createMockWindow(initialHash = ''): MockWindowHandle {
  let hash = initialHash
  let changes = 0
  const listeners = new Set<() => void>()
  const storage = new Map<string, string>()

  const localStorage = {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => {
      storage.set(key, value)
    },
    removeItem: (key: string) => {
      storage.delete(key)
    },
  }

  const emitHashChange = () => {
    changes += 1
    for (const listener of [...listeners]) listener()
  }

  const window: MockWindow = {
    location: {
      get hash() {
        return hash
      },
      set hash(next: string) {
        // A browser fires the event even when the new value is the old one, so
        // this does not early-return on an unchanged write: a redundant write
        // must be observable, and that is one of the things the guard avoids.
        hash = next
        emitHashChange()
      },
      reload: () => {},
    },
    addEventListener: (type: string, listener: () => void) => {
      if (type === 'hashchange') listeners.add(listener)
    },
    removeEventListener: (type: string, listener: () => void) => {
      if (type === 'hashchange') listeners.delete(listener)
    },
    scrollTo: () => {},
    scrollY: 0,
    innerHeight: 800,
    // `react-dom` checks `element instanceof containerInfo.HTMLIFrameElement`
    // while it looks for the active element. A class that nothing is ever an
    // instance of satisfies the check without pretending to be a frame.
    HTMLIFrameElement: class {
      contentWindow = null
    } as unknown as new () => unknown,
  }

  const { container } = createMockDocument(window)

  return {
    window,
    container,
    localStorage,
    readHash: () => hash,
    hashChanges: () => changes,
    emitHashChange,
  }
}
