/**
 * Syntax highlighting, at runtime, code-split.
 *
 * The previous visualizer used Prism at runtime and then split its output on
 * `\n`, one line at a time -- which cut a multi-line token in half. The reference's
 * opening docstring is exactly such a token, so the very first code block on the
 * site rendered mangled. The lesson, recorded in `docs/ARCHITECTURE.md`, is to
 * highlight *whole tokens* and never re-split the result afterwards.
 *
 * This component guarantees that structurally rather than by discipline: Shiki's
 * HTML is rendered verbatim inside one `<pre>`, and the line numbers live in a
 * separate gutter column. Nothing splits Shiki's output, so nothing can cut a
 * token in half. The anchor is also what makes highlighting the *focused* range
 * work without reformatting: see `CodePanel`.
 *
 * Shiki is loaded lazily. A visitor who never opens a code panel never
 * downloads its grammars, so the initial bundle carries only this module's
 * types.
 */

import { useEffect, useRef, useState } from 'react'

/**
 * Shiki, imported *finely*.
 *
 * The obvious `import('shiki')` bundles every language Shiki ships. The first
 * build of this site did that, and the output contained 785 kB of C++ and 790 kB
 * of Emacs Lisp for a site that only ever shows Python, C, Go, Rust, and
 * TypeScript -- grammars that are downloaded in a lazy chunk, so they are not in
 * the initial payload, but they are still in the deployed site and in anyone's
 * bandwidth bill for the first code panel they open.
 *
 * `shiki/core` plus five explicit grammar imports and two themes is the documented
 * way to avoid that, and the chunk drops from megabytes to about 100 kB.
 *
 * The type is declared by hand rather than imported, for the same reason the
 * import is dynamic: keeping this module free of a *value* dependency on Shiki
 * means the initial bundle does not reference it at all.
 */
interface Highlighter {
  codeToHtml: (code: string, options: { lang: string; theme: string }) => string
}

let highlighterPromise: Promise<Highlighter> | null = null

/**
 * One highlighter for the whole session, carrying both themes.
 *
 * A module-level promise rather than a React context: the grammars are stateless,
 * there is exactly one of them, and a provider would mean threading a context
 * through every code panel for no benefit. Both themes are created once, so
 * switching theme is a change of argument rather than a second highlighter.
 */
function loadHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const [{ createHighlighterCore }, engine, python, c, go, rust, typescript, light, dark] =
        await Promise.all([
          import('shiki/core'),
          // The JavaScript regex engine, imported on its own.
          // `createHighlighterCore` needs an engine explicitly;
          // `createHighlighter` would have supplied one, at the cost of also
          // pulling in every language. This is the engine to use precisely
          // *because* it does not need the oniguruma wasm blob -- which the
          // bundler would otherwise emit as a 600 kB `wasm-*.js` chunk.
          import('shiki/engine/javascript'),
          import('@shikijs/langs/python'),
          import('@shikijs/langs/c'),
          import('@shikijs/langs/go'),
          import('@shikijs/langs/rust'),
          import('@shikijs/langs/typescript'),
          import('@shikijs/themes/github-light'),
          import('@shikijs/themes/github-dark'),
        ])

      // `createdBundledHighlighter`'s return type is narrower than the shape used
      // here, so it is adapted rather than the caller being widened to accept
      // anything Shiki might return.
      // No `loadWasm`: the five grammars here are all TextMate regex grammars
      // with no embedded wasm, so the TextMate engine alone is enough. Passing
      // `loadWasm` anyway is accepted by the runtime and rejected by the type,
      // which is a useful signal that it is not needed.
      return createHighlighterCore({
        langs: [python, c, go, rust, typescript],
        themes: [light, dark],
        // The engine object, not a factory: shiki 4's `engine` option is typed
        // `Awaitable<RegexEngine>`, and `createJavaScriptRegexEngine()` returns
        // one directly. Passing a factory (the shape older examples use) is a type
        // error, and the object is what is wanted anyway.
        engine: engine.createJavaScriptRegexEngine(),
      }) as unknown as Highlighter
    })()
  }
  return highlighterPromise
}

const cache = new Map<string, string>()

export function useHighlight(
  code: string,
  language: string,
  theme: 'light' | 'dark',
): { html: string | null; ready: boolean } {
  const key = `${theme}:${language}:${code.length}:${code.slice(0, 64)}`
  const [state, setState] = useState<{ html: string | null; ready: boolean }>(() => {
    const cached = cache.get(key)
    return { html: cached ?? null, ready: cached !== undefined }
  })
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    const cached = cache.get(key)
    if (cached !== undefined) {
      setState({ html: cached, ready: true })
      return
    }
    let cancelled = false
    void loadHighlighter().then((highlighter) => {
      const themeName = theme === 'dark' ? 'github-dark' : 'github-light'
      const html = highlighter.codeToHtml(code, {
        lang: language,
        theme: themeName,
      })
      cache.set(key, html)
      if (!cancelled && mounted.current) setState({ html, ready: true })
    })
    return () => {
      cancelled = true
    }
  }, [code, language, theme, key])

  return state
}
