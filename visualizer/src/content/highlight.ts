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
  codeToHtml: (
    code: string,
    options: { lang: string; theme: string; transformers: Transformer[] },
  ) => string
}

/** The slice of a Shiki transformer this module uses; Shiki is not imported. */
interface Transformer {
  name: string
  pre: (node: { properties?: Record<string, unknown> }) => void
}

/**
 * Hand the `<pre>` no background of its own.
 *
 * Shiki writes the theme's background onto the `<pre>` as an inline style, and an
 * inline style beats any stylesheet. Shiki 4 also ignores its own `bg` option and
 * a patched `editor.background`, both of which are the obvious fixes and both of
 * which leave `background-color:#fff` in the output -- so a code panel is a white
 * box inside a tinted one, with a visible seam, and the panel's own `--surface-1`
 * never applies.
 *
 * A transformer is the structural way to do this: it rewrites the hast node Shiki
 * built, rather than string-editing its HTML afterwards, which is the one thing
 * this module exists to avoid (see the note at the top about splitting output on
 * newlines). The e2e suite asserts the result.
 */
const OWN_BACKGROUND: Transformer = {
  name: 'atlas-panel-background',
  pre(node) {
    const style = String(node.properties?.style ?? '')
    node.properties = {
      ...node.properties,
      style: style.replace(/background(?:-color)?:[^;]*(?:;|$)/g, '').trim(),
    }
  },
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
          /*
           * Vitesse, not GitHub. Two reasons, and the second is the testable one.
           *
           * The obvious choice was github-light/github-dark, and it is what this
           * used: the syntax colours read well, but the *page* around them was
           * also GitHub's greys and its blue, so the code blocks were the only
           * part of the site with any character. Vitesse is muted and warm, its
           * dark background is #121212 -- within a hair of this site's dark
           * surface -- and its comments and punctuation recede the way they should
           * in a block a reader is reading rather than scanning.
           */
          import('@shikijs/themes/vitesse-light'),
          import('@shikijs/themes/vitesse-dark'),
        ])

      /*
       * Theme interop, and it is the kind of thing that fails silently.
       *
       * In Node, `await import('@shikijs/themes/github-light')` gives a module
       * namespace with the theme under `default`. Vite pre-bundles the CJS
       * dependency and hands back the theme *directly*, with no `default`. Code
       * that only handles the first case registers a theme called `undefined`,
       * every `codeToHtml` throws `ShikiError: Theme not found`, and
       * `CodePanel` quietly falls back to unhighlighted text -- so the page looks
       * fine and nothing says otherwise.
       *
       * Hence `themeOf`, and hence the e2e test asserting the code really is
       * highlighted rather than merely present.
       */
      type ThemeRegistration = { name: string; colors: Record<string, string> }
      const themeOf = (module: unknown): ThemeRegistration => {
        const candidate = module as { default?: ThemeRegistration } & Partial<ThemeRegistration>
        const theme = (candidate.default ?? candidate) as ThemeRegistration
        if (!theme?.name || !theme?.colors) {
          throw new Error('a Shiki theme module did not expose a theme with a name')
        }
        return theme
      }

      return createHighlighterCore({
        langs: [python, c, go, rust, typescript],
        themes: [themeOf(light), themeOf(dark)],
        // The engine object, not a factory: shiki 4's `engine` option is typed
        // `Awaitable<RegexEngine>`, and `createJavaScriptRegexEngine()` returns
        // one directly. Passing a factory (the shape older examples use) is a
        // type error, and the object is what is wanted anyway.
        //
        // The JavaScript regex engine rather than oniguruma, which is the point
        // of importing it explicitly: oniguruma is a 622 kB wasm blob the bundler
        // would emit whether or not any of the five grammars needed it.
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
    // A cache hit needs no setState: `useState`'s initialiser already read it
    // above, and re-setting it here would be a synchronous state update in an
    // effect -- a cascading render for no change.
    if (cache.has(key)) return
    let cancelled = false
    void loadHighlighter().then((highlighter) => {
      const themeName = theme === 'dark' ? 'vitesse-dark' : 'vitesse-light'
      const html = highlighter.codeToHtml(code, {
        lang: language,
        theme: themeName,
        transformers: [OWN_BACKGROUND],
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
