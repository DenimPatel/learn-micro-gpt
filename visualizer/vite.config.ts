import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'
import { copyFileSync, existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))

/**
 * Emit `dist/404.html` as a copy of `index.html`.
 *
 * GitHub Pages serves `404.html` for any path it cannot resolve, so a copy means
 * the site keeps working if it is ever moved to history routing: a deep link
 * loads the app, which reads `location.hash` and shows the right route.
 *
 * It costs one extra file and buys the option to drop the hash. The router still
 * uses the hash today, because that needs no server configuration at all -- see
 * `src/app/router.ts` for why that is the right default.
 *
 * A post-build copy would be simpler, but it would be invisible to anything
 * reading the bundle, and a plugin that says what it does is a plugin you can
 * reason about.
 */
function copyIndexAs404(): Plugin {
  return {
    name: 'copy-index-as-404',
    apply: 'build',
    /*
     * `closeBundle` with a read from disk, rather than `generateBundle` and
     * `this.getFile`. The plugin context in Vite 8 / Rolldown does not expose
     * `getFile`, and the alternative -- inspecting the bundle object in
     * `generateBundle` -- couples this to a bundle's internal shape. Reading the
     * emitted file is one line longer and depends on nothing.
     */
    closeBundle() {
      const outDir = resolve(here, 'dist')
      const source = join(outDir, 'index.html')
      if (existsSync(source)) {
        copyFileSync(source, join(outDir, '404.html'))
      }
    },
  }
}

/**
 * Vite config.
 *
 * `base: './'` is belt and braces. The single load-bearing decision is that
 * every source file is imported with `?raw` from `../implementations` and
 * `../reference` at build time (see `src/data/sources.ts`) and never fetched at
 * runtime. That removes the entire class of subpath 404s that broke the previous
 * visualizer on GitHub Pages: with nothing fetched, there is no URL to get wrong.
 * The relative base then covers the *one* remaining runtime URL, the route.
 *
 * `server.fs.allow: ['..']` is required for the dev server to serve files from
 * outside the visualizer directory. In production it is irrelevant -- the imports
 * are resolved and inlined at build time, so `dist/` contains no cross-directory
 * references at all.
 */
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), copyIndexAs404()],
  server: {
    port: 5173,
    fs: {
      // The concept code panels and the anchors both come from outside this
      // directory. Dev only: production inlines them.
      allow: [resolve(here, '..')],
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    /*
     * Measured, not aspirational: the initial route is about 180 kB gzipped
     * (React 68, KaTeX 77, the app the rest). The five language sources are
     * code-split into their own chunks and are not in that figure.
     *
     * The ceiling sits above the measured value so a regression is visible
     * without firing on every build.
     *
     * Pyodide is excluded by construction: it is fetched from a pinned CDN only
     * when someone presses Run, and it is the largest single thing this site can
     * optionally download.
     */
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        /*
         * Keep the initial route cacheable across content edits: a change to a
         * concept should not invalidate the React chunk.
         *
         * A function rather than an object map, because Rollup 4 (which Vite 8
         * uses) removed the `Record<string, string[]>` overload in favour of
         * `ManualChunksFunction` for exactly this case.
         */
        manualChunks(id: string) {
          if (id.includes('node_modules/react') || id.includes('node_modules/scheduler')) {
            return 'react'
          }
          if (id.includes('node_modules/katex')) return 'math'
          if (id.includes('node_modules/dagre')) return 'layout'
          // Shiki is loaded on demand by `src/content/highlight.ts`; leaving it
          // out of any manual chunk lets the dynamic import own it.
          return undefined
        },
      },
    },
  },
})
