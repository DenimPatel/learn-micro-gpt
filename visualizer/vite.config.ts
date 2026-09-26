import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))

/**
 * Vite config.
 *
 * `base: './'` is belt and braces. The single load-bearing decision is that
 * every source file is imported with `?raw` from `../implementations` and
 * `../reference` at build time (see `src/data/sources.ts`) and never fetched at
 * runtime. That is what removes the entire class of subpath 404s that broke the
 * previous visualizer on GitHub Pages: with nothing fetched, there is no URL to
 * get wrong. The relative base then covers the *one* remaining runtime URL, the
 * route.
 *
 * `server.fs.allow: ['..']` is required for the dev server to serve files from
 * outside the visualizer directory. In production this is irrelevant -- the
 * imports are resolved and inlined at build time, so `dist/` contains no
 * cross-directory references at all.
 */
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
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
    // A budget enforced in CI. The initial route must stay under 200 KB gzip,
    // excluding Pyodide, which is lazy by construction -- it is only fetched
    // when someone presses Run, and only from a pinned CDN.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        // Keep the initial route small and cacheable: the vendor split means a
        // content-only change does not invalidate React.
        manualChunks: {
          react: ['react', 'react-dom'],
          math: ['katex'],
          layout: ['dagre'],
        },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    reporters: 'default',
  },
})
