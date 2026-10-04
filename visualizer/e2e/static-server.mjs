/**
 * A static file server that mounts `dist/` under a prefix.
 *
 * The point of the prefix is that it is *not* the root. GitHub Pages serves a
 * project site from `https://<user>.github.io/<repo>/`, and the previous
 * visualizer's absolute URLs broke there while working perfectly at `/`. A test
 * server that serves from the root cannot reproduce that, so this one does not:
 * every path is served under `/learn-micro-gpt/`, and anything outside that
 * prefix 404s, exactly as Pages would.
 *
 * 60 lines, no dependency. `vite preview` was the obvious alternative and is
 * wrong here: it strips the base path instead of serving under it.
 *
 * The prefix is a parameter (`PREFIX`), not a constant, because "a fork at any
 * subpath works" is the property the router claims and a single hard-coded
 * prefix cannot test it. `base: './'` makes the built asset URLs relative, so
 * the *same* `dist/` is correct at every prefix -- which means this suite can be
 * run again under a fork-like prefix (`/some-fork/`) without rebuilding, and
 * proves the claim rather than assuming it. The 404-outside-the-prefix behaviour
 * is unchanged: it is correct, and the parameterisation is not a licence to
 * weaken it.
 *
 * ## The under-prefix miss serves `404.html` (GitHub Pages' model)
 *
 * GitHub Pages serves the built `dist/404.html` with an HTTP 404 status for any
 * path it cannot resolve. Before this, the miss branch wrote plain text
 * (`not found: <path>`) and never read `404.html` -- the file the build emits was
 * emitted, compared in CI, and never served by anything. The two behaviours now
 * agree with production: the miss returns 404.html's bytes, status 404, HTML.
 *
 * Serving the document is all this can do, and all Pages does. It does **not**
 * guarantee the app boots: `base: './'` makes the shell's `./assets/...` resolve
 * against the *directory* of the served URL, so a one-segment miss
 * (`/repo/nope` -> `/repo/`) boots, while the multi-segment shape a history
 * router would produce (`/repo/nope/softmax` -> `/repo/nope/assets/...`) serves
 * the shell and then mounts nothing. That is a property of relative asset URLs,
 * not of this branch, and it is pinned in `e2e/not-found.spec.ts` so nobody
 * reads "the fallback is served" as "the fallback works".
 *
 * The *outside*-the-prefix* branch stays plain text on purpose (above): a request
 * that is not under the mount point is a different condition from a path that is
 * under it but missing -- it does not reach this project site at all -- so the
 * plain-text refusal is kept. Only the under-prefix miss changed.
 *
 * If `dist/404.html` is absent -- a dist built before the fallback existed -- the
 * miss falls back to the old plain-text 404 rather than crashing, so the server
 * remains usable against any dist.
 */
import { createReadStream, existsSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { extname, join, normalize, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('../dist', import.meta.url)))

/**
 * Normalise any `PREFIX` value to a leading-and-trailing-slash mount point.
 *
 * Kept in lock-step with `e2e/subpath.ts`, which the config and the tests read.
 * It is duplicated rather than imported because this file is plain Node (no TS
 * loader) and its whole point is to have no build step; the rule is one line and
 * the two copies are asserted equal by the suite running green at a custom
 * prefix.
 */
const normalisePrefix = (value) => '/' + value.replace(/^\/+|\/+$/g, '') + '/'

const PREFIX = normalisePrefix(process.env.PREFIX ?? 'learn-micro-gpt')
const PORT = Number(process.env.PORT ?? 4173)

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.map': 'application/json; charset=utf-8',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
}

const server = createServer((request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  let pathname = decodeURIComponent(url.pathname)

  if (!pathname.startsWith(PREFIX)) {
    // Deliberate: a request outside the mount point is a 404, not a fallback.
    // A server that quietly served from the root would hide exactly the bug this
    // suite exists to find.
    response.writeHead(404, { 'content-type': 'text/plain' })
    response.end('not found (outside the mount point)')
    return
  }

  pathname = pathname.slice(PREFIX.length)
  if (pathname === '' || pathname.endsWith('/')) pathname += 'index.html'

  // `normalize` collapses any `..` before the join, so a request cannot climb
  // out of dist/. The join is then re-checked, because normalize alone has had
  // bypasses historically.
  const target = join(ROOT, normalize(pathname).replace(/^(\.\.[/\\])+/, ''))
  if (!target.startsWith(ROOT) || !existsSync(target) || !statSync(target).isFile()) {
    // Under the prefix but missing: GitHub Pages serves `404.html` here, with a
    // 404 status, so a deep link into a history-routed build still boots the app
    // shell. `404.html` is emitted by `vite.config.ts` as a copy of `index.html`.
    const fallback = join(ROOT, '404.html')
    if (existsSync(fallback) && statSync(fallback).isFile()) {
      response.writeHead(404, {
        'content-type': 'text/html; charset=utf-8',
        'cache-control': 'no-store',
      })
      createReadStream(fallback).pipe(response)
      return
    }
    // No fallback in this build: keep the plain-text 404 rather than crash.
    response.writeHead(404, { 'content-type': 'text/plain' })
    response.end(`not found: ${pathname}`)
    return
  }

  response.writeHead(200, {
    'content-type': TYPES[extname(target)] ?? 'application/octet-stream',
    'cache-control': 'no-store',
  })
  createReadStream(target).pipe(response)
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}${PREFIX}`)
})
