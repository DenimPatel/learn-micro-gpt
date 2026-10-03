/**
 * The one place the e2e subpath is decided.
 *
 * The suite runs against a production build served under a mount point, because
 * serving from `/` hides exactly the class of bug (`fetch('/microgpt.py')`,
 * `<script src="/main.js">`) that shipped this site broken on GitHub Pages. The
 * mount point used to be the constant `/learn-micro-gpt/` in three files
 * (`static-server.mjs`, `playwright.config.ts`, and the escaped-request test); a
 * single hard-coded value can only ever prove that *one* prefix works, while the
 * router's claim is that a fork at *any* subpath works.
 *
 * So the prefix is an environment variable, read here, and every consumer -- the
 * static server (`PREFIX`), the Playwright config (`baseURL` and the web-server
 * URL) and the tests -- reads it from the same place. Setting `PREFIX=some-fork`
 * re-runs the entire suite under `/some-fork/` through the *same* code path as
 * the default, which is what turns the claim into a check.
 */

/** Normalise any prefix spelling (`some-fork`, `/some-fork`, `/some-fork/`) to a mount point. */
export function normalisePrefix(value: string): string {
  return '/' + value.replace(/^\/+|\/+$/g, '') + '/'
}

/** The mount point the suite runs under. Defaults to GitHub Pages' `/learn-micro-gpt/`. */
export const PREFIX = normalisePrefix(process.env.PREFIX ?? 'learn-micro-gpt')

/** The port the static server listens on and the tests point at. */
export const PORT = Number(process.env.PORT ?? 4173)

/** The origin the static server serves from. */
export const ORIGIN = `http://127.0.0.1:${PORT}`
