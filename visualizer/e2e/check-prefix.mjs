/**
 * Guard: the e2e prefix must be the prefix GitHub Pages actually serves from.
 *
 * The subpath suite (see `e2e/subpath.ts`, and IMP-053) proves the site works
 * when served under a mount point. That is only half of the deploy story. If the
 * repository is renamed -- or the Pages site moves to a custom domain -- the
 * *real* base path changes while the suite keeps passing against the old literal.
 * The tests stay green, the assertion they stand for is gone, and production
 * 404s. That is the same class of bug as the absolute `fetch('/microgpt.py')`
 * the suite was built to catch, one level up: a name in the build that no longer
 * matches the name at runtime.
 *
 * So the prefix is declared once, as a workflow-level variable
 * (`env.PAGES_BASE_PATH`), and `prefixFailures` checks it against the repository
 * name -- which is exactly what a GitHub Pages *project* site's base path is:
 * `https://<user>.github.io/<repo>/`. `actions/configure-pages` reports the same
 * value as `steps.pages.outputs.base_path` in the `deploy` job, and the optional
 * third argument lets this check compare against it when it is available.
 *
 * CLI (run from the repository root):
 *   node visualizer/e2e/check-prefix.mjs <declared-prefix> [repo-name] [pages-base-path]
 *
 * The repository name defaults to the one git knows (the last path segment of
 * `remote.origin.url`, falling back to the working directory's name), so a local
 * run needs only the declared prefix. Comparison is by *mount point* (leading and
 * trailing slashes are not significant), so `/learn-micro-gpt` and
 * `learn-micro-gpt/` agree. When a `pages-base-path` is given (the value
 * `configure-pages` reported), it must agree too -- that is the case that catches
 * a custom domain, which changes the base path without changing the repo name.
 *
 * `prefixFailures` is exported and unit-tested (`tests/check-prefix.test.ts`),
 * which also asserts this workflow coupling still exists, so the check cannot be
 * removed from `ci.yml` without a test going red.
 */
import { execFileSync } from 'node:child_process'
import { basename } from 'node:path'

/** Normalise any spelling to a comparison key: a mount point without edge slashes. */
export const key = (value) => value.replace(/^\/+|\/+$/g, '')

/** Render a mount-point key for a message: the site root, or `/name/`. */
export const describe = (value) => (key(value) === '' ? 'the site root (/)' : `/${key(value)}/`)

/**
 * The reasons the declared e2e prefix disagrees with where the site is served.
 * Empty means agreement. Pure: no IO, no process state, so it is testable.
 *
 * @param {string} declared  the prefix the e2e suite is mounted at
 * @param {string} repo      the repository name (a Pages project site is served
 *                           from `/<repo>/`)
 * @param {string} [pagesBasePath]  the base path `configure-pages` reported, when
 *                           known; the authority for a custom domain
 * @returns {string[]} human-readable failures, or `[]`
 */
export function prefixFailures(declared, repo, pagesBasePath) {
  const failures = []

  // A Pages *project* site is served from `/<repo>/`. The workflow's declared
  // prefix exists so a rename is a one-line change, but the single line it must
  // agree with is the repository name itself.
  if (key(declared) !== key(repo)) {
    failures.push(
      `the e2e prefix is ${describe(declared)} but the repository is "${repo}", so ` +
        `GitHub Pages serves this site from ${describe(repo)}. The subpath suite is ` +
        `passing against a prefix the site is no longer served from. Update ` +
        `env.PAGES_BASE_PATH in .github/workflows/ci.yml to "${repo}".`,
    )
  }

  // When `configure-pages` reported a base path, it is the authority: it accounts
  // for a custom domain or a non-default Pages configuration, which a repo-name
  // check alone cannot see.
  if (pagesBasePath && key(pagesBasePath) !== key(declared)) {
    failures.push(
      `actions/configure-pages reports the site is served from ${describe(pagesBasePath)}, ` +
        `but the e2e prefix is ${describe(declared)}. This happens when the Pages site ` +
        `has a custom domain or a non-default base path. They must be the same or the ` +
        `suite is testing a URL nobody visits.`,
    )
  }

  return failures
}

/** The repository name: from the git remote if there is one, else the directory name. */
export function repoName() {
  try {
    const url = execFileSync('git', ['config', '--get', 'remote.origin.url'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
    if (url) return basename(url.replace(/\.git$/, ''))
  } catch {
    // No git, or no remote -- fall through to the directory name.
  }
  return basename(process.cwd())
}

// --------------------------------------------------------------------------
// CLI. Only runs when this file is the entry point, so importing it (the test)
// has no side effects.
// --------------------------------------------------------------------------

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  const [declared, repoArg, pagesBasePath] = process.argv.slice(2)
  if (!declared) {
    console.error(
      'usage: node visualizer/e2e/check-prefix.mjs <declared-prefix> [repo-name] [pages-base-path]',
    )
    process.exit(2)
  }

  const repo = repoArg || repoName()
  const failures = prefixFailures(declared, repo, pagesBasePath)

  if (failures.length > 0) {
    console.error('e2e prefix does not match where the site is actually served:\n')
    for (const failure of failures) console.error(`  - ${failure}`)
    process.exit(1)
  }

  console.log(`e2e prefix ${describe(declared)} matches the served base path (repo "${repo}")`)
}
