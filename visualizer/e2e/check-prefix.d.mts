/**
 * Types for `check-prefix.mjs`.
 *
 * The guard is deliberately plain ESM (a few regexes and one `console.error`,
 * with no build step, so `make web-check`-style local runs and CI both execute
 * the same bytes). It is imported by `tests/check-prefix.test.ts`, so this
 * declaration gives that import a type instead of an implicit `any`. Keep it in
 * step with the exports in `check-prefix.mjs`.
 */

/** Normalise any spelling to a comparison key: a mount point without edge slashes. */
export declare function key(value: string): string

/** Render a mount-point key for a message: the site root, or `/name/`. */
export declare function describe(value: string): string

/** The reasons the declared e2e prefix disagrees with the served base path. */
export declare function prefixFailures(
  declared: string,
  repo: string,
  pagesBasePath?: string,
): string[]

/** The repository name: from the git remote if there is one, else the directory name. */
export declare function repoName(): string
