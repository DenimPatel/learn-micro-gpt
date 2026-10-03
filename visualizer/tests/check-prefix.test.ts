import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

import { describe as describeMount, prefixFailures } from '../e2e/check-prefix.mjs'

/**
 * The e2e prefix guard.
 *
 * The subpath suite proves the site works *under a mount point*; it cannot prove
 * the mount point is the one GitHub Pages serves from. If the repository is
 * renamed, `env.PAGES_BASE_PATH` in `ci.yml` is a literal that keeps the suite
 * green against a URL nobody visits -- the deploy bug the suite exists to catch,
 * one level up. `prefixFailures` closes that, and this test pins it two ways:
 *
 *  1. the comparison itself (agreement, both mismatch directions, the optional
 *     `configure-pages` base path), and
 *  2. that `ci.yml` still declares `PAGES_BASE_PATH` *and* feeds it to the e2e
 *     run -- because a guard that is not wired into CI guards nothing, which is
 *     exactly the shape of the bug this item is about.
 */

const CI = fileURLToPath(new URL('../../.github/workflows/ci.yml', import.meta.url))
const REPO = 'learn-micro-gpt'

describe('the e2e prefix against the served base path', () => {
  it('agrees when the prefix is the repository name', () => {
    expect(prefixFailures(REPO, REPO)).toEqual([])
  })

  it('treats leading and trailing slashes as noise, because a mount point is a mount point', () => {
    expect(prefixFailures(`/${REPO}/`, REPO)).toEqual([])
    expect(prefixFailures('learn-micro-gpt/', REPO)).toEqual([])
  })

  it('fails a rename, and says which file to fix', () => {
    const failures = prefixFailures(REPO, 'some-fork')
    expect(failures).toHaveLength(1)
    // Names the drift, the consequence, and the one place to change.
    expect(failures[0]).toContain('/learn-micro-gpt/')
    expect(failures[0]).toContain('some-fork')
    expect(failures[0]).toContain('env.PAGES_BASE_PATH')
  })

  it('fails a custom domain that `configure-pages` reports, even when the repo name matches', () => {
    // A custom domain serves from `/`, not `/<repo>/`, so the repo-name check
    // alone would pass while the site is really at the root.
    const failures = prefixFailures(REPO, REPO, '/')
    expect(failures).toHaveLength(1)
    expect(failures[0]).toContain('configure-pages')
    expect(failures[0]).toContain('the site root (/)')
  })

  it('treats a missing `configure-pages` base path as "no opinion", not a failure', () => {
    // The `web` job has no Pages environment; only `deploy` does. So the third
    // argument is absent there and must not invent a failure.
    expect(prefixFailures(REPO, REPO, undefined)).toEqual([])
    expect(prefixFailures(REPO, REPO, '')).toEqual([])
  })

  it('reports the mount point readably in the message', () => {
    expect(describeMount('learn-micro-gpt')).toBe('/learn-micro-gpt/')
    expect(describeMount('/')).toBe('the site root (/)')
    expect(describeMount('')).toBe('the site root (/)')
  })
})

describe('the guard is wired into CI', () => {
  const ci = readFileSync(CI, 'utf8')

  it('declares PAGES_BASE_PATH as the single source of the prefix', () => {
    // At the workflow level (indent 2), not scoped to one job.
    expect(ci).toMatch(/^ {2}PAGES_BASE_PATH: 'learn-micro-gpt'$/m)
  })

  it('runs the guard, and instead of hard-coding the e2e prefix feeds it PAGES_BASE_PATH', () => {
    expect(ci).toContain('node visualizer/e2e/check-prefix.mjs "$PAGES_BASE_PATH"')
    expect(ci).toContain('PREFIX: ${{ env.PAGES_BASE_PATH }}')
    // The whole point: the default e2e run must not name a literal prefix.
    expect(ci).not.toMatch(/PREFIX: '\/learn-micro-gpt\/'/)
  })
})
