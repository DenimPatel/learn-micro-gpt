import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

/**
 * The `dangerouslySetInnerHTML` allow-list.
 *
 * Three live call sites are known and each one is documented at the line that
 * uses it (see the "Trust boundary" comments in `src/content/markdown.tsx` and
 * `src/components/CodePanel.tsx`):
 *
 *  - `content/markdown.tsx:1` -- KaTeX's rendered output for a frontmatter
 *    formula literal;
 *  - `components/CodePanel.tsx:2` -- Shiki's syntax-highlight HTML (`html` for
 *    the light theme, `darkHtml` for the dark one).
 *
 * None of them is fed raw markdown or anything a reader can type, and each one
 * is an injection path the moment that stops being true. A comment saying so is
 * not a guard -- the whole class of bug is a *fourth* site added without anyone
 * noticing, and no rendering test would fail for one. So this test reads the
 * source tree from disk and fails if the set of files with live call sites is
 * not exactly the allow-list below. Adding a call site means adding it here,
 * with its provenance, in the same commit.
 *
 * Counting note: comments in these files name the identifier (the trust-boundary
 * notes above do, deliberately), so the scan is per *line*: a line counts only
 * if it contains a JSX attribute (`dangerouslySetInnerHTML={`) or an object
 * property (`dangerouslySetInnerHTML:`) followed by code that is not a comment.
 */

const here = fileURLToPath(new URL('.', import.meta.url))

/** Where each file's call sites come from. Kept next to the counts on purpose. */
const ALLOW_LIST = [
  {
    file: 'src/content/markdown.tsx',
    sites: 1,
    provenance: "KaTeX's rendered output for a frontmatter formula literal (`__html: html`)",
  },
  {
    file: 'src/components/CodePanel.tsx',
    sites: 2,
    provenance: 'Shiki syntax-highlight HTML (`__html: html`, `__html: darkHtml`)',
  },
] as const

interface Occurrence {
  file: string
  line: number
  text: string
}

/**
 * A live call site: the attribute or object key is real code, not a comment.
 *
 * `//` comments are stripped before matching (there are no string or JSX text
 * literals in these files that contain `//` before a call site, and a false
 * negative here would be caught by the exact-count assertion below); `*`-style
 * block-comment lines inside JSX comments are skipped for the same reason.
 */
function liveCallSites(source: string): number {
  let count = 0
  for (const raw of source.split('\n')) {
    const line = raw.replace(/\/\/.*$/, '')
    const trimmed = line.trim()
    if (trimmed.startsWith('*') || trimmed.startsWith('/*')) continue
    if (/dangerouslySetInnerHTML\s*[=:]\s*\{/.test(line)) count += 1
  }
  return count
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return /\.(ts|tsx)$/.test(entry) ? [path] : []
  })
}

describe('the dangerouslySetInnerHTML trust boundary', () => {
  const allowListed = new Set(ALLOW_LIST.map((entry) => entry.file))
  const found: Occurrence[] = []
  for (const path of sourceFiles(join(here, '../src'))) {
    const source = readFileSync(path, 'utf8')
    source.split('\n').forEach((line, index) => {
      const stripped = line.replace(/\/\/.*$/, '')
      const trimmed = stripped.trim()
      if (trimmed.startsWith('*') || trimmed.startsWith('/*')) return
      if (/dangerouslySetInnerHTML\s*[=:]\s*\{/.test(stripped)) {
        found.push({ file: relative(join(here, '..'), path), line: index + 1, text: trimmed })
      }
    })
  }

  const describeAllowList = () =>
    ALLOW_LIST.map((entry) => `${entry.file} (${entry.sites}): ${entry.provenance}`).join('; ')

  it('has exactly the documented call sites, and no fourth one', () => {
    const actual = Object.fromEntries(ALLOW_LIST.map((entry) => [entry.file, 0])) as Record<
      string,
      number
    >
    for (const occurrence of found) {
      actual[occurrence.file] = (actual[occurrence.file] ?? 0) + 1
    }
    const expected = Object.fromEntries(ALLOW_LIST.map((entry) => [entry.file, entry.sites]))
    expect(
      actual,
      `dangerouslySetInnerHTML allow-list is ${describeAllowList()}. Found: ` +
        found.map((o) => `${o.file}:${o.line} ${o.text}`).join(' | '),
    ).toEqual(expected)
  })

  it('counts code, not comment prose', () => {
    // The three trust-boundary comments name the identifier. If the scan counted
    // prose it would report 6+, not 3 -- so this pins the counting at 3 total and
    // at least one named mention that is *not* counted.
    const markdown = readFileSync(join(here, '../src/content/markdown.tsx'), 'utf8')
    expect(liveCallSites(markdown)).toBe(1)
    expect(markdown.match(/dangerouslySetInnerHTML/g)?.length ?? 0).toBeGreaterThan(1)
  })

  it('fails loudly when a file outside the allow-list gains a call site', () => {
    // The guard's own guard: a synthetic fourth site must not be allowed through
    // by the same regex the scan uses.
    expect(findCallSites('const x = <span dangerouslySetInnerHTML={{ __html: prose }} />')).toEqual(
      ['const x = <span dangerouslySetInnerHTML={{ __html: prose }} />'],
    )
  })

  it('names every allow-listed file in the source tree', () => {
    for (const entry of ALLOW_LIST) {
      expect(allowListed.has(entry.file)).toBe(true)
      expect(liveCallSites(readFileSync(join(here, '..', entry.file), 'utf8'))).toBe(entry.sites)
    }
  })
})

/** The scan's predicate, exposed so the synthetic case can exercise it. */
function findCallSites(source: string): string[] {
  return source
    .split('\n')
    .map((line) => line.replace(/\/\/.*$/, '').trim())
    .filter(
      (line) =>
        !line.startsWith('*') &&
        !line.startsWith('/*') &&
        /dangerouslySetInnerHTML\s*[=:]\s*\{/.test(line),
    )
}
