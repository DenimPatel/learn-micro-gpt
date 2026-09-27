/**
 * Source files, imported at build time with Vite's `?raw`.
 *
 * ## Nothing here is ever fetched at runtime
 *
 * This is the single most important decision in the file, and it is a direct
 * response to how the previous visualizer broke: it did `fetch('/microgpt.py')`
 * with a `../microgpt.py` fallback. The absolute URL ignores whatever subpath
 * GitHub Pages served the site from; the relative one breaks on any route whose
 * depth differs from the file's. Both are gone as a *category*: every import here
 * is resolved by the bundler at build time and inlined into JavaScript, so there
 * is no URL left to get wrong.
 *
 * `server.fs.allow: ['..']` in `vite.config.ts` is what lets the dev server read
 * outside this directory. In production it is irrelevant.
 *
 * ## Why the language sources are lazily imported
 *
 * They are dynamic `?raw` imports rather than static ones. That does **not**
 * reintroduce a runtime fetch -- the chunks are still resolved and inlined at
 * build time, so the "no URL to get wrong" property is untouched -- but the five
 * files (about 152 KB of source together) land in their own chunks instead of the
 * initial payload, and a reader who only reads Python never downloads the C port.
 *
 * Measured effect: the initial bundle goes from 602 kB to about 450 kB, and from
 * 213 kB to roughly 160 kB gzipped.
 */

import type { Language } from './types'

/**
 * The five tracks, lazily.
 *
 * `() => import('... ?raw')` rather than a pre-resolved map, so that a static
 * analysis could not collapse them back into the main chunk.
 */
const LOADERS: Record<Language, () => Promise<string>> = {
  python: () => import('../../../reference/microgpt.py?raw').then((m) => m.default),
  c: () => import('../../../implementations/c/microgpt.c?raw').then((m) => m.default),
  go: () => import('../../../implementations/go/main.go?raw').then((m) => m.default),
  rust: () => import('../../../implementations/rust/src/lib.rs?raw').then((m) => m.default),
  typescript: () =>
    import('../../../implementations/typescript/src/index.ts?raw').then((m) => m.default),
}

const cache = new Map<Language, string>()

/** Load one source, once. The `python` case is awaited eagerly; see `useSource`. */
export async function loadSource(language: Language): Promise<string> {
  const cached = cache.get(language)
  if (cached !== undefined) return cached
  const text = await LOADERS[language]()
  cache.set(language, text)
  return text
}

export function loadedSource(language: Language): string | undefined {
  return cache.get(language)
}

export const TRACK_ORDER: readonly Language[] = ['python', 'c', 'rust', 'go', 'typescript']

/*
 * The research candidate, lazily.
 *
 * It is a *separate* file from the Rust track and it is not a `Language`: it has
 * no entry in `anchors.json`, no concept resolves into it, and nothing else on
 * the site links it. Only the research page shows it, and only to put the frozen
 * track and the tuned one next to each other.
 *
 * Kept out of `LOADERS` on purpose. `LOADERS` is keyed by `Language`, and adding
 * a sixth key would make the type lie -- it would say "these are the five
 * cross-linked tracks" while containing one that is not. A separate loader with a
 * separate cache keeps the type honest.
 */
const CANDIDATE_LOADER = () =>
  import('../../../autoresearch/candidate/src/lib.rs?raw').then((m) => m.default)

let candidateCache: string | undefined

export async function loadCandidate(): Promise<string> {
  if (candidateCache !== undefined) return candidateCache
  candidateCache = await CANDIDATE_LOADER()
  return candidateCache
}

export function loadedCandidate(): string | undefined {
  return candidateCache
}

/*
 * Per-experiment patches, one chunk each, loaded on demand.
 *
 * `eager: false` is the important half. A hundred experiments is a hundred
 * patches, and a reader who opens the page to look at the chart should not
 * download all of them to look at the best one. Same reasoning, and the same
 * "still no runtime URL to get wrong" property, as the lazy language sources
 * above: the bundler resolves every one of these at build time.
 *
 * The negation is not optional. `*.patch` also matches the best-vs-baseline
 * patch, and having it in both this map and the eager one below makes the
 * bundler warn that the module is in two chunks and resolve the ambiguity
 * itself. One owner per file.
 */
const PATCH_LOADERS = import.meta.glob(
  ['../../../autoresearch/diffs/*.patch', '!../../../autoresearch/diffs/best-vs-baseline.patch'],
  {
    query: '?raw',
    import: 'default',
  },
)

const PATCH_BEST = import.meta.glob('../../../autoresearch/diffs/best-vs-baseline.patch', {
  query: '?raw',
  eager: true,
  import: 'default',
}) as Record<string, string>

const PATCH_BEST_TEXT = Object.values(PATCH_BEST)[0] ?? ''

/** The frozen-track-to-current-candidate patch. Eager, and always present. */
export const bestVsBaselinePatch = PATCH_BEST_TEXT

/** The patch for one experiment, by run id. Rejected if that run never happened. */
export async function loadPatch(runId: string): Promise<string | null> {
  const key = `../../../autoresearch/diffs/${runId}.patch`
  const loader = PATCH_LOADERS[key]
  if (!loader) return null
  return (await loader()) as string
}

export const LANGUAGES: Record<Language, { label: string; id: string; note: string }> = {
  python: { id: 'python', label: 'Python', note: 'the reference — 199 lines, stdlib only' },
  c: { id: 'c', label: 'C', note: 'the parity track — same config, hand-written backward pass' },
  go: { id: 'go', label: 'Go', note: 'the same algorithm, statically typed' },
  rust: { id: 'rust', label: 'Rust', note: 'the same algorithm, over a typed arena' },
  typescript: {
    id: 'typescript',
    label: 'TypeScript',
    note: 'the same algorithm, in the language this page is written in',
  },
}

export const HIGHLIGHT_LANG: Record<Language, string> = {
  python: 'python',
  c: 'c',
  go: 'go',
  rust: 'rust',
  typescript: 'typescript',
}

// --- generated data (small enough to be eager) -------------------------------

import contentJson from './generated/content.json'
import indexJson from './generated/index.json'
import anchorsJson from './generated/anchors.json'
import benchmarksJson from '../../../benchmarks/results.json'
import researchJson from '../../../autoresearch/results.json'

import type {
  AttnRow,
  Benchmarks,
  Concept,
  ContentIndex,
  ProbsRow,
  ResearchResults,
  ResolvedAnchors,
  SampleRow,
  StepRow,
  TraceMeta,
} from './types'

export const content = contentJson as unknown as { concepts: Concept[] }
export const index = indexJson as unknown as ContentIndex
export const anchors = anchorsJson as unknown as ResolvedAnchors
export const benchmarks = benchmarksJson as unknown as Benchmarks

/*
 * The research track's data, eager for the same reason `benchmarks` is: it is a
 * few kilobytes, and a page that cannot render is worse than a page that is 4 kB
 * heavier. The two things that are *not* eager are the candidate's source and the
 * per-experiment patches, both of which grow without bound -- see `loadCandidate`
 * and `PATCH_LOADERS` below.
 */
export const research = researchJson as unknown as ResearchResults

export const concepts: Concept[] = content.concepts
export const conceptsById: Record<string, Concept> = Object.fromEntries(
  concepts.map((concept) => [concept.id, concept]),
)

// --- traces ------------------------------------------------------------------

import traceMeta from '../../../traces/python/micro/meta.json'
import traceSteps from '../../../traces/python/micro/steps.jsonl?raw'
import traceAttn from '../../../traces/python/micro/attn.jsonl?raw'
import traceProbs from '../../../traces/python/micro/probs.jsonl?raw'
import traceSamples from '../../../traces/python/micro/samples.jsonl?raw'

/** Parse a JSONL import. One line, one object, no splitting surprises. */
function parseJsonl<T>(raw: string): T[] {
  const rows: T[] = []
  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (trimmed) rows.push(JSON.parse(trimmed) as T)
  }
  return rows
}

export const trace = {
  meta: traceMeta as unknown as TraceMeta,
  steps: parseJsonl<StepRow>(traceSteps),
  attn: parseJsonl<AttnRow>(traceAttn),
  probs: parseJsonl<ProbsRow>(traceProbs),
  samples: parseJsonl<SampleRow>(traceSamples),
}

export const lossSeries = trace.steps.map((row) => row.loss)

/** Exponential moving average, matching `tools/trace.py` exactly. */
export function ema(values: number[], alpha = 0.05): number[] {
  const out: number[] = []
  let current = values[0] ?? 0
  for (const value of values) {
    current = alpha * value + (1 - alpha) * current
    out.push(current)
  }
  return out
}

// --- anchors -----------------------------------------------------------------

/** The concept's resolved range in one language, if it has one. */
export function anchorFor(conceptId: string, language: Language) {
  return anchors.concepts[conceptId]?.[language]
}

/** Every language this concept is anchored in, reference first. */
export function languagesFor(conceptId: string): Language[] {
  const declared = Object.keys(anchors.concepts[conceptId] ?? {}) as Language[]
  return TRACK_ORDER.filter((language) => declared.includes(language))
}
