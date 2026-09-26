/**
 * Source files, imported at build time with Vite's `?raw`.
 *
 * **Nothing here is ever fetched at runtime.** That is the single most important
 * decision in this file, and it is a direct response to how the previous
 * visualizer broke: it did `fetch('/microgpt.py')` with a `../microgpt.py`
 * fallback, which is an absolute URL that ignores whatever subpath GitHub Pages
 * served the site from, and a relative URL that breaks on any route whose depth
 * differs from the file's. Both are gone as a category: the imports below are
 * resolved by the bundler at build time, inlined into the JavaScript, and there
 * is no URL left to get wrong.
 *
 * `server.fs.allow: ['..']` in vite.config.ts is what lets the dev server read
 * outside this directory. In production it is irrelevant.
 *
 * The only reason all five languages are bundled rather than lazily loaded is
 * size: together the five sources are about 130 KB of text, and gzip takes that
 * to roughly 35 KB. Splitting them per concept would add a loading state to every
 * language tab in exchange for saving little, and would reintroduce the failure
 * mode we are avoiding.
 */
import type { Language } from './types'

import microgptPy from '../../../reference/microgpt.py?raw'
import microgptC from '../../../implementations/c/microgpt.c?raw'
import microgptGo from '../../../implementations/go/main.go?raw'
import microgptRust from '../../../implementations/rust/src/lib.rs?raw'
import microgptTs from '../../../implementations/typescript/src/index.ts?raw'

import contentJson from './generated/content.json'
import indexJson from './generated/index.json'
import anchorsJson from './generated/anchors.json'

import traceMeta from '../../../traces/python/micro/meta.json'
import traceSteps from '../../../traces/python/micro/steps.jsonl?raw'
import traceAttn from '../../../traces/python/micro/attn.jsonl?raw'
import traceProbs from '../../../traces/python/micro/probs.jsonl?raw'
import traceSamples from '../../../traces/python/micro/samples.jsonl?raw'
import benchmarksJson from '../../../benchmarks/results.json'

import type {
  AttnRow,
  Concept,
  ContentIndex,
  ProbsRow,
  ResolvedAnchors,
  SampleRow,
  StepRow,
} from './types'

/** Display metadata per track. Kept here so no component hardcodes a label. */
export const LANGUAGES: Record<Language, { label: string; id: string; note: string }> = {
  python: { id: 'python', label: 'Python', note: 'the reference — 199 lines, stdlib only' },
  c: { id: 'c', label: 'C', note: 'the parity track — same config as the reference' },
  go: { id: 'go', label: 'Go', note: 'the same algorithm, statically typed' },
  rust: { id: 'rust', label: 'Rust', note: 'the same algorithm, with owned buffers' },
  typescript: {
    id: 'typescript',
    label: 'TypeScript',
    note: 'the same algorithm, in the language the UI is written in',
  },
}

export const SOURCE_TEXT: Record<Language, string> = {
  python: microgptPy,
  c: microgptC,
  go: microgptGo,
  rust: microgptRust,
  typescript: microgptTs,
}

/** Shiki language ids. The reference is the only file called `Python` here. */
export const HIGHLIGHT_LANG: Record<Language, string> = {
  python: 'python',
  c: 'c',
  go: 'go',
  rust: 'rust',
  typescript: 'typescript',
}

export const content = contentJson as unknown as { concepts: Concept[] }
export const index = indexJson as unknown as ContentIndex
export const anchors = anchorsJson as unknown as ResolvedAnchors
export const benchmarks = benchmarksJson

export const concepts: Concept[] = content.concepts
export const conceptsById: Record<string, Concept> = Object.fromEntries(
  concepts.map((concept) => [concept.id, concept]),
)

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
  meta: traceMeta,
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

/** 1-based inclusive line range -> array of source lines, 0-based. */
export function linesFor(language: Language, anchor?: { start: number; end: number }): string[] {
  const text = SOURCE_TEXT[language] ?? ''
  const all = text.split('\n')
  if (!anchor) return []
  return all.slice(Math.max(0, anchor.start - 1), Math.min(all.length, anchor.end))
}

export function anchorFor(conceptId: string, language: Language) {
  return anchors.concepts[conceptId]?.[language]
}

/** Every language this concept is anchored in, reference first. */
export function languagesFor(conceptId: string): Language[] {
  const declared = Object.keys(anchors.concepts[conceptId] ?? {}) as Language[]
  const order: Language[] = ['python', 'c', 'go', 'rust', 'typescript']
  return order.filter((language) => declared.includes(language))
}
