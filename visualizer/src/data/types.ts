/**
 * Data types, mirroring what `tools/render_content.py` and `tools/gen_anchors.py`
 * emit.
 *
 * These are hand-written rather than generated, deliberately: they are the
 * contract, and if the generator's shape changes without this file changing, the
 * typechecker should complain. That is the whole value of having them separate.
 */

export type Language = 'python' | 'c' | 'go' | 'rust' | 'typescript'

export type ChapterId = 'data' | 'machinery' | 'architecture' | 'training' | 'inference'

/** A concept's declared location in one language. Never a line number. */
export interface AnchorDecl {
  selector: string
  offset?: [number, number]
}

/** A selector resolved against a real source file, at build time. */
export interface ResolvedAnchor {
  start: number
  end: number
  selector: string
}

export interface ResolvedAnchors {
  $comment?: string
  generator?: string
  sources: Record<string, { path: string; lines: number; bytes: number }>
  languages: string[]
  concepts: Record<string, Partial<Record<Language, ResolvedAnchor>>>
}

export interface Shape {
  name: string
  shape: string
  note?: string
}

export type Block =
  | { kind: 'prose'; markdown: string }
  | { kind: 'callout' | 'note'; line?: number; title?: string; markdown: string }
  | { kind: 'term'; line?: number; term: string; title?: string; markdown?: string }
  | { kind: 'shape'; line?: number; name: string; title?: string; markdown?: string }
  | {
      kind: 'trace'
      line?: number
      title?: string
      markdown?: string
      args: Record<string, string>
    }
  | { kind: 'lang'; line?: number; args: Record<string, string>; markdown?: string }

export interface Concept {
  id: string
  title: string
  chapter: ChapterId
  order: number
  summary: string
  difficulty?: number
  math?: string
  shapes?: Shape[]
  related: string[]
  prereqs: string[]
  anchors: Partial<Record<Language, AnchorDecl>>
  blocks: Block[]
}

export interface ChapterEntry {
  id: string
  title: string
  summary: string
  order: number
  difficulty: number
}

export interface Chapter {
  id: ChapterId
  order: number
  title: string
  blurb: string
  concepts: ChapterEntry[]
}

export interface Edge {
  from: string
  to: string
  kind?: 'related'
}

export interface ContentIndex {
  reference: { language: string; path: string; lines: number; sha256: string; author: string }
  hyperparameters: Record<string, number | string>
  chapters: Chapter[]
  edges: Edge[]
  glossary: Record<string, string>
}

// --- traces -----------------------------------------------------------------

export interface TraceMeta {
  trace_version: number
  tool_version: string
  lang: string
  config: string
  steps: number
  seed: number
  dataset: { path: string; sha256: string; docs: number }
  reference: { path: string; sha256: string }
  hyperparameters: Record<string, number | string>
  selection: { positions: number[]; heads: number[]; steps: number[]; top_k: number }
  host: { os: string; machine: string; python: string }
  timing: { wall_seconds: number; honesty: string }
  loss_stats: {
    steps: number
    mean: number
    stdev: number
    min: number
    max: number
    first: number
    final: number
    upward_moves: number
    total_moves: number
    ema_alpha: number
    ema_first50: number
    ema_step200: number
    ema_final50: number
  }
  created: string
}

export interface StepRow {
  type: 'step'
  step: number
  loss: number
}

export interface AttnRow {
  type: 'attn'
  step: number
  pos: number
  head: number
  /** One weight per position seen so far. Length is always `pos + 1`. */
  weights: number[]
  sum: number
}

export interface ProbsRow {
  type: 'probs'
  step: number
  pos: number
  top: { token: string; p: number }[]
}

export interface SampleRow {
  type: 'sample'
  step: number
  text: string
}

export interface Benchmarks {
  $comment?: string
  runner: Runner
  configs: { id: string; label: string; hyperparameters: Record<string, number> }[]
  results: BenchResult[]
  methodology?: string
  caveats?: string[]
}

export interface Runner {
  os: string
  cpu: string
  cores: number
  memory_gb: number
  compilers: Record<string, string>
  flags: Record<string, string>
  measured: string
  pinned: boolean
}

export interface BenchResult {
  lang: string
  config: string
  steps: number
  step_ms: number
  steps_per_sec: number
  peak_rss_mb: number | null
  build_flags: string
  build_mode: string
}
