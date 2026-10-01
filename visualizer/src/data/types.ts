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
  /** When the numbers were taken. Never a substitute for `runner`. */
  measured?: string
  /** Steps per measurement. */
  steps: number
  runner: Runner
  config: { id: string; label: string; hyperparameters: Record<string, number> }
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
  wall_seconds: number
  step_ms: number
  /** Null when the measurement failed; a zero here would be a lie. */
  steps_per_sec: number | null
  /** Present only when the reference run is in the same row set. */
  relative_to_python?: number | null
  peak_rss_mb: number | null
  build_flags: string
  build_mode: string
}

/*
 * The research track's contract, mirroring `autoresearch/results.json`.
 *
 * Hand-written rather than inferred from the JSON, for the same reason every other
 * type in this file is: `results.json` is generated by `tools/autoresearch.py`, and
 * a type derived from the thing it types is not a check on it. When the generator
 * changes shape, `tsc` fails here and the fix is deliberate.
 *
 * Two fields carry meaning that a number does not, and are the reason this is a
 * contract rather than a convenience:
 *
 *  - `status` is a *verdict*, not an outcome. `crash` means no measurement
 *    exists, which is why `loss` and `steps_per_sec` are zero on those rows. A
 *    zero there is not a fast model that scored nothing; it is the absence of a
 *    measurement, and the page must not draw it as a point.
 *  - `grad_ratio` is a finite-difference check on the autograd tape. A loss
 *    number cannot tell a correct gradient from a wrong one (see
 *    `docs/KNOWN-ISSUES.md` issue 1), so this is a gate rather than a score, and
 *    a drift in it is worth seeing even while it passes.
 */

export type ResearchVerdict = 'baseline' | 'keep' | 'discard' | 'crash'

/**
 * Which implementation the loop is optimising.
 *
 * A track is a *comparator pair*, not a label: the Rust candidate's speed is a
 * ratio against the frozen Rust build, and the Go candidate's speed is a ratio
 * against the frozen Go build. The two are not comparable, so a row's track is
 * the first thing every selector here filters on -- putting a Go loss and a Rust
 * loss on one scatter would be a chart about nothing.
 */
export type ResearchTrack = 'rust' | 'go' | 'typescript' | 'c'

export interface ResearchRow {
  run_id: string
  /** The commit the experiment started from, not the commit it produced. */
  parent: string
  /** Which implementation this run optimised. Absent in rows written before the
   *  loop learned about Go and TypeScript, which are all Rust. */
  track?: ResearchTrack
  loss: number
  steps_per_sec: number
  loss_gain: number
  speed_gain: number
  grad_ratio: number
  status: ResearchVerdict
  reason: string
  description: string
  /** Loss on documents held out of training, or null for a run that predates the
   *  split. Never 0.0 for "missing" -- an absent measurement and a loss of zero are
   *  the two numbers this column exists to keep apart, and keeping them the same is
   *  how the C track once recorded 0.000000 as a 99.9% improvement. See
   *  docs/KNOWN-ISSUES.md issue 7. */
  val_loss?: number | null
  measured?: string
  has_record?: boolean
}

export interface ResearchThresholds {
  loss_material: number
  loss_tolerance: number
  speed_material: number
  speed_tolerance: number
  min_relative_improvement: number
  rule: string
  note: string
}

export interface ResearchProtocol {
  steps: number
  seed: number
  repeats: number
  trend_window: number
  run_timeout_seconds: number
  loss_axis: string
  speed_axis: string
}

export interface ResearchRunner {
  os?: string
  machine?: string
  processor?: string
  python?: string
  rustc?: string
  measured?: string
  note?: string
}

export interface ResearchProvenance {
  baseline_source: string
  baseline_sha256: string
  candidate_source: string
  candidate_sha256: string
  gradient_probe: string
  gradient_probe_sha256: string
}

/** What one track is, for the page's switcher. */
export interface ResearchTrackInfo {
  language: string
  candidate_dir: string
  source: string
  probe: string
  build: string
}

export interface ResearchModel {
  model?: string
  model_display_name?: string
  reasoning_parameter?: string
  reasoning_effort?: string
  verified?: string
  verification?: string
  pricing_note?: string
}

export interface ResearchResults {
  $comment?: string
  generator?: string
  /** "multi" now that there is more than one. */
  track?: string
  tracks: Record<ResearchTrack, ResearchTrackInfo>
  /** Per track: a baseline digest is a claim about one comparator. */
  provenance_of: Record<ResearchTrack, ResearchProvenance>
  protocol: ResearchProtocol
  thresholds: ResearchThresholds
  objective: { kind: string; axes: string[]; decided_by?: string }
  model?: ResearchModel
  runner?: ResearchRunner
  dataset_sha256?: string
  /** Per track, and null for a track that has not been seeded yet. */
  baselines: Record<ResearchTrack, ResearchRow | null>
  bests: Record<ResearchTrack, ResearchRow | null>
  /** Why a best is missing, per track, when one is. "" when it is not.
   *
   *  A track whose best panel is empty looks like a broken page. This says
   *  whether the honest answer is "nothing kept yet" or "every recorded keep
   *  predates the current protocol, or has been rolled back" -- which is the
   *  state after a protocol change, and a state that comes with a way out. */
  best_status?: Record<ResearchTrack, string>
  counts: { experiments: number; keep: number; discard: number; crash: number }
  /** Per track. Run ids on the frontier: nothing measured is both faster and
   *  lower-loss *for that track's comparator*. */
  pareto: Record<ResearchTrack, string[]>
  runs: ResearchRow[]
  caveats: string[]
}
