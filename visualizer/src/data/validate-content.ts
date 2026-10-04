/**
 * A runtime check on `generated/content.json`.
 *
 * The generated JSON is the pipeline's output, not a hand-written literal, and
 * `import contentJson from './generated/content.json'` gives it a type that
 * describes *some* object, not `Content`. Casting it through `as unknown as`
 * told the typechecker to stop looking, which is exactly the situation where a
 * pipeline regression (a renamed key, a missing chapter, a chapter spelled
 * "trainning") reaches the browser as `undefined` in a heading instead of
 * failing the build. A bad `make validate` should be a red test, not a blank
 * page.
 *
 * So the JSON is parsed by this module and the result -- not the cast -- is what
 * `sources.ts` exports. It runs at import time, before the app renders anything,
 * and it is deliberately *narrow*: it checks the fields the app depends on to
 * render and nothing else. A validator that rejected output the pipeline is
 * entitled to produce would be worse than the cast, because it would break the
 * happy path.
 *
 * No third-party schema library. `visualizer/package.json` has no zod/ajv/
 * valibot, the repo's stated policy is stdlib-only for tooling, and the rules
 * below are a page of code.
 */

import type { Block, ChapterId, Concept } from './types'
import indexJson from './generated/index.json'

/**
 * The chapters, but *not* as a literal list.
 *
 * This used to be `const CHAPTER_IDS = ['data', 'machinery', ...]`, which is
 * exactly the kind of restatement `IMP-092` is about: a chapter added to
 * `content/index.json` would appear in the sidebar and pass `make validate`,
 * while this check -- the one thing standing between a renamed chapter and
 * `undefined` in a heading -- kept rejecting it. The ids now come from the same
 * generated `index.json` the rail renders from, so the validator's idea of a
 * valid chapter and the app's are the same idea. The order is the file's order;
 * it is only ever used in the error message.
 */
const CHAPTER_IDS: readonly string[] = (
  indexJson as { chapters: { id: ChapterId }[] }
).chapters.map((chapter) => chapter.id)

const BLOCK_KINDS: readonly Block['kind'][] = [
  'prose',
  'callout',
  'note',
  'term',
  'shape',
  'trace',
  'lang',
]

/**
 * A malformed generated file.
 *
 * `path` is a JavaScript-ish access path (`concepts[3].title`) rather than a
 * JSON pointer, because the person reading it is looking at TypeScript.
 */
export class ContentValidationError extends Error {
  readonly path: string

  constructor(path: string, problem: string) {
    super(`${path}: ${problem}`)
    this.name = 'ContentValidationError'
    this.path = path
  }
}

/** The object `sources.ts` re-exports as `content`. */
export interface Content {
  concepts: Concept[]
}

function fail(path: string, problem: string): never {
  throw new ContentValidationError(path, problem)
}

/** A non-null, non-array object -- the only shape worth indexing into. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) fail(path, `expected an object, got ${describe(value)}`)
  return value
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(path, `expected an array, got ${describe(value)}`)
  return value
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') fail(path, `expected a string, got ${describe(value)}`)
  return value
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(path, `expected a finite number, got ${describe(value)}`)
  }
  return value
}

/**
 * Optional string: absent is fine, present-but-wrong is a bug.
 *
 * `undefined` and `null` are both accepted as "absent" so the check does not
 * depend on `JSON.stringify`'s treatment of an explicitly-null key.
 */
function optionalString(value: unknown, path: string): void {
  if (value === undefined || value === null) return
  requireString(value, path)
}

function optionalNumber(value: unknown, path: string): void {
  if (value === undefined || value === null) return
  requireNumber(value, path)
}

/** What the offending value was, for the message. Deliberately terse. */
function describe(value: unknown): string {
  if (value === undefined) return 'nothing'
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'an array'
  const type = typeof value
  if (type === 'string') return `a string (${JSON.stringify(value)})`
  if (type === 'number' || type === 'boolean') return `a ${type} (${String(value)})`
  return `a ${type}`
}

/** One `Shape` entry: `name` and `shape` are required, `note` is not. */
function validateShape(value: unknown, path: string): void {
  const shape = requireRecord(value, path)
  requireString(shape.name, `${path}.name`)
  requireString(shape.shape, `${path}.shape`)
  optionalString(shape.note, `${path}.note`)
}

/**
 * One block.
 *
 * `kind` is the discriminant the renderer switches on, so an unknown one is a
 * hard failure. The per-kind payloads are checked only where the renderer would
 * otherwise read a property that does not exist; optional fields are left
 * optional, matching `Block`.
 */
function validateBlock(value: unknown, path: string): void {
  const block = requireRecord(value, path)
  const kind = requireString(block.kind, `${path}.kind`)
  if (!(BLOCK_KINDS as readonly string[]).includes(kind)) {
    fail(`${path}.kind`, `unknown block kind ${JSON.stringify(kind)}`)
  }
  optionalNumber(block.line, `${path}.line`)
  optionalString(block.title, `${path}.title`)
  optionalString(block.markdown, `${path}.markdown`)
  switch (kind) {
    case 'prose':
    case 'callout':
    case 'note':
      // These three are the only blocks whose `markdown` is required by
      // `Block` -- the renderer prints it unconditionally.
      requireString(block.markdown, `${path}.markdown`)
      break
    case 'term':
      requireString(block.term, `${path}.term`)
      break
    case 'shape':
      requireString(block.name, `${path}.name`)
      break
    case 'trace':
    case 'lang': {
      // `args` is spread into the generated snippet, so an array or a string
      // here is a rendering bug rather than a style question.
      const args = requireRecord(block.args, `${path}.args`)
      for (const [key, entry] of Object.entries(args)) {
        requireString(entry, `${path}.args.${key}`)
      }
      break
    }
    default:
      break
  }
}

/** One concept: the fields the rail, the heading, the ordering and the body use. */
function validateConcept(value: unknown, path: string): void {
  const concept = requireRecord(value, path)

  requireString(concept.id, `${path}.id`)
  requireString(concept.title, `${path}.title`)
  requireString(concept.summary, `${path}.summary`)

  const chapter = requireString(concept.chapter, `${path}.chapter`)
  if (!(CHAPTER_IDS as readonly string[]).includes(chapter)) {
    fail(
      `${path}.chapter`,
      `expected one of ${CHAPTER_IDS.join(', ')}, got ${JSON.stringify(chapter)}`,
    )
  }

  requireNumber(concept.order, `${path}.order`)
  optionalNumber(concept.difficulty, `${path}.difficulty`)
  optionalString(concept.math, `${path}.math`)

  if (concept.shapes !== undefined && concept.shapes !== null) {
    requireArray(concept.shapes, `${path}.shapes`).forEach((shape, index) => {
      validateShape(shape, `${path}.shapes[${index}]`)
    })
  }

  for (const list of ['related', 'prereqs'] as const) {
    requireArray(concept[list], `${path}.${list}`).forEach((entry, index) => {
      requireString(entry, `${path}.${list}[${index}]`)
    })
  }

  // `anchors` is `Partial<Record<Language, AnchorDecl>>`: an object whose values
  // carry a `selector`. Nothing resolves an anchor in the browser (the ranges
  // come pre-resolved from `anchors.json`), but a missing selector means the
  // generator silently dropped a concept, so it is worth naming.
  const anchors = requireRecord(concept.anchors, `${path}.anchors`)
  for (const [language, decl] of Object.entries(anchors)) {
    const anchorPath = `${path}.anchors.${language}`
    const anchor = requireRecord(decl, anchorPath)
    requireString(anchor.selector, `${anchorPath}.selector`)
  }

  requireArray(concept.blocks, `${path}.blocks`).forEach((block, index) => {
    validateBlock(block, `${path}.blocks[${index}]`)
  })
}

/**
 * Parse the generated content, or throw the first problem found.
 *
 * Throwing on the first problem rather than collecting all of them is
 * deliberate: the pipeline emits the whole file in one pass, so a broken file is
 * usually one bug repeated, and the first path is the one to fix.
 */
export function validateContent(value: unknown): Content {
  const root = requireRecord(value, 'content')
  const concepts = requireArray(root.concepts, 'content.concepts')
  concepts.forEach((concept, index) => {
    validateConcept(concept, `concepts[${index}]`)
  })
  const seen = new Set<string>()
  concepts.forEach((concept, index) => {
    const id = (concept as Concept).id
    if (seen.has(id)) fail(`concepts[${index}].id`, `duplicate id ${JSON.stringify(id)}`)
    seen.add(id)
  })
  // The cast is to the type this function has just proven, on values it built
  // from the JSON -- not a cast of the JSON itself. `Concept` is asserted per
  // field above; the narrowing here is the check's conclusion, not a way around
  // the check.
  return { concepts: concepts as Concept[] }
}
