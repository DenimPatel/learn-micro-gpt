import { describe, expect, it } from 'vitest'
import { ContentValidationError, validateContent } from '../src/data/validate-content'

/**
 * The runtime check on `generated/content.json`.
 *
 * Two jobs here. The first is that a malformed file fails with the offending
 * path in the message -- a pipeline that renames `summary` or drops a chapter
 * should be a red test naming the field, not `undefined` in a heading. The
 * second, and the one a schema library would have made easy to forget, is that
 * the *real* `content.json` still parses: this validator runs at import time, so
 * a rule that is stricter than the generator breaks the whole site.
 */

/** A minimal concept that passes, so each test can break exactly one field. */
function concept(overrides: Record<string, unknown> = {}) {
  return {
    id: 'adam',
    title: 'Adam',
    chapter: 'training',
    order: 2,
    summary: 'two running averages per parameter',
    related: ['cross-entropy-loss'],
    prereqs: [],
    anchors: {},
    blocks: [],
    ...overrides,
  }
}

function content(concepts: unknown[]) {
  return { $comment: 'GENERATED', concepts }
}

describe('validateContent', () => {
  it('accepts the real generated content.json unchanged', async () => {
    const raw = (await import('../src/data/generated/content.json')).default
    const parsed = validateContent(raw)
    expect(parsed.concepts.length).toBeGreaterThan(0)
    // Every concept survives the parse with the fields the app reads.
    for (const concept of parsed.concepts) {
      expect(typeof concept.id).toBe('string')
      expect(typeof concept.title).toBe('string')
      expect(Array.isArray(concept.blocks)).toBe(true)
    }
  })

  it('accepts a minimal valid concept', () => {
    expect(validateContent(content([concept()])).concepts).toHaveLength(1)
  })

  it('names the index and field of a missing title', () => {
    const bad = content([concept(), concept(), concept({ title: undefined })])
    expect(() => validateContent(bad)).toThrow(ContentValidationError)
    expect(() => validateContent(bad)).toThrow(/concepts\[2\]\.title/)
  })

  it('names a wrong-typed summary', () => {
    const bad = content([concept({ summary: 42 })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.summary/)
  })

  it('names a chapter outside the five ChapterId values', () => {
    const bad = content([concept({ chapter: 'trainning' })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.chapter/)
    expect(() => validateContent(bad)).toThrow(/data, machinery, architecture/)
  })

  it('names a non-numeric order', () => {
    const bad = content([concept({ order: '2' })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.order/)
  })

  it('rejects a concept whose blocks are not an array', () => {
    const bad = content([concept({ blocks: 'nope' })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.blocks/)
  })

  it('names the offending block inside a concept', () => {
    const bad = content([
      concept({ blocks: [{ kind: 'prose', markdown: 'ok' }, { kind: 'prose' }] }),
    ])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.blocks\[1\]\.markdown/)
  })

  it('requires markdown on the blocks whose type requires it', () => {
    // `prose`, `callout` and `note` all carry a required `markdown`; the renderer
    // prints it unconditionally, so a missing one is a blank page, not a style.
    for (const kind of ['prose', 'callout', 'note']) {
      const bad = content([concept({ blocks: [{ kind }] })])
      expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.blocks\[0\]\.markdown/)
    }
  })

  it('leaves markdown optional on the kinds whose type makes it optional', () => {
    // `term`, `shape`, `trace` and `lang` may omit `markdown`; requiring it would
    // reject output the generator is entitled to emit.
    for (const kind of ['term', 'shape']) {
      const bad = content([
        concept({
          blocks: [{ kind, term: 't', name: 'n', args: {} }],
        }),
      ])
      expect(() => validateContent(bad)).not.toThrow()
    }
  })

  it('rejects an unknown block kind', () => {
    const bad = content([concept({ blocks: [{ kind: 'poem', markdown: 'ok' }] })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.blocks\[0\]\.kind/)
  })

  it('names a malformed shape entry', () => {
    const bad = content([concept({ shapes: [{ name: 'm' }] })])
    expect(() => validateContent(bad)).toThrow(/concepts\[0\]\.shapes\[0\]\.shape/)
  })

  it('rejects a concept that is not an object', () => {
    expect(() => validateContent(content(['nope']))).toThrow(/concepts\[0\]/)
  })

  it('rejects a missing concepts array at the root', () => {
    expect(() => validateContent({ $comment: 'GENERATED' })).toThrow(/content\.concepts/)
  })

  it('rejects a duplicate concept id', () => {
    const bad = content([concept(), concept()])
    expect(() => validateContent(bad)).toThrow(/concepts\[1\]\.id/)
  })

  it('carries the offending path on the error object', () => {
    const error = (() => {
      try {
        validateContent(content([concept({ title: 3 })]))
        return null
      } catch (caught) {
        return caught as ContentValidationError
      }
    })()
    expect(error).toBeInstanceOf(ContentValidationError)
    expect(error?.name).toBe('ContentValidationError')
    expect(error?.path).toBe('concepts[0].title')
  })

  it('treats optional fields as optional', () => {
    // `difficulty`, `math` and `shapes` are optional on `Concept`; a validator
    // that required them would reject output the generator is entitled to emit.
    const first = validateContent(content([concept()])).concepts[0]
    expect(first?.difficulty).toBeUndefined()
  })
})
