import { describe, expect, it } from 'vitest'
import { segment } from '../src/content/markdown'

/**
 * The maths segmenter.
 *
 * The bug this exists to prevent: markdown plus maths, where a formula
 * containing `*` or `_` gets read as emphasis. `softmax_{i}` becomes italic
 * soup, and the site looks broken in a way that is hard to report usefully.
 */
describe('segment', () => {
  it('splits inline maths out of a sentence', () => {
    expect(segment('the loss is $-L$ exactly')).toEqual([
      { kind: 'text', value: 'the loss is ' },
      { kind: 'inline', value: '-L' },
      { kind: 'text', value: ' exactly' },
    ])
  })

  it('prefers display maths over two empty inline ones', () => {
    // `$$x$$` is one display formula, not two empty inline ones. Getting this
    // backwards renders two blank spans and drops the formula entirely.
    const parts = segment('$$x^2$$')
    expect(parts).toEqual([{ kind: 'display', value: 'x^2' }])
  })

  it('handles a formula containing markdown punctuation', () => {
    // The case that motivated the segmenter: underscores and asterisks inside a
    // formula must not be treated as emphasis.
    const parts = segment('$\\mathrm{softmax}(Q, K^*)$ is it')
    expect(parts[0]).toEqual({ kind: 'inline', value: '\\mathrm{softmax}(Q, K^*)' })
    expect(parts[1]).toEqual({ kind: 'text', value: ' is it' })
  })

  it('does not match an inline formula across a newline', () => {
    // `$` opening on one line and closing three lines later is almost certainly
    // a dollar sign in prose, not a formula.
    const parts = segment('costs $5\nand $10\nin total')
    expect(parts.every((part) => part.kind === 'text')).toBe(true)
  })

  it('returns a single text segment when there is no maths', () => {
    expect(segment('just words')).toEqual([{ kind: 'text', value: 'just words' }])
  })

  it('returns nothing for an empty string', () => {
    expect(segment('')).toEqual([])
  })

  it('handles several formulas in one block', () => {
    const parts = segment('$a$ and $b$ and $$c$$')
    expect(parts.filter((p) => p.kind === 'inline')).toHaveLength(2)
    expect(parts.filter((p) => p.kind === 'display')).toHaveLength(1)
  })

  it('does not lose text between adjacent formulas', () => {
    const parts = segment('$a$$b$')
    expect(parts).toEqual([
      { kind: 'inline', value: 'a' },
      { kind: 'inline', value: 'b' },
    ])
  })
})
