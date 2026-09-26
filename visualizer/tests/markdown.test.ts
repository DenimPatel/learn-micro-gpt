import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Markdown, headings, segment, splitAtHeading } from '../src/content/markdown'

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

/**
 * The table of contents, and the split behind the home page's hero.
 *
 * These are the two functions that read a concept's markdown before the renderer
 * does. Both are token-level rather than string-level on purpose: a regex over the
 * source would break on a `#` inside a fenced code block, and a heading renamed by
 * its author would silently move the layout.
 */
describe('headings', () => {
  it('lists level two and three headings in document order', () => {
    const found = headings('# Title\n\n## The loop\n\ntext\n\n### 1. Slicing\n')
    expect(found.map((h) => h.text)).toEqual(['The loop', '1. Slicing'])
    expect(found.map((h) => h.depth)).toEqual([2, 3])
  })

  it('slugs a heading into a usable id', () => {
    // The ids are written into the rendered page by `Markdown`; the toc has to
    // produce the same strings, or every link in the rail is dead.
    expect(headings('## 3. Softmax into weights')[0]?.id).toBe('3-softmax-into-weights')
    expect(headings('## RMSNorm')[0]?.id).toBe('rmsnorm')
  })

  it('keeps ids unique when two headings share a name', () => {
    const found = headings('## Reading the traces\n\n## Reading the traces\n')
    expect(found.map((h) => h.id)).toEqual(['reading-the-traces', 'reading-the-traces-2'])
  })

  it('ignores headings inside a fenced code block', () => {
    // A `#` comment in a Python block is not a heading, and treating it as one
    // puts a phantom entry in the rail that scrolls nowhere.
    const found = headings('## Real\n\n```python\n# a comment\n```\n')
    expect(found.map((h) => h.text)).toEqual(['Real'])
  })

  it('returns nothing for a document with no headings', () => {
    expect(headings('just a paragraph\n')).toEqual([])
  })
})

describe('splitAtHeading', () => {
  it('splits at the first heading of the requested depth', () => {
    const [intro, body] = splitAtHeading('one\n\n## Two\n\nthree\n', 2)
    expect(intro).toBe('one\n\n')
    expect(body).toBe('## Two\n\nthree\n')
  })

  it('keeps earlier headings in the first half', () => {
    const [intro] = splitAtHeading('# Title\n\nlede\n\n## Section\n', 2)
    expect(intro).toBe('# Title\n\nlede\n\n')
  })

  it('returns the whole document when there is no such heading', () => {
    const md = 'no headings here\n'
    expect(splitAtHeading(md, 2)).toEqual([md, ''])
  })

  it('loses nothing: the halves re-assemble to the input', () => {
    // Re-assembled from the tokens' own `raw` source, so this is exact rather than
    // approximately right.
    const md = '# a\n\ntext **bold** and $x$\n\n- item `code`\n\n## b\n\n> quote\n'
    const [intro, body] = splitAtHeading(md, 2)
    expect(intro + body).toBe(md)
  })
})

/**
 * The rendered output, for the one class of markdown that is easy to get wrong.
 *
 * `marked` hands a list item's whole inline content over as a single `text` token
 * that carries children. Rendering `token.text` for it -- rather than walking
 * `token.tokens` -- shows the reader the literal `**` and backtick markers, and a
 * site that explains a loss function with visible asterisks around the word
 * "average" is not shippable. It is asserted here because nothing else catches it:
 * the prose still reads, the build still passes, and it is only visible.
 *
 * `react-dom/server` rather than a DOM: this is a pure function of the token tree,
 * and a node-environment test is a thousand times faster than mounting anything.
 */
describe('Markdown', () => {
  const render = (markdown: string) => renderToStaticMarkup(createElement(Markdown, null, markdown))

  it('renders emphasis and code inside a list item', () => {
    const html = render('- **Every row sums to 1**, to floating-point precision.\n')
    expect(html).toContain('<strong>Every row sums to 1</strong>')
    expect(html).toContain('to floating-point precision')
    expect(html).not.toContain('**')
  })

  it('renders a code span inside a list item', () => {
    const html = render('- the row has exactly `pos + 1` entries\n')
    expect(html).toContain('<code>pos + 1</code>')
    expect(html).not.toContain('`')
  })

  it('renders emphasis inside a blockquote', () => {
    const html = render('> a **pull quote**\n')
    expect(html).toContain('<blockquote>')
    expect(html).toContain('<strong>pull quote</strong>')
  })

  it('keeps inline maths intact next to a code span', () => {
    const html = render('the loss is $-L$ and `L` is it\n')
    expect(html).toContain('-L')
    expect(html).toContain('<code>L</code>')
  })

  it('gives every heading an id the toc can target', () => {
    const html = render('## The loop, unrolled\n')
    expect(html).toContain('id="the-loop-unrolled"')
  })
})
