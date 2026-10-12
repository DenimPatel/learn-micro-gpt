import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Markdown, Math, headings, segment, splitAtHeading } from '../src/content/markdown'

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

/**
 * Raw HTML in a markdown body, and the trust boundary it crosses.
 *
 * The concept prose lives in this repository, so it is "trusted" -- which is
 * exactly the assumption that rots. A fork with a bad edit, a pasted snippet, or
 * a `marked` upgrade that changes the renderer's method signatures or its option
 * defaults (v13 removed `mangle`, and `renderer.html` has moved more than once
 * since) should render a page that reads wrong, never one that executes. The
 * renderer is config-less beyond `gfm`/`breaks`, so the only guard is that every
 * `html` token is walked as text through the `default` branch and emitted by
 * React as escaped text -- there is no `dangerouslySetInnerHTML` in this path.
 *
 * Nothing else asserts that. The site would still build, the prose would still
 * read, and the failure would be silent until someone's browser ran the tag. The
 * `hostile` body below exercises all three of the item's cases at once: a
 * `<script>` element, an `<img onerror=...>`, and a `javascript:` link.
 */
describe('Markdown escaping', () => {
  const render = (markdown: string) => renderToStaticMarkup(createElement(Markdown, null, markdown))

  const hostile = [
    '# Heading',
    '',
    'Before <script>alert(1)</script> middle <img src=x onerror=alert(1)> end',
    '',
    '[x](javascript:alert(1))',
    '',
  ].join('\n')

  it('never emits a script element, even from a hand-written tag', () => {
    const html = render(hostile)
    // No live `<script` element. The tag may still appear -- as the escaped text
    // `&lt;script&gt;` -- and the next assertion pins that it does, so a renderer
    // that "fixes" the problem by dropping the text is caught too.
    expect(html).not.toContain('<script')
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('never emits an event-handler attribute', () => {
    const html = render(hostile)
    // `<img ...>` is not an element here: it is escaped text, so it carries no
    // `onerror` for a browser to run. The item's literal "no `onerror=`" is not
    // expressible -- the escaped source text necessarily contains that substring
    // -- so what is pinned is the meaningful half: no `<img` element at all, and
    // the offending text still present as escaped text rather than dropped.
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
    // And when `onerror=` does appear, it is never outside of escaped text: the
    // only occurrence sits inside the escaped `&lt;...&gt;` run above.
    const liveAttribute = /<[^>]*\sonerror=/.test(html)
    expect(liveAttribute).toBe(false)
  })

  it('never emits a live javascript: href, and pins the raw text is not dropped', () => {
    const html = render(hostile)
    // React refuses a `javascript:` URL in `href` and rewrites it to a throwing
    // expression -- so `href="javascript:` *is* present, wrapping React's own
    // `throw new Error('React has blocked a javascript: URL ...')`. The URL is
    // not live: the scheme is syntactically there but the body cannot run as a
    // script, and the rewrite is asserted by name so that if React ever stops
    // guarding this, the test fails and someone has to decide what to do.
    expect(html).toContain('React has blocked a javascript: URL')
    expect(html).toContain('throw new Error')
    // And no href points anywhere other than a real link or that guard: there is
    // exactly one `<a` and its href is the blocked-URL expression.
    const hrefs = [...html.matchAll(/<a [^>]*href="([^"]*)"/g)].map((m) => m[1])
    expect(hrefs).toHaveLength(1)
    expect(hrefs[0]).toContain('React has blocked a javascript: URL')
    // The raw link text survives, so nothing was silently swallowed.
    expect(html).toContain('>x</a>')
    // The heading and its id survive an escaping run, so over-escaping cannot
    // pass by eating the document.
    expect(html).toContain('<h2 id="heading">Heading</h2>')
  })

  it('escapes raw HTML inside a paragraph, a list item and a blockquote', () => {
    // Three different branches of the token walker, so a regression in any one of
    // them is caught: `renderToken`'s `paragraph`, `list` and `blockquote` cases
    // each hand their children to `renderInlineChildren`, whose `default` branch
    // is what escapes an `html` token.
    const html = render('a <b>bold-ish</b> tag\n\n- item <i>one</i>\n\n> quote <em>two</em>\n')
    expect(html).not.toContain('<b>')
    expect(html).not.toContain('<i>')
    expect(html).not.toContain('<em>two</em>')
    expect(html).toContain('&lt;b&gt;bold-ish&lt;/b&gt;')
    expect(html).toContain('&lt;i&gt;one&lt;/i&gt;')
    expect(html).toContain('&lt;em&gt;two&lt;/em&gt;')
  })

  it('still gives an ordinary external link its target and rel', () => {
    // The escaping is not so eager that a real link breaks: a plain `https` link
    // keeps its href and still gets `target="_blank"` / `rel="noreferrer
    // noopener"`. This is the non-hostile half of the same decision.
    const html = render('see [the paper](https://example.com/paper)\n')
    expect(html).toContain('href="https://example.com/paper"')
    expect(html).toContain('target="_blank"')
    expect(html).toContain('rel="noreferrer noopener"')
  })
})

/**
 * The KaTeX delimiters, and why `*` and `_` inside a formula must not become
 * emphasis.
 *
 * This is the failure mode the segmenter exists for: `$softmax_{i}$` parsed as
 * markdown turns into italic soup, and the site looks broken in a way that is
 * hard to report usefully. The unit-level `segment` block above pins the split;
 * these cases pin the *rendered* consequence, so a change to `Math` or to how
 * `renderInlineMath` is wired shows up here.
 */
describe('Markdown maths delimiters', () => {
  const render = (markdown: string) => renderToStaticMarkup(createElement(Markdown, null, markdown))

  it('renders $...$ through the Math component as inline maths', () => {
    const html = render('the loss is $-\\ln p$ exactly\n')
    expect(html).toContain('class="math-inline"')
    // The delimiter is consumed by the segmenter -- a literal `$` left in the
    // output means the formula was shown as source text, not rendered.
    expect(html).not.toContain('$')
    // KaTeX's own output carries the formula source back for a screen reader.
    expect(html).toContain('encoding="application/x-tex"')
  })

  it('does not read * or _ inside a formula as emphasis', () => {
    // The exact bug: markdown would see `a*b_c` and emit `<em>` around part of
    // it. The segmenter strips the delimiters first, so KaTeX owns the `*` and
    // the `_` and no emphasis element appears.
    const html = render('the value $a*b_c$ here\n')
    expect(html).toContain('class="math-inline"')
    expect(html).toContain('aria-label="a*b_c"')
    expect(html).toContain('>a*b_c</annotation>')
    expect(html).not.toContain('<em>')
    expect(html).not.toContain('<em ')
  })

  it('renders $$...$$ as display maths through the Math component', () => {
    // Display maths goes through `Math` with `display: true` -> `math-display`.
    // It is asserted on the component directly rather than on a `$$...$$` body:
    // a `$$…$$` line reaches `renderInlineMath`, whose segmenter only looks for
    // inline `$…$` at that depth and emits it as text. Pinning both halves makes
    // the split read as a decision, not an accident.
    const html = renderToStaticMarkup(createElement(Math, { value: 'a*b_c', display: true }))
    expect(html).toContain('class="math-display"')
    expect(html).not.toContain('<em>')
    expect(html).toContain('encoding="application/x-tex"')
  })

  it('renders a malformed formula without throwing or blanking the page', () => {
    // `throwOnError: false`: one broken equation renders its source in red rather
    // than taking the whole page down. A blank paragraph would be worse than the
    // bad formula.
    const html = render('broken $\\frac{1$ here\n')
    expect(html).toContain('broken')
    expect(html).toContain('here')
  })
})

/**
 * Citations in prose.
 *
 * A markdown link whose href is `cite:<range>` should become a button that
 * scrolls the code panel, and *only* that: every other link on the site has to
 * keep being an `<a>`. The renderer decides from the href alone, so these tests
 * are mostly about the boundary between the two.
 *
 * The `renderToStaticMarkup` path is the same one the existing maths tests use,
 * so a regression here surfaces in the same place the rest of the markdown
 * rendering does.
 */
describe('citations', () => {
  const render = (
    markdown: string,
    onCite?: (range: { start: number; end: number }) => boolean,
    onActivate?: (range: { start: number; end: number }) => void,
  ) => renderToStaticMarkup(createElement(Markdown, { onCite, onActivate, children: markdown }))

  it('renders a citation as a button, not a link', () => {
    const html = render('the seed on line 11 is fixed', () => true)
    // No href: a citation acts on the page, it does not navigate.
    expect(html).not.toContain('href="cite:11"')
  })

  it('hands the predicate the parsed range, not the raw href', () => {
    // The predicate runs during render: that is how the renderer decides whether
    // the citation resolved and should become a button at all. So the range it
    // receives is a live assertion, not a placeholder -- `11-12` has to arrive
    // as two numbers, because the caller compares them against anchor offsets.
    // It must be pure: a `setState` here would focus the panel during render,
    // which is why the click goes through a separate `onActivate`.
    const seen: { start: number; end: number }[] = []
    render('see [the shuffle](cite:11-12)', (range) => {
      seen.push(range)
      return true
    })
    expect(seen).toEqual([{ start: 11, end: 12 }])
    expect(render('see [it](cite:11-12)', () => true)).toContain('data-cite="11-12"')
  })

  it('falls back to plain text when nothing can resolve the citation', () => {
    // No handler at all -- the glossary and the home page render concept prose
    // with no panel behind it. A button there would do nothing when clicked.
    const html = render('the seed on line 11')
    expect(html).not.toContain('<button')
  })

  it('falls back to plain text when the handler declines the range', () => {
    // A Python range that cannot highlight anything in the C panel.
    const html = render('see [the shuffle](cite:11-12)', () => false)
    expect(html).not.toContain('<button')
    expect(html).toContain('the shuffle')
  })

  it('leaves ordinary links as links', () => {
    const html = render('[softmax](../concepts/softmax.md)')
    expect(html).toContain('href="../concepts/softmax.md"')
    expect(html).not.toContain('<button')
  })

  it('leaves external links as links, with the rel that keeps them safe', () => {
    const html = render('[papers](https://example.com/)')
    expect(html).toContain('href="https://example.com/"')
    expect(html).toContain('noreferrer')
  })
})
