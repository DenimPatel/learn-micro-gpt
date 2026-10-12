/**
 * Prose citations: `[the seed on line 11](cite:11)`.
 *
 * A concept's prose quotes line ranges ("the seed on line 11 makes this
 * reproducible"), and before this those numbers were inert text. The reader had
 * to find the line themselves in the code panel, which is exactly the gap
 * between a claim and its source that this site exists to close. A citation is
 * an ordinary markdown link whose href uses a `cite:` scheme, so no new
 * directive had to be taught to the content parser: `marked` already tokenises
 * it as a link and hands the href straight through.
 *
 * ## Why the scheme is parsed in one place
 *
 * The same grammar has to be understood by two runtimes that cannot share code:
 * the React renderer that turns a citation into a control, and the Python
 * content pipeline that validates the range against the concept's resolved
 * anchor (`tools/validate_concepts.py`). Neither can import the other, so the
 * grammar is deliberately small enough to state once here and mirror there:
 * `cite:<start>` or `cite:<start>-<end>`, 1-based, inclusive at both ends, and
 * `start <= end`. The TS side is the one the reader's click goes through, so it
 * is the one that gets the unit tests.
 */

/** The scheme prefix that marks a markdown link as a citation. */
export const CITE_SCHEME = 'cite:'

/** A 1-based inclusive line range, as the `CodePanel` takes it. */
export interface CiteRange {
  start: number
  end: number
}

/**
 * `cite:11` and `cite:11-12` -> a range; anything else -> `undefined`.
 *
 * Returns `undefined` for every form the grammar does not cover -- a missing
 * scheme, a non-numeric span, an inverted range, a bare `cite:`. The renderer
 * needs a total function here: an href it does not recognise must keep its
 * existing behaviour (a plain link) rather than become a control that scrolls
 * somewhere arbitrary. And because the Python side rejects exactly these forms
 * at build time, a `undefined` from here means either hand-written markup or a
 * file that never went through the pipeline -- neither of which should silently
 * render as a citation.
 *
 * Whitespace is not trimmed: `cite: 11` is a typo, and a parser that quietly
 * accepted it would be the kind of lenient that stops catching pasted formats.
 */
export function parseCite(href: string | undefined | null): CiteRange | undefined {
  if (typeof href !== 'string' || !href.startsWith(CITE_SCHEME)) return undefined
  const body = href.slice(CITE_SCHEME.length)

  const single = /^(\d+)$/.exec(body)
  if (single) {
    const line = Number(single[1])
    return line >= 1 ? { start: line, end: line } : undefined
  }

  const range = /^(\d+)-(\d+)$/.exec(body)
  if (range) {
    const start = Number(range[1])
    const end = Number(range[2])
    // 1-based: line 0 does not exist. Inverted ranges are rejected rather than
    // normalised, for the same reason the Python gate rejects them -- silently
    // swapping the ends would make `cite:20-11` resolve to something the author
    // did not write, and the whole point of the feature is that the number in
    // the prose is the number that gets highlighted.
    if (start < 1 || end < start) return undefined
    return { start, end }
  }

  return undefined
}

/**
 * Whether a range sits inside the anchor window a citation can reach.
 *
 * Anchors are resolved to *whole-concept* ranges (`anchorFor(conceptId,
 * primary)`), so "inside the concept's own anchored range" is the strongest
 * containment guarantee available: the panel a citation targets crops to that
 * window, and a range outside it would highlight a line the reader cannot see.
 * That is what makes this check worth having rather than a formality.
 *
 * The window is the **primary** language's -- the panel the prose sits next to
 * (`TRACK_ORDER[0]` that the concept declares, matching `primary_language` in
 * `tools/selectors.py`). Checking every declared language instead would be the
 * easy mistake and the wrong one: `data-loading` anchors both Python (13-20)
 * and C (397-484), the two windows never overlap, and a citation valid in the C
 * window would pass such a check while the visible Python panel scrolls nowhere.
 *
 * `tools/validate_concepts.py` fails the build on the same condition against the
 * same language; this copy exists so the renderer can decide *not* to render a
 * control for a range the panel could never show, which keeps the failure from
 * being a dead click in the browser as well as a red build.
 */
export function withinAnchor(
  range: CiteRange,
  anchor: { start: number; end: number } | undefined,
): boolean {
  if (!anchor) return false
  return range.start >= anchor.start && range.end <= anchor.end
}
